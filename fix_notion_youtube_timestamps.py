"""Varre a coluna ``youtube_link`` no Notion, identifica os links SEM marcador de
tempo (``&t=<segundos>``) e conserta-os colocando o timestamp de início do
julgamento daquele processo.

Fontes do timestamp correto (nesta ordem de preferência), ambas baratas:
1. Artefatos locais das execuções (``source_start_seconds`` em preview rows);
2. Capítulos da DESCRIÇÃO do vídeo no YouTube (formato ``HH:MM:SS <classe> <numero>``),
   recuperados por (video_id, número canônico).

O conserto é seguro: só aplica quando o número canônico do processo casa exatamente
com um capítulo/artefato. Registros sem casamento são deixados como estão e contados
à parte.

Uso:
    python fix_notion_youtube_timestamps.py                 # dry-run (relatório)
    python fix_notion_youtube_timestamps.py --apply         # aplica os consertos
    python fix_notion_youtube_timestamps.py --no-youtube    # só artefatos locais
    python fix_notion_youtube_timestamps.py --max-videos 15 # piloto: limita vídeos buscados
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import (
    build_timestamped_youtube_link,
    canonicalize_numero_processo,
    extract_youtube_video_id,
)
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NotionSessoesClient,
    parse_youtube_chapter_timestamps,
)


LOGGER = logging.getLogger("fix_notion_youtube_timestamps")
ARTIFACT_ROOT = Path("artifacts") / "notion_youtube_timestamps"
APPLY_SLEEP_SECONDS = 0.2
FETCH_SLEEP_SECONDS = 0.3
PREVIEW_GLOBS = (
    "artifacts/**/04_preview_rows.json",
    "artifacts/**/04b_enriched_preview_rows.json",
)
_DESC_RE_PRIMARY = re.compile(r'"shortDescription":"(.*?)","isCrawlable"', re.DOTALL)
_DESC_RE_FALLBACK = re.compile(r'"shortDescription":"(.*?)"', re.DOTALL)


@dataclass
class LinkFix:
    page_id: str
    page_url: str
    numero_processo: str
    old: str
    new: str
    source: str
    status: str = "would_update"
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_url": self.page_url,
            "numero_processo": self.numero_processo,
            "old": self.old,
            "new": self.new,
            "source": self.source,
            "status": self.status,
            "error": self.error,
        }


def link_has_timestamp(url: str) -> bool:
    parsed = urlparse(url or "")
    query = parse_qs(parsed.query)
    if query.get("t") or query.get("start"):
        return True
    return parsed.fragment.startswith("t=")


def build_artifact_recovery_map() -> dict[tuple[str, str], int]:
    recovery: dict[tuple[str, str], int] = {}
    files: list[str] = []
    for pattern in PREVIEW_GLOBS:
        files.extend(glob.glob(pattern, recursive=True))
    for file_path in sorted(set(files)):
        try:
            data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            video_id = extract_youtube_video_id(str(row.get("youtube_link", "") or ""))
            numero = canonicalize_numero_processo(str(row.get("numero_processo", "") or ""))
            start = row.get("source_start_seconds", -1)
            if video_id and numero and isinstance(start, int) and start >= 0:
                recovery.setdefault((video_id, numero), start)
    return recovery


def fetch_video_description(video_id: str, session: requests.Session) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    for attempt in range(1, 4):
        try:
            response = session.get(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "pt-BR,pt;q=0.9"},
                timeout=30,
            )
            text = response.text
            match = _DESC_RE_PRIMARY.search(text) or _DESC_RE_FALLBACK.search(text)
            if not match:
                return ""
            return match.group(1).encode("utf-8").decode("unicode_escape", errors="replace")
        except Exception as exc:
            if attempt == 3:
                LOGGER.warning("Falha ao buscar descrição de %s: %s", video_id, exc)
                return ""
            time.sleep(1.0 * attempt)
    return ""


def build_fixes(
    client: NotionSessoesClient,
    *,
    use_youtube: bool,
    max_videos: int,
) -> tuple[list[LinkFix], dict[str, int]]:
    schema = client.fetch_schema()
    artifact_map = build_artifact_recovery_map()
    pages = client.query_data_source()

    pending: list[dict[str, str]] = []
    for page in pages:
        link = client._extract_property_text(page, schema, "youtube_link")
        if not link or link_has_timestamp(link):
            continue
        pending.append(
            {
                "page_id": str(page.get("id", "")),
                "page_url": str(page.get("url", "")),
                "link": link,
                "video_id": extract_youtube_video_id(link),
                "numero_raw": client._extract_property_text(page, schema, "numero_processo"),
                "numero": canonicalize_numero_processo(client._extract_property_text(page, schema, "numero_processo")),
            }
        )

    fixes: list[LinkFix] = []
    resolved_ids: set[str] = set()

    # 1) artefatos locais
    for entry in pending:
        start = artifact_map.get((entry["video_id"], entry["numero"]))
        if start is None:
            continue
        new_link = build_timestamped_youtube_link(entry["link"], start)
        if new_link != entry["link"]:
            fixes.append(_make_fix(entry, new_link, "artifact"))
            resolved_ids.add(entry["page_id"])

    # 2) descrições do YouTube (capítulos)
    fetched_videos = 0
    if use_youtube:
        chapter_cache: dict[str, dict[str, int]] = {}
        session = requests.Session()
        unresolved_videos = sorted(
            {entry["video_id"] for entry in pending if entry["page_id"] not in resolved_ids and entry["video_id"]}
        )
        for video_id in unresolved_videos:
            if max_videos and fetched_videos >= max_videos:
                break
            description = fetch_video_description(video_id, session)
            fetched_videos += 1
            chapter_cache[video_id] = parse_youtube_chapter_timestamps(description) if description else {}
            if FETCH_SLEEP_SECONDS:
                time.sleep(FETCH_SLEEP_SECONDS)
            if fetched_videos % 20 == 0:
                LOGGER.info("Descrições buscadas: %s/%s vídeos", fetched_videos, len(unresolved_videos))
        for entry in pending:
            if entry["page_id"] in resolved_ids:
                continue
            start = chapter_cache.get(entry["video_id"], {}).get(entry["numero"])
            if start is None:
                continue
            new_link = build_timestamped_youtube_link(entry["link"], start)
            if new_link != entry["link"]:
                fixes.append(_make_fix(entry, new_link, "youtube_description"))
                resolved_ids.add(entry["page_id"])

    stats = {
        "missing_timestamp": len(pending),
        "resolved": len(fixes),
        "from_artifact": sum(1 for f in fixes if f.source == "artifact"),
        "from_youtube": sum(1 for f in fixes if f.source == "youtube_description"),
        "unresolved": len(pending) - len(fixes),
        "videos_fetched": fetched_videos,
    }
    return fixes, stats


def _make_fix(entry: dict[str, str], new_link: str, source: str) -> LinkFix:
    return LinkFix(
        page_id=entry["page_id"],
        page_url=entry["page_url"],
        numero_processo=entry["numero_raw"],
        old=entry["link"],
        new=new_link,
        source=source,
    )


def apply_fixes(client: NotionSessoesClient, fixes: list[LinkFix]) -> None:
    for index, fix in enumerate(fixes, start=1):
        try:
            notion_request_with_retry(
                client,
                "PATCH",
                f"/pages/{fix.page_id}",
                json={"properties": {"youtube_link": {"url": fix.new}}},
            )
            fix.status = "updated"
        except Exception as exc:
            fix.status = "failed"
            fix.error = str(exc)
        if APPLY_SLEEP_SECONDS:
            time.sleep(APPLY_SLEEP_SECONDS)
        if index % 25 == 0:
            LOGGER.info("Links aplicados: %s/%s", index, len(fixes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conserta youtube_link sem marcador de tempo (artefatos + descrições do YouTube).")
    parser.add_argument("--apply", action="store_true", help="Aplica os consertos (padrão: apenas dry-run).")
    parser.add_argument("--no-youtube", action="store_true", help="Usa apenas artefatos locais (não busca descrições).")
    parser.add_argument("--max-videos", type=int, default=0, help="Limita quantos vídeos buscar (0 = todos). Útil para piloto.")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    api_key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY/NOTION_TOKEN não encontrado.")
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    client = NotionSessoesClient(api_key=api_key, data_source_id=args.data_source_id)
    fixes, stats = build_fixes(client, use_youtube=not args.no_youtube, max_videos=args.max_videos)
    LOGGER.info(
        "Sem timestamp: %s | resolvidos: %s (artefato=%s, youtube=%s) | sem casamento: %s | vídeos buscados: %s",
        stats["missing_timestamp"],
        stats["resolved"],
        stats["from_artifact"],
        stats["from_youtube"],
        stats["unresolved"],
        stats["videos_fetched"],
    )
    if args.apply and fixes:
        apply_fixes(client, fixes)
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        **stats,
        "applied": sum(1 for f in fixes if f.status == "updated"),
        "failed": sum(1 for f in fixes if f.status == "failed"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "fixes.json").write_text(
        json.dumps([f.as_dict() for f in fixes], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Resumo: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    LOGGER.info("Relatórios em %s", artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
