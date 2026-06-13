"""Varre a coluna ``data_sessao`` no Notion e conserta as datas que divergem da data
oficial da sessão — a que consta no TÍTULO do vídeo do TSE (ex.: 'Sessão Plenária - 29
de Fevereiro de 2024'). O título publicado pelo próprio tribunal é a fonte AUTORITATIVA
da data; o modelo ocasionalmente alucina um valor default (ex.: '2024-05-21'), que então
fica gravado em todas as linhas daquela sessão e some das visões por data.

Estratégia (segura):
- agrupa as páginas por vídeo (``youtube_link`` -> video_id);
- busca o título de cada vídeo uma única vez e infere a data da sessão;
- só conserta quando consegue inferir uma data VÁLIDA do título e ela difere da atual.
  Vídeos cujo título não traz data parseável (eventos atípicos) são deixados como estão
  e contados à parte.

Uso:
    python fix_notion_data_sessao_from_title.py                 # dry-run (relatório)
    python fix_notion_data_sessao_from_title.py --apply         # aplica os consertos
    python fix_notion_data_sessao_from_title.py --max-videos 15 # piloto: limita vídeos
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import (
    extract_youtube_video_id,
    infer_session_date_from_video_title,
)
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NotionSessoesClient,
    fetch_youtube_title,
)


LOGGER = logging.getLogger("fix_notion_data_sessao_from_title")
ARTIFACT_ROOT = Path("artifacts") / "notion_data_sessao_title"
APPLY_SLEEP_SECONDS = 0.2
FETCH_WORKERS = 8


@dataclass
class DateFix:
    page_id: str
    page_url: str
    numero_processo: str
    video_id: str
    title: str
    old: str
    new: str
    status: str = "would_update"
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_url": self.page_url,
            "numero_processo": self.numero_processo,
            "video_id": self.video_id,
            "title": self.title,
            "old": self.old,
            "new": self.new,
            "status": self.status,
            "error": self.error,
        }


def _fetch_titles(video_ids: list[str]) -> dict[str, str]:
    """Busca o título de cada vídeo (uma vez), em paralelo."""
    titles: dict[str, str] = {}
    session = requests.Session()

    def _one(vid: str) -> tuple[str, str]:
        try:
            return vid, fetch_youtube_title(vid, session=session)
        except Exception:  # pragma: no cover - rede
            return vid, ""

    done = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        for vid, title in pool.map(_one, video_ids):
            titles[vid] = title
            done += 1
            if done % 50 == 0:
                LOGGER.info("Títulos buscados: %s/%s vídeos", done, len(video_ids))
    return titles


def build_fixes(
    client: NotionSessoesClient,
    *,
    max_videos: int,
) -> tuple[list[DateFix], dict[str, int]]:
    schema = client.fetch_schema()
    pages = client.query_data_source()

    # agrupa por vídeo
    by_video: dict[str, list[dict[str, str]]] = {}
    for page in pages:
        link = client._extract_property_text(page, schema, "youtube_link")
        video_id = extract_youtube_video_id(link or "")
        if not video_id:
            continue
        by_video.setdefault(video_id, []).append(
            {
                "page_id": str(page.get("id", "")),
                "page_url": str(page.get("url", "")),
                "numero": client._extract_property_text(page, schema, "numero_processo"),
                "data": (client._extract_property_text(page, schema, "data_sessao") or "")[:10],
            }
        )

    video_ids = sorted(by_video)
    if max_videos:
        video_ids = video_ids[:max_videos]
    titles = _fetch_titles(video_ids)

    fixes: list[DateFix] = []
    videos_sem_titulo = 0
    videos_sem_data = 0
    videos_ok = 0
    for video_id in video_ids:
        title = titles.get(video_id, "")
        if not title:
            videos_sem_titulo += 1
            continue
        inferred = infer_session_date_from_video_title(title)
        if not inferred:
            videos_sem_data += 1
            continue
        videos_ok += 1
        for entry in by_video[video_id]:
            if entry["data"] != inferred:
                fixes.append(
                    DateFix(
                        page_id=entry["page_id"],
                        page_url=entry["page_url"],
                        numero_processo=entry["numero"],
                        video_id=video_id,
                        title=title,
                        old=entry["data"] or "(vazio)",
                        new=inferred,
                    )
                )

    stats = {
        "total_pages": sum(len(v) for v in by_video.values()),
        "total_videos": len(by_video),
        "videos_analisados": len(video_ids),
        "videos_titulo_com_data": videos_ok,
        "videos_sem_titulo": videos_sem_titulo,
        "videos_sem_data_no_titulo": videos_sem_data,
        "paginas_divergentes": len(fixes),
        "videos_com_divergencia": len({f.video_id for f in fixes}),
    }
    return fixes, stats


def apply_fixes(client: NotionSessoesClient, fixes: list[DateFix]) -> None:
    for index, fix in enumerate(fixes, start=1):
        try:
            notion_request_with_retry(
                client,
                "PATCH",
                f"/pages/{fix.page_id}",
                json={"properties": {"data_sessao": {"date": {"start": fix.new}}}},
            )
            fix.status = "updated"
        except Exception as exc:
            fix.status = "failed"
            fix.error = str(exc)
        if APPLY_SLEEP_SECONDS:
            time.sleep(APPLY_SLEEP_SECONDS)
        if index % 25 == 0:
            LOGGER.info("Datas aplicadas: %s/%s", index, len(fixes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conserta data_sessao divergente do título oficial do vídeo do TSE."
    )
    parser.add_argument("--apply", action="store_true", help="Aplica os consertos (padrão: apenas dry-run).")
    parser.add_argument("--max-videos", type=int, default=0, help="Limita quantos vídeos analisar (0 = todos).")
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
    fixes, stats = build_fixes(client, max_videos=args.max_videos)
    LOGGER.info(
        "Páginas: %s | vídeos: %s (com data no título: %s, sem título: %s, sem data no título: %s) | "
        "divergentes: %s páginas em %s vídeos",
        stats["total_pages"],
        stats["total_videos"],
        stats["videos_titulo_com_data"],
        stats["videos_sem_titulo"],
        stats["videos_sem_data_no_titulo"],
        stats["paginas_divergentes"],
        stats["videos_com_divergencia"],
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
