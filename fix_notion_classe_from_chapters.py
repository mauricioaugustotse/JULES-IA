"""Corrige a coluna ``classe_processo`` no Notion usando a CLASSE que consta nos
capítulos da descrição do vídeo do YouTube (fonte autoritativa do próprio TSE,
ex.: ``01:20:17 AgR no AREspe - 060006171`` -> classe AgRg-AREspe para 0600061-71).

Aplica automaticamente em dois casos (pedido do usuário):
1. classe_processo VAZIA -> preenche com a classe do capítulo;
2. classe_processo == 'PA' -> corrige quando o capítulo indica outra classe.

Outras divergências (classe atual != capítulo, ambas não-PA) são apenas RELATADAS
para revisão manual, não aplicadas. Só usa classes que já são opção do schema
(não cria etiquetas novas sem revisão).

Uso:
    python fix_notion_classe_from_chapters.py            # dry-run
    python fix_notion_classe_from_chapters.py --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from audit_notion_sessoes_round2 import notion_request_with_retry
from fix_notion_youtube_timestamps import fetch_video_description
from local_secrets import get_secret
from tse_normalization import canonicalize_numero_processo, extract_youtube_video_id
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NotionSessoesClient,
    parse_youtube_chapter_entries,
)

LOGGER = logging.getLogger("fix_notion_classe_from_chapters")
ARTIFACT_ROOT = Path("artifacts") / "notion_classe_from_chapters"
CACHE_DIR = Path("artifacts") / "_yt_descriptions_cache"
APPLY_SLEEP_SECONDS = 0.2
FETCH_SLEEP_SECONDS = 0.3
def _is_specificity_downgrade(current: str, chapter: str) -> bool:
    """True quando a classe do capítulo é apenas a BASE de uma classe atual MAIS
    específica (ex.: atual 'ED-PC' / 'AgRg-REspe' / 'ED-AgRg-AREspe' vs capítulo 'PC'
    / 'REspe' / 'AgRg-AREspe'). A classe é composta por segmentos separados por '-'
    (recurso interno + classe-base); se os segmentos do capítulo forem um SUFIXO
    estrito dos segmentos atuais, o capítulo omite um recurso interno que a etiqueta
    atual carrega — então NÃO rebaixamos."""
    current_segments = current.split("-")
    chapter_segments = chapter.split("-")
    return len(chapter_segments) < len(current_segments) and current_segments[-len(chapter_segments):] == chapter_segments


def _fetch_cached(video_id: str, session: requests.Session) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{video_id}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    desc = fetch_video_description(video_id, session)
    cache_path.write_text(desc or "", encoding="utf-8")
    if FETCH_SLEEP_SECONDS and desc:
        time.sleep(FETCH_SLEEP_SECONDS)
    return desc


def build_audit(client: NotionSessoesClient, *, full: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    schema = client.fetch_schema()
    valid_options = {
        str(o.get("name", "")).strip()
        for o in schema.raw_payload.get("properties", {}).get("classe_processo", {}).get("select", {}).get("options", [])
        if str(o.get("name", "")).strip()
    }
    pages = client.query_data_source()
    records = []
    for page in pages:
        link = client._extract_property_text(page, schema, "youtube_link")
        records.append(
            {
                "page_id": str(page.get("id", "")),
                "url": str(page.get("url", "")),
                "video_id": extract_youtube_video_id(link),
                "numero": canonicalize_numero_processo(client._extract_property_text(page, schema, "numero_processo")),
                "numero_raw": client._extract_property_text(page, schema, "numero_processo"),
                "classe": client._extract_property_text(page, schema, "classe_processo").strip(),
            }
        )
    if full:
        videos_to_fetch = sorted({r["video_id"] for r in records if r["video_id"]})
    else:
        videos_to_fetch = sorted({r["video_id"] for r in records if r["video_id"] and (not r["classe"] or r["classe"] == "PA")})
    session = requests.Session()
    chapters: dict[str, dict[str, Any]] = {}
    for i, vid in enumerate(videos_to_fetch, start=1):
        desc = _fetch_cached(vid, session)
        chapters[vid] = parse_youtube_chapter_entries(desc) if desc else {}
        if i % 40 == 0:
            LOGGER.info("Descrições processadas: %s/%s", i, len(videos_to_fetch))

    fixes: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for r in records:
        entry = chapters.get(r["video_id"], {}).get(r["numero"]) if r["video_id"] else None
        if not entry:
            continue
        chapter_classe = entry.get("classe", "")
        if not chapter_classe or chapter_classe not in valid_options:
            continue
        if not r["classe"]:
            fixes.append({**_change(r, chapter_classe), "reason": "blank"})
        elif r["classe"] == chapter_classe:
            continue
        elif r["classe"] == "PA":
            fixes.append({**_change(r, chapter_classe), "reason": "pa_correction"})
        elif _is_specificity_downgrade(r["classe"], chapter_classe):
            review.append({**_change(r, chapter_classe), "reason": "skipped_downgrade"})
        else:
            fixes.append({**_change(r, chapter_classe), "reason": "correction"})
    stats = {"records": len(records), "videos_fetched": len(videos_to_fetch), "fixes": len(fixes), "review": len(review)}
    return fixes, review, stats


def _change(r: dict[str, Any], new: str) -> dict[str, Any]:
    return {"page_id": r["page_id"], "url": r["url"], "numero_processo": r["numero_raw"], "old": r["classe"], "new": new}


def apply_fixes(client: NotionSessoesClient, fixes: list[dict[str, Any]]) -> None:
    for i, fx in enumerate(fixes, start=1):
        try:
            notion_request_with_retry(
                client, "PATCH", f"/pages/{fx['page_id']}",
                json={"properties": {"classe_processo": {"select": {"name": fx["new"]}}}},
            )
            fx["status"] = "updated"
        except Exception as exc:
            fx["status"] = "failed"
            fx["error"] = str(exc)
        if APPLY_SLEEP_SECONDS:
            time.sleep(APPLY_SLEEP_SECONDS)
        if i % 25 == 0:
            LOGGER.info("Aplicados: %s/%s", i, len(fixes))


def main() -> int:
    parser = argparse.ArgumentParser(description="Corrige classe_processo (blanks + PA) via capítulos do YouTube.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--full", action="store_true", help="Varre TODAS as siglas (não só blank/PA), corrigindo divergências reais.")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    api_key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY/NOTION_TOKEN não encontrado.")
    client = NotionSessoesClient(api_key=api_key, data_source_id=args.data_source_id)
    fixes, review, stats = build_audit(client, full=args.full)
    LOGGER.info(
        "Registros: %s | vídeos: %s | correções: %s | pulados/revisão: %s",
        stats["records"], stats["videos_fetched"], stats["fixes"], stats["review"],
    )
    if args.apply and fixes:
        apply_fixes(client, fixes)
    artifact_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "fixes.json").write_text(json.dumps(fixes, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats,
               "applied": sum(1 for f in fixes if f.get("status") == "updated"),
               "failed": sum(1 for f in fixes if f.get("status") == "failed")}
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Resumo: %s | relatórios em %s", json.dumps(summary, ensure_ascii=False), artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
