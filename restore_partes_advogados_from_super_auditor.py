from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tse_youtube_notion_core import (
    NotionSessoesClient,
    PublishPreviewRow,
    build_runtime_context,
    normalize_advogado_list,
    normalize_party_list,
)


LOGGER = logging.getLogger("restore_partes_advogados")
DEFAULT_SUMMARY = Path("artifacts/tse_youtube_notion/super_auditor/20260329_154817/summary.json")
ARTIFACT_ROOT = Path("artifacts/tse_youtube_notion/restore_partes_advogados")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restaura partes/advogados a partir de um summary do super auditor.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--year", dest="years", type=int, action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def _normalize_list(field_name: str, values: Any) -> list[str]:
    if field_name == "partes":
        return normalize_party_list(values or [])
    if field_name == "advogados":
        return normalize_advogado_list(values or [])
    return []


def _extract_restore_targets(summary_payload: dict[str, Any], years: set[int]) -> dict[str, dict[str, list[str]]]:
    targets: dict[str, dict[str, list[str]]] = {}
    for decision in summary_payload.get("decisions", []):
        year = int(decision.get("year") or 0)
        if years and year not in years:
            continue
        page_id = decision.get("page_id") or ""
        if not page_id:
            continue
        diff = decision.get("diff") or {}
        page_fields: dict[str, list[str]] = {}
        for field_name in ("partes", "advogados"):
            if field_name not in diff:
                continue
            after_value = _normalize_list(field_name, (diff.get(field_name) or {}).get("after"))
            if after_value:
                page_fields[field_name] = after_value
        if page_fields:
            targets[page_id] = page_fields
    return targets


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary_path = Path(args.summary)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    years = set(args.years or [])
    targets = _extract_restore_targets(payload, years)

    runtime = build_runtime_context()
    notion_client = NotionSessoesClient(
        api_key=runtime["notion_api_key"],
        data_source_id=runtime["notion_data_source_id"],
        logger=LOGGER,
        normalize_multiselect_colors_post_write=False,
    )
    notion_schema = notion_client.fetch_schema()

    run_root = ARTIFACT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    updated = 0
    failed: list[dict[str, Any]] = []
    per_year = defaultdict(lambda: {"updated_pages": 0, "partes": 0, "advogados": 0})

    processed = 0
    for decision in payload.get("decisions", []):
        page_id = decision.get("page_id") or ""
        if page_id not in targets:
            continue
        year = str(decision.get("year") or "")
        row = PublishPreviewRow(
            tema="",
            classe_processo="",
            tipo_registro="",
            eleicao="",
            origem="",
            tribunal="",
            numero_processo=decision.get("numero_processo") or "",
            youtube_link="",
            relator="",
            resultado="",
            votacao="",
            data_sessao="",
            partes=targets[page_id].get("partes", []),
            advogados=targets[page_id].get("advogados", []),
        )
        try:
            if args.apply:
                notion_client.update_row(notion_schema, page_id, row)
            updated += 1
            processed += 1
            per_year[year]["updated_pages"] += 1
            if row.partes:
                per_year[year]["partes"] += 1
            if row.advogados:
                per_year[year]["advogados"] += 1
            if processed % 25 == 0:
                LOGGER.info("Restauração em progresso: %s páginas atualizadas.", processed)
        except Exception as exc:
            failed.append(
                {
                    "page_id": page_id,
                    "numero_processo": row.numero_processo,
                    "error": str(exc),
                }
            )

    summary = {
        "summary_source": str(summary_path),
        "years": sorted(years) if years else "all-in-summary",
        "apply": bool(args.apply),
        "candidate_pages": len(targets),
        "updated_pages": updated,
        "failed_pages": len(failed),
        "stats_by_year": dict(per_year),
        "failed": failed,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Restauração concluída. Resumo: %s", run_root / "summary.json")


if __name__ == "__main__":
    main()
