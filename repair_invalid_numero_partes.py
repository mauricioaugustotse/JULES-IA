from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tse_backfill_2025_notion import load_existing_pages_for_year_with_retry, update_notion_row_with_retry
from tse_youtube_notion_core import (
    NotionSessoesClient,
    PublishPreviewRow,
    build_runtime_context,
    validate_preview_row,
)


LOGGER = logging.getLogger("repair_invalid_numero_partes")
ARTIFACT_ROOT = Path("artifacts/tse_youtube_notion/repair_invalid_numero_partes")
TEXTUAL_PROCESS_DESCRIPTOR_REGEX = re.compile(r"^(?:recurso|recursos|agravo|agravos)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrige ou remove páginas com numero_processo textual inválido e partes colapsadas."
    )
    parser.add_argument("--year", dest="years", type=int, action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def _is_textual_process_descriptor(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(TEXTUAL_PROCESS_DESCRIPTOR_REGEX.match(text)) and not re.search(r"\d", text)


def _diff_fields(before: PublishPreviewRow, after: PublishPreviewRow) -> list[str]:
    changed: list[str] = []
    for field_name in ("numero_processo", "partes"):
        if getattr(before, field_name, None) != getattr(after, field_name, None):
            changed.append(field_name)
    return changed


def _extract_page_rich_text_property(page_payload: dict[str, Any], property_name: str) -> str:
    prop = (page_payload.get("properties") or {}).get(property_name) or {}
    return "".join(item.get("plain_text", "") for item in prop.get("rich_text", [])).strip()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    years = args.years or [2025, 2024, 2023, 2022]

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

    stats_by_year: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    updated_pages: list[dict[str, Any]] = []
    trashed_pages: list[dict[str, Any]] = []
    unchanged_pages: list[dict[str, Any]] = []

    for year in years:
        grouped = load_existing_pages_for_year_with_retry(
            notion_client,
            notion_schema,
            year,
        )
        for records in grouped.values():
            for record in records:
                before = record.row.model_copy(deep=True)
                after = validate_preview_row(record.row.model_copy(deep=True), notion_schema)
                raw_page = None
                raw_numero_processo = ""
                if not str(before.numero_processo or "").strip():
                    raw_page = notion_client._request("GET", f"/pages/{record.page_id}")
                    raw_numero_processo = _extract_page_rich_text_property(raw_page, "numero_processo")
                textual_numero = _is_textual_process_descriptor(raw_numero_processo or before.numero_processo)
                changed_fields = _diff_fields(before, after)
                if textual_numero and not after.numero_processo:
                    item = {
                        "page_id": record.page_id,
                        "url": record.url,
                        "year": year,
                        "before_numero_processo": raw_numero_processo or before.numero_processo,
                        "before_partes": before.partes,
                    }
                    if args.apply:
                        notion_client._request("PATCH", f"/pages/{record.page_id}", json={"in_trash": True})
                    trashed_pages.append(item)
                    stats_by_year[str(year)]["trashed_invalid_numero"] += 1
                    continue
                if changed_fields:
                    item = {
                        "page_id": record.page_id,
                        "url": record.url,
                        "year": year,
                        "changed_fields": changed_fields,
                        "before": {field: getattr(before, field) for field in changed_fields},
                        "after": {field: getattr(after, field) for field in changed_fields},
                    }
                    if args.apply:
                        update_notion_row_with_retry(notion_client, notion_schema, record.page_id, after)
                    updated_pages.append(item)
                    for field_name in changed_fields:
                        stats_by_year[str(year)][f"updated_{field_name}"] += 1
                else:
                    unchanged_pages.append(
                        {
                            "page_id": record.page_id,
                            "year": year,
                        }
                    )

    summary = {
        "years": years,
        "apply": bool(args.apply),
        "updated_pages": len(updated_pages),
        "trashed_pages": len(trashed_pages),
        "stats_by_year": dict(stats_by_year),
        "updated": updated_pages,
        "trashed": trashed_pages,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Reparo de numero_processo/partes concluído. Resumo: %s", run_root / "summary.json")


if __name__ == "__main__":
    main()
