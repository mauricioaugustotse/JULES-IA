from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient


LOGGER = logging.getLogger("compact_notion_option_schema")
ARTIFACT_ROOT = Path("artifacts") / "notion_schema_compaction"
SUPPORTED_TYPES = {"select", "multi_select"}
DEFAULT_PROPERTIES = ["origem", "partes", "advogados"]
APPLY_SLEEP_SECONDS = 0.08
PAGE_UPDATE_WORKERS = 4

BAD_OPTION_PATTERNS = {
    "origem": re.compile(
        r"(?i)(?:"
        r"^boa vista$|^campo grande$|^goi[aâ]nia$|^macei[oó]$|"
        r"prefeito|vice-prefeito|prefeitura|tribunal|tre/?|tse|zona eleitoral|ju[ií]zo|"
        r"^sp/rj$|^[A-Z]{2}$|^[^/]+$"
        r")"
    ),
    "advogados": re.compile(
        r"(?i)(?:"
        r"\bpelo\b|\bpela\b|representante|recorrente|recorrido|agravante|agravado|"
        r"embargante|embargado|impetrante|impetrado|investigado|coliga[cç][aã]o|"
        r"sociedade\s+de\s+advogad|e\s+outr[oa]s?$"
        r")"
    ),
    "partes": re.compile(r"(?i)(?:\s+x\s+|Dr\.|Dra\.|advogad[oa])"),
}


@dataclass
class PropertyAudit:
    property_name: str
    property_type: str
    schema_options: int
    used_options: int
    unused_options: int
    nondefault_color_options: int
    bad_schema_options_count: int
    bad_used_options_count: int
    bad_schema_examples: list[str]
    bad_used_examples: list[str]
    unused_examples: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "property": self.property_name,
            "type": self.property_type,
            "schema_options": self.schema_options,
            "used_options": self.used_options,
            "unused_options": self.unused_options,
            "nondefault_color_options": self.nondefault_color_options,
            "bad_schema_options_count": self.bad_schema_options_count,
            "bad_used_options_count": self.bad_used_options_count,
            "bad_schema_examples": self.bad_schema_examples,
            "bad_used_examples": self.bad_used_examples,
            "unused_examples": self.unused_examples,
        }


def page_property_values(page: dict[str, Any], property_name: str, property_type: str) -> list[str]:
    prop = page.get("properties", {}).get(property_name, {})
    if property_type == "select":
        value = ((prop.get("select") or {}).get("name") or "").strip()
        return [value] if value else []
    if property_type == "multi_select":
        return [
            str(item.get("name", "")).strip()
            for item in prop.get("multi_select", []) or []
            if str(item.get("name", "")).strip()
        ]
    return []


def payload_for_values(property_type: str, values: list[str]) -> dict[str, Any]:
    if property_type == "select":
        return {"select": {"name": values[0]}} if values else {"select": None}
    if property_type == "multi_select":
        return {"multi_select": [{"name": value} for value in values if value.strip()]}
    raise ValueError(f"Tipo nao suportado: {property_type}")


def collect_used_values(pages: list[dict[str, Any]], property_name: str, property_type: str) -> list[str]:
    used: list[str] = []
    seen: set[str] = set()
    for page in pages:
        for value in page_property_values(page, property_name, property_type):
            if value not in seen:
                seen.add(value)
                used.append(value)
    return used


def schema_option_names(raw_property: dict[str, Any], property_type: str) -> list[str]:
    return [
        str(option.get("name", "")).strip()
        for option in (raw_property.get(property_type, {}).get("options") or [])
        if str(option.get("name", "")).strip()
    ]


def audit_property(
    schema_payload: dict[str, Any],
    pages: list[dict[str, Any]],
    property_name: str,
) -> PropertyAudit | None:
    raw_property = schema_payload.get("properties", {}).get(property_name, {})
    property_type = raw_property.get("type")
    if property_type not in SUPPORTED_TYPES:
        return None
    options = schema_option_names(raw_property, property_type)
    used = collect_used_values(pages, property_name, property_type)
    used_set = set(used)
    unused = [option for option in options if option not in used_set]
    nondefault = [
        option
        for option in (raw_property.get(property_type, {}).get("options") or [])
        if (option.get("color") or "default") != "default"
    ]
    pattern = BAD_OPTION_PATTERNS.get(property_name)
    bad_schema = [option for option in options if pattern and pattern.search(option)]
    bad_used = [option for option in used if pattern and pattern.search(option)]
    return PropertyAudit(
        property_name=property_name,
        property_type=property_type,
        schema_options=len(options),
        used_options=len(used),
        unused_options=len(unused),
        nondefault_color_options=len(nondefault),
        bad_schema_options_count=len(bad_schema),
        bad_used_options_count=len(bad_used),
        bad_schema_examples=bad_schema[:50],
        bad_used_examples=bad_used[:50],
        unused_examples=unused[:50],
    )


def create_option_property(client: NotionSessoesClient, property_name: str, property_type: str) -> None:
    notion_request_with_retry(
        client,
        "PATCH",
        f"/data_sources/{client.data_source_id}",
        json={"properties": {property_name: {property_type: {}}}},
    )


def drop_property(client: NotionSessoesClient, property_name: str) -> None:
    notion_request_with_retry(
        client,
        "PATCH",
        f"/data_sources/{client.data_source_id}",
        json={"properties": {property_name: None}},
    )


def rename_property(client: NotionSessoesClient, current_name: str, new_name: str) -> None:
    payload = notion_request_with_retry(client, "GET", f"/data_sources/{client.data_source_id}")
    prop = payload.get("properties", {}).get(current_name)
    if not prop:
        raise RuntimeError(f"Propriedade {current_name!r} nao encontrada para renomear.")
    prop_id = prop.get("id") or current_name
    notion_request_with_retry(
        client,
        "PATCH",
        f"/data_sources/{client.data_source_id}",
        json={"properties": {prop_id: {"name": new_name}}},
    )


def clear_property_options(client: NotionSessoesClient, property_name: str, property_type: str) -> None:
    notion_request_with_retry(
        client,
        "PATCH",
        f"/data_sources/{client.data_source_id}",
        json={"properties": {property_name: {property_type: {"options": []}}}},
    )


def patch_pages_with_values(
    client: NotionSessoesClient,
    property_name: str,
    property_type: str,
    values_by_page: list[tuple[str, list[str]]],
    *,
    include_empty: bool = False,
) -> int:
    items = [
        (page_id, values)
        for page_id, values in values_by_page
        if page_id and (values or include_empty)
    ]
    total = len(items)

    def patch_one(item: tuple[str, list[str]]) -> str:
        page_id, values = item
        notion_request_with_retry(
            client,
            "PATCH",
            f"/pages/{page_id}",
            json={"properties": {property_name: payload_for_values(property_type, values)}},
        )
        if APPLY_SLEEP_SECONDS:
            time.sleep(APPLY_SLEEP_SECONDS)
        return page_id

    updated = 0
    with ThreadPoolExecutor(max_workers=PAGE_UPDATE_WORKERS) as executor:
        futures = [executor.submit(patch_one, item) for item in items]
        for future in as_completed(futures):
            future.result()
            updated += 1
            if updated % 50 == 0:
                LOGGER.info("%s: paginas atualizadas %s/%s", property_name, updated, total)
    return updated


def values_snapshot(
    pages: list[dict[str, Any]],
    property_name: str,
    property_type: str,
) -> list[tuple[str, list[str]]]:
    return [
        (str(page.get("id", "")), page_property_values(page, property_name, property_type))
        for page in pages
    ]


def assert_copied_values(
    client: NotionSessoesClient,
    temp_name: str,
    property_type: str,
    expected: list[tuple[str, list[str]]],
) -> dict[str, Any]:
    expected_by_page = {page_id: values for page_id, values in expected}
    fresh_pages = client.query_data_source()
    mismatches: list[dict[str, Any]] = []
    for page in fresh_pages:
        page_id = str(page.get("id", ""))
        observed = page_property_values(page, temp_name, property_type)
        expected_values = expected_by_page.get(page_id, [])
        if observed != expected_values:
            mismatches.append({"page_id": page_id, "expected": expected_values, "observed": observed})
            if len(mismatches) >= 25:
                break
    if mismatches:
        raise RuntimeError(f"Copia para {temp_name!r} divergente: {mismatches[:3]}")
    return {"verified_pages": len(expected_by_page), "mismatches": 0}


def cleanup_legacy_property(
    client: NotionSessoesClient,
    legacy_name: str,
    property_type: str,
) -> dict[str, Any]:
    clear_property_options(client, legacy_name, property_type)
    try:
        drop_property(client, legacy_name)
    except Exception as exc:
        return {
            "legacy_status": "left_empty_residue",
            "legacy_property": legacy_name,
            "legacy_clear_updates": 0,
            "drop_error": str(exc),
        }
    return {
        "legacy_status": "dropped_after_clearing",
        "legacy_property": legacy_name,
        "legacy_clear_updates": 0,
    }


def rebuild_property(
    client: NotionSessoesClient,
    property_name: str,
    *,
    timestamp: str,
    apply_changes: bool,
) -> dict[str, Any]:
    schema = client.fetch_schema()
    pages = client.query_data_source()
    audit_before = audit_property(schema.raw_payload, pages, property_name)
    if audit_before is None:
        return {"property": property_name, "status": "unsupported_or_missing"}
    if not apply_changes:
        return {"property": property_name, "status": "would_rebuild", "before": audit_before.as_dict()}

    property_type = audit_before.property_type
    temp_name = f"zz_tmp_{property_name}_{timestamp}"[:90]
    legacy_name = f"zz_residuo_schema_{property_name}_{timestamp}"[:90]
    existing_names = set(schema.raw_payload.get("properties", {}))
    if temp_name in existing_names or legacy_name in existing_names:
        raise RuntimeError(
            f"Nome temporario/residual ja existe para {property_name}: {temp_name!r} ou {legacy_name!r}."
        )

    original_values = values_snapshot(pages, property_name, property_type)
    LOGGER.info("%s: criando propriedade temporaria %s", property_name, temp_name)
    create_option_property(client, temp_name, property_type)
    LOGGER.info("%s: copiando valores ativos para propriedade temporaria", property_name)
    page_updates = patch_pages_with_values(client, temp_name, property_type, original_values)
    verification = assert_copied_values(client, temp_name, property_type, original_values)

    LOGGER.info("%s: trocando propriedade temporaria para o nome oficial", property_name)
    renamed_old = False
    try:
        rename_property(client, property_name, legacy_name)
        renamed_old = True
        rename_property(client, temp_name, property_name)
    except Exception:
        if renamed_old:
            try:
                rename_property(client, legacy_name, property_name)
            except Exception:
                LOGGER.exception("%s: falha tambem ao restaurar nome original", property_name)
        raise

    LOGGER.info("%s: esvaziando propriedade legada para neutralizar opcoes antigas", property_name)
    legacy_result = cleanup_legacy_property(client, legacy_name, property_type)

    fresh_schema = client.fetch_schema()
    fresh_pages = client.query_data_source()
    audit_after = audit_property(fresh_schema.raw_payload, fresh_pages, property_name)
    return {
        "property": property_name,
        "status": "rebuilt",
        "page_updates": page_updates,
        "verification": verification,
        "before": audit_before.as_dict(),
        "after": audit_after.as_dict() if audit_after else None,
        **legacy_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compacta schemas select/multi_select do Notion usando apenas valores ativos.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--properties", nargs="+", default=DEFAULT_PROPERTIES)
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    api_key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY/NOTION_TOKEN nao encontrado.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else ARTIFACT_ROOT / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)
    client = NotionSessoesClient(api_key=api_key, data_source_id=args.data_source_id)

    results: list[dict[str, Any]] = []
    for property_name in args.properties:
        results.append(rebuild_property(client, property_name, timestamp=timestamp, apply_changes=args.apply))

    schema = client.fetch_schema()
    pages = client.query_data_source()
    final_audit = [
        audit.as_dict()
        for property_name in args.properties
        if (audit := audit_property(schema.raw_payload, pages, property_name)) is not None
    ]
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "properties": args.properties,
        "results_by_status": dict(Counter(result.get("status", "unknown") for result in results)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (artifact_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "final_audit.json").write_text(json.dumps(final_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Relatorios gravados em %s", artifact_dir)
    LOGGER.info("Resumo: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
