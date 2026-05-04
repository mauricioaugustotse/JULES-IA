from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from cleanup_notion_sessoes_followup import prop_value
from local_secrets import get_secret
from tse_backfill_2025_notion import notion_page_to_row
from tse_normalization import normalize_class_text, normalize_ministro_name, normalize_pedido_vista_value
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionDataSourceSchema, NotionSessoesClient


LOGGER = logging.getLogger("cleanup_notion_relator_pedido_duplicates")
ARTIFACT_ROOT = Path("artifacts") / "notion_relator_pedido_duplicates"
TARGET_PROPERTIES = ("relator", "pedido_vista")
PAGE_UPDATE_SLEEP_SECONDS = 0.08
PAGE_UPDATE_WORKERS = 2

NON_PERSON_SELECT_KEYS = {
    "presidencia",
    "min presidencia",
}

DUPLICATE_GROUPS: dict[str, dict[str, Any]] = {
    "edson_fachin": {
        "canonical": "Min. Edson Fachin",
        "variants": [
            "Min. Edson Fachin",
            "Min. Luís Edson Fachin",
            "Min. Luiz Edson Fachin",
        ],
    },
    "henrique_neves": {
        "canonical": "Min. Henrique Neves da Silva",
        "variants": [
            "Min. Henrique Neves",
            "Min. Henrique Neves da Silva",
        ],
    },
    "luiz_fux": {
        "canonical": "Min. Luiz Fux",
        "variants": [
            "Min. Luís Fux",
            "Min. Luiz Fux",
        ],
    },
    "luis_felipe_salomao": {
        "canonical": "Min. Luís Felipe Salomão",
        "variants": [
            "Min. Luís Felipe Salomão",
            "Min. Luis Salomão",
        ],
    },
    "maria_claudia_bucchianeri": {
        "canonical": "Min. Maria Cláudia Bucchianeri",
        "variants": [
            "Min. Maria Claudia Bucchianeri",
            "Min. Maria Cláudia Bucchianeri",
            "Min. Maria Claudia Bucchianeri Pinheiro",
            "Min. Maria Cláudia Bucchianeri Pinheiro",
        ],
    },
    "maria_thereza": {
        "canonical": "Min. Maria Thereza de Assis Moura",
        "variants": [
            "Min. Maria Thereza",
            "Min. Maria Thereza de Assis Moura",
        ],
    },
    "napoleao_maia": {
        "canonical": "Min. Napoleão Nunes Maia Filho",
        "variants": [
            "Min. Napoleão Maia",
            "Min. Napoleão Nunes Maia",
            "Min. Napoleão Nunes Maia Filho",
        ],
    },
    "paulo_sanseverino": {
        "canonical": "Min. Paulo de Tarso Sanseverino",
        "variants": [
            "Min. Paulo Tarso Sanseverino",
            "Min. Paulo de Tarso Sanseverino",
        ],
    },
    "tarcisio_vieira": {
        "canonical": "Min. Tarcísio Vieira de Carvalho Neto",
        "variants": [
            "Min. Tarcísio Vieira de Carvalho",
            "Min. Tarcísio Vieira de Carvalho Neto",
        ],
    },
}


@dataclass
class FieldChange:
    page_id: str
    page_url: str
    numero_processo: str
    tema: str
    property_name: str
    old: str
    new: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_url": self.page_url,
            "numero_processo": self.numero_processo,
            "tema": self.tema,
            "property_name": self.property_name,
            "old": self.old,
            "new": self.new,
            "reason": self.reason,
        }


def option_key(value: str) -> str:
    return normalize_class_text(re.sub(r"^Min\.\s*", "", str(value or "").strip()))


def build_variant_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group in DUPLICATE_GROUPS.values():
        canonical = str(group["canonical"])
        for variant in group["variants"]:
            mapping[option_key(str(variant))] = canonical
    return mapping


VARIANT_TO_CANONICAL = build_variant_map()


def canonical_person_select(value: str, property_name: str) -> tuple[str, str]:
    current = str(value or "").strip()
    if not current:
        return "", ""
    normalized_key = option_key(current)
    if normalized_key in NON_PERSON_SELECT_KEYS:
        return "", f"{property_name} continha etiqueta sem pessoa identificada; campo limpo"

    mapped = VARIANT_TO_CANONICAL.get(normalized_key)
    if mapped:
        if mapped != current:
            return mapped, f"{property_name} unificado em etiqueta canonica do mesmo ministro"
        return current, ""

    normalized = normalize_ministro_name(current) if property_name == "relator" else normalize_pedido_vista_value(current)
    if normalized and normalized != current:
        normalized_key = option_key(normalized)
        mapped = VARIANT_TO_CANONICAL.get(normalized_key, normalized)
        if mapped != current:
            return mapped, f"{property_name} normalizado para etiqueta canonica de ministro"
    return current, ""


def collect_used_values(pages: list[dict[str, Any]], property_name: str) -> list[str]:
    used: list[str] = []
    seen: set[str] = set()
    for page in pages:
        value = prop_value(page, property_name)
        values = value if isinstance(value, list) else [value]
        for item in values:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                used.append(text)
    return used


def schema_option_names(schema: NotionDataSourceSchema, property_name: str) -> list[str]:
    prop = schema.properties[property_name]
    raw_prop = schema.raw_payload.get("properties", {}).get(property_name, {})
    return [
        str(option.get("name", "")).strip()
        for option in (raw_prop.get(prop.type, {}).get("options") or [])
        if str(option.get("name", "")).strip()
    ]


def duplicate_cluster_report(schema: NotionDataSourceSchema, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for property_name in TARGET_PROPERTIES:
        used = set(collect_used_values(pages, property_name))
        options = set(schema_option_names(schema, property_name))
        for group_name, group in DUPLICATE_GROUPS.items():
            canonical = str(group["canonical"])
            variants = [str(item) for item in group["variants"]]
            used_present = sorted(value for value in variants if value in used)
            schema_present = sorted(value for value in variants if value in options)
            if used_present or schema_present:
                report.append(
                    {
                        "property": property_name,
                        "group": group_name,
                        "canonical": canonical,
                        "used_present": used_present,
                        "schema_present": schema_present,
                        "used_duplicate_count": max(0, len(used_present) - (1 if canonical in used_present else 0)),
                        "schema_duplicate_count": max(0, len(schema_present) - (1 if canonical in schema_present else 0)),
                    }
                )
    return report


def build_audit(client: NotionSessoesClient) -> tuple[list[dict[str, Any]], list[FieldChange], list[dict[str, Any]]]:
    schema = client.fetch_schema()
    pages = client.query_data_source()
    changes: list[FieldChange] = []
    for page in pages:
        row = notion_page_to_row(client, schema, page)
        page_id = str(page.get("id", ""))
        page_url = str(page.get("url", ""))
        for property_name in TARGET_PROPERTIES:
            current = str(getattr(row, property_name) or "").strip()
            proposed, reason = canonical_person_select(current, property_name)
            if reason and proposed != current:
                changes.append(
                    FieldChange(
                        page_id=page_id,
                        page_url=page_url,
                        numero_processo=row.numero_processo,
                        tema=row.tema,
                        property_name=property_name,
                        old=current,
                        new=proposed,
                        reason=reason,
                    )
                )
    return pages, changes, duplicate_cluster_report(schema, pages)


def select_payload(value: str) -> dict[str, Any]:
    return {"select": {"name": value}} if value else {"select": None}


def apply_page_changes(
    client: NotionSessoesClient,
    changes: list[FieldChange],
    *,
    max_pages: int = 0,
) -> list[dict[str, Any]]:
    by_page: dict[str, list[FieldChange]] = defaultdict(list)
    for change in changes:
        by_page[change.page_id].append(change)
    items = list(by_page.items())
    if max_pages > 0:
        items = items[:max_pages]

    def update_one(item: tuple[str, list[FieldChange]]) -> dict[str, Any]:
        page_id, page_changes = item
        payload = {"properties": {change.property_name: select_payload(change.new) for change in page_changes}}
        notion_request_with_retry(client, "PATCH", f"/pages/{page_id}", json=payload)
        if PAGE_UPDATE_SLEEP_SECONDS:
            time.sleep(PAGE_UPDATE_SLEEP_SECONDS)
        return {"page_id": page_id, "status": "updated", "fields": [change.property_name for change in page_changes]}

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=PAGE_UPDATE_WORKERS) as executor:
        futures = {executor.submit(update_one, item): item for item in items}
        for index, future in enumerate(as_completed(futures), start=1):
            page_id, page_changes = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                LOGGER.warning("Falha ao atualizar pagina %s: %s", page_id, exc)
                results.append(
                    {
                        "page_id": page_id,
                        "status": "failed",
                        "fields": [change.property_name for change in page_changes],
                        "error": str(exc),
                    }
                )
            if index % 50 == 0:
                LOGGER.info("Paginas atualizadas: %s/%s", index, len(items))
    return results


def planned_remaining_options(
    schema: NotionDataSourceSchema,
    pages: list[dict[str, Any]],
    property_name: str,
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    prop = schema.properties[property_name]
    raw_prop = schema.raw_payload.get("properties", {}).get(property_name, {})
    raw_options = raw_prop.get(prop.type, {}).get("options") or []
    used = collect_used_values(pages, property_name)
    used_set = set(used)
    color_by_name = {
        str(option.get("name", "")).strip(): str(option.get("color") or "default")
        for option in raw_options
        if str(option.get("name", "")).strip()
    }
    option_names = [str(option.get("name", "")).strip() for option in raw_options if str(option.get("name", "")).strip()]
    remaining = [
        {"name": name, "color": color_by_name.get(name, "default")}
        for name in option_names
        if name in used_set
    ]
    remaining_names = {item["name"] for item in remaining}
    for name in used:
        if name not in remaining_names:
            remaining.append({"name": name, "color": "default"})
            remaining_names.add(name)
    unused = [name for name in option_names if name not in used_set]
    missing = [name for name in used if name not in set(option_names)]
    return remaining, unused, missing


def cleanup_schema_options(
    client: NotionSessoesClient,
    *,
    apply_changes: bool,
) -> list[dict[str, Any]]:
    schema = client.fetch_schema()
    pages = client.query_data_source()
    results: list[dict[str, Any]] = []
    for property_name in TARGET_PROPERTIES:
        prop = schema.properties.get(property_name)
        if not prop or prop.type != "select":
            results.append({"property": property_name, "status": "missing_or_not_select"})
            continue
        remaining, unused, missing = planned_remaining_options(schema, pages, property_name)
        status = "no_unused_options" if not unused and not missing else "would_patch_options"
        if apply_changes and (unused or missing):
            try:
                notion_request_with_retry(
                    client,
                    "PATCH",
                    f"/data_sources/{client.data_source_id}",
                    json={"properties": {property_name: {"select": {"options": remaining}}}},
                )
                status = "patched_options"
            except Exception as exc:
                results.append(
                    {
                        "property": property_name,
                        "status": "patch_failed",
                        "used_options": len(remaining),
                        "unused_options": len(unused),
                        "missing_schema_options": missing,
                        "error": str(exc),
                    }
                )
                continue
        results.append(
            {
                "property": property_name,
                "status": status,
                "used_options": len(remaining),
                "unused_options": len(unused),
                "unused_option_names": unused,
                "missing_schema_options": missing,
            }
        )
    return results


def readback(
    client: NotionSessoesClient,
    changes: list[FieldChange],
) -> dict[str, Any]:
    schema = client.fetch_schema()
    pages = client.query_data_source()
    by_id = {str(page.get("id", "")): page for page in pages}
    mismatches: list[dict[str, Any]] = []
    for change in changes:
        page = by_id.get(change.page_id)
        if not page:
            mismatches.append({"page_id": change.page_id, "property_name": change.property_name, "reason": "page_not_found"})
            continue
        current = prop_value(page, change.property_name)
        if current != change.new:
            mismatches.append(
                {
                    "page_id": change.page_id,
                    "property_name": change.property_name,
                    "expected": change.new,
                    "actual": current,
                }
            )
    obsolete_values = {
        str(variant)
        for group in DUPLICATE_GROUPS.values()
        for variant in group["variants"]
        if str(variant) != str(group["canonical"])
    } | {"Min. Presidência"}
    obsolete_used: dict[str, list[str]] = {}
    unused_schema_options: dict[str, list[str]] = {}
    obsolete_schema_options: dict[str, list[str]] = {}
    for property_name in TARGET_PROPERTIES:
        used = collect_used_values(pages, property_name)
        used_set = set(used)
        options = schema_option_names(schema, property_name)
        obsolete_used[property_name] = sorted(value for value in obsolete_values if value in used_set)
        unused_schema_options[property_name] = [value for value in options if value not in used_set]
        obsolete_schema_options[property_name] = sorted(value for value in obsolete_values if value in set(options))
    return {
        "checked_changes": len(changes),
        "mismatches": mismatches,
        "obsolete_values_still_used": obsolete_used,
        "unused_schema_options": unused_schema_options,
        "obsolete_schema_options": obsolete_schema_options,
        "duplicate_clusters_after": duplicate_cluster_report(schema, pages),
    }


def write_reports(
    artifact_dir: Path,
    *,
    changes: list[FieldChange],
    apply_results: list[dict[str, Any]],
    schema_results: list[dict[str, Any]],
    readback_result: dict[str, Any],
    duplicate_clusters_before: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    change_dicts = [change.as_dict() for change in changes]
    (artifact_dir / "changes.json").write_text(json.dumps(change_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    with (artifact_dir / "changes.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(change_dicts[0].keys()) if change_dicts else list(FieldChange("", "", "", "", "", "", "", "").as_dict().keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(change_dicts)
    (artifact_dir / "apply_results.json").write_text(json.dumps(apply_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "schema_results.json").write_text(json.dumps(schema_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "readback.json").write_text(json.dumps(readback_result, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "duplicate_clusters_before.json").write_text(json.dumps(duplicate_clusters_before, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unifica duplicidades de relator e pedido_vista no Notion.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    api_key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY/NOTION_TOKEN nao encontrado.")
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    client = NotionSessoesClient(api_key=api_key, data_source_id=args.data_source_id)

    pages, changes, duplicate_clusters_before = build_audit(client)
    LOGGER.info("Paginas lidas: %s; mudancas propostas: %s", len(pages), len(changes))
    apply_results: list[dict[str, Any]] = []
    if args.apply and changes:
        apply_results = apply_page_changes(client, changes, max_pages=args.max_pages)
    schema_results = cleanup_schema_options(client, apply_changes=args.apply)
    readback_result = readback(client, changes) if args.apply else {}
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "total_records": len(pages),
        "total_changes": len(changes),
        "changes_by_property": dict(Counter(change.property_name for change in changes)),
        "changes_by_target": dict(Counter(f"{change.property_name}: {change.old} -> {change.new or '[vazio]'}" for change in changes)),
        "pages_with_changes": len({change.page_id for change in changes}),
        "applied_pages": sum(1 for item in apply_results if item.get("status") == "updated"),
        "failed_pages": sum(1 for item in apply_results if item.get("status") == "failed"),
        "schema_results_by_status": dict(Counter(item.get("status", "unknown") for item in schema_results)),
        "readback_mismatches": len(readback_result.get("mismatches", [])),
        "obsolete_used_after": {
            prop: len(values)
            for prop, values in (readback_result.get("obsolete_values_still_used", {}) or {}).items()
        },
        "obsolete_schema_options_after": {
            prop: len(values)
            for prop, values in (readback_result.get("obsolete_schema_options", {}) or {}).items()
        },
        "unused_schema_options_after": {
            prop: len(values)
            for prop, values in (readback_result.get("unused_schema_options", {}) or {}).items()
        },
        "artifact_dir": str(artifact_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_reports(
        artifact_dir,
        changes=changes,
        apply_results=apply_results,
        schema_results=schema_results,
        readback_result=readback_result,
        duplicate_clusters_before=duplicate_clusters_before,
        summary=summary,
    )
    LOGGER.info("Relatorios gravados em %s", artifact_dir)
    LOGGER.info("Resumo: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
