from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes import canonical_ministro
from audit_notion_sessoes_round2 import (
    UF_CAPITALS,
    infer_uf_from_tribunal,
    normalized_partes_values,
    notion_request_with_retry,
)
from cleanup_notion_sessoes_followup import (
    CONTAMINATED_PEDIDO_RE,
    MINISTER_ALIASES,
    build_valid_minister_names,
    canonical_pedido_vista,
    normalize_token,
    prop_value,
)
from local_secrets import get_secret
from tse_backfill_2025_notion import notion_page_to_row
from tse_normalization import extract_uf_from_text, normalize_class_text
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NotionDataSourceSchema,
    NotionSessoesClient,
    PublishPreviewRow,
    normalize_composition_list,
)


LOGGER = logging.getLogger("cleanup_notion_sessoes_fine")
ARTIFACT_ROOT = Path("artifacts") / "notion_sessoes_fine_cleanup"
APPLY_SLEEP_SECONDS = 0.08

RELATOR_CONTAMINATION_RE = re.compile(
    r"(?i)\b(?:"
    r"discute|entendeu|votou|destacou|ressaltou|concluiu|julgou|"
    r"anuncia|solicita|fundamenta|pelo desprovimento|divergiu|justifica|"
    r"relator|relatora|processo|recurso|implementa|equ[ií]voco|que|para"
    r")\b"
)
ORIGEM_ORGAN_RE = re.compile(
    r"(?i)^(?:"
    r"prefeitura|prefeito(?:\\s+e\\s+vice-prefeito)?|vice-prefeito|"
    r"tribunal|tre|tse|zona eleitoral|ju[ií]zo|secretaria|governo|munic[ií]pio"
    r")\b"
)
UF_PAIR_RE = re.compile(r"^(?P<left>[A-Z]{2})\s*/\s*(?P<right>[A-Z]{2})$")

MINISTER_CANONICAL_ALIASES = {
    **MINISTER_ALIASES,
    "admar": "Min. Admar Gonzaga",
    "alexandre moraes": "Min. Alexandre de Moraes",
    "andre de almeida mendonca": "Min. André Mendonça",
    "andré de almeida mendonça": "Min. André Mendonça",
    "benito goncalves": "Min. Benedito Gonçalves",
    "benito gonçalves": "Min. Benedito Gonçalves",
    "carlos mario velloso": "Min. Carlos Mário Velloso Filho",
    "carlos mário velloso": "Min. Carlos Mário Velloso Filho",
    "fachin": "Min. Edson Fachin",
    "fux": "Min. Luiz Fux",
    "luiz edson fachin": "Min. Edson Fachin",
    "luís edson fachin": "Min. Edson Fachin",
    "maria claudia bucchianeri": "Min. Maria Cláudia Bucchianeri",
    "maria claudia bucchianeri pinheiro": "Min. Maria Cláudia Bucchianeri Pinheiro",
    "mauro campbell": "Min. Mauro Campbell Marques",
    "og": "Min. Og Fernandes",
    "rosa": "Min. Rosa Weber",
    "salomao": "Min. Luís Felipe Salomão",
    "salomão": "Min. Luís Felipe Salomão",
    "sergio": "Min. Sérgio Banhos",
    "sérgio": "Min. Sérgio Banhos",
    "tarcisio": "Min. Tarcísio Vieira de Carvalho Neto",
    "tarcísio": "Min. Tarcísio Vieira de Carvalho Neto",
}


@dataclass
class FieldChange:
    page_id: str
    page_url: str
    numero_processo: str
    tema: str
    field: str
    property_name: str
    old: Any
    new: Any
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_url": self.page_url,
            "numero_processo": self.numero_processo,
            "tema": self.tema,
            "field": self.field,
            "property_name": self.property_name,
            "old": self.old,
            "new": self.new,
            "reason": self.reason,
        }


def report_value(value: Any) -> Any:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return str(value or "").strip()


def canonical_minister_label(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = normalize_token(text.removeprefix("Min.").strip())
    if normalized in MINISTER_CANONICAL_ALIASES:
        return MINISTER_CANONICAL_ALIASES[normalized]
    direct = canonical_ministro(text)
    if direct and not RELATOR_CONTAMINATION_RE.search(direct):
        return direct
    pedido = canonical_pedido_vista(text, build_valid_minister_names(list(MINISTER_CANONICAL_ALIASES.values())))
    if pedido and not RELATOR_CONTAMINATION_RE.search(pedido):
        return pedido
    return ""


def sanitize_relator(row: PublishPreviewRow, valid_names: list[str]) -> tuple[str, str]:
    current = str(row.relator or "").strip()
    if not current:
        return "", ""
    if RELATOR_CONTAMINATION_RE.search(current) or not re.fullmatch(r"Min\.\s+[^,.;:()\[\]\n]{2,80}", current):
        candidate = canonical_pedido_vista(current, valid_names)
        if candidate and candidate != current and not RELATOR_CONTAMINATION_RE.search(candidate):
            return candidate, "relator contaminado reduzido ao nome canonico do ministro"
        return "", "relator contaminado sem ministro inferivel; campo limpo"
    canonical = canonical_minister_label(current)
    if canonical and canonical != current:
        return canonical, "relator padronizado para nome canonico"
    return current, ""


def capital_for_uf(uf: str) -> str:
    return UF_CAPITALS.get(str(uf or "").upper(), "")


def sanitize_origem(row: PublishPreviewRow) -> tuple[str, str]:
    current = str(row.origem or "").strip()
    if not current:
        return "", ""
    uf_pair = UF_PAIR_RE.match(current)
    if uf_pair:
        uf = infer_uf_from_tribunal(row.tribunal) or uf_pair.group("right")
        capital = capital_for_uf(uf)
        if capital:
            return capital, "origem em formato UF/UF substituida pela capital da UF"

    pref_match = re.match(
        r"(?i)^prefeitura\s+de\s+(?P<city>.+?)/(?P<uf>[A-Z]{2})$",
        current,
    )
    if pref_match:
        return f"{pref_match.group('city').strip()}/{pref_match.group('uf').upper()}", "origem continha Prefeitura; mantida apenas cidade/UF"

    mayor_match = re.match(
        r"(?i)^prefeito(?:\s+e\s+vice-prefeito)?\s+de\s+(?P<city>.+?)/(?P<uf>[A-Z]{2})$",
        current,
    )
    if mayor_match:
        return f"{mayor_match.group('city').strip()}/{mayor_match.group('uf').upper()}", "origem continha cargo politico; mantida apenas cidade/UF"

    normalized = normalize_class_text(current)
    if ORIGEM_ORGAN_RE.search(current):
        uf = extract_uf_from_text(current) or infer_uf_from_tribunal(row.tribunal)
        capital = capital_for_uf(uf)
        if capital:
            return capital, "origem continha orgao em vez de cidade; substituida pela capital da UF"

    if "/" not in current:
        uf = infer_uf_from_tribunal(row.tribunal)
        if uf:
            return f"{current}/{uf}", "origem trazia cidade sem UF; UF inferida do tribunal"

    if normalized in {"sp rj", "go go"}:
        uf = infer_uf_from_tribunal(row.tribunal) or current.rsplit("/", 1)[-1]
        capital = capital_for_uf(uf)
        if capital:
            return capital, "origem em formato UF/UF substituida pela capital da UF"
    return current, ""


def sanitize_composicao(values: list[str]) -> tuple[list[str], str]:
    if not values:
        return [], ""
    normalized = normalize_composition_list(values)
    repaired: list[str] = []
    seen: set[str] = set()
    for value in normalized:
        canonical = canonical_minister_label(value) or value
        if canonical and canonical not in seen:
            seen.add(canonical)
            repaired.append(canonical)
    if repaired != values:
        return repaired, "composicao padronizada por nomes canonicos de ministros"
    return values, ""


def sanitize_partes(values: list[str]) -> tuple[list[str], str]:
    if not values:
        return [], ""
    normalized = normalized_partes_values(values)
    if normalized and normalized != values:
        return normalized, "partes normalizadas para reduzir duplicidades falsas"
    return values, ""


def build_payload(change: FieldChange) -> dict[str, Any]:
    if change.property_name in {"relator", "origem"}:
        return {"select": {"name": str(change.new)}} if change.new else {"select": None}
    if change.property_name in {"composicao", "partes"}:
        return {"multi_select": [{"name": str(item)} for item in change.new if str(item).strip()]}
    raise ValueError(f"Propriedade sem payload: {change.property_name}")


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
    results: list[dict[str, Any]] = []
    for index, (page_id, page_changes) in enumerate(items, start=1):
        payload = {"properties": {change.property_name: build_payload(change) for change in page_changes}}
        try:
            notion_request_with_retry(client, "PATCH", f"/pages/{page_id}", json=payload)
            results.append({"page_id": page_id, "status": "updated", "fields": [change.field for change in page_changes]})
        except Exception as exc:
            LOGGER.warning("Falha ao atualizar %s: %s", page_id, exc)
            results.append({"page_id": page_id, "status": "failed", "fields": [change.field for change in page_changes], "error": str(exc)})
        if index % 50 == 0:
            LOGGER.info("Paginas processadas: %s/%s", index, len(items))
        time.sleep(APPLY_SLEEP_SECONDS)
    return results


def collect_usage(pages: list[dict[str, Any]], property_name: str, prop_type: str) -> list[str]:
    used: list[str] = []
    seen: set[str] = set()
    for page in pages:
        value = prop_value(page, property_name)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item and item not in seen:
                seen.add(item)
                used.append(item)
    return used


def cleanup_small_schema_options(
    client: NotionSessoesClient,
    schema: NotionDataSourceSchema,
    pages: list[dict[str, Any]],
    *,
    apply_changes: bool,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for property_name in ["relator", "composicao"]:
        prop = schema.properties.get(property_name)
        raw_prop = schema.raw_payload.get("properties", {}).get(property_name, {})
        if not prop or prop.type not in {"select", "multi_select"}:
            continue
        used = collect_usage(pages, property_name, prop.type)
        options = ((raw_prop.get(prop.type) or {}).get("options") or [])
        used_set = set(used)
        unused = [str(option.get("name", "")).strip() for option in options if str(option.get("name", "")).strip() not in used_set]
        if not unused:
            findings.append({"property": property_name, "status": "no_unused_options", "unused_options": 0, "used_options": len(used)})
            continue
        if len(used) > 100:
            findings.append({"property": property_name, "status": "skipped_over_100_used_options", "unused_options": len(unused), "used_options": len(used)})
            continue
        option_color_by_name = {
            str(option.get("name", "")).strip(): str(option.get("color") or "default")
            for option in options
            if str(option.get("name", "")).strip()
        }
        remaining_options = [{"name": value, "color": option_color_by_name.get(value, "default")} for value in used]
        if not apply_changes:
            findings.append({"property": property_name, "status": "would_patch_options", "unused_options": len(unused), "used_options": len(used)})
            continue
        try:
            notion_request_with_retry(
                client,
                "PATCH",
                f"/data_sources/{client.data_source_id}",
                json={"properties": {property_name: {prop.type: {"options": remaining_options}}}},
            )
            findings.append({"property": property_name, "status": "patched_options", "unused_options": len(unused), "used_options": len(used)})
        except Exception as exc:
            findings.append({"property": property_name, "status": "patch_failed", "unused_options": len(unused), "used_options": len(used), "error": str(exc)})
    return findings


def audit_schema_color_limitations(schema: NotionDataSourceSchema) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for property_name in ["origem", "partes", "advogados"]:
        prop = schema.properties.get(property_name)
        raw_prop = schema.raw_payload.get("properties", {}).get(property_name, {})
        if not prop or prop.type not in {"select", "multi_select"}:
            continue
        options = ((raw_prop.get(prop.type) or {}).get("options") or [])
        nondefault = [option for option in options if (option.get("color") or "default") != "default"]
        findings.append(
            {
                "property": property_name,
                "type": prop.type,
                "options": len(options),
                "nondefault_color_options": len(nondefault),
                "status": "blocked_by_notion_api_existing_option_color_immutable" if nondefault else "all_default",
                "source": "https://developers.notion.com/reference/update-data-source-properties",
            }
        )
    return findings


def build_audit(client: NotionSessoesClient) -> tuple[list[dict[str, Any]], list[FieldChange]]:
    schema = client.fetch_schema()
    pages = client.query_data_source()
    valid_names = build_valid_minister_names(schema.properties.get("relator").options if "relator" in schema.properties else [])
    changes: list[FieldChange] = []
    for page in pages:
        row = notion_page_to_row(client, schema, page)
        page_id = str(page.get("id", ""))
        page_url = str(page.get("url", ""))

        new_relator, reason = sanitize_relator(row, valid_names)
        if reason and new_relator != row.relator:
            changes.append(FieldChange(page_id, page_url, row.numero_processo, row.tema, "relator", "relator", row.relator, new_relator, reason))

        new_origem, reason = sanitize_origem(row)
        if reason and new_origem != row.origem:
            changes.append(FieldChange(page_id, page_url, row.numero_processo, row.tema, "origem", "origem", row.origem, new_origem, reason))

        new_composicao, reason = sanitize_composicao(list(row.composicao or []))
        if reason and new_composicao != row.composicao:
            changes.append(FieldChange(page_id, page_url, row.numero_processo, row.tema, "composicao", "composicao", list(row.composicao or []), new_composicao, reason))

        new_partes, reason = sanitize_partes(list(row.partes or []))
        if reason and new_partes != row.partes:
            changes.append(FieldChange(page_id, page_url, row.numero_processo, row.tema, "partes", "partes", list(row.partes or []), new_partes, reason))
    return pages, changes


def write_reports(
    artifact_dir: Path,
    changes: list[FieldChange],
    apply_results: list[dict[str, Any]],
    schema_results: list[dict[str, Any]],
    color_results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    change_dicts = [change.as_dict() for change in changes]
    (artifact_dir / "changes.json").write_text(json.dumps(change_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    with (artifact_dir / "changes.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(change_dicts[0].keys()) if change_dicts else list(FieldChange("", "", "", "", "", "", "", "", "").as_dict().keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(change_dicts)
    (artifact_dir / "apply_results.json").write_text(json.dumps(apply_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "schema_results.json").write_text(json.dumps(schema_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "color_results.json").write_text(json.dumps(color_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ajustes finos da database Notion sessoes.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--artifact-dir", default="")
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

    LOGGER.info("Carregando paginas e montando auditoria fina...")
    pages, changes = build_audit(client)
    LOGGER.info("Paginas lidas: %s; mudancas propostas: %s", len(pages), len(changes))
    apply_results: list[dict[str, Any]] = []
    if args.apply and changes:
        apply_results = apply_page_changes(client, changes, max_pages=args.max_pages)
    schema = client.fetch_schema()
    fresh_pages = client.query_data_source()
    schema_results = cleanup_small_schema_options(client, schema, fresh_pages, apply_changes=args.apply)
    color_results = audit_schema_color_limitations(client.fetch_schema())
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "total_records": len(pages),
        "total_changes": len(changes),
        "changes_by_field": dict(Counter(change.field for change in changes)),
        "pages_with_changes": len({change.page_id for change in changes}),
        "applied_pages": sum(1 for item in apply_results if item.get("status") == "updated"),
        "failed_pages": sum(1 for item in apply_results if item.get("status") == "failed"),
        "schema_results_by_status": dict(Counter(item.get("status", "unknown") for item in schema_results)),
        "color_results_by_status": dict(Counter(item.get("status", "unknown") for item in color_results)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_reports(artifact_dir, changes, apply_results, schema_results, color_results, summary)
    LOGGER.info("Relatorios gravados em %s", artifact_dir)
    LOGGER.info("Resumo: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
