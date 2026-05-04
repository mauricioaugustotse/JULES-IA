from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from audit_notion_sessoes_round2 import strict_origem_value
from local_secrets import get_secret
from tse_backfill_2025_notion import notion_page_to_row
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    DEFAULT_NOTION_VERSION,
    NotionSessoesClient,
)


LOGGER = logging.getLogger("cleanup_notion_sessoes_followup")
ARTIFACT_ROOT = Path("artifacts") / "notion_sessoes_followup"
RESIDUE_RENAMES = {
    "origem__legacy_cleanup": "zz_residuo_lixeira_origem",
    "partes__legacy_cleanup": "zz_residuo_lixeira_partes",
    "composicao__tmp_cleanup": "zz_residuo_lixeira_composicao",
}
RESIDUE_NAME_PREFIX = "zz_residuo_lixeira_"
REQUEST_RETRIES = 5
RETRY_BASE_SECONDS = 1.7

CONTAMINATED_PEDIDO_RE = re.compile(
    r"(?i)(?:"
    r"\bfoi\b|\bvisa\b|\bsuspendeu\b|\bsuspender\b|\bpara\b|"
    r"\bdivergiu\b|\bacompanhou\b|\bressaltou\b|\btamb[eé]m\b|"
    r"\bque\b|\brelatoria\b|\ban[aá]lise\b|motiv|aguardar|"
    r"defini[cç][aã]o|\btese\b|mat[eé]ria|\brelator\b|\brelatora\b"
    r")|['\"]|\be\s+Ministro\b"
)
MINISTER_ALIASES = {
    "admar gonzaga": "Min. Admar Gonzaga",
    "alexandre de moraes": "Min. Alexandre de Moraes",
    "andre mendonca": "Min. André Mendonça",
    "andré mendonça": "Min. André Mendonça",
    "antonio carlos ferreira": "Min. Antônio Carlos Ferreira",
    "antônio carlos ferreira": "Min. Antônio Carlos Ferreira",
    "carmen lucia": "Min. Cármen Lúcia",
    "cármen lúcia": "Min. Cármen Lúcia",
    "carlos horbach": "Min. Carlos Horbach",
    "dias toffoli": "Min. Dias Toffoli",
    "edilene lobo": "Min. Edilene Lôbo",
    "edilene lôbo": "Min. Edilene Lôbo",
    "edson fachin": "Min. Edson Fachin",
    "estela aranha": "Min. Estela Aranha",
    "floriano de azevedo marques": "Min. Floriano de Azevedo Marques",
    "gilmar mendes": "Min. Gilmar Mendes",
    "herman benjamin": "Min. Herman Benjamin",
    "henrique neves": "Min. Henrique Neves",
    "henrique neves da silva": "Min. Henrique Neves da Silva",
    "humberto martins": "Min. Humberto Martins",
    "isabel gallotti": "Min. Isabel Gallotti",
    "jorge mussi": "Min. Jorge Mussi",
    "luciana lossio": "Min. Luciana Lóssio",
    "luciana lóssio": "Min. Luciana Lóssio",
    "luis felipe salomao": "Min. Luís Felipe Salomão",
    "luís felipe salomão": "Min. Luís Felipe Salomão",
    "luis salomao": "Min. Luís Felipe Salomão",
    "luís salomão": "Min. Luís Felipe Salomão",
    "luiz fux": "Min. Luiz Fux",
    "luís fux": "Min. Luiz Fux",
    "luiz edson fachin": "Min. Luiz Edson Fachin",
    "luís edson fachin": "Min. Luís Edson Fachin",
    "maria claudia bucchianeri": "Min. Maria Claudia Bucchianeri",
    "maria claudia bucchianeri pinheiro": "Min. Maria Claudia Bucchianeri Pinheiro",
    "marco aurelio": "Min. Marco Aurélio",
    "marco aurélio": "Min. Marco Aurélio",
    "mauro campbell": "Min. Mauro Campbell Marques",
    "mauro campbell marques": "Min. Mauro Campbell Marques",
    "napoleao maia": "Min. Napoleão Maia",
    "napoleão maia": "Min. Napoleão Maia",
    "napoleao nunes maia filho": "Min. Napoleão Nunes Maia Filho",
    "napoleão nunes maia filho": "Min. Napoleão Nunes Maia Filho",
    "nunes marques": "Min. Nunes Marques",
    "og fernandes": "Min. Og Fernandes",
    "ramos tavares": "Min. Ramos Tavares",
    "raul araujo": "Min. Raul Araújo",
    "raul araújo": "Min. Raul Araújo",
    "ricardo lewandowski": "Min. Ricardo Lewandowski",
    "rosa weber": "Min. Rosa Weber",
    "sebastiao reis junior": "Min. Sebastião Reis Júnior",
    "sebastião reis júnior": "Min. Sebastião Reis Júnior",
    "sergio banhos": "Min. Sérgio Banhos",
    "sérgio banhos": "Min. Sérgio Banhos",
    "tarcisio vieira": "Min. Tarcísio Vieira de Carvalho Neto",
    "tarcísio vieira": "Min. Tarcísio Vieira de Carvalho Neto",
    "teori zavascki": "Min. Teori Zavascki",
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


def normalize_token(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def report_value(value: Any) -> Any:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return str(value or "").strip()


def prop_value(page: dict[str, Any], property_name: str) -> str | list[str]:
    prop = page.get("properties", {}).get(property_name, {})
    prop_type = prop.get("type")
    if prop_type == "select":
        return ((prop.get("select") or {}).get("name") or "").strip()
    if prop_type == "multi_select":
        return [
            item.get("name", "").strip()
            for item in prop.get("multi_select", []) or []
            if item.get("name", "").strip()
        ]
    return ""


def empty_payload_for_property(prop_type: str) -> dict[str, Any]:
    if prop_type == "select":
        return {"select": None}
    if prop_type == "multi_select":
        return {"multi_select": []}
    raise ValueError(f"Tipo nao suportado para limpeza: {prop_type}")


def request_with_retry(
    api_key: str,
    method: str,
    path: str,
    *,
    json_payload: dict[str, Any] | None = None,
    data_source_id: str = DEFAULT_NOTION_DATA_SOURCE_ID,
) -> dict[str, Any]:
    url = f"https://api.notion.com/v1{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": DEFAULT_NOTION_VERSION,
        "Content-Type": "application/json",
    }
    last_error = ""
    for attempt in range(1, REQUEST_RETRIES + 1):
        response = requests.request(method, url, headers=headers, json=json_payload, timeout=60)
        if response.status_code < 400:
            return response.json() if response.content else {}
        last_error = response.text
        retryable = response.status_code in {429, 500, 502, 503, 504}
        if not retryable or attempt == REQUEST_RETRIES:
            raise RuntimeError(f"Notion API error {response.status_code}: {response.text}")
        retry_after = response.headers.get("Retry-After")
        try:
            retry_delay = float(retry_after or 0)
        except ValueError:
            retry_delay = 0
        time.sleep(max(retry_delay, RETRY_BASE_SECONDS ** attempt))
    raise RuntimeError(f"Falha na API do Notion: {last_error}")


def build_valid_minister_names(schema_options: list[str]) -> list[str]:
    names = set(MINISTER_ALIASES.values())
    for value in schema_options:
        text = str(value or "").strip()
        if not text.startswith("Min."):
            continue
        if CONTAMINATED_PEDIDO_RE.search(text):
            continue
        if len(text.split()) < 2 or normalize_token(text) in {"min", "min que"}:
            continue
        names.add(text)
    return sorted(names, key=lambda item: len(normalize_token(item)), reverse=True)


def first_minister_match(value: str, valid_names: list[str]) -> str:
    normalized = normalize_token(value)
    matches: list[tuple[int, int, str]] = []
    for name in valid_names:
        bare = normalize_token(name.removeprefix("Min.").strip())
        if not bare:
            continue
        match = re.search(rf"\b{re.escape(bare)}\b", normalized)
        if match:
            matches.append((match.start(), -len(bare), name))
    if matches:
        return sorted(matches)[0][2]
    for alias, canonical in MINISTER_ALIASES.items():
        alias_norm = normalize_token(alias)
        match = re.search(rf"\b{re.escape(alias_norm)}\b", normalized)
        if match:
            matches.append((match.start(), -len(alias_norm), canonical))
    return sorted(matches)[0][2] if matches else ""


def canonical_pedido_vista(value: str, valid_names: list[str]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    matched = first_minister_match(text, valid_names)
    if CONTAMINATED_PEDIDO_RE.search(text):
        return matched
    if matched and normalize_token(text) == normalize_token(matched):
        return matched
    if matched and text.startswith("Min.") and not CONTAMINATED_PEDIDO_RE.search(text):
        return matched
    return matched


def build_audit(client: NotionSessoesClient) -> tuple[list[dict[str, Any]], list[FieldChange], Counter[str]]:
    schema = client.fetch_schema()
    pages = client.query_data_source()
    valid_names = build_valid_minister_names(schema.properties.get("pedido_vista").options if "pedido_vista" in schema.properties else [])
    changes: list[FieldChange] = []
    counters: Counter[str] = Counter()
    residue_props = {
        name: prop
        for name, prop in schema.properties.items()
        if name in RESIDUE_RENAMES or name.startswith(RESIDUE_NAME_PREFIX)
    }

    for page in pages:
        row = notion_page_to_row(client, schema, page)
        page_id = page.get("id", "")
        page_url = page.get("url", "")

        proposed_origin, reason = strict_origem_value(row)
        if proposed_origin and proposed_origin != row.origem:
            changes.append(
                FieldChange(page_id, page_url, row.numero_processo, row.tema, "origem", "origem", row.origem, proposed_origin, reason)
            )
            counters["origem"] += 1

        if row.pedido_vista:
            proposed_pedido = canonical_pedido_vista(row.pedido_vista, valid_names)
            if proposed_pedido != row.pedido_vista:
                reason = "pedido_vista reduzido ao nome canonico do ministro" if proposed_pedido else "pedido_vista contaminado sem ministro inferivel; campo limpo"
                changes.append(
                    FieldChange(
                        page_id,
                        page_url,
                        row.numero_processo,
                        row.tema,
                        "pedido_vista",
                        "pedido_vista",
                        row.pedido_vista,
                        proposed_pedido,
                        reason,
                    )
                )
                counters["pedido_vista"] += 1

        for property_name, prop_schema in residue_props.items():
            current = prop_value(page, property_name)
            if current not in ("", []):
                changes.append(
                    FieldChange(
                        page_id,
                        page_url,
                        row.numero_processo,
                        row.tema,
                        "residue",
                        property_name,
                        report_value(current),
                        [],
                        "valor de coluna residual limpo para evitar leitura como dado ativo",
                    )
                )
                counters[f"residue:{property_name}"] += 1
    return pages, changes, counters


def build_payload_for_changes(schema: Any, page_changes: list[FieldChange]) -> dict[str, Any]:
    payload: dict[str, Any] = {"properties": {}}
    for change in page_changes:
        prop_name = change.property_name
        prop = schema.properties[prop_name]
        if change.field in {"origem", "pedido_vista"}:
            if change.new:
                payload["properties"][prop_name] = {"select": {"name": str(change.new)}}
            else:
                payload["properties"][prop_name] = {"select": None}
            continue
        payload["properties"][prop_name] = empty_payload_for_property(prop.type)
    return payload


def apply_page_changes(
    api_key: str,
    client: NotionSessoesClient,
    changes: list[FieldChange],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    schema = client.fetch_schema()
    by_page: dict[str, list[FieldChange]] = {}
    for change in changes:
        by_page.setdefault(change.page_id, []).append(change)

    def update_one(item: tuple[str, list[FieldChange]]) -> dict[str, Any]:
        page_id, page_changes = item
        payload = build_payload_for_changes(schema, page_changes)
        request_with_retry(api_key, "PATCH", f"/pages/{page_id}", json_payload=payload)
        return {"page_id": page_id, "status": "updated", "fields": [change.property_name for change in page_changes]}

    results: list[dict[str, Any]] = []
    items = list(by_page.items())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(update_one, item): item for item in items}
        for index, future in enumerate(as_completed(futures), start=1):
            page_id, page_changes = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "page_id": page_id,
                        "status": "failed",
                        "fields": [change.property_name for change in page_changes],
                        "error": str(exc),
                    }
                )
                LOGGER.warning("Falha ao atualizar pagina %s: %s", page_id, exc)
            if index % 50 == 0:
                LOGGER.info("Paginas atualizadas: %s/%s", index, len(items))
    return results


def patch_schema_options_and_residue_names(
    api_key: str,
    client: NotionSessoesClient,
    *,
    apply_changes: bool,
) -> list[dict[str, Any]]:
    schema = client.fetch_schema()
    pages = client.query_data_source()
    findings: list[dict[str, Any]] = []

    for property_name in ["pedido_vista", *RESIDUE_RENAMES.keys(), *RESIDUE_RENAMES.values()]:
        if property_name not in schema.properties:
            continue
        prop = schema.properties[property_name]
        raw_prop = schema.raw_payload.get("properties", {}).get(property_name, {})
        if prop.type not in {"select", "multi_select"}:
            continue
        used: list[str] = []
        seen: set[str] = set()
        for page in pages:
            value = prop_value(page, property_name)
            values = value if isinstance(value, list) else [value]
            for item in values:
                if item and item not in seen:
                    seen.add(item)
                    used.append(item)
        options = ((raw_prop.get(prop.type) or {}).get("options") or [])
        remaining = [
            {"name": str(option.get("name", "")).strip(), "color": str(option.get("color") or "default")}
            for option in options
            if str(option.get("name", "")).strip() in seen
        ]
        unused_count = len([option for option in options if str(option.get("name", "")).strip() not in seen])
        status = "no_unused_options" if unused_count == 0 else "would_patch_options"
        if apply_changes and unused_count:
            try:
                request_with_retry(
                    api_key,
                    "PATCH",
                    f"/data_sources/{client.data_source_id}",
                    json_payload={"properties": {property_name: {prop.type: {"options": remaining}}}},
                )
                status = "patched_options"
            except Exception as exc:
                status = "patch_options_failed"
                findings.append({"property": property_name, "status": status, "unused_options": unused_count, "error": str(exc)})
                continue
        findings.append({"property": property_name, "status": status, "used_options": len(used), "unused_options": unused_count})

    if apply_changes:
        schema = client.fetch_schema()
    for old_name, new_name in RESIDUE_RENAMES.items():
        if old_name not in schema.properties:
            continue
        if new_name in schema.properties:
            findings.append({"property": old_name, "status": "rename_skipped_target_exists", "target": new_name})
            continue
        if not apply_changes:
            findings.append({"property": old_name, "status": "would_rename_residue", "target": new_name})
            continue
        try:
            client.rename_property(old_name, new_name)
            findings.append({"property": old_name, "status": "renamed_residue", "target": new_name})
        except Exception as exc:
            findings.append({"property": old_name, "status": "rename_failed", "target": new_name, "error": str(exc)})
    return findings


def write_reports(
    artifact_dir: Path,
    changes: list[FieldChange],
    apply_results: list[dict[str, Any]],
    schema_results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    change_dicts = [change.as_dict() for change in changes]
    (artifact_dir / "changes.json").write_text(json.dumps(change_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    with (artifact_dir / "changes.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(change_dicts[0].keys()) if change_dicts else list(
            FieldChange("", "", "", "", "", "", "", "", "").as_dict().keys()
        )
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(change_dicts)
    (artifact_dir / "apply_results.json").write_text(json.dumps(apply_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "schema_results.json").write_text(json.dumps(schema_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limpeza complementar da database Notion sessoes.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    api_key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY/NOTION_TOKEN nao encontrado.")
    client = NotionSessoesClient(api_key=api_key, data_source_id=args.data_source_id)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")

    LOGGER.info("Carregando paginas e montando auditoria complementar...")
    pages, changes, counters = build_audit(client)
    LOGGER.info("Paginas lidas: %s; mudancas propostas: %s", len(pages), len(changes))
    apply_results: list[dict[str, Any]] = []
    schema_results: list[dict[str, Any]] = []
    if args.apply and changes:
        LOGGER.info("Aplicando mudancas em paginas com %s workers...", args.workers)
        apply_results = apply_page_changes(api_key, client, changes, workers=args.workers)
    LOGGER.info("Normalizando opcoes de schema e nomes de residuos...")
    schema_results = patch_schema_options_and_residue_names(api_key, client, apply_changes=args.apply)
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "total_records": len(pages),
        "total_changes": len(changes),
        "changes_by_kind": dict(counters),
        "pages_with_changes": len({change.page_id for change in changes}),
        "applied_pages": sum(1 for item in apply_results if item.get("status") == "updated"),
        "failed_pages": sum(1 for item in apply_results if item.get("status") == "failed"),
        "schema_results_by_status": dict(Counter(item.get("status", "unknown") for item in schema_results)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_reports(artifact_dir, changes, apply_results, schema_results, summary)
    LOGGER.info("Relatorios gravados em %s", artifact_dir)
    LOGGER.info("Resumo: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
