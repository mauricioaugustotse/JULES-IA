from __future__ import annotations

import argparse
import json
import logging
import re
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient


LOGGER = logging.getLogger("cleanup_notion_composicao_numbered")
ARTIFACT_ROOT = Path("artifacts") / "notion_composicao_numbered_cleanup"
NUMBERED_RE = re.compile(r"^(?P<base>.+?)\s*\((?P<number>\d+)\)\s*$")
PAGE_UPDATE_WORKERS = 2
APPLY_SLEEP_SECONDS = 0.2
CANONICAL_ALIASES = {
    "min edilene lobo": "Min. Edilene Lôbo",
    "min estella aranha": "Min. Estela Aranha",
    "min maria claudia bucchianeri de pinheiro": "Min. Maria Cláudia Bucchianeri Pinheiro",
    "min maria claudia bucchianeri pinheiro": "Min. Maria Cláudia Bucchianeri Pinheiro",
}


@dataclass
class PageChange:
    page_id: str
    page_url: str
    old: list[str]
    new: list[str]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_url": self.page_url,
            "old": self.old,
            "new": self.new,
            "reason": self.reason,
        }


def composicao_values(page: dict[str, Any], property_name: str = "composicao") -> list[str]:
    return [
        str(item.get("name", "")).strip()
        for item in page.get("properties", {}).get(property_name, {}).get("multi_select", []) or []
        if str(item.get("name", "")).strip()
    ]


def strip_number_suffix(value: str) -> tuple[str, bool]:
    text = str(value or "").strip()
    match = NUMBERED_RE.match(text)
    if not match:
        return text, False
    return match.group("base").strip(), True


def normalize_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def canonical_composicao_value(value: str) -> tuple[str, bool]:
    stripped, numbered = strip_number_suffix(value)
    canonical = CANONICAL_ALIASES.get(normalize_key(stripped), stripped)
    return canonical, numbered or canonical != value


def normalize_values(values: list[str]) -> tuple[list[str], bool]:
    normalized: list[str] = []
    changed = False
    seen: set[str] = set()
    for value in values:
        candidate, item_changed = canonical_composicao_value(value)
        if item_changed:
            changed = True
        if candidate in seen:
            changed = True
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized, changed or normalized != values


def build_audit(client: NotionSessoesClient) -> tuple[list[dict[str, Any]], list[PageChange], dict[str, Any]]:
    schema = client.fetch_schema()
    pages = client.query_data_source()
    raw_prop = schema.raw_payload.get("properties", {}).get("composicao", {})
    options = [
        str(option.get("name", "")).strip()
        for option in raw_prop.get("multi_select", {}).get("options", []) or []
        if str(option.get("name", "")).strip()
    ]
    numbered_options = [option for option in options if NUMBERED_RE.match(option)]
    used_counter: Counter[str] = Counter()
    changes: list[PageChange] = []
    for page in pages:
        old_values = composicao_values(page)
        used_counter.update(old_values)
        new_values, changed = normalize_values(old_values)
        if changed:
            changes.append(
                PageChange(
                    page_id=str(page.get("id", "")),
                    page_url=str(page.get("url", "")),
                    old=old_values,
                    new=new_values,
                    reason="sufixo numerico removido de etiquetas de composicao",
                )
            )
    clusters: dict[str, list[str]] = defaultdict(list)
    for option in numbered_options:
        base, _changed = canonical_composicao_value(option)
        clusters[base].append(option)
    metadata = {
        "schema_options": len(options),
        "numbered_schema_options": len(numbered_options),
        "numbered_used_options": sum(1 for option in numbered_options if used_counter[option]),
        "clusters": {
            base: {
                "variants": variants,
                "used_counts": {variant: used_counter[variant] for variant in variants},
                "base_used": used_counter[base],
            }
            for base, variants in clusters.items()
        },
    }
    return pages, changes, metadata


def patch_page(client: NotionSessoesClient, change: PageChange) -> dict[str, Any]:
    notion_request_with_retry(
        client,
        "PATCH",
        f"/pages/{change.page_id}",
        json={"properties": {"composicao": {"multi_select": [{"name": value} for value in change.new]}}},
    )
    if APPLY_SLEEP_SECONDS:
        time.sleep(APPLY_SLEEP_SECONDS)
    return {"page_id": change.page_id, "status": "updated"}


def apply_page_changes(client: NotionSessoesClient, changes: list[PageChange], *, max_pages: int = 0) -> list[dict[str, Any]]:
    selected = changes[:max_pages] if max_pages > 0 else changes
    results: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=PAGE_UPDATE_WORKERS) as executor:
        futures = {executor.submit(patch_page, client, change): change for change in selected}
        for future in as_completed(futures):
            change = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"page_id": change.page_id, "status": "failed", "error": str(exc)}
            results.append(result)
            completed += 1
            if completed % 50 == 0:
                LOGGER.info("Paginas processadas: %s/%s", completed, len(selected))
    return results


def cleanup_schema_options(client: NotionSessoesClient, *, apply_changes: bool) -> dict[str, Any]:
    schema = client.fetch_schema()
    pages = client.query_data_source()
    raw_prop = schema.raw_payload.get("properties", {}).get("composicao", {})
    if raw_prop.get("type") != "multi_select":
        return {"property": "composicao", "status": "missing_or_not_multiselect"}
    options = raw_prop.get("multi_select", {}).get("options", []) or []
    color_by_name = {
        str(option.get("name", "")).strip(): str(option.get("color") or "default")
        for option in options
        if str(option.get("name", "")).strip()
    }
    used: list[str] = []
    seen: set[str] = set()
    for page in pages:
        for value in composicao_values(page):
            if value not in seen:
                seen.add(value)
                used.append(value)
    option_names = [str(option.get("name", "")).strip() for option in options if str(option.get("name", "")).strip()]
    unused = [name for name in option_names if name not in seen]
    numbered_left = [name for name in option_names if NUMBERED_RE.match(name)]
    payload_options = [{"name": value, "color": color_by_name.get(value, "default")} for value in used]
    if not apply_changes:
        return {
            "property": "composicao",
            "status": "would_patch_options" if unused else "no_unused_options",
            "used_options": len(used),
            "unused_options": len(unused),
            "numbered_schema_options": len(numbered_left),
            "unused_examples": unused[:50],
        }
    if unused and len(used) <= 100:
        notion_request_with_retry(
            client,
            "PATCH",
            f"/data_sources/{client.data_source_id}",
            json={"properties": {"composicao": {"multi_select": {"options": payload_options}}}},
        )
        return {
            "property": "composicao",
            "status": "patched_options",
            "used_options": len(used),
            "unused_options": len(unused),
            "numbered_schema_options_before": len(numbered_left),
        }
    return {
        "property": "composicao",
        "status": "skipped_schema_patch",
        "used_options": len(used),
        "unused_options": len(unused),
        "numbered_schema_options": len(numbered_left),
    }


def write_reports(
    artifact_dir: Path,
    changes: list[PageChange],
    apply_results: list[dict[str, Any]],
    schema_result: dict[str, Any],
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "changes.json").write_text(
        json.dumps([change.as_dict() for change in changes], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "apply_results.json").write_text(json.dumps(apply_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "schema_result.json").write_text(json.dumps(schema_result, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove sufixos numericos de etiquetas da coluna composicao.")
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
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M")
    client = NotionSessoesClient(api_key=api_key, data_source_id=args.data_source_id)
    pages, changes, metadata = build_audit(client)
    LOGGER.info("Paginas lidas: %s; mudancas propostas: %s", len(pages), len(changes))
    apply_results: list[dict[str, Any]] = []
    if args.apply and changes:
        apply_results = apply_page_changes(client, changes, max_pages=args.max_pages)
    schema_result = cleanup_schema_options(client, apply_changes=args.apply)
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "total_records": len(pages),
        "total_changes": len(changes),
        "pages_with_changes": len({change.page_id for change in changes}),
        "applied_pages": sum(1 for result in apply_results if result.get("status") == "updated"),
        "failed_pages": sum(1 for result in apply_results if result.get("status") == "failed"),
        "schema_status": schema_result.get("status"),
        "numbered_schema_options": metadata.get("numbered_schema_options"),
        "numbered_used_options": metadata.get("numbered_used_options"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_reports(artifact_dir, changes, apply_results, schema_result, metadata, summary)
    LOGGER.info("Relatorios gravados em %s", artifact_dir)
    LOGGER.info("Resumo: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
