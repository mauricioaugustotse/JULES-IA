from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_backfill_2025_notion import notion_page_to_row
from tse_normalization import (
    dedupe_preserve_order,
    is_empty_advogados_value,
    is_mpe_noise_entry,
    normalize_advogados_list,
    normalize_text,
    normalize_token,
    split_csv_like_text,
)
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient


LOGGER = logging.getLogger("cleanup_notion_sessoes_advogados")
ARTIFACT_ROOT = Path("artifacts") / "notion_sessoes_advogados_cleanup"
APPLY_SLEEP_SECONDS = 0.08

LAWYER_TITLE_RE = re.compile(r"(?i)^(?:drs?\.?|dras?\.?|doutores|doutoras|doutor|doutora)\s+")
TRAILING_CONTEXT_RE = re.compile(r"\s*\(([^()]*)\)\s*\.?\s*$")
EXPLICIT_TITLED_CONJUNCTION_RE = re.compile(
    r"\s+(?:e|/)\s+(?=(?:dr\.|dra\.|drs\.|dras\.|doutor|doutora)\s+)",
    flags=re.IGNORECASE,
)
OTHER_LAWYERS_RE = re.compile(r"(?i)\s+e\s+outr[oa]s?\s*$")
ADVOGADO_PREFIX_RE = re.compile(r"(?i)^(?:advogad[oa]s?|adv\.?|patron[oa]s?|defensor(?:a|es|as)?)\s*:?\s*")
INVALID_NAME_RE = re.compile(
    r"(?i)^(?:"
    r"sem\s+informa[cç][aã]o|n[ãa]o\s+informad[oa]s?|n[ãa]o\s+mencionad[oa]s?|"
    r"advogad[oa]s?|mpe|mp|ministerio publico eleitoral|minist[eé]rio p[uú]blico eleitoral"
    r")$"
)
LAW_FIRM_RE = re.compile(r"(?i)\b(?:sociedade\s+de\s+)?advogad[oa]s?\b")
NAME_PARTICLES = {"de", "da", "do", "das", "dos", "e", "di", "del", "della"}
FEMALE_FIRST_NAMES = {
    "angela",
    "camila",
    "carla",
    "daniane",
    "edna",
    "erika",
    "gabriela",
    "isiquiele",
    "juliana",
    "karina",
    "larissa",
    "luciana",
    "mariana",
    "marilda",
    "maria",
    "marina",
    "natalia",
    "tamara",
    "taynara",
    "valeska",
    "virginia",
}
CURATED_ALIAS_TARGETS = {
    "andre maimoni": "andre brandao henrique maimoni",
    "angelo ferraro": "angelo longo ferraro",
    "eduardo alckmin": "jose eduardo rangel alckmin",
    "eugenio aragao": "eugenio jose guilherme aragao",
    "fabiano feitosa": "fabiano freire feitosa",
    "fernando neisser": "fernando gaspar neisser",
    "henrique neves": "henrique neves silva",
    "joelson dias": "joelson costa dias",
    "jose eduardo alckmin": "jose eduardo rangel alckmin",
    "karina kufa": "karina paula kufa",
    "luciana lossio": "luciana christina guimaraes lossio",
    "maria claudia": "maria claudia bucchianeri pinheiro",
    "maria claudia bucchianeri": "maria claudia bucchianeri pinheiro",
    "mauro menezes": "mauro azevedo menezes",
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


@dataclass
class ParsedAdvogado:
    raw: str
    display: str
    key: str
    changed: bool
    reasons: list[str]


def strip_accents(value: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch))


def name_without_title(value: str) -> str:
    text = normalize_text(str(value or "")).strip()
    text = re.sub(r"(?i)^(dr\.|dra\.)\s+prof(?:essor|essora)?\.?\s+", r"\1 ", text)
    text = LAWYER_TITLE_RE.sub("", text).strip()
    text = re.sub(r"(?i)^prof(?:essor|essora)?\.?\s+", "", text).strip()
    return text.strip(" .;:,")


def title_case_if_all_caps(name: str) -> str:
    letters = [ch for ch in name if ch.isalpha()]
    if not letters or any(ch.islower() for ch in letters):
        return name
    words: list[str] = []
    for word in name.split():
        normalized = normalize_token(word)
        if normalized in NAME_PARTICLES:
            words.append(normalized)
        else:
            words.append(word[:1].upper() + word[1:].lower())
    return " ".join(words)


def standardize_advogado_display(value: str, *, title_case_all_caps: bool = True) -> str:
    text = normalize_text(str(value or "")).strip()
    text = re.sub(r"(?i)^(dr\.|dra\.)\s+prof(?:essor|essora)?\.?\s+", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip(" .;:,")
    prefix_match = LAWYER_TITLE_RE.match(text)
    prefix = prefix_match.group(0).strip() if prefix_match else ""
    name = name_without_title(text)
    if title_case_all_caps:
        name = title_case_if_all_caps(name)
    first = normalize_token(name.split()[0]) if name.split() else ""
    if first in FEMALE_FIRST_NAMES:
        prefix = "Dra."
    elif prefix:
        prefix = "Dra." if prefix.lower().startswith(("dra", "doutora")) else "Dr."
    else:
        prefix = "Dr."
    return f"{prefix} {name}".strip()


def name_tokens(value: str, *, keep_particles: bool = False) -> list[str]:
    normalized = normalize_token(name_without_title(value).replace("-", " "))
    tokens = [token for token in re.split(r"\s+", normalized) if token]
    if keep_particles:
        return tokens
    return [token for token in tokens if token not in NAME_PARTICLES]


def advogado_key(value: str) -> str:
    tokens = name_tokens(value)
    if not tokens:
        return ""
    return " ".join(tokens)


def looks_invalid_advogado(value: str) -> bool:
    text = normalize_text(str(value or "")).strip()
    if not text:
        return True
    without_title = normalize_token(name_without_title(text))
    if not without_title or INVALID_NAME_RE.fullmatch(without_title):
        return True
    if LAW_FIRM_RE.search(name_without_title(text)):
        return True
    if is_empty_advogados_value(text) or is_mpe_noise_entry(text):
        return True
    return False


def strip_context_suffix(value: str) -> tuple[str, bool]:
    text = normalize_text(str(value or "")).strip()
    changed = False
    while True:
        match = TRAILING_CONTEXT_RE.search(text)
        if not match:
            break
        text = text[: match.start()].strip()
        changed = True
    text = text.rstrip(" .;:,").strip()
    return text, changed


def remove_other_lawyers_marker(value: str) -> tuple[str, bool]:
    text = normalize_text(str(value or "")).strip()
    new_text = OTHER_LAWYERS_RE.sub("", text).strip()
    return new_text, new_text != text


def split_explicit_compound_label(value: str) -> list[str]:
    text = normalize_text(str(value or "")).strip()
    if not text:
        return []
    parts = [part.strip() for part in EXPLICIT_TITLED_CONJUNCTION_RE.split(text) if part.strip()]
    return parts or [text]


def normalize_single_advogado(value: str) -> list[str]:
    normalized = normalize_advogados_list(value)
    if not normalized:
        return []
    return split_csv_like_text(normalized)


def expand_advogado_value(value: str) -> list[ParsedAdvogado]:
    raw = normalize_text(str(value or "")).strip()
    if not raw or looks_invalid_advogado(raw):
        return []

    candidates = normalize_single_advogado(raw) or [raw]
    parsed: list[ParsedAdvogado] = []
    for candidate in candidates:
        no_context, context_changed = strip_context_suffix(candidate)
        no_context = ADVOGADO_PREFIX_RE.sub("", no_context).strip()
        for piece in split_explicit_compound_label(no_context):
            without_others, others_changed = remove_other_lawyers_marker(piece)
            normalized_parts = normalize_single_advogado(without_others) or [without_others]
            for normalized in normalized_parts:
                display, second_context_changed = strip_context_suffix(normalized)
                display, second_others_changed = remove_other_lawyers_marker(display)
                display = display.rstrip(" .;:,").strip()
                structural_change = (
                    context_changed
                    or second_context_changed
                    or others_changed
                    or second_others_changed
                    or piece != no_context
                    or bool(re.search(r"(?i)\bprof(?:essor|essora)?\.?\b", display))
                )
                display = standardize_advogado_display(display, title_case_all_caps=structural_change)
                if not display or looks_invalid_advogado(display):
                    continue
                key = advogado_key(display)
                if not key:
                    continue
                reasons: list[str] = []
                if context_changed or second_context_changed:
                    reasons.append("sufixo contextual removido")
                if others_changed or second_others_changed:
                    reasons.append("marcador 'e outros' removido")
                if piece != no_context:
                    reasons.append("etiqueta com mais de um advogado desmembrada")
                if display != raw:
                    reasons.append("nome normalizado")
                parsed.append(ParsedAdvogado(raw=raw, display=display, key=key, changed=display != raw, reasons=dedupe_preserve_order(reasons)))
    deduped: list[ParsedAdvogado] = []
    seen: set[str] = set()
    for item in parsed:
        if item.display in seen:
            continue
        seen.add(item.display)
        deduped.append(item)
    return deduped


def tokens_are_subsequence(short: list[str], long: list[str]) -> bool:
    if not short:
        return False
    cursor = 0
    for token in long:
        if token == short[cursor]:
            cursor += 1
            if cursor == len(short):
                return True
    return False


def build_alias_roots(keys: list[str]) -> dict[str, str]:
    unique_keys = sorted(set(key for key in keys if key), key=lambda item: (len(item.split()), item))
    tokens_by_key = {key: key.split() for key in unique_keys}
    root_by_key = {key: key for key in unique_keys}
    for key in unique_keys:
        curated_target = CURATED_ALIAS_TARGETS.get(key, "")
        if curated_target in root_by_key:
            root_by_key[key] = curated_target
            continue
        short_tokens = tokens_by_key[key]
        if len(short_tokens) < 3:
            continue
        candidates: list[tuple[int, str]] = []
        for other in unique_keys:
            if other == key:
                continue
            long_tokens = tokens_by_key[other]
            if len(long_tokens) <= len(short_tokens):
                continue
            if tokens_are_subsequence(short_tokens, long_tokens):
                candidates.append((len(long_tokens), other))
        if len(candidates) == 1:
            root_by_key[key] = candidates[0][1]
        elif candidates:
            candidates.sort(reverse=True)
            top = candidates[0]
            if len(candidates) == 1 or top[0] > candidates[1][0]:
                root_by_key[key] = top[1]
    changed = True
    while changed:
        changed = False
        for key, root in list(root_by_key.items()):
            final = root_by_key.get(root, root)
            if final != root:
                root_by_key[key] = final
                changed = True
    return root_by_key


def display_quality(display: str, count: int) -> tuple[int, int, int, int, int]:
    tokens = name_tokens(display, keep_particles=True)
    accent_count = sum(1 for ch in display if ord(ch) > 127)
    has_title = 1 if LAWYER_TITLE_RE.match(display) else 0
    return (len([token for token in tokens if token not in NAME_PARTICLES]), has_title, accent_count, count, len(display))


def build_canonical_map(all_values: list[list[str]]) -> tuple[dict[str, str], dict[str, str], dict[str, Counter[str]]]:
    displays_by_key: dict[str, Counter[str]] = defaultdict(Counter)
    keys: list[str] = []
    for values in all_values:
        for raw in values:
            for parsed in expand_advogado_value(raw):
                displays_by_key[parsed.key][parsed.display] += 1
                keys.append(parsed.key)
    root_by_key = build_alias_roots(keys)
    displays_by_root: dict[str, Counter[str]] = defaultdict(Counter)
    for key, displays in displays_by_key.items():
        root = root_by_key.get(key, key)
        displays_by_root[root].update(displays)
    canonical_by_root: dict[str, str] = {}
    for root, displays in displays_by_root.items():
        canonical_by_root[root] = max(
            displays,
            key=lambda display: display_quality(display, displays[display]),
        )
    canonical_by_key = {key: canonical_by_root[root_by_key.get(key, key)] for key in displays_by_key}
    return canonical_by_key, root_by_key, displays_by_root


def sanitize_advogados_values(
    values: list[str],
    canonical_by_key: dict[str, str],
    root_by_key: dict[str, str],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    new_values: list[str] = []
    reasons: list[str] = []
    transformations: list[dict[str, Any]] = []
    for raw in values:
        parsed_values = expand_advogado_value(raw)
        if not parsed_values:
            if raw:
                reasons.append("etiqueta vazia ou espuria removida")
                transformations.append({"old": raw, "new": [], "reason": "etiqueta vazia ou espuria removida"})
            continue
        for parsed in parsed_values:
            canonical = canonical_by_key.get(parsed.key, parsed.display)
            item_reasons = list(parsed.reasons)
            if canonical != parsed.display:
                item_reasons.append("alias consolidado em nome canonico")
            if root_by_key.get(parsed.key, parsed.key) != parsed.key:
                item_reasons.append("duplicidade falsa reduzida ao nome completo")
            item_reasons = dedupe_preserve_order(item_reasons)
            if canonical not in new_values:
                new_values.append(canonical)
            reasons.extend(item_reasons)
            if canonical != raw:
                transformations.append({"old": raw, "new": canonical, "reason": "; ".join(item_reasons) or "nome normalizado"})
    return dedupe_preserve_order(new_values), dedupe_preserve_order(reasons), transformations


def ambiguous_advogado_values(all_values: list[list[str]], canonical_by_key: dict[str, str]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for values in all_values:
        for raw in values:
            for parsed in expand_advogado_value(raw):
                canonical = canonical_by_key.get(parsed.key, parsed.display)
                tokens = name_tokens(canonical)
                if len(tokens) <= 1:
                    counter[canonical] += 1
                    examples.setdefault(canonical, raw)
    return [
        {"value": value, "count": count, "example_raw": examples.get(value, "")}
        for value, count in counter.most_common()
    ]


def build_payload(change: FieldChange) -> dict[str, Any]:
    return {"multi_select": [{"name": str(item)} for item in change.new if str(item).strip()]}


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
            results.append(
                {"page_id": page_id, "status": "failed", "fields": [change.field for change in page_changes], "error": str(exc)}
            )
        if index % 50 == 0:
            LOGGER.info("Paginas processadas: %s/%s", index, len(items))
        time.sleep(APPLY_SLEEP_SECONDS)
    return results


def collect_used_values(pages: list[dict[str, Any]], property_name: str) -> list[str]:
    used: list[str] = []
    seen: set[str] = set()
    for page in pages:
        values = [
            str(item.get("name", "")).strip()
            for item in page.get("properties", {}).get(property_name, {}).get("multi_select", [])
            if str(item.get("name", "")).strip()
        ]
        for value in values:
            if value not in seen:
                seen.add(value)
                used.append(value)
    return used


def audit_advogados_schema(client: NotionSessoesClient, pages: list[dict[str, Any]]) -> dict[str, Any]:
    schema = client.fetch_schema()
    prop = schema.raw_payload.get("properties", {}).get("advogados", {})
    options = prop.get("multi_select", {}).get("options", []) if prop.get("type") == "multi_select" else []
    used = collect_used_values(pages, "advogados")
    used_set = set(used)
    unused = [
        str(option.get("name", "")).strip()
        for option in options
        if str(option.get("name", "")).strip() and str(option.get("name", "")).strip() not in used_set
    ]
    nondefault = [option for option in options if (option.get("color") or "default") != "default"]
    status = "no_unused_options" if not unused else "skipped_over_100_used_options"
    if unused and len(used) <= 100:
        status = "cleanup_possible_but_not_run"
    return {
        "property": "advogados",
        "schema_options": len(options),
        "used_options": len(used),
        "unused_options": len(unused),
        "nondefault_color_options": len(nondefault),
        "status": status,
        "note": (
            "A API do Notion nao permite alterar cores de opcoes existentes; remocao fisica de opcoes "
            "ociosas so e segura quando o payload final fica dentro do limite de opcoes."
        ),
        "source": "https://developers.notion.com/reference/update-data-source-properties",
    }


def build_audit(client: NotionSessoesClient) -> tuple[list[dict[str, Any]], list[FieldChange], dict[str, Any]]:
    schema = client.fetch_schema()
    pages = client.query_data_source()
    rows = [notion_page_to_row(client, schema, page) for page in pages]
    all_values = [list(row.advogados or []) for row in rows]
    canonical_by_key, root_by_key, displays_by_root = build_canonical_map(all_values)
    changes: list[FieldChange] = []
    transformations_by_page: dict[str, list[dict[str, Any]]] = {}
    for page, row in zip(pages, rows):
        old_values = list(row.advogados or [])
        new_values, reasons, transformations = sanitize_advogados_values(old_values, canonical_by_key, root_by_key)
        if new_values != old_values:
            page_id = str(page.get("id", ""))
            changes.append(
                FieldChange(
                    page_id=page_id,
                    page_url=str(page.get("url", "")),
                    numero_processo=row.numero_processo,
                    tema=row.tema,
                    field="advogados",
                    property_name="advogados",
                    old=old_values,
                    new=new_values,
                    reason="; ".join(reasons) or "advogados padronizados",
                )
            )
            transformations_by_page[page_id] = transformations
    metadata = {
        "canonical_groups": [
            {
                "root_key": root,
                "canonical": max(displays, key=lambda display: display_quality(display, displays[display])),
                "variants": dict(displays.most_common()),
            }
            for root, displays in sorted(displays_by_root.items())
            if len(displays) > 1
        ],
        "ambiguous_values": ambiguous_advogado_values(all_values, canonical_by_key),
        "transformations_by_page": transformations_by_page,
    }
    return pages, changes, metadata


def write_reports(
    artifact_dir: Path,
    changes: list[FieldChange],
    apply_results: list[dict[str, Any]],
    schema_result: dict[str, Any],
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    change_dicts = [change.as_dict() for change in changes]
    (artifact_dir / "changes.json").write_text(json.dumps(change_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    with (artifact_dir / "changes.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(change_dicts[0].keys()) if change_dicts else list(FieldChange("", "", "", "", "", "", [], [], "").as_dict().keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(change_dicts)
    (artifact_dir / "apply_results.json").write_text(json.dumps(apply_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "schema_result.json").write_text(json.dumps(schema_result, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Padroniza nomes da coluna advogados na database Notion sessoes.")
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

    LOGGER.info("Carregando paginas e montando auditoria de advogados...")
    pages, changes, metadata = build_audit(client)
    LOGGER.info("Paginas lidas: %s; mudancas propostas: %s", len(pages), len(changes))
    apply_results: list[dict[str, Any]] = []
    if args.apply and changes:
        apply_results = apply_page_changes(client, changes, max_pages=args.max_pages)
    fresh_pages = client.query_data_source() if args.apply else pages
    schema_result = audit_advogados_schema(client, fresh_pages)
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "total_records": len(pages),
        "total_changes": len(changes),
        "changes_by_field": dict(Counter(change.field for change in changes)),
        "pages_with_changes": len({change.page_id for change in changes}),
        "applied_pages": sum(1 for item in apply_results if item.get("status") == "updated"),
        "failed_pages": sum(1 for item in apply_results if item.get("status") == "failed"),
        "schema_status": schema_result.get("status"),
        "ambiguous_values": len(metadata.get("ambiguous_values", [])),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_reports(artifact_dir, changes, apply_results, schema_result, metadata, summary)
    LOGGER.info("Relatorios gravados em %s", artifact_dir)
    LOGGER.info("Resumo: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
