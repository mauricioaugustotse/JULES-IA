from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from local_secrets import get_secret
from tse_backfill_2025_notion import notion_page_to_row
from tse_normalization import (
    extract_uf_from_text,
    normalize_origem_value,
    parse_multi_value_text,
)
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    GENERAL_NEWS_LIMIT,
    NOTION_PROPERTY_MAP,
    NotionDataSourceSchema,
    NotionSessoesClient,
    PublishPreviewRow,
    build_fallback_tema,
    extract_youtube_video_id,
    infer_classe_from_row_text,
    infer_pedido_vista_from_row_text,
    infer_punchline_from_row_text,
    infer_relator_from_row_text,
    infer_resultado_from_row_text,
    infer_theme_from_row_text,
    infer_votacao_from_row_text,
    is_plausible_ministro_name,
    normalize_class_text,
    normalize_classe_processo,
    normalize_eleicao_value,
    normalize_ministro_name,
    normalize_resultado_final,
    normalize_tre,
    normalize_votacao,
    punchline_looks_generic,
    should_replace_classe_processo,
    tema_looks_generic,
)


LOGGER = logging.getLogger("audit_notion_sessoes")
ARTIFACT_ROOT = Path("artifacts") / "notion_sessoes_audit"
APPLY_SLEEP_SECONDS = 0.12
TEXT_FIELDS = {"tema", "punchline"}
STRUCTURED_FIELDS = {
    "classe_processo",
    "eleicao",
    "origem",
    "tribunal",
    "relator",
    "pedido_vista",
    "resultado",
    "votacao",
}
SUPPORTED_UPDATE_FIELDS = TEXT_FIELDS | STRUCTURED_FIELDS | {"tipo_registro"}


@dataclass
class PageRecord:
    page: dict[str, Any]
    row: PublishPreviewRow
    index: int
    page_id: str
    page_url: str
    video_id: str
    timestamp_seconds: int | None


@dataclass
class FieldChange:
    page_id: str
    page_url: str
    data_sessao: str
    video_id: str
    timestamp_seconds: int | None
    numero_processo: str
    tema: str
    field: str
    old: str
    new: str
    reason: str
    confidence: str = "high"

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_url": self.page_url,
            "data_sessao": self.data_sessao,
            "video_id": self.video_id,
            "timestamp_seconds": self.timestamp_seconds,
            "numero_processo": self.numero_processo,
            "tema": self.tema,
            "field": self.field,
            "old": self.old,
            "new": self.new,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class PageChangeSet:
    record: PageRecord
    changes: dict[str, FieldChange] = field(default_factory=dict)

    def add(self, field_name: str, new_value: str, reason: str, confidence: str = "high") -> None:
        old_value = string_value(getattr(self.record.row, field_name, ""))
        new_value = string_value(new_value)
        if old_value == new_value:
            return
        if field_name not in SUPPORTED_UPDATE_FIELDS:
            raise ValueError(f"Campo nao suportado para update: {field_name}")
        self.changes[field_name] = FieldChange(
            page_id=self.record.page_id,
            page_url=self.record.page_url,
            data_sessao=self.record.row.data_sessao,
            video_id=self.record.video_id,
            timestamp_seconds=self.record.timestamp_seconds,
            numero_processo=self.record.row.numero_processo,
            tema=self.record.row.tema,
            field=field_name,
            old=old_value,
            new=new_value,
            reason=reason,
            confidence=confidence,
        )
        setattr(self.record.row, field_name, new_value)


def string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    return str(value or "").strip()


def extract_youtube_timestamp_or_none(youtube_link: str) -> int | None:
    match = re.search(r"[?&]t=(\d+)", str(youtube_link or ""))
    if not match:
        return None
    try:
        return max(0, int(match.group(1)))
    except ValueError:
        return None


def first_lower(text: str) -> str:
    value = string_value(text).rstrip(".")
    if not value:
        return ""
    return value[:1].lower() + value[1:]


def clean_sentence(text: str, max_len: int = 180) -> str:
    value = re.sub(r"\s+", " ", string_value(text)).strip(" .;:-")
    if not value:
        return ""
    if len(value) > max_len:
        value = value[:max_len].rsplit(" ", 1)[0].rstrip(" .;:-")
    value = value[:1].upper() + value[1:]
    return value if value.endswith((".", "!", "?")) else f"{value}."


def canonical_ministro(value: str) -> str:
    candidate = normalize_ministro_name(value) if value else ""
    candidate = re.sub(r"^Min\.\s+Min\s+", "Min. ", candidate).strip()
    if not candidate or not is_plausible_ministro_name(candidate):
        return ""
    return candidate


def looks_copied_from_analysis(punchline: str, analysis: str) -> bool:
    punch = normalize_class_text(punchline)
    body = normalize_class_text(analysis)
    if not punch or not body:
        return False
    if body.startswith(punch[: min(len(punch), 80)]):
        return True
    if len(punch) >= 60 and punch[:100] in body[:260]:
        return True
    return False


def weak_punchline_reason(row: PublishPreviewRow) -> str:
    current = string_value(row.punchline)
    normalized = normalize_class_text(current)
    if punchline_looks_generic(current, row):
        return "punchline generica, truncada ou metajuridica"
    if looks_copied_from_analysis(current, row.analise_do_conteudo_juridico):
        return "punchline copiada do inicio da analise"
    if re.match(r"^(?:o processo|o caso|o recurso|a consulta|o julgamento)\s+(?:trata|discute|versa)\b", normalized):
        return "punchline descritiva fraca"
    if len(current) < 55 and re.search(r"(?i)\b(?:RE|REspe|AREspe|ADI|ADO|art\.?|lei|resolu[cç][aã]o)\b", current):
        return "punchline curta demais e baseada em citacao"
    return ""


def candidate_looks_bad_punchline(value: str, row: PublishPreviewRow) -> bool:
    text = string_value(value)
    normalized = normalize_class_text(text)
    if punchline_looks_generic(text, row):
        return True
    if re.match(r"^\d+[º°]?\s+(?:do|da|de|,|$)", text, flags=re.IGNORECASE):
        return True
    if re.match(r"^(?:sumula|súmula|tema\s+\d+|RE\s+\d+|REspe\s+\d+|AREspe\s+\d+|ADI\s+\d+|ADO\s+\d+)\b", text, flags=re.IGNORECASE):
        return True
    if re.match(r"^(?:art(?:\.|igo)?|lei|resolu[cç][aã]o|codigo eleitoral|código eleitoral)\b", normalized):
        return True
    if normalized.endswith((" por", " de", " do", " da", " em", " que")):
        return True
    return False


def theme_subject(row: PublishPreviewRow) -> str:
    preferred = string_value(row.tema)
    if weak_theme_reason(row):
        inferred = infer_theme_from_row_text(row)
        if inferred and not weak_theme_reason(row.model_copy(update={"tema": inferred})):
            preferred = inferred
    return first_lower(preferred)


def fallback_editorial_punchline(row: PublishPreviewRow) -> str:
    subject = theme_subject(row)
    if not subject:
        return ""
    result = string_value(row.resultado)
    if result == "Suspenso por vista":
        if string_value(row.pedido_vista):
            return clean_sentence(f"Pedido de vista de {row.pedido_vista} interrompe julgamento sobre {subject}")
        return clean_sentence(f"Pedido de vista interrompe julgamento sobre {subject}")
    if result in {"Desprovido", "Desprovida", "Desprovadas"}:
        return clean_sentence(f"A Corte mantém a decisão no julgamento sobre {subject}")
    if result in {"Provido", "Provido em parte", "Parcialmente deferido"}:
        return clean_sentence(f"A Corte reforma o resultado no julgamento sobre {subject}")
    if result in {"Nao conhecido", "Nao conhecida", "Não conhecido", "Não conhecida"}:
        return clean_sentence(f"A Corte não conhece da discussão sobre {subject}")
    if result in {"Deferido", "Aprovada", "Aprovadas", "Referendada", "Referendado"}:
        return clean_sentence(f"A Corte acolhe o encaminhamento sobre {subject}")
    if result in {"Indeferido", "Indeferida", "Rejeitada", "Rejeitados"}:
        return clean_sentence(f"A Corte rejeita a pretensão ligada a {subject}")
    if result in {"Procedente", "Procedente em parte", "Improcedente"}:
        return clean_sentence(f"A Corte define o mérito da controvérsia sobre {subject}")
    return clean_sentence(f"Julgamento concentra a discussão em {subject}")


def better_punchline(row: PublishPreviewRow) -> str:
    candidate = infer_punchline_from_row_text(row)
    if candidate and not candidate_looks_bad_punchline(candidate, row):
        current_norm = normalize_class_text(row.punchline)
        candidate_norm = normalize_class_text(candidate)
        if candidate_norm and candidate_norm != current_norm:
            return clean_sentence(candidate)
    fallback = fallback_editorial_punchline(row)
    if fallback and not candidate_looks_bad_punchline(fallback, row):
        return fallback
    return ""


def weak_theme_reason(row: PublishPreviewRow) -> str:
    current = string_value(row.tema)
    normalized = normalize_class_text(current)
    if tema_looks_generic(current, row):
        return "tema generico ou metajuridico"
    if normalized in {
        "adiamento de julgamento por pedido de vista",
        "julgamento adiado por pedido de vista",
    }:
        return "tema generico de adiamento"
    if len(current) > 125:
        return "tema longo demais"
    if re.match(r"^(?:o processo|o caso|o recurso|a consulta|o julgamento)\s+(?:trata|discute|versa)\b", normalized):
        return "tema em formato de frase descritiva"
    if re.match(r"^(?:quanto ao|no merito|alem disso|por fim)\b", normalized):
        return "tema iniciado por conectivo de analise"
    if normalized.endswith(" que") or normalized.endswith(" de"):
        return "tema truncado"
    if re.match(r"^decisao do tre\b.*\bque$", normalized):
        return "tema truncado"
    if re.match(r"^(?:min\.?\s+)?[a-z]+(?:\s+[a-z]+){0,3},\s+(?:votou|entendeu|afirmou|destacou)\b", normalized):
        return "tema capturado a partir de fala ou voto"
    if re.search(r"\b(?:votou pelo|entendeu que|destacou que|foi aprovado|foram aprovadas|foram desaprovadas)\b", normalized):
        return "tema com formula de resultado"
    if re.match(r"^(?:RE|REspe|AREspe|ADI|ADO|AgR|ED)\s+\d", current, flags=re.IGNORECASE):
        return "tema baseado em citacao processual"
    return ""


def better_theme(row: PublishPreviewRow) -> str:
    inferred = infer_theme_from_row_text(row)
    if normalize_class_text(inferred) in {
        "adiamento de julgamento por pedido de vista",
        "julgamento adiado por pedido de vista",
    }:
        return ""
    if normalize_class_text(inferred) in {
        "inelegibilidade dos investigados",
        "inelegibilidade dos envolvidos",
        "inelegibilidade do vice-prefeito",
    } and len(string_value(row.tema)) > 70:
        return ""
    if inferred and not weak_theme_reason(row.model_copy(update={"tema": inferred})):
        return clean_sentence(inferred, max_len=125).rstrip(".")
    fallback = build_fallback_tema(row)
    if fallback and not weak_theme_reason(row.model_copy(update={"tema": fallback})):
        return clean_sentence(fallback, max_len=125).rstrip(".")
    return ""


def is_suspicious_schema_option(property_name: str, option_name: str) -> bool:
    option = string_value(option_name)
    normalized = normalize_class_text(option)
    if not option:
        return False
    if property_name in {"relator", "pedido_vista"}:
        if any(
            marker in normalized
            for marker in [
                "ressaltou",
                "suspendeu",
                "divergiu",
                "para aguardar",
                "para analise",
                "para análise",
                "visa ",
                " que",
                "fundamenta",
                "conclusoes do relator",
                "conclusões do relator",
            ]
        ):
            return True
        return option.startswith("Min.") and not is_plausible_ministro_name(option)
    if property_name == "origem":
        return any(
            marker in normalized
            for marker in [
                "prefeito e vice",
                "assistencia e cidadania",
                "assistência e cidadania",
                "financeiro do",
                "resolucao do",
                "resolução do",
                "tribunal regional federal",
            ]
        )
    if property_name == "tipo_registro":
        return not bool(re.match(r"^Julgamento\s+\d+$", option, flags=re.IGNORECASE))
    return False


def copy_row(row: PublishPreviewRow) -> PublishPreviewRow:
    return row.model_copy(deep=True)


def load_records(client: NotionSessoesClient, schema: NotionDataSourceSchema) -> list[PageRecord]:
    records: list[PageRecord] = []
    for index, page in enumerate(client.query_data_source()):
        row = notion_page_to_row(client, schema, page)
        page_id = string_value(page.get("id"))
        row.page_id = page_id
        youtube_link = string_value(row.youtube_link)
        records.append(
            PageRecord(
                page=page,
                row=row,
                index=index,
                page_id=page_id,
                page_url=string_value(page.get("url")),
                video_id=extract_youtube_video_id(youtube_link) or "",
                timestamp_seconds=extract_youtube_timestamp_or_none(youtube_link),
            )
        )
    return records


def add_structured_changes(change_set: PageChangeSet) -> None:
    row = change_set.record.row
    working = copy_row(row)

    normalized_classe = normalize_classe_processo(working.classe_processo)
    inferred_classe = infer_classe_from_row_text(working)
    if inferred_classe and should_replace_classe_processo(normalized_classe, inferred_classe, working):
        normalized_classe = inferred_classe
    elif not normalized_classe:
        normalized_classe = inferred_classe
    if normalized_classe and normalized_classe != row.classe_processo:
        change_set.add("classe_processo", normalized_classe, "classe_processo normalizado ou inferido da linha")
        working.classe_processo = normalized_classe

    normalized_eleicao = normalize_eleicao_value(working.eleicao)
    if normalized_eleicao and normalized_eleicao != row.eleicao:
        change_set.add("eleicao", normalized_eleicao, "eleicao normalizada")
        working.eleicao = normalized_eleicao

    normalized_origem = normalize_origem_value(working.origem)
    if normalized_origem and normalized_origem != row.origem:
        change_set.add("origem", normalized_origem, "origem normalizada")
        working.origem = normalized_origem

    uf = extract_uf_from_text(normalized_origem or working.origem)
    normalized_tribunal = normalize_tre(working.tribunal, uf)
    if normalized_tribunal and normalized_tribunal != row.tribunal:
        change_set.add("tribunal", normalized_tribunal, "tribunal normalizado a partir da origem/UF")
        working.tribunal = normalized_tribunal

    normalized_relator = canonical_ministro(working.relator)
    if not normalized_relator:
        normalized_relator = canonical_ministro(infer_relator_from_row_text(working))
    if normalized_relator and normalized_relator != row.relator:
        change_set.add("relator", normalized_relator, "relator normalizado ou inferido da linha")
        working.relator = normalized_relator

    normalized_pedido_vista = canonical_ministro(working.pedido_vista)
    if not normalized_pedido_vista:
        normalized_pedido_vista = canonical_ministro(infer_pedido_vista_from_row_text(working))
    if normalized_pedido_vista and normalized_pedido_vista != row.pedido_vista:
        change_set.add("pedido_vista", normalized_pedido_vista, "pedido_vista normalizado ou inferido da linha")
        working.pedido_vista = normalized_pedido_vista

    normalized_resultado = normalize_resultado_final(working.resultado, working.classe_processo)
    if not normalized_resultado:
        normalized_resultado = infer_resultado_from_row_text(working)
    if normalized_resultado and normalized_resultado != row.resultado:
        change_set.add("resultado", normalized_resultado, "resultado normalizado ou inferido da linha")
        working.resultado = normalized_resultado

    normalized_votacao = normalize_votacao(working.votacao)
    if not normalized_votacao:
        normalized_votacao = infer_votacao_from_row_text(working)
    if working.resultado == "Suspenso por vista" and not normalized_votacao:
        normalized_votacao = "Suspenso"
    if normalized_votacao and normalized_votacao != row.votacao:
        change_set.add("votacao", normalized_votacao, "votacao normalizada ou inferida da linha")


def add_text_changes(change_set: PageChangeSet) -> None:
    row = change_set.record.row
    theme_reason = weak_theme_reason(row)
    if theme_reason:
        candidate_theme = better_theme(row)
        if candidate_theme and normalize_class_text(candidate_theme) != normalize_class_text(row.tema):
            change_set.add("tema", candidate_theme, theme_reason, confidence="medium")

    punch_reason = weak_punchline_reason(row)
    if punch_reason:
        candidate_punchline = better_punchline(row)
        if candidate_punchline and normalize_class_text(candidate_punchline) != normalize_class_text(row.punchline):
            change_set.add("punchline", candidate_punchline, punch_reason, confidence="medium")


def current_tipo_number(record: PageRecord) -> int:
    match = re.search(r"(\d+)", string_value(record.row.tipo_registro))
    if not match:
        return 10**9
    return int(match.group(1))


def tipo_sort_key(record: PageRecord) -> tuple[int, int, int, str]:
    if record.timestamp_seconds is None:
        return (1, current_tipo_number(record), record.index, record.page_id)
    return (0, record.timestamp_seconds, current_tipo_number(record), record.page_id)


def add_tipo_registro_changes(change_sets: dict[str, PageChangeSet], records: list[PageRecord]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    groups: dict[str, list[PageRecord]] = defaultdict(list)
    for record in records:
        if record.video_id:
            groups[record.video_id].append(record)
        else:
            warnings.append(
                {
                    "page_id": record.page_id,
                    "page_url": record.page_url,
                    "reason": "sem video_id no youtube_link; tipo_registro nao sequenciado automaticamente",
                    "tipo_registro": record.row.tipo_registro,
                    "youtube_link": record.row.youtube_link,
                }
            )

    for video_id, group in groups.items():
        ordered = sorted(group, key=tipo_sort_key)
        has_untimestamped = any(record.timestamp_seconds is None for record in ordered)
        for position, record in enumerate(ordered, start=1):
            expected = f"Julgamento {position}"
            if record.row.tipo_registro != expected:
                reason = "tipo_registro resequenciado por video_id e timestamp do YouTube"
                confidence = "high"
                if record.timestamp_seconds is None or has_untimestamped:
                    confidence = "medium"
                    reason += "; grupo contem linha sem timestamp confiavel"
                change_sets[record.page_id].add("tipo_registro", expected, reason, confidence=confidence)
        if has_untimestamped:
            warnings.extend(
                {
                    "page_id": record.page_id,
                    "page_url": record.page_url,
                    "video_id": video_id,
                    "reason": "linha sem timestamp ficou apos linhas timestampadas no mesmo video",
                    "tipo_registro": record.row.tipo_registro,
                    "youtube_link": record.row.youtube_link,
                }
                for record in ordered
                if record.timestamp_seconds is None
            )
    return warnings


def build_audit(records: list[PageRecord]) -> tuple[dict[str, PageChangeSet], list[dict[str, Any]]]:
    change_sets = {record.page_id: PageChangeSet(record=copy.deepcopy(record)) for record in records}
    for change_set in change_sets.values():
        add_structured_changes(change_set)
        add_text_changes(change_set)
    tipo_warnings = add_tipo_registro_changes(change_sets, [change_set.record for change_set in change_sets.values()])
    return change_sets, tipo_warnings


def collect_schema_usage(records: list[PageRecord], property_name: str) -> Counter[str]:
    usage: Counter[str] = Counter()
    for record in records:
        value = record.page.get("properties", {}).get(property_name, {})
        prop_type = value.get("type")
        if prop_type == "select":
            name = string_value((value.get("select") or {}).get("name"))
            if name:
                usage[name] += 1
        elif prop_type == "status":
            name = string_value((value.get("status") or {}).get("name"))
            if name:
                usage[name] += 1
        elif prop_type == "multi_select":
            for item in value.get("multi_select", []) or []:
                name = string_value(item.get("name"))
                if name:
                    usage[name] += 1
        else:
            text = string_value(getattr(record.row, property_name, ""))
            if text:
                for item in parse_multi_value_text(text):
                    usage[item] += 1
    return usage


def audit_schema_options(schema: NotionDataSourceSchema, records: list[PageRecord]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for property_name in ["relator", "pedido_vista", "origem", "tipo_registro"]:
        prop = schema.properties.get(property_name)
        if not prop or prop.type not in {"select", "status"}:
            continue
        usage = collect_schema_usage(records, property_name)
        for option_name in prop.options:
            if not is_suspicious_schema_option(property_name, option_name):
                continue
            findings.append(
                {
                    "property": property_name,
                    "option": option_name,
                    "used_count": usage[option_name],
                    "cleanup_status": "pending",
                }
            )
    return findings


def property_payload_for_change(
    client: NotionSessoesClient,
    schema: NotionDataSourceSchema,
    field_name: str,
    new_value: str,
) -> tuple[str, dict[str, Any]]:
    if field_name == "tema":
        property_name = schema.title_property_name
    else:
        property_name = NOTION_PROPERTY_MAP[field_name]
    if property_name not in schema.properties:
        raise RuntimeError(f"Propriedade {property_name!r} nao encontrada no schema.")
    if new_value:
        built = client._build_property_value(schema, property_name, new_value)
    else:
        built = client._build_empty_property_value(schema, property_name)
    if built is None:
        raise RuntimeError(f"Nao foi possivel montar payload para {property_name!r}.")
    return property_name, built


def apply_page_changes(
    client: NotionSessoesClient,
    schema: NotionDataSourceSchema,
    page_id: str,
    changes: list[FieldChange],
) -> dict[str, Any]:
    payload: dict[str, Any] = {"properties": {}}
    for change in changes:
        property_name, built = property_payload_for_change(client, schema, change.field, change.new)
        payload["properties"][property_name] = built
    return client._request("PATCH", f"/pages/{page_id}", json=payload)


def cleanup_schema_options(
    client: NotionSessoesClient,
    schema: NotionDataSourceSchema,
    schema_findings: list[dict[str, Any]],
    *,
    apply_changes: bool,
) -> list[dict[str, Any]]:
    updated_findings = copy.deepcopy(schema_findings)
    by_property: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for finding in updated_findings:
        by_property[finding["property"]].append(finding)

    for property_name, findings in by_property.items():
        prop = schema.raw_payload.get("properties", {}).get(property_name, {})
        prop_type = prop.get("type")
        if prop_type not in {"select", "status"}:
            for finding in findings:
                finding["cleanup_status"] = "skipped_unsupported_property_type"
            continue
        options = ((prop.get(prop_type) or {}).get("options") or [])
        if len(options) > 100:
            for finding in findings:
                finding["cleanup_status"] = "skipped_option_payload_over_100"
            continue
        remove_names = {finding["option"] for finding in findings if int(finding.get("used_count") or 0) == 0}
        if not remove_names:
            for finding in findings:
                finding["cleanup_status"] = "kept_still_used"
            continue
        remaining_options = [
            {"name": string_value(option.get("name")), "color": string_value(option.get("color") or "default")}
            for option in options
            if string_value(option.get("name")) and string_value(option.get("name")) not in remove_names
        ]
        if not apply_changes:
            for finding in findings:
                finding["cleanup_status"] = (
                    "would_remove" if finding["option"] in remove_names else "kept_still_used"
                )
            continue
        try:
            client._request(
                "PATCH",
                f"/data_sources/{client.data_source_id}",
                json={"properties": {property_name: {prop_type: {"options": remaining_options}}}},
            )
        except Exception as exc:
            for finding in findings:
                finding["cleanup_status"] = f"failed: {exc}"
            continue
        for finding in findings:
            finding["cleanup_status"] = "removed" if finding["option"] in remove_names else "kept_still_used"
    return updated_findings


def write_reports(
    artifact_dir: Path,
    changes: list[FieldChange],
    schema_findings: list[dict[str, Any]],
    tipo_warnings: list[dict[str, Any]],
    apply_results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    change_dicts = [change.as_dict() for change in changes]
    (artifact_dir / "changes.json").write_text(json.dumps(change_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    with (artifact_dir / "changes.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(change_dicts[0].keys()) if change_dicts else list(FieldChange("", "", "", "", None, "", "", "", "", "", "").as_dict().keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(change_dicts)
    (artifact_dir / "schema_findings.json").write_text(json.dumps(schema_findings, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "tipo_registro_warnings.json").write_text(json.dumps(tipo_warnings, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "apply_results.json").write_text(json.dumps(apply_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_summary(
    records: list[PageRecord],
    changes: list[FieldChange],
    schema_findings: list[dict[str, Any]],
    apply_results: list[dict[str, Any]],
    apply_mode: bool,
) -> dict[str, Any]:
    changes_by_field = Counter(change.field for change in changes)
    changes_by_confidence = Counter(change.confidence for change in changes)
    applied_pages = sum(1 for item in apply_results if item.get("status") == "updated")
    failed_pages = sum(1 for item in apply_results if item.get("status") == "failed")
    return {
        "mode": "apply" if apply_mode else "dry-run",
        "total_records": len(records),
        "total_field_changes": len(changes),
        "pages_with_changes": len({change.page_id for change in changes}),
        "changes_by_field": dict(sorted(changes_by_field.items())),
        "changes_by_confidence": dict(sorted(changes_by_confidence.items())),
        "schema_findings": len(schema_findings),
        "schema_findings_by_status": dict(Counter(item.get("cleanup_status", "unknown") for item in schema_findings)),
        "applied_pages": applied_pages,
        "failed_pages": failed_pages,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita e corrige a database Notion de sessoes do TSE.")
    parser.add_argument("--apply", action="store_true", help="Aplica as correcoes no Notion. Sem isto, roda em dry-run.")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--max-pages", type=int, default=0, help="Limita paginas atualizadas no modo apply.")
    parser.add_argument("--only-field", action="append", choices=sorted(SUPPORTED_UPDATE_FIELDS), help="Aplica/relata apenas um campo especifico. Pode repetir.")
    parser.add_argument("--skip-schema-cleanup", action="store_true")
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
    LOGGER.info("Carregando schema e paginas do Notion...")
    schema = client.fetch_schema()
    records = load_records(client, schema)
    LOGGER.info("Paginas carregadas: %s", len(records))

    change_sets, tipo_warnings = build_audit(records)
    changes = [change for change_set in change_sets.values() for change in change_set.changes.values()]
    if args.only_field:
        selected = set(args.only_field)
        changes = [change for change in changes if change.field in selected]

    schema_findings = audit_schema_options(schema, records)
    if args.skip_schema_cleanup:
        for finding in schema_findings:
            finding["cleanup_status"] = "skipped_by_flag"
    else:
        schema_findings = cleanup_schema_options(client, schema, schema_findings, apply_changes=args.apply)

    apply_results: list[dict[str, Any]] = []
    if args.apply:
        by_page: dict[str, list[FieldChange]] = defaultdict(list)
        for change in changes:
            by_page[change.page_id].append(change)
        page_items = list(by_page.items())
        if args.max_pages > 0:
            page_items = page_items[: args.max_pages]
        LOGGER.info("Aplicando correcoes em %s paginas...", len(page_items))
        for page_index, (page_id, page_changes) in enumerate(page_items, start=1):
            try:
                apply_page_changes(client, schema, page_id, page_changes)
                apply_results.append(
                    {
                        "page_id": page_id,
                        "status": "updated",
                        "fields": [change.field for change in page_changes],
                    }
                )
            except Exception as exc:
                apply_results.append(
                    {
                        "page_id": page_id,
                        "status": "failed",
                        "fields": [change.field for change in page_changes],
                        "error": str(exc),
                    }
                )
                LOGGER.warning("Falha ao atualizar pagina %s: %s", page_id, exc)
            if page_index % 25 == 0:
                LOGGER.info("Paginas processadas: %s/%s", page_index, len(page_items))
            time.sleep(APPLY_SLEEP_SECONDS)

    summary = build_summary(records, changes, schema_findings, apply_results, apply_mode=args.apply)
    write_reports(artifact_dir, changes, schema_findings, tipo_warnings, apply_results, summary)
    LOGGER.info("Relatorios gravados em %s", artifact_dir)
    LOGGER.info("Resumo: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
