from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from openai import APIError, OpenAI
from pydantic import BaseModel, Field

from tse_backfill_2025_notion import (
    BACKFILL_ROOT,
    ExistingPageRecord,
    RepairArtifactContext,
    _special_process_lookup_key,
    audit_existing_year,
    load_existing_pages_for_year_with_retry,
    load_repair_artifact_context,
)
from tse_normalization import (
    canonicalize_numero_processo,
    dedupe_preserve_order,
    normalize_origem_value,
    parse_multi_value_text,
)
from tse_youtube_notion_core import (
    NotionDataSourceSchema,
    NotionSessoesClient,
    PublishPreviewRow,
    build_runtime_context,
    infer_classe_from_row_text,
    infer_origin_from_row_text,
    infer_punchline_from_row_text,
    normalize_composition_list,
    punchline_looks_generic,
    tema_looks_generic,
    validate_preview_row,
)


LOGGER = logging.getLogger("super_auditor")
DEFAULT_MODEL = os.getenv("SUPER_AUDITOR_OPENAI_MODEL") or "gpt-5.4"
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("SUPER_AUDITOR_TIMEOUT_SECONDS") or "90")
DEFAULT_RETRIES = int(os.getenv("SUPER_AUDITOR_RETRIES") or "3")
DEFAULT_BASE_DELAY = float(os.getenv("SUPER_AUDITOR_BASE_DELAY_SECONDS") or "1.5")
ARTIFACT_ROOT = Path("artifacts/tse_youtube_notion/super_auditor")

GPT_REVIEW_FIELDS = (
    "tema",
    "punchline",
    "origem",
    "classe_processo",
    "votacao",
    "resultado",
    "pedido_vista",
    "advogados",
    "partes",
    "tribunal",
    "precedentes_citados",
    "resolucoes_citadas",
    "eleicao",
    "composicao",
)
LIST_FIELDS = {"partes", "advogados", "composicao", "materia_semelhante"}
HARD_FIELDS = {
    "origem",
    "classe_processo",
    "votacao",
    "resultado",
    "pedido_vista",
    "tribunal",
    "eleicao",
    "composicao",
}
CLEARABLE_FIELDS = {
    "tema",
    "punchline",
    "partes",
    "advogados",
    "precedentes_citados",
    "resolucoes_citadas",
    "pedido_vista",
}
CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}
ISSUE_TO_FIELDS = {
    "tema_empty": {"tema"},
    "tema_generic": {"tema"},
    "punchline_empty": {"punchline"},
    "origem_empty": {"origem", "tribunal"},
    "origem_state_only": {"origem"},
    "origem_tre_extenso": {"origem"},
    "origem_invalid_label": {"origem"},
    "origem_downgraded_tre": {"origem", "tribunal"},
    "classe_empty": {"classe_processo"},
    "classe_mismatch": {"classe_processo"},
    "resultado_empty": {"resultado"},
    "votacao_empty": {"votacao"},
    "votacao_inconsistent": {"resultado", "votacao", "pedido_vista"},
    "relator_empty": {"relator"},
    "composicao_incomplete": {"composicao"},
}
THEME_META_PATTERNS = (
    re.compile(r"(?i)^(?:o\s+)?caso\s+(?:do|da|de)\b"),
    re.compile(r"(?i)^julgamento\s+(?:de|do|da)\b"),
    re.compile(r"(?i)\brelator(?:a)?\s+minist"),
    re.compile(
        r"(?i)^(?:(?:al[eé]m disso|ainda|por fim|nesse contexto|nesse ponto)(?:,\s+|\s+))?"
        r"(?:(?:o|a)\s+relator(?:a)?\s+)?"
        r"(?:destacou|assinalou|observou|ressaltou|salientou|pontuou|consignou|"
        r"afirmou|entendeu|registrou|reafirmou|frisou|anotou)\s+que\b"
    ),
    re.compile(r"(?i)^presta[cç][aã]o de contas partid[aá]rias$"),
    re.compile(r"(?i)^(?:proposta|minuta)\s+de\s+resolu[cç][aã]o\b"),
    re.compile(
        r"(?i)^(?:negad[oa]|desprovido|provido|deferid[oa]|indeferid[oa]|rejeitad[oa]|aprovad[oa]|"
        r"homologad[oa]|mantid[oa]|conhecido|n[aã]o\s+conhecido|deu|dado)\b"
    ),
)
PUNCHLINE_WEAK_PATTERNS = (
    re.compile(r"(?i)^(?:o|a)\s+(?:processo|caso|recurso|a[cç][aã]o)\s+(?:trata|discute|versa)\s+(?:de|sobre)\b"),
    re.compile(r"(?i)^trata(?:-|\s+)se de\b"),
    re.compile(r"(?i)^(?:o|a)\s+relator(?:a)?\b"),
    re.compile(r"(?i)^(?:o\s+tse|o\s+tribunal)\s+(?:aprovou|homologou|julgou|negou|deu)\b"),
    re.compile(r"(?i)^(?:o\s+)?voto\s+d[ao]\s+relator(?:a)?\b"),
    re.compile(r"(?i)\bfoi\s+acompanhad[oa]\s+pel(?:o|a)s?\s+ministros?\b"),
)

SYSTEM_PROMPT = """
Você é um auditor jurídico-eleitoral especializado em saneamento de base de dados de sessões do TSE.

Use exclusivamente as evidências locais fornecidas no prompt. Não use nenhuma fonte externa. Não invente fatos ausentes.

Seu trabalho é revisar apenas os campos solicitados e devolver JSON estruturado por campo.

Para cada campo:
- action: keep | update | clear | review
- suggested_value: valor canônico do Notion
- confidence: high | medium | low
- evidence_snippets: trechos curtos da evidência que sustentam a sugestão
- reason: justificativa objetiva e curta

Regras obrigatórias:
- Não altere número do processo, youtube_link, tipo_registro nem matéria_semelhante.
- Priorize preencher campos em branco ou incompletos quando a evidência local for suficiente.
- tema deve refletir a controvérsia jurídica, não frase meta, localidade, número de processo ou proclamação do resultado.
- punchline deve ser uma frase completa, autocontida e jurídica, coerente com análise/raciocínio/fundamentação. Nunca se limite a proclamação de resultado nem a informar quem acompanhou o voto.
- origem deve ser apenas Cidade/UF, TRE/UF ou TSE. Se houver município ou cidade identificável na evidência local, prefira Cidade/UF em vez de TRE/UF.
- classe_processo deve usar rótulos canônicos como REspe, AREspe, CTA, AIJE, ADO, ADI, PC, PA, Lista Tríplice.
- votacao só pode ser Unânime, Por maioria ou Suspenso.
- resultado deve usar o vocabulário padronizado do pipeline.
- tribunal deve ser TSE ou TRE-UF canônico, coerente com a origem.
- eleicao deve sair como ano canônico de quatro dígitos quando estiver explicitamente identificável.
- relator e pedido_vista devem ser nomes canônicos de ministros apenas quando houver base textual clara.
- partes, advogados e composição devem sair como listas de strings.
- composicao deve refletir a composição efetiva da sessão consolidada no material local; se a sessão consolidada estiver disponível, use-a para completar lacunas.
- partes e advogados nunca devem sair como objeto agrupado por polos ou papéis; devolva sempre uma lista plana de strings.
- Se não houver evidência suficiente, use keep ou review, nunca chute.
"""


class FieldSuggestion(BaseModel):
    action: Literal["keep", "update", "clear", "review"] = "keep"
    suggested_value: Any = ""
    confidence: Literal["low", "medium", "high"] = "low"
    evidence_snippets: list[str] = Field(default_factory=list)
    reason: str = ""


class SuperAuditSuggestion(BaseModel):
    tema: FieldSuggestion = Field(default_factory=FieldSuggestion)
    punchline: FieldSuggestion = Field(default_factory=FieldSuggestion)
    origem: FieldSuggestion = Field(default_factory=FieldSuggestion)
    classe_processo: FieldSuggestion = Field(default_factory=FieldSuggestion)
    votacao: FieldSuggestion = Field(default_factory=FieldSuggestion)
    resultado: FieldSuggestion = Field(default_factory=FieldSuggestion)
    pedido_vista: FieldSuggestion = Field(default_factory=FieldSuggestion)
    advogados: FieldSuggestion = Field(default_factory=FieldSuggestion)
    partes: FieldSuggestion = Field(default_factory=FieldSuggestion)
    tribunal: FieldSuggestion = Field(default_factory=FieldSuggestion)
    precedentes_citados: FieldSuggestion = Field(default_factory=FieldSuggestion)
    resolucoes_citadas: FieldSuggestion = Field(default_factory=FieldSuggestion)
    eleicao: FieldSuggestion = Field(default_factory=FieldSuggestion)
    composicao: FieldSuggestion = Field(default_factory=FieldSuggestion)
    general_notes: str = ""


class AuditCandidate(BaseModel):
    year: int
    video_id: str
    page_id: str
    page_url: str = ""
    numero_processo: str = ""
    target_fields: list[str] = Field(default_factory=list)
    issue_keys: list[str] = Field(default_factory=list)


class SuperAuditDecision(BaseModel):
    year: int
    video_id: str
    page_id: str
    numero_processo: str
    target_fields: list[str] = Field(default_factory=list)
    changed_fields: list[str] = Field(default_factory=list)
    deterministic_fields: list[str] = Field(default_factory=list)
    review_fields: list[str] = Field(default_factory=list)
    skipped_fields: list[str] = Field(default_factory=list)
    openai_called: bool = False
    applied: bool = False
    diff: dict[str, dict[str, Any]] = Field(default_factory=dict)
    notes: str = ""


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def discover_playlist_url_for_year(year: int) -> str:
    candidates: list[tuple[str, str, Path]] = []
    for path in sorted(BACKFILL_ROOT.glob(f"{year}_*")):
        if not path.is_dir() or "_smoke_" in path.name:
            continue
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        updated_at = str(payload.get("updated_at") or payload.get("started_at") or "")
        playlist_url = str(payload.get("playlist_url") or "").strip()
        if playlist_url:
            candidates.append((updated_at, playlist_url, path))
    if not candidates:
        raise RuntimeError(f"Não foi possível inferir a playlist de {year} a partir dos manifests locais.")
    candidates.sort(key=lambda item: (item[0], str(item[2])))
    return candidates[-1][1]


def build_offender_index(audit_summary: dict[str, Any]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    offenders = audit_summary.get("offenders") or {}
    for issue_key, items in offenders.items():
        for item in items or []:
            page_id = str(item.get("page_id") or "").strip()
            if page_id:
                index[page_id].add(issue_key)
    return index


def _looks_like_suspicious_multivalue(values: list[str]) -> bool:
    joined = " | ".join(str(value or "") for value in values)
    return any(token in joined for token in ("{", "}", "Não especificado", "não especificado", "[]"))


def tema_looks_weak_for_super_audit(row: PublishPreviewRow) -> bool:
    text = str(row.tema or "").strip()
    if tema_looks_generic(text, row):
        return True
    if not text:
        return True
    normalized = text.strip()
    if len(normalized) < 28:
        return True
    return any(pattern.search(normalized) for pattern in THEME_META_PATTERNS)


def punchline_looks_weak_for_super_audit(row: PublishPreviewRow) -> bool:
    text = str(row.punchline or "").strip()
    if punchline_looks_generic(text, row):
        return True
    if not text:
        return True
    if len(text) < 90:
        return True
    if any(pattern.search(text) for pattern in PUNCHLINE_WEAK_PATTERNS):
        return True
    normalized_tema = re.sub(r"\s+", " ", str(row.tema or "").strip().lower())
    normalized_punch = re.sub(r"\s+", " ", text.lower())
    if normalized_tema and normalized_punch.startswith(normalized_tema):
        return True
    return False


def origin_can_be_upgraded_from_local_text(row: PublishPreviewRow) -> str:
    current = str(row.origem or "").strip()
    current_normalized = normalize_origem_value(current)
    inferred = str(infer_origin_from_row_text(row) or "").strip()
    if not inferred or inferred.startswith("TRE/") or inferred == "TSE":
        return ""
    if not re.search(r"/[A-Z]{2}$", inferred):
        return ""
    if not current:
        return inferred
    if current_normalized.startswith("TRE/"):
        return inferred
    if current_normalized != current and current_normalized and current_normalized.startswith("TRE/"):
        return inferred
    return ""


def _target_fields_for_record(
    record: ExistingPageRecord,
    issue_keys: set[str],
    artifact_context: RepairArtifactContext,
    *,
    focus: str = "all",
) -> tuple[list[str], list[str]]:
    row = record.row
    fields: set[str] = set()
    reasons: list[str] = []
    if focus == "all":
        reasons.extend(sorted(issue_keys))
        for issue_key in issue_keys:
            fields.update(ISSUE_TO_FIELDS.get(issue_key, set()))

    process_key = canonicalize_numero_processo(row.numero_processo)
    special_key = _special_process_lookup_key(
        row.numero_processo,
        row.classe_processo or infer_classe_from_row_text(row),
    )
    artifact_item = artifact_context.item_by_process.get(process_key) or artifact_context.item_by_special_process.get(
        special_key
    )

    origin_issue_keys = sorted(issue_key for issue_key in issue_keys if issue_key.startswith("origem_"))
    if focus == "origem":
        reasons.extend(origin_issue_keys)
        if origin_issue_keys:
            fields.add("origem")
        if origin_can_be_upgraded_from_local_text(row):
            fields.add("origem")
            reasons.append("origem_city_from_local_text")
        ordered_fields = [field for field in ("origem",) if field in fields]
        ordered_reasons = dedupe_preserve_order(reasons)
        return ordered_fields, ordered_reasons

    if focus == "relator":
        if "relator_empty" in issue_keys or not row.relator:
            fields.add("relator")
            reasons.append("relator_missing")
        ordered_fields = [field for field in ("relator",) if field in fields]
        ordered_reasons = dedupe_preserve_order(reasons)
        return ordered_fields, ordered_reasons

    if focus in {"all", "quality-core"}:
        if tema_looks_weak_for_super_audit(row):
            fields.add("tema")
            reasons.append("tema_weak_local")
        if punchline_looks_weak_for_super_audit(row):
            fields.add("punchline")
            reasons.append("punchline_weak_local")
    if focus == "quality-core":
        reasons.extend(origin_issue_keys)
        if origin_issue_keys or origin_can_be_upgraded_from_local_text(row):
            fields.add("origem")
            if origin_can_be_upgraded_from_local_text(row):
                reasons.append("origem_city_from_local_text")
        if not row.partes and artifact_item is not None and artifact_item.partes:
            fields.add("partes")
            reasons.append("partes_missing_with_artifact")
        if _looks_like_suspicious_multivalue(row.partes):
            fields.add("partes")
            reasons.append("partes_suspicious")
        if (not row.advogados and artifact_item is not None and artifact_item.advogados) or _looks_like_suspicious_multivalue(row.advogados):
            fields.add("advogados")
            reasons.append("advogados_improvable")
        if "relator_empty" in issue_keys or not row.relator:
            fields.add("relator")
            reasons.append("relator_missing")
        if "resultado_empty" in issue_keys or not row.resultado:
            fields.add("resultado")
            reasons.append("resultado_missing")
        if "votacao_empty" in issue_keys or "votacao_inconsistent" in issue_keys or not row.votacao:
            fields.add("votacao")
            reasons.append("votacao_improvable")
        if not row.eleicao:
            fields.add("eleicao")
            reasons.append("eleicao_missing")
        if ("tribunal_empty_with_origin" in issue_keys) or (not row.tribunal and row.origem):
            fields.add("tribunal")
            reasons.append("tribunal_missing")
        if ("vista_missing_name" in issue_keys) or ((row.resultado == "Suspenso por vista" or row.votacao == "Suspenso") and not row.pedido_vista):
            fields.add("pedido_vista")
            reasons.append("pedido_vista_missing")
        session_composicao = [str(value).strip() for value in (artifact_context.session_composicao or []) if str(value).strip()]
        current_composicao = [str(value).strip() for value in (row.composicao or []) if str(value).strip()]
        if "composicao_incomplete" in issue_keys or (session_composicao and len(current_composicao) < len(session_composicao)):
            fields.add("composicao")
            reasons.append("composicao_incomplete")
        ordered_fields = [
            field
            for field in (
                "origem",
                "tema",
                "punchline",
                "partes",
                "advogados",
                "relator",
                "pedido_vista",
                "tribunal",
                "resultado",
                "votacao",
                "eleicao",
                "composicao",
            )
            if field in fields
        ]
        ordered_reasons = dedupe_preserve_order(reasons)
        return ordered_fields, ordered_reasons

    if focus == "residual-core":
        reasons.extend(origin_issue_keys)
        if origin_issue_keys or origin_can_be_upgraded_from_local_text(row):
            fields.add("origem")
            if origin_can_be_upgraded_from_local_text(row):
                reasons.append("origem_city_from_local_text")
        if "classe_empty" in issue_keys or "classe_mismatch" in issue_keys or not row.classe_processo:
            fields.add("classe_processo")
            reasons.append("classe_improvable")
        if "relator_empty" in issue_keys or not row.relator:
            fields.add("relator")
            reasons.append("relator_missing")
        if "resultado_empty" in issue_keys or not row.resultado:
            fields.add("resultado")
            reasons.append("resultado_missing")
        if "votacao_empty" in issue_keys or "votacao_inconsistent" in issue_keys or not row.votacao:
            fields.add("votacao")
            reasons.append("votacao_improvable")
        ordered_fields = [
            field
            for field in (
                "origem",
                "classe_processo",
                "relator",
                "resultado",
                "votacao",
            )
            if field in fields
        ]
        ordered_reasons = dedupe_preserve_order(reasons)
        return ordered_fields, ordered_reasons

    if not row.partes and artifact_item is not None and artifact_item.partes:
        fields.add("partes")
        reasons.append("partes_missing_with_artifact")
    if _looks_like_suspicious_multivalue(row.partes):
        fields.add("partes")
        reasons.append("partes_suspicious")
    if (not row.advogados and artifact_item is not None and artifact_item.advogados) or _looks_like_suspicious_multivalue(row.advogados):
        fields.add("advogados")
        reasons.append("advogados_improvable")
    if not row.precedentes_citados and artifact_item is not None and artifact_item.precedentes_citados:
        fields.add("precedentes_citados")
        reasons.append("precedentes_missing_with_artifact")
    if not row.resolucoes_citadas and artifact_item is not None and artifact_item.resolucoes_citadas:
        fields.add("resolucoes_citadas")
        reasons.append("resolucoes_missing_with_artifact")
    if not row.tribunal and row.origem:
        fields.add("tribunal")
        reasons.append("tribunal_empty_with_origin")
    if row.resultado == "Suspenso por vista" and not row.pedido_vista:
        fields.add("pedido_vista")
        reasons.append("vista_missing_name")

    ordered_fields = [field for field in GPT_REVIEW_FIELDS if field in fields]
    ordered_reasons = dedupe_preserve_order(reasons)
    return ordered_fields, ordered_reasons


def build_global_relation_targets(grouped_by_year: dict[int, dict[str, list[ExistingPageRecord]]]) -> dict[str, list[str]]:
    records_by_process: dict[str, list[ExistingPageRecord]] = defaultdict(list)
    for grouped in grouped_by_year.values():
        for records in grouped.values():
            for record in records:
                canonical = canonicalize_numero_processo(record.row.numero_processo)
                if canonical:
                    records_by_process[canonical].append(record)
    relation_targets: dict[str, list[str]] = {}
    for records in records_by_process.values():
        unique_records: list[ExistingPageRecord] = []
        seen_page_ids: set[str] = set()
        for record in records:
            if record.page_id and record.page_id not in seen_page_ids:
                seen_page_ids.add(record.page_id)
                unique_records.append(record)
        if len(unique_records) < 2:
            continue
        for record in unique_records:
            related_ids = [
                candidate.page_id
                for candidate in unique_records
                if candidate.page_id != record.page_id
                and (
                    candidate.video_id != record.video_id
                    or candidate.row.data_sessao != record.row.data_sessao
                )
            ]
            if related_ids:
                relation_targets[record.page_id] = dedupe_preserve_order(related_ids)
    return relation_targets


def _truncate_text(value: str, limit: int = 5000) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit]


def build_row_context(record: ExistingPageRecord, artifact_context: RepairArtifactContext, target_fields: list[str]) -> str:
    row = record.row
    process_key = canonicalize_numero_processo(row.numero_processo)
    special_lookup = _special_process_lookup_key(
        row.numero_processo,
        row.classe_processo or infer_classe_from_row_text(row),
    )
    artifact_item = artifact_context.item_by_process.get(process_key) or artifact_context.item_by_special_process.get(
        special_lookup
    )
    evidence_blocks = []
    if artifact_item is not None:
        evidence_blocks.append(
            "Item extraído do artefato:\n" + json.dumps(artifact_item.model_dump(mode="json"), ensure_ascii=False, indent=2)
        )
    theme_text = artifact_context.theme_text_by_process.get(process_key) or artifact_context.theme_text_by_special_process.get(
        special_lookup
    )
    if theme_text:
        evidence_blocks.append("Texto consolidado do artefato:\n" + _truncate_text(theme_text, limit=6000))
    evidence_blocks.append(
        "Row atual no Notion:\n" + json.dumps(record.row.to_editor_record(), ensure_ascii=False, indent=2)
    )
    if artifact_context.session_date:
        evidence_blocks.append(f"Data autoritativa da sessão: {artifact_context.session_date}")
    if artifact_context.session_composicao:
        evidence_blocks.append("Composição consolidada da sessão: " + ", ".join(artifact_context.session_composicao))
    evidence_blocks.append("Campos a revisar: " + ", ".join(target_fields))
    return "\n\n".join(block for block in evidence_blocks if block.strip())


def _extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_openai_super_auditor(
    client: OpenAI,
    *,
    model: str,
    context_text: str,
) -> SuperAuditSuggestion:
    last_error: Exception | None = None
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_text},
        ],
    }
    for attempt in range(1, DEFAULT_RETRIES + 1):
        try:
            try:
                response = client.chat.completions.create(timeout=DEFAULT_TIMEOUT_SECONDS, **payload)
            except TypeError:
                response = client.chat.completions.create(**payload)
            content = (response.choices[0].message.content or "").strip()
            parsed = _extract_json_payload(content)
            return SuperAuditSuggestion.model_validate(parsed)
        except Exception as exc:
            last_error = exc
            if attempt < DEFAULT_RETRIES:
                time.sleep(DEFAULT_BASE_DELAY ** attempt)
                continue
            break
    raise RuntimeError(f"Falha ao obter sugestão do super auditor: {last_error}") from last_error


def _normalize_suggested_value(field_name: str, value: Any) -> str | list[str]:
    if field_name in LIST_FIELDS:
        if isinstance(value, dict):
            flattened: list[str] = []
            for dict_value in value.values():
                if isinstance(dict_value, list):
                    flattened.extend(str(item).strip() for item in dict_value if str(item).strip())
                elif isinstance(dict_value, dict):
                    for nested in dict_value.values():
                        if isinstance(nested, list):
                            flattened.extend(str(item).strip() for item in nested if str(item).strip())
                        elif str(nested).strip():
                            flattened.append(str(nested).strip())
                elif str(dict_value).strip():
                    flattened.append(str(dict_value).strip())
            return dedupe_preserve_order(flattened)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return parse_multi_value_text(value)
    return str(value or "").strip()


def should_apply_suggestion(field_name: str, suggestion: FieldSuggestion, min_confidence: str) -> bool:
    if suggestion.action not in {"update", "clear"}:
        return False
    if CONFIDENCE_RANK.get(suggestion.confidence, 0) < CONFIDENCE_RANK[min_confidence]:
        return False
    if field_name in HARD_FIELDS and not suggestion.evidence_snippets:
        return False
    if suggestion.action == "clear":
        return field_name in CLEARABLE_FIELDS and bool(suggestion.evidence_snippets)
    normalized = _normalize_suggested_value(field_name, suggestion.suggested_value)
    if field_name in LIST_FIELDS:
        return bool(normalized)
    return bool(str(normalized).strip())


def apply_super_audit_suggestions(
    row: PublishPreviewRow,
    suggestion: SuperAuditSuggestion,
    notion_schema: NotionDataSourceSchema,
    *,
    min_confidence: str,
    deterministic_relations: list[str],
) -> tuple[PublishPreviewRow, list[str], list[str], list[str]]:
    repaired = row.model_copy(deep=True)
    proposed_fields: list[str] = []
    review_fields: list[str] = []
    skipped_fields: list[str] = []

    for field_name in GPT_REVIEW_FIELDS:
        field_suggestion = getattr(suggestion, field_name)
        if field_suggestion.action == "review":
            review_fields.append(field_name)
            continue
        if not should_apply_suggestion(field_name, field_suggestion, min_confidence):
            if field_suggestion.action in {"update", "clear"}:
                skipped_fields.append(field_name)
            continue
        if field_suggestion.action == "clear":
            if field_name in LIST_FIELDS:
                setattr(repaired, field_name, [])
            else:
                setattr(repaired, field_name, "")
                repaired.clear_properties = dedupe_preserve_order([*repaired.clear_properties, field_name])
            proposed_fields.append(field_name)
            continue
        normalized = _normalize_suggested_value(field_name, field_suggestion.suggested_value)
        if field_name == "composicao":
            normalized_composicao = normalize_composition_list(list(normalized or []))
            if len(normalized_composicao) not in {6, 7}:
                skipped_fields.append(field_name)
                continue
            normalized = normalized_composicao
        setattr(repaired, field_name, normalized)
        proposed_fields.append(field_name)

    if "materia_semelhante" in notion_schema.properties and deterministic_relations != repaired.materia_semelhante:
        repaired.materia_semelhante = deterministic_relations
        proposed_fields.append("materia_semelhante")

    repaired = validate_preview_row(repaired, notion_schema)
    changed_fields = [
        field_name
        for field_name in dedupe_preserve_order(proposed_fields)
        if getattr(row, field_name, None) != getattr(repaired, field_name, None)
    ]
    return repaired, changed_fields, dedupe_preserve_order(review_fields), dedupe_preserve_order(skipped_fields)


def build_decision_diff(before: PublishPreviewRow, after: PublishPreviewRow, changed_fields: list[str]) -> dict[str, dict[str, Any]]:
    diff: dict[str, dict[str, Any]] = {}
    for field_name in changed_fields:
        before_value = getattr(before, field_name, None)
        after_value = getattr(after, field_name, None)
        if before_value == after_value:
            continue
        diff[field_name] = {"before": before_value, "after": after_value}
    return diff


def parse_playlist_overrides(values: list[str]) -> dict[int, str]:
    parsed: dict[int, str] = {}
    for value in values:
        year_text, _, playlist_url = value.partition("=")
        if not year_text or not playlist_url:
            raise RuntimeError(f"Formato inválido em --playlist-override: {value}")
        parsed[int(year_text)] = playlist_url.strip()
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Super auditor GPT-5.4 para saneamento dos backfills do Notion.")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--playlist-override", action="append", default=[])
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--focus", choices=["all", "quality-core", "residual-core", "origem", "relator"], default="quality-core")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--video-id", dest="video_ids", action="append", default=[])
    parser.add_argument("--page-id", dest="page_ids", action="append", default=[])
    parser.add_argument("--min-confidence", choices=["high", "medium", "low"], default="high")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    runtime = build_runtime_context()
    if not runtime["openai_api_key"]:
        raise RuntimeError("OPENAI_API_KEY não encontrado.")

    notion_client = NotionSessoesClient(
        api_key=runtime["notion_api_key"],
        data_source_id=runtime["notion_data_source_id"],
        logger=LOGGER,
        normalize_multiselect_colors_post_write=False,
    )
    notion_schema = notion_client.fetch_schema()
    openai_client = OpenAI(api_key=runtime["openai_api_key"], max_retries=0)
    playlist_overrides = parse_playlist_overrides(args.playlist_override)

    grouped_by_year: dict[int, dict[str, list[ExistingPageRecord]]] = {}
    audit_by_year: dict[int, dict[str, Any]] = {}
    playlist_url_by_year: dict[int, str] = {}
    for year in args.years:
        playlist_url = playlist_overrides.get(year) or discover_playlist_url_for_year(year)
        playlist_url_by_year[year] = playlist_url
        grouped = load_existing_pages_for_year_with_retry(
            notion_client,
            notion_schema,
            year,
            playlist_url=playlist_url,
        )
        grouped_by_year[year] = grouped
        audit_by_year[year] = audit_existing_year(grouped, playlist_url=playlist_url, year=year)

    relation_targets = build_global_relation_targets(grouped_by_year)
    run_root = ARTIFACT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    decisions: list[SuperAuditDecision] = []
    stats = {
        "candidates": 0,
        "openai_calls": 0,
        "updated_pages": 0,
        "review_pages": 0,
        "skipped_pages": 0,
        "deterministic_relation_updates": 0,
    }
    stats_by_year: dict[str, dict[str, Any]] = {}
    field_changes_by_year: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    review_fields_by_year: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for year in args.years:
        offender_index = build_offender_index(audit_by_year[year])
        year_stats = {
            "candidates": 0,
            "openai_calls": 0,
            "updated_pages": 0,
            "review_pages": 0,
            "skipped_pages": 0,
            "deterministic_relation_updates": 0,
        }
        stats_by_year[str(year)] = year_stats
        for video_id, records in sorted(grouped_by_year[year].items()):
            if args.video_ids and video_id not in set(args.video_ids):
                continue
            artifact_context = load_repair_artifact_context(playlist_url_by_year[year], year, video_id)
            for record in records:
                if args.page_ids and record.page_id not in set(args.page_ids):
                    continue
                target_fields, issue_keys = _target_fields_for_record(
                    record,
                    offender_index.get(record.page_id, set()),
                    artifact_context,
                    focus=args.focus,
                )
                deterministic_relations = relation_targets.get(record.page_id, [])
                if not target_fields and (args.focus != "all" or deterministic_relations == record.row.materia_semelhante):
                    continue
                stats["candidates"] += 1
                year_stats["candidates"] += 1
                candidate = AuditCandidate(
                    year=year,
                    video_id=video_id,
                    page_id=record.page_id,
                    page_url=record.url,
                    numero_processo=record.row.numero_processo,
                    target_fields=target_fields,
                    issue_keys=issue_keys,
                )
                suggestion = SuperAuditSuggestion()
                openai_called = False
                notes = ""
                if target_fields:
                    context_text = build_row_context(record, artifact_context, target_fields)
                    suggestion = call_openai_super_auditor(
                        openai_client,
                        model=args.model,
                        context_text=context_text,
                    )
                    openai_called = True
                    stats["openai_calls"] += 1
                    year_stats["openai_calls"] += 1
                    (run_root / f"{record.page_id}_request.txt").write_text(context_text, encoding="utf-8")
                    (run_root / f"{record.page_id}_response.json").write_text(
                        json.dumps(suggestion.model_dump(mode="json"), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                repaired, changed_fields, review_fields, skipped_fields = apply_super_audit_suggestions(
                    record.row,
                    suggestion,
                    notion_schema,
                    min_confidence=args.min_confidence,
                    deterministic_relations=deterministic_relations,
                )
                diff = build_decision_diff(record.row, repaired, changed_fields)
                applied = False
                if "materia_semelhante" in changed_fields:
                    stats["deterministic_relation_updates"] += 1
                    year_stats["deterministic_relation_updates"] += 1
                if changed_fields and args.apply and not args.review_only:
                    notion_client.update_row(notion_schema, record.page_id, repaired)
                    applied = True
                    stats["updated_pages"] += 1
                    year_stats["updated_pages"] += 1
                elif changed_fields:
                    stats["skipped_pages"] += 1
                    year_stats["skipped_pages"] += 1
                if review_fields:
                    stats["review_pages"] += 1
                    year_stats["review_pages"] += 1
                for field_name in changed_fields:
                    field_changes_by_year[str(year)][field_name] += 1
                for field_name in review_fields:
                    review_fields_by_year[str(year)][field_name] += 1
                decision = SuperAuditDecision(
                    year=year,
                    video_id=video_id,
                    page_id=record.page_id,
                    numero_processo=record.row.numero_processo,
                    target_fields=target_fields,
                    changed_fields=changed_fields,
                    deterministic_fields=[field for field in changed_fields if field == "materia_semelhante"],
                    review_fields=review_fields,
                    skipped_fields=skipped_fields,
                    openai_called=openai_called,
                    applied=applied,
                    diff=diff,
                    notes=notes,
                )
                decisions.append(decision)
                (run_root / f"{record.page_id}_decision.json").write_text(
                    json.dumps(decision.model_dump(mode="json"), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if args.limit > 0 and len(decisions) >= args.limit:
                    summary = {
                        "model": args.model,
                        "years": args.years,
                        "apply": bool(args.apply and not args.review_only),
                        "focus": args.focus,
                        "min_confidence": args.min_confidence,
                        "stats": stats,
                        "stats_by_year": stats_by_year,
                        "field_changes_by_year": {
                            year_key: dict(values)
                            for year_key, values in field_changes_by_year.items()
                        },
                        "review_fields_by_year": {
                            year_key: dict(values)
                            for year_key, values in review_fields_by_year.items()
                        },
                        "decisions": [decision.model_dump(mode="json") for decision in decisions],
                    }
                    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    LOGGER.info("Super auditor concluído. Resumo: %s", run_root / "summary.json")
                    return

    summary = {
        "model": args.model,
        "years": args.years,
        "apply": bool(args.apply and not args.review_only),
        "focus": args.focus,
        "min_confidence": args.min_confidence,
        "stats": stats,
        "stats_by_year": stats_by_year,
        "field_changes_by_year": {
            year_key: dict(values)
            for year_key, values in field_changes_by_year.items()
        },
        "review_fields_by_year": {
            year_key: dict(values)
            for year_key, values in review_fields_by_year.items()
        },
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
    }
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Super auditor concluído. Resumo: %s", run_root / "summary.json")


if __name__ == "__main__":
    main()
