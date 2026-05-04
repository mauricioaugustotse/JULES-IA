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
from typing import Any, Literal

from pydantic import BaseModel, Field

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import normalize_class_text
from tse_youtube_notion_core import (
    DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_NOTION_DATA_SOURCE_ID,
    GEMINI_CALL_RETRIES,
    GEMINI_RETRY_BASE_DELAY,
    NotionSessoesClient,
    PublishPreviewRow,
    _build_gemini_rest_part,
    call_gemini_generate_content_rest,
    extract_retry_delay_seconds,
    should_disable_model,
)
from tse_backfill_2025_notion import notion_page_to_row


LOGGER = logging.getLogger("rewrite_notion_tema_punchline")
ARTIFACT_ROOT = Path("artifacts") / "notion_tema_punchline_rewrite"
DEFAULT_BATCH_SIZE = 12
PAGE_UPDATE_SLEEP_SECONDS = 0.12
PAGE_UPDATE_WORKERS = 2
PROCESS_NUMBER_RE = re.compile(
    r"\b(?:\d{3,7}-\d{2}(?:\.\d{4}\.\d\.\d{2}\.\d{4})?|ADI\s*\d+|ADO\s*\d+)\b",
    flags=re.IGNORECASE,
)
GENERIC_THEME_PATTERNS = [
    re.compile(r"(?i)^\s*(?:julgamento|processo|caso|recurso|pedido|tema)\b"),
    re.compile(r"(?i)\b(?:tse|tribunal|corte|plen[aá]rio)\s+(?:decide|mant[eé]m|acolhe|rejeita|aprova|define)\b"),
    re.compile(r"(?i)\b(?:pedido de vista|adiamento de julgamento|julgamento adiado)\b"),
]
GENERIC_PUNCH_PATTERNS = [
    re.compile(r"(?i)^\s*(?:tse|a corte|o tribunal)\s+(?:acolhe|mant[eé]m|decide|define|aprova|rejeita)\s+(?:tese|encaminhamento|controv[eé]rsia|decis[aã]o)\b"),
    re.compile(r"(?i)^\s*julgamento sobre\b"),
    re.compile(r"(?i)\bem\s+Bras[ií]lia/DF\.?$"),
]


SYSTEM_PROMPT = """
Você é editor jurídico-eleitoral de uma base de dados de julgamentos do Tribunal Superior Eleitoral.

Use exclusivamente as evidências locais fornecidas. Não use fonte externa e não invente fato, sanção, parte, resultado ou fundamento ausente.

Para cada item, reescreva obrigatoriamente:

1) tema
- Deve ser cabeçalho de ficha de catalogação média.
- Deve identificar a questão jurídica submetida a julgamento.
- Deve ser frase nominal específica, indexável e jurídica.
- Evite resultado, número do processo, nomes de ministros e fórmulas como "TSE decide", "julgamento", "processo", "pedido de vista".
- Tamanho ideal: 7 a 18 palavras.

2) punchline
- Deve ser uma frase editorial, completa e autocontida.
- Deve contextualizar o caso, sintetizar o debate jurídico e indicar a suma do julgamento quando houver resultado.
- Deve permitir visão global do processo em conjunto com o tema.
- Não repita literalmente o tema; complemente-o.
- Não use frase genérica como "TSE mantém decisão sobre...".
- Se o julgamento foi suspenso, indique o objeto jurídico debatido e a razão pública da suspensão, sem fingir julgamento definitivo.
- Tamanho ideal: 28 a 55 palavras.

Regras de fidelidade:
- Se a fonte local for insuficiente para uma conclusão, explicite a limitação de forma sóbria na punchline e mantenha tema pelo núcleo jurídico disponível.
- Não copie literalmente os valores atuais de tema/punchline; eles são contexto secundário e devem ser melhorados.
- Retorne apenas JSON no formato {"items":[...]}.
- Devolva exatamente um item por entrada, preservando o campo key.
"""


class RewriteItem(BaseModel):
    key: str
    tema: str = ""
    punchline: str = ""
    confidence: Literal["low", "medium", "high"] = "medium"
    source_insufficient: bool = False
    reason: str = ""


class RewriteBatchResult(BaseModel):
    items: list[RewriteItem] = Field(default_factory=list)


@dataclass
class PageRecord:
    key: str
    index: int
    page_id: str
    page_url: str
    row: PublishPreviewRow


@dataclass
class PageChange:
    key: str
    index: int
    page_id: str
    page_url: str
    numero_processo: str
    data_sessao: str
    classe_processo: str
    old_tema: str
    new_tema: str
    old_punchline: str
    new_punchline: str
    confidence: str
    source_insufficient: bool
    reason: str
    validation_notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "index": self.index,
            "page_id": self.page_id,
            "page_url": self.page_url,
            "numero_processo": self.numero_processo,
            "data_sessao": self.data_sessao,
            "classe_processo": self.classe_processo,
            "old_tema": self.old_tema,
            "new_tema": self.new_tema,
            "old_punchline": self.old_punchline,
            "new_punchline": self.new_punchline,
            "confidence": self.confidence,
            "source_insufficient": self.source_insufficient,
            "reason": self.reason,
            "validation_notes": self.validation_notes,
        }


def query_data_source_with_retry(client: NotionSessoesClient, filter_payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    start_cursor: str | None = None
    while True:
        payload: dict[str, Any] = {"page_size": 100}
        if filter_payload:
            payload["filter"] = filter_payload
        if start_cursor:
            payload["start_cursor"] = start_cursor
        page = notion_request_with_retry(client, "POST", f"/data_sources/{client.data_source_id}/query", json=payload)
        results.extend(page.get("results", []))
        if not page.get("has_more"):
            break
        start_cursor = page.get("next_cursor")
    return results


def load_records(
    client: NotionSessoesClient,
    *,
    start_index: int = 0,
    max_records: int = 0,
    page_ids: list[str] | None = None,
) -> list[PageRecord]:
    schema = client.fetch_schema()
    pages = query_data_source_with_retry(client)
    if page_ids:
        wanted = {str(page_id).strip() for page_id in page_ids if str(page_id).strip()}
        selected_pages = [page for page in pages if str(page.get("id", "")) in wanted]
        found = {str(page.get("id", "")) for page in selected_pages}
        missing = sorted(wanted - found)
        if missing:
            LOGGER.warning("Page IDs não encontrados no Notion: %s", ", ".join(missing))
    else:
        selected_pages = pages[start_index : start_index + max_records] if max_records > 0 else pages[start_index:]
    records: list[PageRecord] = []
    for offset, page in enumerate(selected_pages, start=start_index):
        page_id = str(page.get("id", ""))
        row = notion_page_to_row(client, schema, page)
        records.append(
            PageRecord(
                key=f"r{offset:04d}",
                index=offset,
                page_id=page_id,
                page_url=str(page.get("url", "")),
                row=row,
            )
        )
    return records


def compact_text(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].strip()
    return f"{cut}..."


def record_source_payload(record: PageRecord) -> dict[str, Any]:
    row = record.row
    return {
        "key": record.key,
        "numero_processo": row.numero_processo,
        "classe_processo": row.classe_processo,
        "tipo_registro": row.tipo_registro,
        "data_sessao": row.data_sessao,
        "origem": row.origem,
        "tribunal": row.tribunal,
        "eleicao": row.eleicao,
        "relator": row.relator,
        "pedido_vista": row.pedido_vista,
        "resultado": row.resultado,
        "votacao": row.votacao,
        "partes": row.partes[:8],
        "tema_atual_nao_copiar": compact_text(row.tema, 220),
        "punchline_atual_nao_copiar": compact_text(row.punchline, 320),
        "analise_factual": compact_text(row.analise_do_conteudo_juridico, 1100),
        "raciocinio_juridico": compact_text(row.raciocinio_juridico, 900),
        "fundamentacao_normativa": compact_text(row.fundamentacao_normativa, 420),
        "precedentes_citados": compact_text(row.precedentes_citados, 300),
        "resolucoes_citadas": compact_text(row.resolucoes_citadas, 260),
    }


def build_batch_prompt(records: list[PageRecord]) -> str:
    payload = {"items": [record_source_payload(record) for record in records]}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def load_cached_batch(path: Path) -> RewriteBatchResult | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return RewriteBatchResult.model_validate(payload["parsed"])
    except Exception:
        return None


def call_rewrite_batch(
    *,
    api_key: str,
    model: str,
    records: list[PageRecord],
    cache_path: Path,
    force: bool,
) -> RewriteBatchResult:
    cached = None if force else load_cached_batch(cache_path)
    if cached is not None:
        return cached
    prompt = build_batch_prompt(records)
    last_error: Exception | None = None
    for attempt in range(1, GEMINI_CALL_RETRIES + 1):
        try:
            parsed, response_text, response_payload = call_gemini_generate_content_rest(
                api_key=api_key,
                model_name=model,
                contents=[{"parts": [_build_gemini_rest_part(text=prompt)]}],
                system_instruction=SYSTEM_PROMPT,
                response_model=RewriteBatchResult,
                temperature=0.25,
                timeout_seconds=DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS,
            )
            result = RewriteBatchResult.model_validate(parsed)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "keys": [record.key for record in records],
                        "prompt": prompt,
                        "response_text": response_text,
                        "parsed": result.model_dump(),
                        "raw_candidate_count": len(response_payload.get("candidates") or []),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return result
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Falha no lote %s-%s tentativa %s/%s: %s", records[0].key, records[-1].key, attempt, GEMINI_CALL_RETRIES, exc)
            if should_disable_model(exc):
                break
            if attempt < GEMINI_CALL_RETRIES:
                retry_delay = extract_retry_delay_seconds(exc)
                time.sleep(max(GEMINI_RETRY_BASE_DELAY**attempt, retry_delay))
    raise RuntimeError(f"Falha definitiva no lote {records[0].key}-{records[-1].key}: {last_error}") from last_error


def clean_sentence(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.strip(" \t\r\n\"'“”‘’`")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def clean_theme(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.strip(" \t\r\n\"'“”‘’`.;:-")
    text = PROCESS_NUMBER_RE.sub("", text)
    text = re.sub(r"\b(?:no|na|nos autos do)\s+(?:REspe|AREspe|RO|PC|PA|CTA|RPP|HC|MS|RMS|RHC)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .;:-")
    if text:
        text = text[:1].upper() + text[1:]
    return text


def theme_invalid(theme: str, row: PublishPreviewRow) -> list[str]:
    notes: list[str] = []
    normalized = normalize_class_text(theme)
    if not theme:
        notes.append("tema vazio")
    if len(theme) < 24:
        notes.append("tema curto demais")
    if len(theme) > 150:
        notes.append("tema longo demais")
    if PROCESS_NUMBER_RE.search(theme):
        notes.append("tema contem numero de processo")
    if any(pattern.search(theme) for pattern in GENERIC_THEME_PATTERNS):
        notes.append("tema parece generico ou decisorio")
    if normalize_class_text(row.numero_processo) and normalize_class_text(row.numero_processo) in normalized:
        notes.append("tema reaproveita numero_processo")
    return notes


def punchline_invalid(punchline: str, theme: str) -> list[str]:
    notes: list[str] = []
    normalized_punch = normalize_class_text(punchline)
    normalized_theme = normalize_class_text(theme)
    if not punchline:
        notes.append("punchline vazia")
    if len(punchline) < 90:
        notes.append("punchline curta demais")
    if len(punchline) > 520:
        notes.append("punchline longa demais")
    if any(pattern.search(punchline) for pattern in GENERIC_PUNCH_PATTERNS):
        notes.append("punchline parece formulaica")
    if normalized_theme and normalized_punch == normalized_theme:
        notes.append("punchline igual ao tema")
    if normalized_theme and normalized_punch.startswith(normalized_theme) and len(normalized_punch) < len(normalized_theme) + 50:
        notes.append("punchline repete o tema sem contexto suficiente")
    return notes


def fallback_theme(row: PublishPreviewRow) -> str:
    candidates = [
        row.analise_do_conteudo_juridico,
        row.raciocinio_juridico,
        row.punchline,
        row.tema,
    ]
    for source in candidates:
        sentences = re.split(r"(?<=[.!?])\s+", str(source or "").strip())
        for sentence in sentences[:3]:
            sentence = clean_theme(sentence)
            sentence = re.sub(r"(?i)^o\s+(?:processo|caso|julgamento)\s+(?:trata|discute|refere-se)\s+(?:de|a|ao|à)?\s*", "", sentence).strip()
            sentence = re.sub(r"(?i)^a\s+controv[eé]rsia\s+(?:gira em torno|reside)\s+(?:de|em|na|no)\s*", "", sentence).strip()
            sentence = re.sub(
                r"(?i)^(?:recurso especial eleitoral|agravo em recurso especial eleitoral|recurso ordin[aá]rio|agravo regimental|embargos de declara[cç][aã]o)\s+(?:sobre|em|contra|interposto contra)\s+",
                "",
                sentence,
            ).strip()
            if sentence and not theme_invalid(sentence, row):
                return sentence[:150].strip()
    base = "Questão jurídica eleitoral sem núcleo suficientemente descrito"
    if row.classe_processo:
        return f"{base} em {row.classe_processo}"
    return base


def fallback_punchline(row: PublishPreviewRow, theme: str) -> str:
    result = str(row.resultado or "").strip()
    vote = str(row.votacao or "").strip()
    source = compact_text(row.raciocinio_juridico or row.analise_do_conteudo_juridico or row.punchline, 260)
    if source:
        if result:
            return clean_sentence(f"A linha trata de {theme[:1].lower() + theme[1:]}; {source} O desfecho registrado foi {result.lower()}" + (f", com votação {vote.lower()}" if vote else ""))
        return clean_sentence(f"A linha trata de {theme[:1].lower() + theme[1:]}; {source}")
    if result:
        return clean_sentence(f"O registro cataloga {theme[:1].lower() + theme[1:]}, com desfecho anotado como {result.lower()}" + (f" e votação {vote.lower()}" if vote else ""))
    return clean_sentence(f"O registro cataloga {theme[:1].lower() + theme[1:]}, mas a fonte local não descreve com segurança todos os fundamentos do julgamento")


def validate_and_build_changes(records: list[PageRecord], results: dict[str, RewriteItem]) -> tuple[list[PageChange], list[dict[str, Any]]]:
    changes: list[PageChange] = []
    issues: list[dict[str, Any]] = []
    for record in records:
        row = record.row
        item = results.get(record.key)
        notes: list[str] = []
        if item is None:
            item = RewriteItem(
                key=record.key,
                tema=fallback_theme(row),
                punchline="",
                confidence="low",
                source_insufficient=True,
                reason="modelo nao devolveu item para a chave; fallback local aplicado",
            )
            notes.append("item ausente na resposta do modelo")
        theme = clean_theme(item.tema)
        theme_notes = theme_invalid(theme, row)
        if theme_notes:
            notes.extend(theme_notes)
            theme = fallback_theme(row)
        punchline = clean_sentence(item.punchline)
        punch_notes = punchline_invalid(punchline, theme)
        if punch_notes:
            notes.extend(punch_notes)
            punchline = fallback_punchline(row, theme)
        second_punch_notes = punchline_invalid(punchline, theme)
        if second_punch_notes:
            notes.extend(f"fallback: {note}" for note in second_punch_notes)
        if normalize_class_text(theme) == normalize_class_text(row.tema) and normalize_class_text(punchline) == normalize_class_text(row.punchline):
            notes.append("saida final ficou identica aos valores atuais")
        change = PageChange(
            key=record.key,
            index=record.index,
            page_id=record.page_id,
            page_url=record.page_url,
            numero_processo=row.numero_processo,
            data_sessao=row.data_sessao,
            classe_processo=row.classe_processo,
            old_tema=row.tema,
            new_tema=theme,
            old_punchline=row.punchline,
            new_punchline=punchline,
            confidence=item.confidence,
            source_insufficient=item.source_insufficient or item.confidence == "low",
            reason=item.reason,
            validation_notes=notes,
        )
        changes.append(change)
        if notes:
            issues.append(change.as_dict())
    return changes, issues


def generate_changes(
    *,
    api_key: str,
    model: str,
    records: list[PageRecord],
    artifact_dir: Path,
    batch_size: int,
    force: bool,
) -> tuple[list[PageChange], list[dict[str, Any]]]:
    result_items: dict[str, RewriteItem] = {}
    batch_dir = artifact_dir / "llm_batches"
    total_batches = (len(records) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(records), batch_size), start=1):
        batch_records = records[start : start + batch_size]
        cache_path = batch_dir / f"batch_{batch_index:04d}_{batch_records[0].key}_{batch_records[-1].key}.json"
        result = call_rewrite_batch(
            api_key=api_key,
            model=model,
            records=batch_records,
            cache_path=cache_path,
            force=force,
        )
        for item in result.items:
            result_items[item.key] = item
        if batch_index % 10 == 0 or batch_index == total_batches:
            LOGGER.info("Lotes gerados: %s/%s", batch_index, total_batches)
    return validate_and_build_changes(records, result_items)


def patch_page(client: NotionSessoesClient, title_property_name: str, change: PageChange) -> dict[str, Any]:
    notion_request_with_retry(
        client,
        "PATCH",
        f"/pages/{change.page_id}",
        json={
            "properties": {
                title_property_name: {"title": [{"text": {"content": change.new_tema[:2000]}}]},
                "punchline": {"rich_text": [{"text": {"content": change.new_punchline[:2000]}}]},
            }
        },
    )
    if PAGE_UPDATE_SLEEP_SECONDS:
        time.sleep(PAGE_UPDATE_SLEEP_SECONDS)
    return {"page_id": change.page_id, "status": "updated"}


def apply_changes(client: NotionSessoesClient, changes: list[PageChange], *, max_pages: int = 0) -> list[dict[str, Any]]:
    schema = client.fetch_schema()
    selected = changes[:max_pages] if max_pages > 0 else changes
    results: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=PAGE_UPDATE_WORKERS) as executor:
        futures = {executor.submit(patch_page, client, schema.title_property_name, change): change for change in selected}
        for future in as_completed(futures):
            change = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"page_id": change.page_id, "status": "failed", "error": str(exc)}
            results.append(result)
            completed += 1
            if completed % 100 == 0:
                LOGGER.info("Paginas atualizadas: %s/%s", completed, len(selected))
    return results


def readback_sample(client: NotionSessoesClient, changes: list[PageChange], *, limit: int = 25) -> dict[str, Any]:
    pages = query_data_source_with_retry(client)
    by_id = {str(page.get("id", "")): page for page in pages}
    schema = client.fetch_schema()
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for change in changes[:limit]:
        page = by_id.get(change.page_id)
        if not page:
            mismatches.append({"page_id": change.page_id, "reason": "page_not_found"})
            continue
        row = notion_page_to_row(client, schema, page)
        checked += 1
        if row.tema != change.new_tema or row.punchline != change.new_punchline:
            mismatches.append(
                {
                    "page_id": change.page_id,
                    "expected_tema": change.new_tema,
                    "actual_tema": row.tema,
                    "expected_punchline": change.new_punchline,
                    "actual_punchline": row.punchline,
                }
            )
    return {"checked": checked, "mismatches": mismatches, "total_pages": len(pages)}


def write_reports(
    artifact_dir: Path,
    *,
    records: list[PageRecord],
    changes: list[PageChange],
    validation_issues: list[dict[str, Any]],
    apply_results: list[dict[str, Any]],
    readback: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "records_snapshot.json").write_text(
        json.dumps([record_source_payload(record) | {"page_id": record.page_id, "page_url": record.page_url, "index": record.index} for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "changes.json").write_text(json.dumps([change.as_dict() for change in changes], ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "validation_issues.json").write_text(json.dumps(validation_issues, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "apply_results.json").write_text(json.dumps(apply_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "readback.json").write_text(json.dumps(readback, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reescreve integralmente tema e punchline da base sessoes no Notion.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--page-id", dest="page_ids", action="append", default=[])
    parser.add_argument("--max-apply-pages", type=int, default=0)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    notion_key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    gemini_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not notion_key:
        raise RuntimeError("NOTION_API_KEY/NOTION_TOKEN nao encontrado.")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY nao encontrado.")
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M")
    client = NotionSessoesClient(api_key=notion_key, data_source_id=args.data_source_id)
    records = load_records(client, start_index=args.start_index, max_records=args.max_records, page_ids=args.page_ids)
    LOGGER.info("Registros carregados: %s", len(records))
    changes, validation_issues = generate_changes(
        api_key=gemini_key,
        model=args.model,
        records=records,
        artifact_dir=artifact_dir,
        batch_size=max(1, args.batch_size),
        force=args.force_regenerate,
    )
    LOGGER.info("Mudancas preparadas: %s; saidas com ajustes locais: %s", len(changes), len(validation_issues))
    apply_results: list[dict[str, Any]] = []
    readback: dict[str, Any] = {}
    if args.apply:
        apply_results = apply_changes(client, changes, max_pages=args.max_apply_pages)
        applied_changes = changes[: args.max_apply_pages] if args.max_apply_pages > 0 else changes
        readback = readback_sample(client, applied_changes)
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "total_records": len(records),
        "total_changes": len(changes),
        "validation_issues": len(validation_issues),
        "source_insufficient": sum(1 for change in changes if change.source_insufficient),
        "confidence": dict(Counter(change.confidence for change in changes).most_common()),
        "applied_pages": sum(1 for result in apply_results if result.get("status") == "updated"),
        "failed_pages": sum(1 for result in apply_results if result.get("status") == "failed"),
        "readback_checked": readback.get("checked", 0),
        "readback_mismatches": len(readback.get("mismatches", [])),
        "artifact_dir": str(artifact_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_reports(
        artifact_dir,
        records=records,
        changes=changes,
        validation_issues=validation_issues,
        apply_results=apply_results,
        readback=readback,
        summary=summary,
    )
    LOGGER.info("Relatorios gravados em %s", artifact_dir)
    LOGGER.info("Resumo: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
