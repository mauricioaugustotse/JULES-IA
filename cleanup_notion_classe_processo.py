from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import normalize_class_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient, PublishPreviewRow
from tse_backfill_2025_notion import notion_page_to_row


LOGGER = logging.getLogger("cleanup_notion_classe_processo")
ARTIFACT_ROOT = Path("artifacts") / "notion_classe_processo_cleanup"
PAGE_UPDATE_SLEEP_SECONDS = 0.2
SCHEMA_SLEEP_SECONDS = 0.2

INVALID_TSE_CLASSES = {"ADI", "ADO"}
APPELLATE_OR_INCIDENT_CLASSES = {
    "AgRg-AREspe",
    "AgRg-REspe",
    "AgRg-RO",
    "AgRg-MS",
    "AgRg-PC",
    "AgRg-AR",
    "AgRg-HC",
    "AgR-HC",
    "ED-AgRg-AREspe",
    "ED-AREspe",
    "ED-REspe",
    "ED-RO",
    "ED-PC",
    "ED-Lista Tríplice",
    "AREspe",
    "REspe",
    "RO",
    "RHC",
    "RMS",
    "MS",
}
GENERIC_REPLACEABLE_CLASSES = {"PA"}


@dataclass
class PageRecord:
    page_id: str
    page_url: str
    index: int
    row: PublishPreviewRow


@dataclass
class ClasseProposal:
    value: str
    reason: str
    confidence: str
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class PageChange:
    page_id: str
    page_url: str
    index: int
    numero_processo: str
    data_sessao: str
    tema: str
    old: str
    new: str
    reason: str
    confidence: str
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_url": self.page_url,
            "index": self.index,
            "numero_processo": self.numero_processo,
            "data_sessao": self.data_sessao,
            "tema": self.tema,
            "field": "classe_processo",
            "old": self.old,
            "new": self.new,
            "reason": self.reason,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


def _squash(value: str, *, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def build_inference_text(row: PublishPreviewRow) -> tuple[str, str]:
    parts = [
        row.numero_processo,
        row.tema,
        row.punchline,
        row.analise_do_conteudo_juridico,
        row.fundamentacao_normativa,
        row.precedentes_citados,
        row.raciocinio_juridico,
        row.resolucoes_citadas,
        row.resultado,
        row.votacao,
    ]
    raw = "\n".join(part for part in parts if str(part or "").strip())
    return raw, normalize_class_text(raw)


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


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _proposal(value: str, reason: str, evidence: str, confidence: str = "high") -> ClasseProposal:
    return ClasseProposal(value=value, reason=reason, confidence=confidence, evidence=_squash(evidence))


def infer_classe_processo_for_cleanup(row: PublishPreviewRow) -> ClasseProposal | None:
    raw, text = build_inference_text(row)
    if not text:
        return None

    # Incidents and appellate classes must come before subject-matter classes:
    # "recurso especial sobre prestacao de contas" is REspe, not PC.
    if _has(text, r"\bembargos de declaracao\b.*\b(agravo regimental|agravo interno|agrg)\b.*\b(agravo em recurso especial eleitoral|arespe)\b"):
        return _proposal("ED-AgRg-AREspe", "embargos de declaracao em agravo regimental no AREspe", raw)
    if _has(text, r"\bembargos de declaracao\b.*\b(agravo em recurso especial eleitoral|arespe)\b"):
        return _proposal("ED-AREspe", "embargos de declaracao em agravo em recurso especial eleitoral", raw)
    if _has(text, r"\bembargos de declaracao\b.*\b(recurso especial eleitoral|respe)\b"):
        return _proposal("ED-REspe", "embargos de declaracao em recurso especial eleitoral", raw)
    if _has(text, r"\bembargos de declaracao\b.*\b(recurso ordinario(?: eleitoral)?|ro)\b"):
        return _proposal("ED-RO", "embargos de declaracao em recurso ordinario", raw)
    if _has(text, r"\bembargos de declaracao\b.*\b(prestacao de contas|pc)\b"):
        return _proposal("ED-PC", "embargos de declaracao em prestacao de contas", raw)
    if _has(text, r"\bembargos de declaracao\b.*\blista triplice\b"):
        return _proposal("ED-Lista Tríplice", "embargos de declaracao em lista triplice", raw)

    if _has(text, r"\b(agravo regimental|agravo interno|agrg)\b.*\b(agravo em recurso especial eleitoral|arespe)\b"):
        return _proposal("AgRg-AREspe", "agravo regimental em agravo em recurso especial eleitoral", raw)
    if _has(text, r"\b(agravo regimental|agravo interno|agrg)\b.*\b(recurso especial eleitoral|respe)\b"):
        return _proposal("AgRg-REspe", "agravo regimental em recurso especial eleitoral", raw)
    if _has(text, r"\b(agravo regimental|agravo interno|agrg)\b.*\b(recurso ordinario(?: eleitoral)?|ro)\b"):
        return _proposal("AgRg-RO", "agravo regimental em recurso ordinario", raw)
    if _has(text, r"\b(agravo regimental|agravo interno|agrg)\b.*\bmandado de seguranca\b"):
        return _proposal("AgRg-MS", "agravo regimental em mandado de seguranca", raw)
    if _has(text, r"\b(agravo regimental|agravo interno|agrg)\b.{0,140}\b(?:em|na|nos autos da)?\s*prestacao de contas\b|\bprestacao de contas\b.{0,140}\b(agravo regimental|agravo interno|agrg)\b"):
        return _proposal("AgRg-PC", "agravo regimental em prestacao de contas", raw)
    if _has(text, r"\b(agravo regimental|agravo interno|agrg)\b.*\bacao rescisoria\b"):
        return _proposal("AgRg-AR", "agravo regimental em acao rescisoria", raw)
    if _has(text, r"\b(agravo regimental|agravo interno|agrg)\b.*\bhabeas corpus\b"):
        return _proposal("AgRg-HC", "agravo regimental em habeas corpus", raw)

    if _has(text, r"\bagravo(?:s)? em recurso(?:s)? especial(?:is)? eleitoral(?:is)?\b|\barespe\b"):
        return _proposal("AREspe", "agravo em recurso especial eleitoral identificado no texto", raw)
    if _has(text, r"\brecurso(?:s)? especial(?:is)? eleitoral(?:is)?\b|\brespe\b"):
        return _proposal("REspe", "recurso especial eleitoral identificado no texto", raw)
    if _has(text, r"\brecurso ordinario(?: eleitoral)?\b|\brecursos ordinarios eleitorais\b"):
        return _proposal("RO", "recurso ordinario identificado no texto", raw)
    if _has(text, r"\breferend\w*\b.*\btutela cautelar antecedente\b"):
        return _proposal("Ref-TutCautAnt", "referendo em tutela cautelar antecedente", raw)
    if _has(text, r"\btutela cautelar antecedente\b"):
        return _proposal("TutCautAnt", "tutela cautelar antecedente identificada no texto", raw)
    if _has(text, r"\brecurso em habeas corpus\b|\brhc\b"):
        return _proposal("RHC", "recurso em habeas corpus identificado no texto", raw)
    if _has(text, r"\brecurso em mandado de seguranca\b|\brms\b"):
        return _proposal("RMS", "recurso em mandado de seguranca identificado no texto", raw)
    if _has(text, r"\bmandado de seguranca\b"):
        return _proposal("MS", "mandado de seguranca identificado no texto", raw)

    if _has(text, r"\bpedido de registro (?:do|de) partido politico\b|\bpedido de registro de alterac(?:ao|oes) estatutaria(?:s)?\b|\bpedido de anotacao de alteracao estatutaria\b|\bregistro do estatuto\b|\bregistro de mudanca estatutaria\b|\balteracao estatutaria partidaria\b|\bregistro de partido politico\b|\bregistro de federacao partidaria\b"):
        return _proposal("RPP", "registro ou alteracao estatutaria de partido politico", raw)
    if _has(text, r"\bconsulta formulada\b|\btrata-se de consulta\b|\bprocesso trata de uma consulta\b|\bobjeto da consulta\b|\bconsulta eleitoral\b"):
        return _proposal("CTA", "consulta eleitoral identificada no texto", raw)
    if _has(text, r"\bprestacao de contas\b|\bcontas partidarias\b|\bcontas de campanha\b|\bdesaprovacao de contas\b"):
        return _proposal("PC", "prestacao de contas identificada no texto", raw)
    if _has(text, r"\blista triplice\b|\bindicados em lista triplice\b"):
        return _proposal("Lista Tríplice", "lista triplice identificada no texto", raw)
    if _has(text, r"\bacao de investigacao judicial eleitoral\b|\baije\b"):
        return _proposal("AIJE", "acao de investigacao judicial eleitoral identificada no texto", raw)
    if _has(text, r"\brepresentacao eleitoral\b|\brepresentacao por propaganda\b|\brepresentacao ajuizada\b|\bprocesso trata de uma representacao\b|\brp\s+\d+\b"):
        return _proposal("Rp", "representacao eleitoral identificada no texto", raw)
    if _has(text, r"\bquestao de ordem\b"):
        return _proposal("QO", "questao de ordem identificada no texto", raw)
    if _has(text, r"\bpeticao civel\b|\bpetciv\b"):
        return _proposal("PetCiv", "peticao civel identificada no texto", raw)
    if _has(text, r"\bacao rescisoria\b"):
        return _proposal("AR", "acao rescisoria identificada no texto", raw)
    if _has(text, r"\b(?:pedido|requerimento) de registro de candidatura\b.*\b(?:presidente da republica|vice-presidente da republica|presidencia da republica|vice-presidencia da republica)\b|\bregistro de candidatura presidencial\b|\bregistro de candidatura a presidencia da republica\b|\bregistro de candidatura a vice-presidencia da republica\b"):
        return _proposal("RCand", "registro de candidatura presidencial identificado no texto", raw)
    if _has(text, r"\btrata-se de demonstrativo de regularidade de atos partidarios\b|\bprocesso trata d[ao] demonstrativo de regularidade de atos partidarios\b|\bregularidade de drap\b|\bdemonstrativo de regularidade de atos partidarios \(drap\)\b"):
        return _proposal("DRAP", "demonstrativo de regularidade de atos partidarios identificado no texto", raw)
    if _has(text, r"\bcriacao de zona eleitoral\b|\bremanejamento\b.*\bzona eleitoral\b"):
        return _proposal("Czer", "criacao ou remanejamento de zona eleitoral", raw)
    if _has(text, r"\brevisao do eleitorado\b"):
        return _proposal("RvE", "revisao do eleitorado identificada no texto", raw)
    if _has(text, r"\bprocesso administrativo\b|\bproposta de alteracao (?:na|da|de) resolucao\b|\balteracao (?:na|da|de) resolucao tse\b|\binstrucao para regulamentacao\b|\bminuta\b.*\bresolucao\b|\bforca federal\b|\bestrutura organica\b|\batos gerais do processo eleitoral\b|\bsistema eletronico de votacao\b"):
        return _proposal("PA", "processo administrativo ou instrucao normativa do TSE", raw)
    if _has(text, r"\bhabeas corpus\b"):
        return _proposal("HC", "habeas corpus identificado no texto", raw)
    if _has(text, r"\b(?:ajuizou|interpos|apresentou|propos|formulou)\s+reclamacao\b|\breclamacao\s+(?:ajuizada|apresentada|interposta|proposta)\b|\bprocesso trata de uma reclamacao\b|\btrata-se de reclamacao\b"):
        return _proposal("Rcl", "reclamacao identificada no texto", raw)

    return None


def explicit_current_cleanup_proposal(row: PublishPreviewRow, current: str) -> ClasseProposal | None:
    if current != "PC":
        return None
    raw, text = build_inference_text(row)
    tema_text = normalize_class_text(" ".join([row.tema, row.analise_do_conteudo_juridico]))
    account_core = _has(
        tema_text,
        r"\bprestacao de contas partidarias\b|\bprestacao de contas de campanha\b|\bcontas de campanha\b|\bcontas partidarias\b|\bdesaprovacao de contas\b",
    )
    registration_core = _has(tema_text, r"\bregistro de candidatura\b|\bfiliacao partidaria\b")
    incidental_accounts = _has(text, r"\bausencia de prestacao de contas\b|\bprestacao de contas como inovacao recursal\b")
    if registration_core and incidental_accounts and not account_core:
        return _proposal(
            "",
            "classe PC removida: o nucleo textual e registro de candidatura/filiacao partidaria, e a mencao a prestacao de contas e incidental",
            raw,
            confidence="review",
        )
    return None


def should_apply_proposal(current: str, proposal: ClasseProposal | None) -> tuple[bool, str, str, str, str]:
    current = str(current or "").strip()
    if not proposal:
        if current in INVALID_TSE_CLASSES:
            return (
                True,
                "",
                "classe ADI/ADO removida: a etiqueta nao corresponde a classe processual julgada pelo TSE e o texto nao permite inferencia segura",
                "review",
                "",
            )
        return False, current, "", "", ""

    candidate = proposal.value
    if candidate == current:
        return False, current, "", "", ""

    if not candidate and current:
        return True, "", proposal.reason, proposal.confidence, proposal.evidence

    if current in INVALID_TSE_CLASSES:
        return (
            True,
            candidate,
            f"{proposal.reason}; substitui etiqueta ADI/ADO indevida",
            proposal.confidence,
            proposal.evidence,
        )

    if not current:
        return True, candidate, proposal.reason, proposal.confidence, proposal.evidence

    if current in GENERIC_REPLACEABLE_CLASSES and proposal.confidence == "high" and candidate not in GENERIC_REPLACEABLE_CLASSES:
        return True, candidate, f"{proposal.reason}; substitui classe generica {current}", proposal.confidence, proposal.evidence

    if current == "REspe" and candidate in {"AgRg-REspe", "ED-REspe"}:
        return True, candidate, f"{proposal.reason}; torna incidente processual explicito", proposal.confidence, proposal.evidence
    if current == "AREspe" and candidate in {"AgRg-AREspe", "ED-AREspe", "ED-AgRg-AREspe"}:
        return True, candidate, f"{proposal.reason}; torna incidente processual explicito", proposal.confidence, proposal.evidence
    if current == "RO" and candidate in {"AgRg-RO", "ED-RO"}:
        return True, candidate, f"{proposal.reason}; torna incidente processual explicito", proposal.confidence, proposal.evidence
    if current == "PC" and candidate in {"AgRg-PC", "ED-PC"}:
        return True, candidate, f"{proposal.reason}; torna incidente processual explicito", proposal.confidence, proposal.evidence
    if current == "MS" and candidate == "AgRg-MS":
        return True, candidate, f"{proposal.reason}; torna incidente processual explicito", proposal.confidence, proposal.evidence
    if current == "Lista Tríplice" and candidate == "ED-Lista Tríplice":
        return True, candidate, f"{proposal.reason}; torna incidente processual explicito", proposal.confidence, proposal.evidence

    return False, current, "", "", ""


def load_records(client: NotionSessoesClient) -> tuple[list[PageRecord], Counter[str], list[str]]:
    schema = client.fetch_schema()
    pages = query_data_source_with_retry(client)
    records: list[PageRecord] = []
    distribution: Counter[str] = Counter()
    for index, page in enumerate(pages):
        row = notion_page_to_row(client, schema, page)
        current = str(row.classe_processo or "").strip()
        distribution[current] += 1
        records.append(
            PageRecord(
                page_id=str(page.get("id", "")),
                page_url=str(page.get("url", "")),
                index=index,
                row=row,
            )
        )
    schema_options = list(schema.properties["classe_processo"].options)
    return records, distribution, schema_options


def build_audit(records: list[PageRecord]) -> tuple[list[PageChange], list[dict[str, Any]], Counter[str]]:
    changes: list[PageChange] = []
    pending: list[dict[str, Any]] = []
    proposal_distribution: Counter[str] = Counter()
    for record in records:
        current = str(record.row.classe_processo or "").strip()
        proposal = explicit_current_cleanup_proposal(record.row, current) or infer_classe_processo_for_cleanup(record.row)
        if proposal:
            proposal_distribution[proposal.value] += 1
        should_apply, new_value, reason, confidence, evidence = should_apply_proposal(current, proposal)
        if should_apply:
            changes.append(
                PageChange(
                    page_id=record.page_id,
                    page_url=record.page_url,
                    index=record.index,
                    numero_processo=record.row.numero_processo,
                    data_sessao=record.row.data_sessao,
                    tema=record.row.tema,
                    old=current,
                    new=new_value,
                    reason=reason,
                    confidence=confidence,
                    evidence=evidence,
                )
            )
            continue
        if current in INVALID_TSE_CLASSES or (not current and not proposal):
            pending.append(
                {
                    "page_id": record.page_id,
                    "page_url": record.page_url,
                    "index": record.index,
                    "numero_processo": record.row.numero_processo,
                    "data_sessao": record.row.data_sessao,
                    "tema": record.row.tema,
                    "current": current,
                    "proposal": proposal.as_dict() if proposal else None,
                    "reason": "sem evidencia textual suficiente para preencher com seguranca" if not proposal else "proposta nao aplicada por prudencia",
                    "evidence": _squash(build_inference_text(record.row)[0]),
                }
            )
    return changes, pending, proposal_distribution


def patch_page(client: NotionSessoesClient, change: PageChange) -> dict[str, Any]:
    select_value = {"name": change.new} if change.new else None
    notion_request_with_retry(
        client,
        "PATCH",
        f"/pages/{change.page_id}",
        json={"properties": {"classe_processo": {"select": select_value}}},
    )
    if PAGE_UPDATE_SLEEP_SECONDS:
        time.sleep(PAGE_UPDATE_SLEEP_SECONDS)
    return {"page_id": change.page_id, "status": "updated", "new": change.new}


def apply_page_changes(client: NotionSessoesClient, changes: list[PageChange], *, max_pages: int = 0) -> list[dict[str, Any]]:
    selected = changes[:max_pages] if max_pages > 0 else changes
    schema = client.fetch_schema()
    existing = set(schema.properties["classe_processo"].options)
    missing = sorted({change.new for change in selected if change.new and change.new not in existing})
    if missing:
        LOGGER.info("Criando opcoes ausentes em classe_processo: %s", ", ".join(missing))
        client.ensure_select_options_default({"classe_processo": missing})
    results: list[dict[str, Any]] = []
    for index, change in enumerate(selected, start=1):
        try:
            results.append(patch_page(client, change))
        except Exception as exc:
            results.append({"page_id": change.page_id, "status": "failed", "error": str(exc), "new": change.new})
        if index % 50 == 0:
            LOGGER.info("Paginas processadas: %s/%s", index, len(selected))
    return results


def cleanup_schema_options(client: NotionSessoesClient, *, apply_changes: bool) -> dict[str, Any]:
    schema = client.fetch_schema()
    pages = query_data_source_with_retry(client)
    raw_prop = schema.raw_payload.get("properties", {}).get("classe_processo", {})
    if raw_prop.get("type") != "select":
        return {"property": "classe_processo", "status": "missing_or_not_select"}
    options = raw_prop.get("select", {}).get("options", []) or []
    color_by_name = {
        str(option.get("name", "")).strip(): str(option.get("color") or "default")
        for option in options
        if str(option.get("name", "")).strip()
    }
    used: list[str] = []
    seen: set[str] = set()
    used_counter: Counter[str] = Counter()
    for page in pages:
        value = page.get("properties", {}).get("classe_processo", {}).get("select") or {}
        name = str(value.get("name", "")).strip()
        if not name:
            continue
        used_counter[name] += 1
        if name not in seen:
            seen.add(name)
            used.append(name)
    option_names = [str(option.get("name", "")).strip() for option in options if str(option.get("name", "")).strip()]
    unused = [name for name in option_names if name not in seen]
    invalid_used = [name for name in INVALID_TSE_CLASSES if used_counter[name]]
    payload_options = [{"name": value, "color": color_by_name.get(value, "default")} for value in used]
    result = {
        "property": "classe_processo",
        "used_options": len(used),
        "schema_options_before": len(option_names),
        "unused_options": len(unused),
        "unused_examples": unused[:50],
        "invalid_used": invalid_used,
    }
    if not apply_changes:
        result["status"] = "would_patch_options" if unused else "no_unused_options"
        return result
    if unused and len(used) <= 100:
        time.sleep(SCHEMA_SLEEP_SECONDS)
        notion_request_with_retry(
            client,
            "PATCH",
            f"/data_sources/{client.data_source_id}",
            json={"properties": {"classe_processo": {"select": {"options": payload_options}}}},
        )
        result["status"] = "patched_options"
        return result
    result["status"] = "skipped_schema_patch"
    return result


def readback_summary(client: NotionSessoesClient, expected_changes: list[PageChange]) -> dict[str, Any]:
    records, distribution, schema_options = load_records(client)
    by_id = {record.page_id: record for record in records}
    mismatches: list[dict[str, Any]] = []
    for change in expected_changes:
        record = by_id.get(change.page_id)
        if not record:
            mismatches.append({"page_id": change.page_id, "expected": change.new, "actual": None, "reason": "page_not_found"})
            continue
        actual = str(record.row.classe_processo or "").strip()
        if actual != change.new:
            mismatches.append({"page_id": change.page_id, "expected": change.new, "actual": actual})
    invalid_active = {name: distribution[name] for name in sorted(INVALID_TSE_CLASSES) if distribution[name]}
    return {
        "total_records": len(records),
        "distribution": dict(distribution.most_common()),
        "schema_options": schema_options,
        "invalid_active": invalid_active,
        "mismatches": mismatches,
    }


def write_reports(
    artifact_dir: Path,
    *,
    changes: list[PageChange],
    pending: list[dict[str, Any]],
    apply_results: list[dict[str, Any]],
    schema_result: dict[str, Any],
    readback: dict[str, Any],
    distribution_before: Counter[str],
    proposal_distribution: Counter[str],
    schema_options_before: list[str],
    summary: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "changes.json").write_text(
        json.dumps([change.as_dict() for change in changes], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "pending_review.json").write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "apply_results.json").write_text(json.dumps(apply_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "schema_result.json").write_text(json.dumps(schema_result, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "readback_summary.json").write_text(json.dumps(readback, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "distribution_before": dict(distribution_before.most_common()),
        "proposal_distribution": dict(proposal_distribution.most_common()),
        "schema_options_before": schema_options_before,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita e corrige classe_processo na base sessoes do Notion.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--skip-schema-cleanup", action="store_true")
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
    records, distribution_before, schema_options_before = load_records(client)
    changes, pending, proposal_distribution = build_audit(records)
    LOGGER.info("Paginas lidas: %s; mudancas propostas: %s; pendencias: %s", len(records), len(changes), len(pending))
    apply_results: list[dict[str, Any]] = []
    if args.apply and changes:
        apply_results = apply_page_changes(client, changes, max_pages=args.max_pages)
    schema_result: dict[str, Any] = {"status": "skipped_by_flag"}
    if not args.skip_schema_cleanup:
        schema_result = cleanup_schema_options(client, apply_changes=args.apply)
    readback: dict[str, Any] = {}
    if args.apply:
        readback = readback_summary(client, changes[: args.max_pages] if args.max_pages > 0 else changes)
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "total_records": len(records),
        "total_changes": len(changes),
        "changes_by_old": dict(Counter(change.old for change in changes).most_common()),
        "changes_by_new": dict(Counter(change.new for change in changes).most_common()),
        "pending_review": len(pending),
        "applied_pages": sum(1 for result in apply_results if result.get("status") == "updated"),
        "failed_pages": sum(1 for result in apply_results if result.get("status") == "failed"),
        "schema_status": schema_result.get("status"),
        "invalid_active_after": readback.get("invalid_active", {}),
        "readback_mismatches": len(readback.get("mismatches", [])),
        "artifact_dir": str(artifact_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_reports(
        artifact_dir,
        changes=changes,
        pending=pending,
        apply_results=apply_results,
        schema_result=schema_result,
        readback=readback,
        distribution_before=distribution_before,
        proposal_distribution=proposal_distribution,
        schema_options_before=schema_options_before,
        summary=summary,
    )
    LOGGER.info("Relatorios gravados em %s", artifact_dir)
    LOGGER.info("Resumo: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
