"""Correção local auditável de uma proposta, sem gravar no Notion."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from tse_official_session import compare_official_rows
from tse_session_reconciliation import reconcile_session_rows
from tse_youtube_notion_core import PublishPreviewRow, validate_preview_row
from vistoria_presenter import coverage_issues


EDITABLE_FIELDS = ("numero_processo", "classe_processo", "origem", "relator",
                   "resultado", "votacao", "pedido_vista", "composicao")


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def prepare_edited_item(item: dict[str, Any], edits: dict[str, Any], *,
                        inventory: dict | None = None, evidence: dict | None = None) -> dict[str, Any]:
    """Valida e devolve o patch; a chamada decide quando persistir na fila."""
    if item.get("status") != "pending" or not item.get("row"):
        raise ValueError("Selecione um caso pendente com proposta de julgamento.")
    unknown = set(edits) - set(EDITABLE_FIELDS)
    if unknown:
        raise ValueError("Campo não editável: " + ", ".join(sorted(unknown)))
    original = deepcopy(item["row"])
    proposal = deepcopy(original)
    for field, value in edits.items():
        proposal[field] = ([str(v).strip() for v in value if str(v).strip()] if field == "composicao"
                           and isinstance(value, list) else str(value or "").strip())
    row = validate_preview_row(PublishPreviewRow.model_validate(proposal), None)
    directory = Path(item.get("artifact_dir") or ".")
    inventory = inventory if inventory is not None else _read(directory / "00_official_session_inventory.json")
    evidence = evidence if evidence is not None else _read(directory / "03_analysis.json")
    original_issues = coverage_issues(item)
    fresh_issues = list(original_issues)
    audit: dict = {}
    available = inventory.get("status") == "available" and isinstance(inventory.get("processes"), list) and bool(inventory.get("session_date"))
    if available:
        row.errors = [error for error in row.errors if not str(error).startswith("[oficial:")]
        corrected, audit = reconcile_session_rows([row], inventory, evidence=evidence)
        if len(corrected) != 1:
            raise ValueError("A proposta foi identificada como citação. Confira o trecho antes de continuar.")
        row = corrected[0]
        comparison = compare_official_rows(inventory, [row.model_dump(mode="json")])
        related = [issue for issue in comparison if issue.get("row_index") == 0 or 0 in issue.get("row_indices", [])]
        fresh_issues = [issue for issue in original_issues if not str(issue.get("code", "")).startswith("official_")] + related
        for issue in related:
            if issue.get("severity") == "error":
                row.add_error(f"[oficial:{issue['code']}] {issue['message']}"
                              + (f" Esperado: {issue['expected']!r}; observado: {issue.get('actual')!r}." if "expected" in issue else ""))
    extra = deepcopy(item.get("extra") or {})
    history = list(extra.get("manual_edits") or [])
    history.append({"at": datetime.now(timezone.utc).isoformat(), "source": "GUI: Corrigir dados",
                    "changes": [{"field": field, "before": original.get(field), "requested": proposal.get(field),
                                 "after": getattr(row, field)} for field in EDITABLE_FIELDS if original.get(field) != proposal.get(field)],
                    "official_reconciliation": audit})
    extra.update(manual_edits=history, coverage_issues=fresh_issues)
    extra["coverage_issue"] = fresh_issues[0] if fresh_issues else None
    old_row_reasons = set(original.get("errors") or []) | set(original.get("warnings") or [])
    old_official_messages = [str(issue.get("message", "")) for issue in original_issues
                             if str(issue.get("code", "")).startswith("official_") and issue.get("message")]
    reasons = [reason for reason in item.get("reasons") or [] if reason not in old_row_reasons and not (
        available and (str(reason).startswith("[oficial:") or any(message in reason for message in old_official_messages)))]
    reasons = list(dict.fromkeys([*reasons, *row.errors, *row.warnings]))
    return {"row": row.model_dump(mode="json"), "extra": extra, "reasons": reasons,
            "triagem": "Proposta corrigida na GUI e revalidada; aguarda publicação."}
