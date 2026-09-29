"""Restore session facts proven before publication if later maintenance changes them."""
from __future__ import annotations

from typing import Any

from tse_workflow_monitor import _field_matches, _property_value


SESSION_FIELDS = {"numero_processo", "data_sessao", "relator", "composicao",
                  "classe_processo", "origem", "resultado", "votacao", "pedido_vista"}


def repair_confirmed_notion_fields(rows, results, client, schema, reconciliation, *, checkpoint=None):
    """Patch only fields supported by the reconciliation audit, then let callers verify.

    A journal checkpoint is persisted before each write. Unconfirmed extraction
    values are never restored automatically. Repeating the call is a no-op once
    the protected properties agree.
    """
    protected: dict[tuple[str, str], set[str]] = {}
    for phase in (reconciliation or {}).get("phases", [reconciliation or {}]):
        for match in phase.get("matches", []):
            key = (str(phase.get("session_date") or ""), str(match.get("numero_processo") or ""))
            protected.setdefault(key, set()).update(set(match.get("confirmed_fields", [])) & SESSION_FIELDS)
    events: list[dict[str, Any]] = []
    for index, (row, result) in enumerate(zip(rows, results)):
        fields = protected.get((row.data_sessao, row.numero_processo), set())
        if not fields or result.get("status") not in {"created", "updated"}:
            continue
        event = {"row_index": index, "page_id": result.get("page_id"), "numero_processo": row.numero_processo}
        try:
            page_id = event["page_id"]
            if not page_id:
                raise ValueError("Publicação sem identificador da página")
            page = client._request("GET", f"/pages/{page_id}")
            if page.get("archived") or page.get("in_trash"):
                raise ValueError("Página arquivada; restauração automática não aplicável")
            if str(page.get("id", "")).replace("-", "") != page_id.replace("-", ""):
                raise ValueError("Identificador diverge do diário da publicação")
            payload = client.build_properties_payload(schema, row)
            changes, before = {}, {}
            for name, expected in payload.items():
                if name not in fields:
                    continue
                kind = next(iter(expected))
                actual = page.get("properties", {}).get(name, {})
                try:
                    same = _field_matches(name, _property_value(expected, kind), _property_value(actual, kind))
                except (KeyError, ValueError, TypeError):
                    same = False
                if not same:
                    changes[name] = expected
                    before[name] = actual
            if not changes:
                continue
            event.update(status="planned", before=before, properties=changes)
            events.append(event)
            if checkpoint:
                checkpoint(events)
            response = client._request("PATCH", f"/pages/{page_id}", json={"properties": changes})
            if str(response.get("id", "")).replace("-", "") != page_id.replace("-", ""):
                raise ValueError("Notion não confirmou a página corrigida")
            event["status"] = "applied"
        except Exception as exc:
            event.update(status="error", error=str(exc))
            if not any(existing is event for existing in events):
                events.append(event)
        if checkpoint:
            checkpoint(events)
    return events
