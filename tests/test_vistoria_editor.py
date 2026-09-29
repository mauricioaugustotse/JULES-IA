import pytest

from vistoria_editor import prepare_edited_item


def proposal(**overrides):
    row = {"numero_processo": "0600113-04.2026.6.00.0000", "classe_processo": "PA",
           "data_sessao": "2026-09-24", "origem": "Doutor Severiano/RN", "relator": "Min. Nunes Marques",
           "resultado": "Deferido", "votacao": "Unânime", "tema": "Requisição de forças federais",
           "punchline": "Requisição de forças federais para a zona eleitoral.",
           "errors": ["[oficial:official_unexpected_row] Identidade incerta.", "Bloqueio de auditoria independente."]}
    row.update(overrides)
    return {"id": "candidate", "status": "pending", "row": row,
            "extra": {"coverage_issue": {"code": "official_unexpected_row", "severity": "error", "row_index": 2}},
            "reasons": list(row["errors"])}


def inventory(*processes):
    return {"status": "available", "session_date": "2026-09-24", "processes": list(processes)}


PA = {"numeroProcesso": "0601130-04.2026.6.20.0000", "situacaoProcesso": "Julgado",
      "segredoJustica": True, "relator": "NUNES MARQUES"}


def test_edit_rechecks_identity_preserves_unrelated_blockers_and_records_audit():
    original = proposal()
    patch = prepare_edited_item(original, {"numero_processo": PA["numeroProcesso"]}, inventory=inventory(PA), evidence={})
    assert patch["row"]["numero_processo"] == PA["numeroProcesso"]
    assert not any(error.startswith("[oficial:") for error in patch["row"]["errors"])
    assert "Bloqueio de auditoria independente." in patch["row"]["errors"]
    assert not patch["extra"]["coverage_issues"]
    change = patch["extra"]["manual_edits"][-1]["changes"][0]
    assert change["before"] == "0600113-04.2026.6.00.0000"
    assert change["after"] == PA["numeroProcesso"]
    assert original["row"]["numero_processo"] == change["before"]


def test_unavailable_inventory_does_not_erase_official_blockers():
    patch = prepare_edited_item(proposal(), {"numero_processo": PA["numeroProcesso"]}, inventory={"status": "unavailable"}, evidence={})
    assert any(error.startswith("[oficial:") for error in patch["row"]["errors"])
    assert patch["extra"]["coverage_issues"][0]["code"] == "official_unexpected_row"


def test_recheck_does_not_block_candidate_for_other_missing_cases():
    other = {"numeroProcesso": "0600877-71.2026.6.00.0000", "situacaoProcesso": "Julgado"}
    patch = prepare_edited_item(proposal(), {"numero_processo": PA["numeroProcesso"]}, inventory=inventory(PA, other), evidence={})
    assert not any(issue.get("code") == "official_missing_judgment" for issue in patch["extra"]["coverage_issues"])


def test_edit_cannot_turn_withdrawn_case_into_final_judgment():
    withdrawn = {**PA, "situacaoProcesso": "Não julgado", "motivoRetiradaPauta": "Adiado"}
    patch = prepare_edited_item(proposal(), {"numero_processo": PA["numeroProcesso"]}, inventory=inventory(withdrawn), evidence={})
    assert any("official_excluded_row" in error for error in patch["row"]["errors"])


def test_diagnostic_only_and_nonpending_items_cannot_be_edited():
    with pytest.raises(ValueError):
        prepare_edited_item({"status": "pending", "row": None}, {})
    with pytest.raises(ValueError):
        prepare_edited_item({**proposal(), "status": "published"}, {})
