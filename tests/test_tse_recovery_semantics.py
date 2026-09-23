"""Regressions discovered while recovering the September session batch."""
import json

import pytest

import tse_youtube_notion_core as core


@pytest.mark.parametrize("outcome", ["Deferido", "Indeferido"])
def test_federal_force_administrative_request_accepts_actual_outcome(outcome):
    row = core.PublishPreviewRow(
        classe_processo="PA", numero_processo="0600001-11",
        tema="Emprego de força federal nas eleições", resultado=outcome,
    )
    core.apply_rag_consistency_checks(row)
    assert not any("incompat" in warning.lower() for warning in row.warnings)
    assert outcome in core.resultado_allowed_for_classe("PA")
    assert "Desprovido" not in core.resultado_allowed_for_classe("PA")


def test_cached_rito_refresh_corrects_old_count_without_rewriting_windows(tmp_path):
    store = core.RunArtifacts(tmp_path)
    old = {"apregoamentos_individuais": 6, "events": [], "adjustments": [{"type": "original"}]}
    store.write_json("01b_rito_refinement.json", old)
    snippets = []
    for base in (0, 200, 400):
        snippets.extend([
            {"text": "Encerrada a fala anterior.", "start_seconds": base, "end_seconds": base + 50},
            {"text": "Chamo a julgamento o recurso especial.", "start_seconds": base + 60, "end_seconds": base + 70},
        ])
    store.write_json("raw_transcript_fetch.json", {"snippets": snippets})
    store.write_text("01_session_windows.json", '{"judgments": []}')
    windows_before = (tmp_path / "01_session_windows.json").read_bytes()
    report = core.refresh_cached_rito_report(store)
    assert report["apregoamentos_individuais"] == 3
    assert report["adjustments_are_historical"] is True
    assert store.read_json("01b_rito_refinement.original.json") == old
    assert (tmp_path / "01_session_windows.json").read_bytes() == windows_before
    current_before = (tmp_path / "01b_rito_refinement.json").stat().st_mtime_ns
    assert core.refresh_cached_rito_report(store) == report
    assert (tmp_path / "01b_rito_refinement.json").stat().st_mtime_ns == current_before


def test_cached_rito_without_local_transcript_does_not_modify_history(tmp_path):
    store = core.RunArtifacts(tmp_path)
    store.write_json("01b_rito_refinement.json", {"apregoamentos_individuais": 6})
    assert core.refresh_cached_rito_report(store) is None
    assert store.read_json("01b_rito_refinement.json")["apregoamentos_individuais"] == 6
    assert not store.exists("01b_rito_refinement.original.json")


@pytest.mark.parametrize("rationale", [
    "O processo foi retirado de pauta por falta de quórum.",
    "O julgamento foi adiado por ausência de quórum na sessão.",
])
def test_metadata_nonjudgment_is_not_mislabeled_as_precedent(tmp_path, rationale):
    row = core.PublishPreviewRow(numero_processo="0600015-33", classe_processo="REspe")
    response = core.ProcessMetadataResult(is_judged_process=False, rationale=rationale)
    core._apply_nonjudgment_metadata_assessment(row, response, core.RunArtifacts(tmp_path))
    state, reasons = core.assess_row_publishability(row)
    assert state == "skipped"
    assert rationale in " ".join(reasons)
    assert "precedente" not in " ".join(reasons).lower()


def test_metadata_false_without_reason_requires_review_instead_of_silent_skip(tmp_path):
    row = core.PublishPreviewRow(numero_processo="0600015-33", classe_processo="REspe")
    core._apply_nonjudgment_metadata_assessment(
        row, core.ProcessMetadataResult(is_judged_process=False), core.RunArtifacts(tmp_path),
    )
    state, reasons = core.assess_row_publishability(row)
    assert row.blocked
    assert state != "skipped"
    assert any("sem esclarecer" in error for error in row.errors)


def test_cached_metadata_false_reason_is_reclassified_without_new_request(tmp_path):
    store = core.RunArtifacts(tmp_path)
    row = core.PublishPreviewRow(
        numero_processo="0600015-33", classe_processo="REspe", data_sessao="2026-09-15",
        youtube_link="https://www.youtube.com/watch?v=gY6-5zPrFlo", errors=[
        "Busca Google indicou que o número consultado aparece como precedente citado, não como processo julgado.",
    ])
    saved = {
        "applied": row.model_dump(),
        "parsed": {"is_judged_process": False, "rationale": "Retirado de pauta por falta de quórum."},
    }
    store.write_json("04a_process_metadata_01.json", saved)
    obj = core.GeminiProcessMetadataEnricher.__new__(core.GeminiProcessMetadataEnricher)
    obj.artifact_store = store
    obj._call_grounded_json = lambda **kw: pytest.fail("Cached evidence must not cause a paid request")
    result = obj.enrich_rows([row])[0]
    state, reasons = core.assess_row_publishability(result)
    assert state == "skipped"
    assert "quórum" in " ".join(reasons)
    assert "precedente" not in " ".join(reasons)
    assert store.read_json("04a_process_metadata_01.json") == saved


def test_current_cached_rito_evidence_is_left_untouched(tmp_path):
    store = core.RunArtifacts(tmp_path)
    snippets = [core.TranscriptSnippet("Chamo a julgamento o recurso.", 100, 110)]
    events = core.detect_rito_events(snippets)
    previous = {
        "events": [event.model_dump() for event in events],
        "apregoamentos_individuais": core.count_individual_apregoamentos(events),
        "adjustments": [], "detector_version": 2,
    }
    store.write_json("raw_transcript_fetch.json", {"snippets": [
        {"text": "Chamo a julgamento o recurso.", "start_seconds": 100, "end_seconds": 110},
    ]})
    store.write_json("01b_rito_refinement.json", previous)
    original_bytes = (tmp_path / "01b_rito_refinement.json").read_bytes()
    assert core.refresh_cached_rito_report(store) == previous
    assert (tmp_path / "01b_rito_refinement.json").read_bytes() == original_bytes
    assert not store.exists("01b_rito_refinement.original.json")
