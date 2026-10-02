import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_tse_youtube_notion_core import FakeNotionClient, make_schema
from tse_youtube_notion_core import PublishPreviewRow
import vistoria_queue as queue


def row(**overrides):
    return PublishPreviewRow(**{
        "tema": "Requisição de força federal", "numero_processo": "0601130-04.2026.6.20.0000",
        "source_start_seconds": 2752, "source_bundle_index": 3, "source_item_index": 1,
        "data_sessao": "2026-09-24", "classe_processo": "PA", "relator": "Min. Cármen Lúcia",
        "resultado": "Aprovada", "votacao": "Unânime", "origem": "Doutor Severiano/RN",
        "youtube_link": "https://youtube.com/watch?v=video&t=2752", **overrides,
    })


def item(candidate=None, source="batch"):
    return queue.make_vistoria_item(source=source, video_id="video", youtube_url="https://youtube.com/watch?v=video",
                                   disposition="blocked", reasons=["Conferir extração."],
                                   row=(candidate or row()).model_dump(mode="json"))


def issue(code="official_field_mismatch", **overrides):
    return {"code": code, "message": "Conferir resultado oficial.", "severity": "error",
            "numero_processo": row().numero_processo, "row_index": 0, **overrides}


def sync(path, issues, rows=None):
    return queue.sync_monitor_issues(issues, video_id="video", youtube_url="https://youtube.com/watch?v=video",
                                     artifact_dir="artifacts/run/video", rows=rows, queue_file=path)


def test_multiple_checks_share_one_candidate_and_repeat_does_not_write(tmp_path):
    path = tmp_path / "queue.jsonl"
    issues = [issue(), issue("official_composition_mismatch", message="Conferir composição.")]
    first = sync(path, issues, [row()])
    assert first["added"] == 1
    current = queue.load_items("pending", path)
    assert len(current) == 1
    assert current[0]["row"]["origem"] == "Doutor Severiano/RN"
    assert current[0]["data_sessao"] == "2026-09-24"
    assert len(current[0]["extra"]["coverage_issues"]) == 2
    before = path.read_bytes()
    repeated = sync(path, issues, [row()])
    assert repeated["added"] == repeated["updated"] == repeated["resolved"] == 0
    assert path.read_bytes() == before


def test_latest_empty_snapshot_resolves_legacy_alerts_and_keeps_history(tmp_path):
    path = tmp_path / "queue.jsonl"
    old = queue.make_vistoria_item(source="monitor", video_id="video", youtube_url="url",
                                  disposition="cobertura", reasons=["Número antigo incompleto"],
                                  extra={"coverage_issue": issue()}, dedupe_key="legacy")
    queue.append_items([old], path)
    assert sync(path, [], [row()])["resolved"] == 1
    assert queue.load_items("pending", path) == []
    history = queue.load_items(None, path)
    assert history[0]["status"] == "resolved"
    assert history[0]["reasons"] == old["reasons"]
    assert len(path.read_text().splitlines()) == 2


def test_existing_batch_absorbs_monitor_and_removes_obsolete_reasons(tmp_path):
    path = tmp_path / "queue.jsonl"
    original = item()
    queue.append_items([original], path)
    sync(path, [issue()], [row()])
    pending = queue.load_items("pending", path)
    assert len(pending) == 1
    assert pending[0]["id"] == original["id"]
    assert len(pending[0]["reasons"]) == 2
    sync(path, [], [row()])
    pending = queue.load_items("pending", path)
    assert pending[0]["reasons"] == original["reasons"]
    assert not pending[0]["extra"].get("coverage_issues")


def test_final_batch_item_absorbs_earlier_monitor_group(tmp_path):
    path = tmp_path / "queue.jsonl"
    sync(path, [issue()], [row()])
    old_id = queue.load_items("pending", path)[0]["id"]
    original = item()
    queue.append_items([original], path)
    pending = queue.load_items("pending", path)
    assert [candidate["id"] for candidate in pending] == [original["id"]]
    assert pending[0]["extra"]["coverage_issues"][0]["code"] == "official_field_mismatch"
    old = next(candidate for candidate in queue.load_items(None, path) if candidate["id"] == old_id)
    assert old["status"] == "resolved"
    assert old["superseded_by"] == original["id"]
    sync(path, [], [row()])
    assert queue.load_items("pending", path)[0]["reasons"] == original["reasons"]


def test_preserves_rejected_decision_even_with_corrected_number(tmp_path):
    path = tmp_path / "queue.jsonl"
    original = item(row(numero_processo="0600113-04.2026.6.00.0000"))
    queue.append_items([original], path)
    queue.update_status([original["id"]], "rejected", queue_file=path)
    sync(path, [issue()], [row()])
    queue.append_items([item()], path)
    assert queue.load_items("pending", path) == []
    assert queue.load_items(None, path)[0]["status"] == "rejected"


def test_preserves_publication_history_but_surfaces_new_errors_on_published_page(tmp_path):
    path = tmp_path / "queue.jsonl"
    original = item()
    queue.append_items([original], path)
    queue.update_status([original["id"]], "published", queue_file=path)
    sync(path, [issue()], [row()])
    assert queue.load_items("published", path)[0]["id"] == original["id"]
    assert len(queue.load_items("pending", path)) == 1
    assert queue.load_items("pending", path)[0]["source"] == "monitor"
    sync(path, [], [row()])
    assert not queue.load_items("pending", path)


def test_corrected_identity_updates_candidate_without_duplicate(tmp_path):
    path = tmp_path / "queue.jsonl"
    original = item(row(numero_processo="0600113-04.2026.6.00.0000"))
    queue.append_items([original], path)
    queue.append_items([item()], path)
    pending = queue.load_items("pending", path)
    assert len(pending) == 1
    assert pending[0]["id"] == original["id"]
    assert pending[0]["row"]["numero_processo"] == row().numero_processo


def test_citations_with_shared_timestamp_are_not_merged_with_main_judgment(tmp_path):
    path = tmp_path / "queue.jsonl"
    queue.append_items([item(), item(row(numero_processo="0600484-08", source_item_index=2))], path)
    assert len(queue.load_items("pending", path)) == 2


def test_diagnostic_alert_does_not_gain_blank_row_or_approval(tmp_path):
    path = tmp_path / "queue.jsonl"
    sync(path, [issue("official_missing_judgment", row_index=None)], [])
    diagnostic = queue.load_items("pending", path)[0]
    assert diagnostic["row"] is None
    assert queue.approval_eligibility(diagnostic)[0] is False
    assert queue.approval_eligibility({"row": {"numero_processo": row().numero_processo}})[0] is False


def test_official_conflict_cannot_be_bypassed_by_approval():
    notion = FakeNotionClient()
    result = queue.publish_approved_items(
        [item(row(errors=["[oficial:official_field_mismatch] Resultado diverge da ata."]))],
        notion, make_schema(), apply=True,
    )
    assert result[0]["status"] == "blocked"
    assert not notion.created


def test_revalidation_block_does_not_fall_back_to_direct_create(monkeypatch):
    notion = FakeNotionClient()
    def revalidate(candidate, _schema):
        candidate.add_error("Novo conflito encontrado na validação.")
        return candidate
    monkeypatch.setattr(queue, "validate_preview_row", revalidate)
    result = queue.publish_approved_items([item()], notion, make_schema(), apply=True)
    assert result[0]["status"] == "blocked"
    assert not notion.created


def test_approved_retry_uses_existing_page():
    notion = FakeNotionClient()
    proposal = item(row(numero_processo="0600249-07", origem="Brasília/DF"))
    result = queue.publish_approved_items([proposal], notion, make_schema(), apply=True)
    assert result[0]["status"] == "updated"
    assert result[0]["page_id"] == "page-123"
    assert not notion.created


def test_publication_returns_readback_for_exact_written_row(monkeypatch):
    import tse_workflow_monitor
    seen = []
    def verify(rows, results, client, schema):
        seen.append((rows[0], results[0]["page_id"]))
        return [{"status": "verified", "page_id": results[0]["page_id"]}]
    monkeypatch.setattr(tse_workflow_monitor, "verify_notion_rows", verify)
    notion = FakeNotionClient()
    proposal = item(row(numero_processo="0600249-07", origem="Brasília/DF"))
    result = queue.publish_approved_items([proposal], notion, make_schema(), apply=True)
    assert result[0]["verification"]["status"] == "verified"
    assert seen[0][0] is notion.updated[0][1]


def test_later_automatic_publication_closes_candidate_only_after_readback(tmp_path):
    path = tmp_path / "queue.jsonl"
    old = item(row(numero_processo="0600113-04.2026.6.00.0000"))
    queue.append_items([old], path)
    results = [{"status": "created", "page_id": "page"}]
    checks = [{"row_index": 0, "page_id": "page", "status": "unverified"}]
    assert queue.reconcile_published_items([row()], results, checks, video_id="video", queue_file=path) == 0
    assert queue.load_items("pending", path)
    checks[0]["status"] = "verified"
    assert queue.reconcile_published_items([row()], results, checks, video_id="video", queue_file=path) == 1
    assert not queue.load_items("pending", path)
    published = queue.load_items("published", path)[0]
    assert published["published_page_id"] == "page"
    assert published["row"]["numero_processo"] == row().numero_processo


def citation_audit(candidate, **overrides):
    return {"exclusions": [{"code": "cited_process_number", "numero_processo": candidate.numero_processo,
                            "parent_numero_processo": row().numero_processo,
                            "evidence": ["Mesmo bloco; condenação antecedente citada como fundamento."],
                            "row": candidate.model_dump(mode="json"), **overrides}]}


def test_verified_parent_resolves_old_batch_and_monitor_citations_with_audit(tmp_path):
    path = tmp_path / "queue.jsonl"
    citation = row(numero_processo="0013595-14", source_item_index=2)
    batch, monitor = item(citation), item(citation, source="monitor")
    queue.append_items([batch, monitor], path)
    results = [{"status": "created", "page_id": "parent-page"}]
    checks = [{"row_index": 0, "page_id": "parent-page", "status": "verified"}]
    reconciliation = {"phases": [citation_audit(citation), {"exclusions": []}]}
    assert queue.reconcile_published_items([row()], results, checks, video_id="video", queue_file=path,
                                           reconciliation=reconciliation) == 2
    assert not queue.load_items("pending", path)
    resolved = queue.load_items("resolved", path)
    assert len(resolved) == 2
    assert all(candidate["resolution_kind"] == "verified_citation" for candidate in resolved)
    assert all(candidate["parent_page_id"] == "parent-page" for candidate in resolved)
    assert all(candidate["automatic_exclusion"]["evidence"] for candidate in resolved)
    before = path.read_bytes()
    assert queue.reconcile_published_items([row()], results, checks, video_id="video", queue_file=path,
                                           reconciliation=reconciliation) == 0
    assert path.read_bytes() == before


@pytest.mark.parametrize("override, verification, published_rows", [
    ({"evidence": []}, "verified", True),
    ({"code": "unmatched"}, "verified", True),
    ({}, "unverified", True),
    ({}, "verified", False),
    ({"parent_numero_processo": "0600957-40.2026.6.07.0000"}, "verified", True),
])
def test_citation_resolution_needs_evidence_and_its_verified_parent(tmp_path, override, verification, published_rows):
    path = tmp_path / "queue.jsonl"
    citation = row(numero_processo="0013595-14", source_item_index=2)
    queue.append_items([item(citation)], path)
    results = [{"status": "created", "page_id": "parent-page"}]
    checks = [{"row_index": 0, "page_id": "parent-page", "status": verification}]
    assert queue.reconcile_published_items([row()] if published_rows else [], results, checks,
                                           video_id="video", queue_file=path,
                                           reconciliation=citation_audit(citation, **override)) == 0
    assert len(queue.load_items("pending", path)) == 1


@pytest.mark.parametrize("status", ["rejected", "published", "approved"])
def test_citation_resolution_preserves_explicit_decision(tmp_path, status):
    path = tmp_path / "queue.jsonl"
    citation = row(numero_processo="0013595-14", source_item_index=2)
    original = item(citation)
    queue.append_items([original], path)
    queue.update_status([original["id"]], status, queue_file=path)
    results = [{"status": "created", "page_id": "parent-page"}]
    checks = [{"row_index": 0, "page_id": "parent-page", "status": "verified"}]
    assert queue.reconcile_published_items([row()], results, checks, video_id="video", queue_file=path,
                                           reconciliation=citation_audit(citation)) == 0
    assert queue.load_items(None, path)[0]["status"] == status


def institutional_audit(candidate):
    return {"status": "complete", "session_date": candidate.data_sessao,
            "exclusions": [{"code": "institutional_act", "row": candidate.model_dump(mode="json"),
                            "reason": "Eleição interna sem processo.", "evidence": ["Varredura e detalhe originais confirmam eleição interna."]}]}


def test_institutional_exclusion_closes_only_its_pending_candidate_with_history(tmp_path):
    path = tmp_path / "queue.jsonl"
    ceremony = row(numero_processo="", classe_processo="", relator="")
    original = item(ceremony)
    queue.append_items([original, item(row(source_start_seconds=3000))], path)
    assert queue.reconcile_published_items([], [], [], video_id="video", queue_file=path,
                                           reconciliation=institutional_audit(ceremony)) == 1
    assert len(queue.load_items("pending", path)) == 1
    resolved = queue.load_items("resolved", path)[0]
    assert resolved["resolution_kind"] == "institutional_exclusion"
    assert resolved["reasons"] == original["reasons"]
    assert queue.reconcile_published_items([], [], [], video_id="video", queue_file=path,
                                           reconciliation=institutional_audit(ceremony)) == 0


@pytest.mark.parametrize("failure", ["approved", "rejected", "published", "no_evidence", "wrong_date", "unavailable", "number", "class", "no_position"])
def test_institutional_resolution_preserves_decisions_and_requires_anchored_audit(tmp_path, failure):
    path = tmp_path / "queue.jsonl"
    ceremony = row(numero_processo="", classe_processo="", relator="")
    original = item(ceremony)
    queue.append_items([original], path)
    audit = institutional_audit(ceremony)
    if failure in {"approved", "rejected", "published"}:
        queue.update_status([original["id"]], failure, queue_file=path)
    elif failure == "no_evidence":
        audit["exclusions"][0]["evidence"] = []
    elif failure == "wrong_date":
        audit["session_date"] = "2026-10-01"
    elif failure == "unavailable":
        audit["status"] = "unavailable"
    else:
        field, value = {"number": ("numero_processo", "0601130-04"), "class": ("classe_processo", "PA"), "no_position": ("source_bundle_index", 0)}[failure]
        audit["exclusions"][0]["row"][field] = value
    assert queue.reconcile_published_items([], [], [], video_id="video", queue_file=path, reconciliation=audit) == 0
    assert queue.load_items(None, path)[0]["status"] == (failure if failure in {"approved", "rejected", "published"} else "pending")
