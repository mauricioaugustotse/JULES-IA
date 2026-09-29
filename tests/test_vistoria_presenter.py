import json

from vistoria_presenter import (detail_text, group_review_items, is_notice,
                               process_number, timestamp_seconds, video_url)


def item(item_id, number="", *, issue=None, row=None, status="pending"):
    return {"id": item_id, "video_id": "session-video", "status": status,
            "row": row, "numero_hint": number, "reasons": [f"Motivo {item_id}"],
            "extra": {"coverage_issue": issue} if issue else {}}


def test_one_case_keeps_all_alerts_and_the_publishable_proposal():
    full = "0600014-52.2025.6.00.0000"
    proposal = item("proposal", row={"numero_processo": "0600014-52", "tema": "Lista tríplice"})
    alert = item("alert", full, issue={"code": "official_incomplete_process_number", "field": "numero_processo", "expected": full, "actual": "0600014-52"})
    groups = group_review_items([alert, proposal])
    assert len(groups) == 1
    assert groups[0]["id"] == "proposal"
    assert set(groups[0]["member_ids"]) == {"proposal", "alert"}
    assert len(groups[0]["reasons"]) == 2
    assert "Registro oficial: " + full in detail_text(groups[0])


def test_ambiguous_short_number_does_not_merge_different_full_processes():
    groups = group_review_items([
        item("a", "0600014-52.2025.6.00.0000"),
        item("b", "0600014-52.2026.6.20.0000"),
        item("short", "0600014-52"),
    ])
    assert len(groups) == 3


def test_different_videos_and_closed_statuses_are_kept_separate():
    first = item("first", "0600014-52")
    other_video = {**item("other", "0600014-52"), "video_id": "another-session"}
    closed = item("closed", "0600014-52", status="published")
    assert len(group_review_items([first, other_video, closed])) == 3


def test_local_evidence_is_visible_without_promoting_alert_to_publishable(tmp_path):
    number = "0601130-04.2026.6.20.0000"
    (tmp_path / "00_official_session_inventory.json").write_text(json.dumps({
        "session_date": "2026-09-24", "source_url": "https://example.org/session",
        "processes": [{"numeroProcesso": number, "situacaoProcesso": "Julgado", "segredoJustica": True}],
    }), encoding="utf-8")
    (tmp_path / "04h_publish_preview_rows.json").write_text(json.dumps([
        {"numero_processo": number, "origem": "Doutor Severiano/RN", "classe_processo": "PA",
         "punchline": "Requisição de forças federais deferida.", "source_start_seconds": 2752},
    ]), encoding="utf-8")
    source = {**item("missing", number, issue={"code": "official_missing_judgment", "row_index": 0}),
              "artifact_dir": str(tmp_path)}
    case = group_review_items([source])[0]
    assert not case["row"]
    assert case["display_row"]["origem"] == "Doutor Severiano/RN"
    text = detail_text(case)
    assert "Requisição de forças federais deferida." in text
    assert "Situação: Julgado" in text
    assert "Campos vazios não provam ausência de julgamento" in text
    assert "00:45:52" in text
    assert "t=2752" in video_url(case)


def test_warning_is_notice_only_when_it_has_no_proposal():
    warning = item("notice", issue={"severity": "warning", "code": "official_unknown_status"})
    assert is_notice(warning)
    warning["row"] = {"numero_processo": "0600162-45", "resultado": "Suspenso por vista"}
    assert not is_notice(warning)


def test_process_number_and_timestamp_from_structured_evidence():
    source = item("missing", issue={"numero_processo": "06011300420266200000"})
    assert process_number(source) == "0601130-04.2026.6.20.0000"
    source["row"] = {"source_start_seconds": 0, "youtube_link": "https://www.youtube.com/watch?v=abc"}
    assert timestamp_seconds(source) == 0
    assert video_url(source).endswith("&t=0")
