"""Regression checks for silent partial scans and judgments lost between stages."""
import logging

import pytest

import tse_youtube_notion_core as core


URL = "https://www.youtube.com/watch?v=abc123"


def extractor(tmp_path):
    obj = core.GeminiSessionExtractor.__new__(core.GeminiSessionExtractor)
    obj.logger = logging.getLogger(__name__)
    obj.artifact_store = core.RunArtifacts(tmp_path)
    obj.allow_transcript_fallback = False
    return obj


def scan_result(start=0, number="0600175-16"):
    return core.SessionExtraction(judgments=[core.SessionWindow(
        title_hint=number, start_seconds=start, end_seconds=start + 60,
        mentioned_process_numbers=[number],
    )])


def test_partial_scan_recovers_only_unread_intervals(tmp_path, monkeypatch):
    obj = extractor(tmp_path)
    monkeypatch.setattr(core, "fetch_youtube_duration_seconds", lambda _: 600)
    monkeypatch.setattr(core, "chunk_video_windows", lambda _, window_seconds=None, **kw:
                        [(0, 300), (270, 600)] if window_seconds is None else
                        [(0, 150), (120, 300), (270, 450), (420, 600)])
    calls = []

    def call(**kwargs):
        calls.append((kwargs["start_seconds"], kwargs["end_seconds"]))
        if len(calls) == 1:
            raise RuntimeError("timeout")
        return scan_result(kwargs["start_seconds"])

    obj._call_gemini = call
    result = obj._extract_session_windows(URL)
    assert result.judgments
    assert calls == [(0, 300), (270, 600), (0, 150), (120, 300)]
    report = core.load_scan_coverage_report(obj.artifact_store)
    assert report["status"] == "complete"
    assert report["uncovered_intervals"] == []
    assert report["attempts"][0]["windows"][0]["status"] == "failed"


def test_partial_scan_never_returns_success_when_recovery_fails(tmp_path, monkeypatch):
    obj = extractor(tmp_path)
    monkeypatch.setattr(core, "fetch_youtube_duration_seconds", lambda _: 600)
    monkeypatch.setattr(core, "chunk_video_windows", lambda *args, **kw: [(0, 300), (270, 600)])

    def call(**kwargs):
        if kwargs["start_seconds"] == 0:
            raise RuntimeError("timeout")
        return scan_result(300)

    obj._call_gemini = call
    with pytest.raises(RuntimeError, match="falhou"):
        obj._extract_session_windows(URL)
    report = core.load_scan_coverage_report(obj.artifact_store)
    assert report["status"] == "incomplete"
    assert report["uncovered_intervals"] == [[0, 270]]


def test_legacy_cache_with_failed_chunk_is_not_trusted(tmp_path):
    obj = extractor(tmp_path)
    obj.artifact_store.write_json("00_scan_windows.json", {
        "duration_seconds": 600, "plans": [{"label": "primary", "windows": [[0, 300], [270, 600]]}],
    })
    obj.artifact_store.write_json("raw_global_response_chunk_02.json", scan_result(300).model_dump())
    obj.artifact_store.write_json("01_session_windows.json", scan_result(300).model_dump())
    with pytest.raises(RuntimeError, match="Cache.*cobertura incompleta"):
        obj.analyze_session(URL)
    assert core.load_scan_coverage_report(obj.artifact_store)["uncovered_intervals"] == [[0, 270]]


def test_successful_empty_scan_window_counts_as_analyzed(tmp_path, monkeypatch):
    obj = extractor(tmp_path)
    monkeypatch.setattr(core, "fetch_youtube_duration_seconds", lambda _: 600)
    obj._call_gemini = lambda **kw: core.SessionExtraction()
    assert obj._extract_session_windows(URL).judgments == []
    assert core.load_scan_coverage_report(obj.artifact_store)["status"] == "complete"


def test_empty_bundle_gets_one_recovery_and_pending_placeholder(tmp_path):
    obj = extractor(tmp_path)
    obj.artifact_store.write_json("01_session_windows.json", scan_result().model_dump())
    calls = []

    def call(*args, **kw):
        calls.append(kw)
        return core.JudgmentBundleExtraction(title_hint="0600175-16")

    obj._extract_judgment_bundle = call
    analysis = obj.analyze_session(URL)
    assert len(calls) == 2
    assert "coverage_retry_context" in calls[1]
    assert analysis.bundles[0].items[0].numero_processo == "0600175-16"
    report = obj.artifact_store.read_json("02_detail_coverage.json")
    assert report["status"] == "incomplete"
    assert report["blocks"][0]["issues"] == ["empty_bundle"]
    # Restarting the same run does not loop forever or repeat paid recovery.
    obj.analyze_session(URL)
    assert len(calls) == 2
    assert obj.artifact_store.read_json("02_detail_coverage.json")["status"] == "incomplete"


def test_scan_identity_mismatch_is_recovered_before_deduplication(tmp_path):
    obj = extractor(tmp_path)
    obj.artifact_store.write_json("01_session_windows.json", scan_result(1228).model_dump())
    calls = []

    def call(*args, **kw):
        calls.append(kw)
        number = "0600175-16" if kw.get("coverage_retry_context") else "0601041-36"
        return core.JudgmentBundleExtraction(
            title_hint="0600175-16", start_seconds=1228,
            items=[core.JudgmentItemExtraction(numero_processo=number)],
        )

    obj._extract_judgment_bundle = call
    result = obj.analyze_session(URL)
    assert len(calls) == 2
    assert result.bundles[0].items[0].numero_processo == "0600175-16"
    assert obj.artifact_store.read_json("02_detail_coverage.json")["status"] == "complete"


def test_explicit_legitimate_exclusion_is_preserved(tmp_path):
    obj = extractor(tmp_path)
    obj.artifact_store.write_json("01_session_windows.json", scan_result().model_dump())
    calls = []

    def call(*args, **kw):
        calls.append(kw)
        return core.JudgmentBundleExtraction(should_ignore=True, ignore_reason="Julgamento em lista.")

    obj._extract_judgment_bundle = call
    assert obj.analyze_session(URL).bundles[0].should_ignore
    assert len(calls) == 1
    assert obj.artifact_store.read_json("02_detail_coverage.json")["blocks"][0]["status"] == "excluded"


def test_chapter_inventory_recovers_missing_case_with_collective_list_boundary(tmp_path):
    obj = extractor(tmp_path)
    obj.artifact_store.write_json("00_chapter_inventory.json", {
        "status": "available", "chapters": [{
            "numero_processo": "0600175-16", "start_seconds": 1228,
            "end_seconds": 1404, "classe": "AREspe",
        }],
    })
    obj.artifact_store.write_json("00_scan_windows.json", {"duration_seconds": 3000})
    result = obj._add_missing_chapter_windows(core.SessionExtraction())
    assert len(result.judgments) == 1
    assert result.judgments[0].end_seconds == 1404
    assert result.judgments[0].mentioned_process_numbers == ["0600175-16"]


def test_chapter_recovery_does_not_reuse_shifted_bundle_cache(tmp_path):
    obj = extractor(tmp_path)
    obj.artifact_store.write_json("01_session_windows.json", scan_result(500, "0601041-36").model_dump())
    obj.artifact_store.write_json("02_judgment_01.json", core.JudgmentBundleExtraction(
        title_hint="0601041-36", start_seconds=500,
        items=[core.JudgmentItemExtraction(numero_processo="0601041-36")],
    ).model_dump())
    obj.artifact_store.write_json("00_chapter_inventory.json", {
        "chapters": [{"numero_processo": "0600175-16", "start_seconds": 100, "end_seconds": 400}],
    })
    seen = []

    def call(url, session, window, index, **kw):
        seen.append(window.mentioned_process_numbers)
        return core.JudgmentBundleExtraction(
            title_hint=window.title_hint, start_seconds=window.start_seconds,
            items=[core.JudgmentItemExtraction(numero_processo=window.mentioned_process_numbers[0])],
        )

    obj._extract_judgment_bundle = call
    result = obj.analyze_session(URL)
    assert [bundle.items[0].numero_processo for bundle in result.bundles] == ["0600175-16", "0601041-36"]
    assert seen[0] == ["0600175-16"]


def test_chapter_end_limits_detail_before_collective_list(tmp_path):
    obj = extractor(tmp_path)
    obj.artifact_store.write_json("00_chapter_inventory.json", {
        "chapters": [{"numero_processo": "0600175-16", "start_seconds": 100, "end_seconds": 400}],
    })
    session = obj._add_missing_chapter_windows(core.SessionExtraction())
    obj._refine_bundle_start_seconds = lambda **kw: 100
    captured = []

    def call(**kw):
        captured.append(kw)
        return core.JudgmentBundleExtraction()

    obj._call_gemini = call
    obj._extract_judgment_bundle(URL, session, session.judgments[0], 1)
    assert captured[0]["end_seconds"] == 400


def test_touching_next_window_limits_detail_padding():
    first = core.SessionWindow(start_seconds=100, end_seconds=400)
    second = core.SessionWindow(start_seconds=400, end_seconds=600)
    session = core.SessionExtraction(judgments=[first, second])
    assert core.GeminiSessionExtractor._detail_end_seconds(session, first) == 400


def test_unresolved_joint_process_stays_pending_without_fabrication(tmp_path):
    obj = extractor(tmp_path)
    session = scan_result()
    session.judgments[0].mentioned_process_numbers.append("0601041-36")
    obj.artifact_store.write_json("01_session_windows.json", session.model_dump())
    obj._extract_judgment_bundle = lambda *a, **kw: core.JudgmentBundleExtraction(
        title_hint="0600175-16", items=[core.JudgmentItemExtraction(numero_processo="0600175-16")],
    )
    analysis = obj.analyze_session(URL)
    assert [item.numero_processo for item in analysis.bundles[0].items] == ["0600175-16"]
    report = obj.artifact_store.read_json("02_detail_coverage.json")
    assert report["status"] == "incomplete"
    assert report["blocks"][0]["issues"] == ["missing_scan_process"]


def test_partial_transcript_scan_does_not_return_success(tmp_path, monkeypatch):
    obj = extractor(tmp_path)
    obj._get_transcript_snippets = lambda _: [core.TranscriptSnippet("texto", 0, 600)]
    monkeypatch.setattr(core, "build_transcript_chunks", lambda _: [
        core.TranscriptChunk(0, 300, "primeiro", 1),
        core.TranscriptChunk(270, 600, "segundo", 1),
    ])
    calls = []

    def call(**kw):
        calls.append(kw)
        if len(calls) == 2:
            raise RuntimeError("timeout")
        return scan_result()

    obj._call_gemini_text = call
    with pytest.raises(RuntimeError, match="Cobertura incompleta da transcrição"):
        obj._extract_session_windows_from_transcript(URL)
    assert core.load_scan_coverage_report(obj.artifact_store)["status"] == "incomplete"


def test_scan_rejects_case_returned_from_unrelated_official_chapter():
    # Actual 10 September failure: a case at 1578s was hallucinated at 378s.
    result = core.scan_chunk_chapter_conflicts(
        scan_result(378, "REspe 060117962").judgments,
        chapters=[{"numero_processo": "0601179-62", "start_seconds": 1578, "end_seconds": 1678}],
        window_start_seconds=315, window_end_seconds=435,
    )
    assert len(result) == 1
    assert result[0]["chapter_intervals"] == [[1578, 1678]]


def test_chapter_check_preserves_long_case_repeated_chapters_and_unknown_cases():
    chapters = [
        {"numero_processo": "0600224-87", "start_seconds": 2051, "end_seconds": 5287},
        {"numero_processo": "0600883-78", "start_seconds": 1190, "end_seconds": 1505},
        {"numero_processo": "0600883-78", "start_seconds": 4000, "end_seconds": 4300},
    ]
    windows = [
        core.SessionWindow(start_seconds=2051, end_seconds=5287, mentioned_process_numbers=["REspe 060022487"]),
        core.SessionWindow(start_seconds=4000, end_seconds=4300, mentioned_process_numbers=["600883-78"]),
        core.SessionWindow(start_seconds=4050, mentioned_process_numbers=["0600999-99"]),
    ]
    assert core.scan_chunk_chapter_conflicts(
        windows, chapters=chapters, window_start_seconds=4050, window_end_seconds=4350,
    ) == []


def test_conflicting_chapter_time_does_not_mark_scan_complete(tmp_path):
    obj = extractor(tmp_path)
    obj.artifact_store.write_json("00_chapter_inventory.json", {
        "status": "available", "chapters": [
            {"numero_processo": "0601179-62", "start_seconds": 1578, "end_seconds": 1678},
        ],
    })
    obj._call_gemini = lambda **kw: scan_result(378, "0601179-62")
    obj._scan_duration_seconds = 1800
    with pytest.raises(RuntimeError, match="Nenhum chunk"):
        obj._extract_session_windows_for_plan(
            youtube_url=URL, windows=[(315, 435)], artifact_prefix="probe",
            plan_label="probe", duration_seconds=1800,
        )
    report = core.load_scan_coverage_report(obj.artifact_store)
    assert report["attempts"][0]["windows"][0]["status"] == "rejected"
    assert obj.artifact_store.exists("probe_chunk_01.chapter_conflicts.json")


@pytest.mark.parametrize("requested,expected", [
    ("gemini-3.1-pro-preview", "gemini-3.1-pro-preview"),
    ("models/gemini-2.5-pro", "gemini-2.5-pro"),
    ("", core.DEFAULT_GEMINI_MODEL),
    ("   ", core.DEFAULT_GEMINI_MODEL),
])
def test_explicit_recovery_model_is_respected(requested, expected):
    assert core.resolve_gemini_model(None, requested) == expected
    assert core.build_gemini_model_candidates(None, requested) == [expected]


def test_legacy_cache_rebuild_preserves_chapter_conflict_rejection(tmp_path):
    store = core.RunArtifacts(tmp_path)
    store.write_json("00_scan_windows.json", {
        "duration_seconds": 600,
        "plans": [{"label": "primary", "windows": [[0, 600]]}],
    })
    store.write_json("raw_global_response_chunk_01.json", scan_result().model_dump())
    store.write_json("raw_global_response_chunk_01.chapter_conflicts.json", {
        "conflicts": [{"numero_processo": "0600175-16", "chapter_intervals": [[1500, 1800]]}],
    })
    report = core.load_scan_coverage_report(store)
    assert report["status"] == "incomplete"
    assert report["attempts"][0]["windows"][0]["status"] == "rejected"
