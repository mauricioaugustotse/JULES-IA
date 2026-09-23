import tse_chapter_coverage as coverage
from tse_youtube_notion_core import RunArtifacts


DESCRIPTION = """00:00:00 Início da transmissão
00:20:28 LT 060017516
00:23:15 LT 060104136
00:25:48 Julgamento em lista
00:27:00 Encerramento
"""


def test_inventory_caches_source_and_bounds_cases_before_collective_list(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(coverage, "fetch_youtube_description", lambda vid: calls.append(vid) or DESCRIPTION)
    store = RunArtifacts(tmp_path)
    first = coverage.ensure_chapter_inventory(store, "https://www.youtube.com/watch?v=gY6-5zPrFlo&t=80")
    second = coverage.ensure_chapter_inventory(store, "https://www.youtube.com/watch?v=gY6-5zPrFlo")
    assert calls == ["gY6-5zPrFlo"]
    assert first == second == store.read_json(coverage.INVENTORY_FILENAME)
    assert first["description"] == DESCRIPTION
    assert [(row["numero_processo"], row["start_seconds"], row["end_seconds"]) for row in first["chapters"]] == [
        ("0600175-16", 1228, 1395), ("0601041-36", 1395, 1548),
    ]


def test_unavailable_inventory_is_retried_and_not_treated_as_complete(tmp_path, monkeypatch):
    replies = iter(["", DESCRIPTION])
    monkeypatch.setattr(coverage, "fetch_youtube_description", lambda vid: next(replies))
    store = RunArtifacts(tmp_path)
    missing = coverage.ensure_chapter_inventory(store, "https://www.youtube.com/watch?v=gY6-5zPrFlo")
    assert missing["status"] == "unavailable"
    assert coverage.compare_chapters(missing, [])[0]["kind"] == "chapter_inventory_unavailable"
    assert coverage.ensure_chapter_inventory(store, "https://www.youtube.com/watch?v=gY6-5zPrFlo")["status"] == "available"


def test_same_count_wrong_case_still_reports_actual_missing_case(tmp_path, monkeypatch):
    # Regressão do lote 17/09: 0600175-16 foi substituído pelo caso seguinte,
    # 0601041-36. Duas linhas do segundo caso não cobrem o primeiro.
    monkeypatch.setattr(coverage, "fetch_youtube_description", lambda vid: DESCRIPTION)
    inventory = coverage.ensure_chapter_inventory(RunArtifacts(tmp_path), "https://www.youtube.com/watch?v=gY6-5zPrFlo")
    issues = coverage.compare_chapters(inventory, [
        {"numero_processo": "0601041-36"}, {"numero_processo": "0601041-36.2026.6.00.0000"},
    ])
    assert len(issues) == 1
    assert issues[0]["numero_processo"] == "0600175-16"
    assert issues[0]["start_seconds"] == 1228


def test_completed_numbers_and_omitted_leading_zero_match_without_extra_row_rejection():
    inventory = {"status": "available", "chapters": [
        {"numero_processo": "0600175-16", "start_seconds": 1228},
        {"numero_processo": "0601041-36", "start_seconds": 1395},
    ]}
    rows = [{"numero_processo": "0600175-16.2026.6.00.0000"},
            {"numero_processo": "601041-36"}, {"numero_processo": "0600559-88"}]
    assert coverage.compare_chapters(inventory, rows) == []


def test_corrupt_or_different_video_cache_does_not_suppress_capture(tmp_path, monkeypatch):
    store = RunArtifacts(tmp_path)
    (tmp_path / coverage.INVENTORY_FILENAME).write_text("{", encoding="utf-8")
    calls = []
    monkeypatch.setattr(coverage, "fetch_youtube_description", lambda vid: calls.append(vid) or DESCRIPTION)
    coverage.ensure_chapter_inventory(store, "https://www.youtube.com/watch?v=gY6-5zPrFlo")
    coverage.ensure_chapter_inventory(store, "https://www.youtube.com/watch?v=pNK-L5X6fXw")
    assert calls == ["gY6-5zPrFlo", "pNK-L5X6fXw"]


def test_fetch_exception_preserves_unknown_coverage_without_leaking_error_details(tmp_path, monkeypatch):
    def fail(vid):
        raise RuntimeError("sensitive configuration")
    monkeypatch.setattr(coverage, "fetch_youtube_description", fail)
    result = coverage.ensure_chapter_inventory(RunArtifacts(tmp_path), "https://www.youtube.com/watch?v=gY6-5zPrFlo")
    assert result["status"] == "unavailable"
    assert "RuntimeError" in result["error"]
    assert "sensitive" not in result["error"]
