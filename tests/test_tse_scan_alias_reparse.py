"""Regression: English scan aliases must not become zero-second anonymous cases."""
import json
import logging

import pytest

import tse_youtube_notion_core as core
from reparse_tse_scan_cache import reparse_scan_cache


@pytest.mark.parametrize("start_key,end_key", [
    ("start_timestamp", "end_timestamp"),
    ("timestamp_start", "timestamp_end"),
])
@pytest.mark.parametrize("as_list", [False, True])
def test_english_scan_aliases_retain_absolute_times_and_short_process_numbers(start_key, end_key, as_list):
    payload = {
        "session_date": "22 de setembro de 2026",
        "ministers_present": ["Min. Floriano de Azevedo Marques"],
        "judgments": [{start_key: 2970, end_key: 3270,
                       "process_numbers": ["060154852", "060162124"], "should_ignore": False}],
    }
    result = core._coerce_gemini_response_model(core.SessionExtraction, json.dumps([payload] if as_list else payload))
    assert result.data_sessao == "22 de setembro de 2026"
    assert result.composicao == ["Min. Floriano de Azevedo Marques"]
    assert result.judgments[0].start_seconds == 2970
    assert result.judgments[0].end_seconds == 3270
    assert result.judgments[0].mentioned_process_numbers == ["060154852", "060162124"]
    obj = core.GeminiSessionExtractor.__new__(core.GeminiSessionExtractor)
    obj.logger = logging.getLogger(__name__)
    merged = obj._merge_session_chunks([result])
    assert merged.judgments[0].mentioned_process_numbers == ["0601548-52", "0601621-24"]
    assert all(len(''.join(ch for ch in number if ch.isdigit())) == 9
               for number in merged.judgments[0].mentioned_process_numbers)


def create_source(tmp_path, raw_start=300):
    source = tmp_path / "run"
    store = core.RunArtifacts(source)
    store.write_json("00_scan_coverage.json", core._scan_coverage_payload(600, [{
        "label": "primary", "artifact_prefix": "raw_global_response", "source": "video", "windows": [
            {"start_seconds": 0, "end_seconds": 300, "status": "complete"},
            {"start_seconds": 300, "end_seconds": 600, "status": "rejected"},
        ],
    }]))
    store.write_text("raw_global_response_chunk_01.txt", '{"judgments": []}')
    store.write_json("raw_global_response_chunk_02.json", {"judgments": []})
    store.write_text("raw_global_response_chunk_02.txt", json.dumps({
        "session_date": None, "ministers_present": [], "judgments": [{
            "start_timestamp": raw_start, "end_timestamp": max(600, raw_start + 1),
            "process_numbers": ["060154852"],
        }],
    }))
    return source


def test_offline_reparse_recovers_coverage_without_calls_writes_or_content_verification(tmp_path, monkeypatch):
    source = create_source(tmp_path)
    originals = {file.name: file.read_bytes() for file in source.iterdir()}
    monkeypatch.setattr(core.requests, "post", lambda *a, **kw: pytest.fail("No online requests"))
    monkeypatch.setattr(core.requests, "get", lambda *a, **kw: pytest.fail("No online requests"))
    target = tmp_path / "repaired"
    report = reparse_scan_cache(source, target)
    assert report["temporal_coverage_status"] == "complete"
    assert report["uncovered_intervals"] == []
    assert report["content_verified"] is False
    assert report["publication_ready"] is False
    assert report["api_calls"] == report["notion_writes"] == 0
    assert report["changes"][0]["new_status"] == "complete"
    assert {file.name: file.read_bytes() for file in source.iterdir()} == originals
    assert {file.name: file.read_bytes() for file in (target / "original_snapshot").iterdir()} == originals
    assert not (target / "01_session_windows.json").exists()
    assert not (target / "03_analysis.json").exists()
    repaired = json.loads((target / "raw_global_response_chunk_02.json").read_text())
    assert repaired["judgments"][0]["start_seconds"] == 300
    assert repaired["judgments"][0]["mentioned_process_numbers"] == ["060154852"]


def test_offline_reparse_keeps_truly_out_of_window_response_rejected(tmp_path):
    source = create_source(tmp_path, raw_start=9000)
    report = reparse_scan_cache(source, tmp_path / "repaired")
    assert report["temporal_coverage_status"] == "incomplete"
    assert report["uncovered_intervals"] == [[300, 600]]


def test_offline_reparse_refuses_overwriting_or_nesting_in_original_run(tmp_path):
    source = create_source(tmp_path)
    with pytest.raises(ValueError):
        reparse_scan_cache(source, source)
    with pytest.raises(ValueError):
        reparse_scan_cache(source, source / "repaired")
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        reparse_scan_cache(source, target)
