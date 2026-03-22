import json

import tse_backfill_scale_watch as watch


def test_process_snapshot_emits_each_threshold_once(tmp_path):
    log_path = tmp_path / "scale_watch.log"
    state = {"seen_thresholds": [], "last_target": None, "events": []}
    manifest = {
        "updated_at": "2026-03-21T14:30:00",
        "eta_at": "2026-03-21T15:00:00",
        "current_target_workers": 8,
        "videos": {
            "a": {"status": "done"},
            "b": {"status": "running"},
            "c": {"status": "pending"},
        },
    }

    emitted = watch.process_snapshot(
        manifest=manifest,
        thresholds=[6, 8, 10],
        state=state,
        log_path=log_path,
    )

    assert len(emitted) == 2
    assert "threshold=6" in emitted[0]
    assert "threshold=8" in emitted[1]
    assert state["seen_thresholds"] == [6, 8]
    log_lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(log_lines) == 2

    emitted_again = watch.process_snapshot(
        manifest=manifest,
        thresholds=[6, 8, 10],
        state=state,
        log_path=log_path,
    )
    assert emitted_again == []


def test_parse_thresholds_dedupes_and_sorts():
    assert watch.parse_thresholds("10,6,8,8") == [6, 8, 10]
