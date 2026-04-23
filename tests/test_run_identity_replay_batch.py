from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import run_identity_replay_batch as replay


def test_completed_replay_ids_accepts_legacy_done_entries() -> None:
    year_entry = {
        "replayed_videos": [
            {"video_id": "abc123", "summary": {"created": 1}},
            {"video_id": "def456", "status": "done"},
            {"video_id": "ghi789", "status": "error"},
        ]
    }

    assert replay._completed_replay_ids(year_entry) == {"abc123", "def456"}


def test_is_timeout_like_error_matches_worker_stall_messages() -> None:
    assert replay._is_timeout_like_error(RuntimeError("Timeout de 2400s ao processar o vídeo X."))
    assert replay._is_timeout_like_error(RuntimeError("Sem progresso real de artefatos por 300s no vídeo X."))
    assert not replay._is_timeout_like_error(RuntimeError("Falha de autenticação no Notion"))


def test_run_replay_queue_starts_up_to_max_workers(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary = {"years": []}
    year_entry = {"replayed_videos": [], "failed_videos": [], "active_videos": []}
    summary["years"].append(year_entry)
    worker_args = SimpleNamespace()
    started: list[str] = []

    tasks = [
        replay.ReplayTask(video=SimpleNamespace(video_id="a"), video_id="a", reasons=[]),
        replay.ReplayTask(video=SimpleNamespace(video_id="b"), video_id="b", reasons=[]),
        replay.ReplayTask(video=SimpleNamespace(video_id="c"), video_id="c", reasons=[]),
    ]

    def fake_start_worker(video, _args, root_dir):
        started.append(video.video_id)
        return SimpleNamespace(
            video=video,
            artifact_dir=root_dir / video.video_id,
            process=SimpleNamespace(pid=100 + len(started)),
            started_at="2026-04-01T00:00:00",
            last_seen_artifact_name="",
        )

    def fake_poll_worker(handle):
        return {
            "status": "done",
            "summary": {"video_id": handle.video.video_id},
            "progress_changed": False,
        }

    year_had_errors = replay._run_replay_queue(
        summary=summary,
        summary_path=summary_path,
        year=2025,
        year_entry=year_entry,
        tasks=tasks,
        worker_args=worker_args,
        year_root=tmp_path,
        retry_timeouts=1,
        max_workers=2,
        base_video_timeout_seconds=10,
        base_no_progress_timeout_seconds=5,
        start_worker_fn=fake_start_worker,
        poll_worker_fn=fake_poll_worker,
        sleep_fn=lambda *_args, **_kwargs: None,
    )

    assert not year_had_errors
    assert started[:2] == ["a", "b"]
    assert len(year_entry["replayed_videos"]) == 3
    assert year_entry["failed_videos"] == []
    assert year_entry["active_videos"] == []


def test_run_replay_queue_retries_timeout_then_succeeds(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary = {"years": []}
    year_entry = {"replayed_videos": [], "failed_videos": [], "active_videos": []}
    summary["years"].append(year_entry)
    worker_args = SimpleNamespace()
    start_attempts: list[str] = []
    poll_attempts = {"x": 0}

    task = replay.ReplayTask(video=SimpleNamespace(video_id="x"), video_id="x", reasons=[])

    def fake_start_worker(video, _args, root_dir):
        start_attempts.append(video.video_id)
        return SimpleNamespace(
            video=video,
            artifact_dir=root_dir / f"{video.video_id}_{len(start_attempts)}",
            process=SimpleNamespace(pid=200 + len(start_attempts)),
            started_at="2026-04-01T00:00:00",
            last_seen_artifact_name="",
        )

    def fake_poll_worker(handle):
        poll_attempts["x"] += 1
        if poll_attempts["x"] == 1:
            return {
                "status": "error",
                "error": "Timeout de 2400s ao processar o vídeo x.",
                "progress_changed": False,
            }
        return {
            "status": "done",
            "summary": {"video_id": handle.video.video_id},
            "progress_changed": False,
        }

    year_had_errors = replay._run_replay_queue(
        summary=summary,
        summary_path=summary_path,
        year=2025,
        year_entry=year_entry,
        tasks=[task],
        worker_args=worker_args,
        year_root=tmp_path,
        retry_timeouts=1,
        max_workers=2,
        base_video_timeout_seconds=10,
        base_no_progress_timeout_seconds=5,
        start_worker_fn=fake_start_worker,
        poll_worker_fn=fake_poll_worker,
        sleep_fn=lambda *_args, **_kwargs: None,
    )

    assert not year_had_errors
    assert start_attempts == ["x", "x"]
    assert year_entry["failed_videos"] == []
    assert len(year_entry["replayed_videos"]) == 1
    assert year_entry["replayed_videos"][0]["attempts"] == 2


def test_persisted_candidate_map_keeps_failed_video_reasons_for_resume() -> None:
    year_entry = {
        "candidate_videos": [
            {"video_id": "ok123", "reasons": [{"source": "publish-results", "numero_processo": "1", "reason": "x"}]},
        ],
        "failed_videos": [
            {"video_id": "fail456", "reasons": [{"source": "publish-results", "numero_processo": "2", "reason": "y"}]},
        ],
    }

    assert replay._persisted_candidate_map(year_entry) == {
        "ok123": [{"source": "publish-results", "numero_processo": "1", "reason": "x"}],
        "fail456": [{"source": "publish-results", "numero_processo": "2", "reason": "y"}],
    }


def test_filter_candidate_map_restricts_to_requested_video_ids() -> None:
    candidate_map = {
        "ok123": [{"source": "publish-results", "numero_processo": "1", "reason": "x"}],
        "skip456": [{"source": "publish-results", "numero_processo": "2", "reason": "y"}],
    }

    assert replay._filter_candidate_map(candidate_map, {"ok123"}) == {
        "ok123": [{"source": "publish-results", "numero_processo": "1", "reason": "x"}],
    }
    assert replay._filter_candidate_map(candidate_map, set()) == candidate_map


def test_load_playlist_url_prefers_explicit_override() -> None:
    assert replay._load_playlist_url(2020, "https://www.youtube.com/playlist?list=PL_EXPLICIT") == (
        "https://www.youtube.com/playlist?list=PL_EXPLICIT"
    )
