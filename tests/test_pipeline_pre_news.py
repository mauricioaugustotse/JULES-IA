from __future__ import annotations

from argparse import Namespace
import json
import sys
from types import SimpleNamespace

import pipeline_pre_news
from pipeline_pre_news import build_stage_commands, ensure_python_module_available, resolve_playlist_url


def _args(**overrides) -> Namespace:
    base = {
        "playlist_url": "https://www.youtube.com/playlist?list=PL_TEST",
        "year": 2022,
        "limit": 0,
        "max_workers": 3,
        "initial_workers": 3,
        "auto_scale": False,
        "resume": False,
        "skip_initial_backfill": False,
        "skip_rerun": False,
        "skip_audit": False,
        "skip_repair": False,
        "skip_schema_core": False,
        "skip_identity_core": False,
        "skip_identity_replay": False,
        "skip_deterministic_core": False,
        "skip_composition_core": False,
        "skip_super_auditor": False,
        "repair_focus": "all",
        "super_focus": "quality-core",
        "super_model": "gpt-5.4-mini",
        "super_min_confidence": "medium",
        "skip_residual_focal": False,
        "residual_max_rounds": 3,
        "residual_video_timeout_seconds": 2400,
        "residual_no_progress_timeout_seconds": 900,
        "residual_retry_timeouts": 1,
    }
    base.update(overrides)
    return Namespace(**base)


def test_build_stage_commands_default_pipeline_order() -> None:
    stages = build_stage_commands(_args())
    assert [stage.name for stage in stages] == [
        "initial_backfill",
        "rerun_error_videos",
        "audit_existing_year",
        "repair_existing_year",
        "repair_schema_core",
        "repair_identity_core",
        "repair_identity_replay",
        "repair_deterministic_core",
        "repair_composition_core",
        "super_auditor",
    ]
    super_stage = stages[-1]
    assert "--focus" in super_stage.command
    assert "quality-core" in super_stage.command
    assert "gpt-5.4-mini" in super_stage.command
    assert "medium" in super_stage.command
    identity_replay_stage = next(stage for stage in stages if stage.name == "repair_identity_replay")
    assert "--playlist-url" in identity_replay_stage.command
    assert "https://www.youtube.com/playlist?list=PL_TEST" in identity_replay_stage.command


def test_build_stage_commands_can_start_after_gemini() -> None:
    stages = build_stage_commands(_args(skip_initial_backfill=True))
    assert [stage.name for stage in stages] == [
        "rerun_error_videos",
        "audit_existing_year",
        "repair_existing_year",
        "repair_schema_core",
        "repair_identity_core",
        "repair_identity_replay",
        "repair_deterministic_core",
        "repair_composition_core",
        "super_auditor",
    ]


def test_build_stage_commands_can_skip_composition_core() -> None:
    stages = build_stage_commands(_args(skip_composition_core=True))
    assert "repair_composition_core" not in [stage.name for stage in stages]


def test_build_stage_commands_can_skip_identity_replay() -> None:
    stages = build_stage_commands(_args(skip_identity_replay=True))
    assert "repair_identity_replay" not in [stage.name for stage in stages]


def test_resolve_playlist_url_discovers_from_manifest(tmp_path, monkeypatch) -> None:
    backfill_root = tmp_path / "artifacts"
    manifest_dir = backfill_root / "2022_PL_TEST"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "playlist_url": "https://www.youtube.com/playlist?list=PL_DISCOVERED",
                "updated_at": "2026-03-30T10:00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_pre_news, "BACKFILL_ROOT", backfill_root)
    assert (
        resolve_playlist_url(pipeline_pre_news.PLAYLIST_PLACEHOLDER, 2022)
        == "https://www.youtube.com/playlist?list=PL_DISCOVERED"
    )


def test_ensure_python_module_available_raises_clear_error(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stderr="ModuleNotFoundError: No module named 'openai'", stdout="")

    monkeypatch.setattr(pipeline_pre_news.subprocess, "run", fake_run)
    try:
        ensure_python_module_available("python", "openai")
    except RuntimeError as exc:
        assert "openai" in str(exc)
        assert "Módulo obrigatório ausente" in str(exc)
    else:
        raise AssertionError("Era esperado RuntimeError quando o módulo está ausente.")


def test_residual_focal_closure_uses_manifest_after_rerun(tmp_path, monkeypatch) -> None:
    playlist_url = "https://www.youtube.com/playlist?list=PL_TEST"
    backfill_root = tmp_path / "backfill"
    manifest_dir = backfill_root / "2022_PL_TEST"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "playlist_url": playlist_url,
                "videos": {
                    "vid_error": {"position": 1, "status": "error"},
                    "vid_done": {"position": 2, "status": "done"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_pre_news, "BACKFILL_ROOT", backfill_root)

    calls = []

    def fake_run_stage(stage, run_root):
        calls.append(stage)
        if stage.name == "residual_rerun_round_1":
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["videos"]["vid_error"]["status"] = "done"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "name": stage.name,
            "status": "done",
            "returncode": 0,
            "log_path": str(run_root / f"{stage.name}.log"),
        }

    monkeypatch.setattr(pipeline_pre_news, "run_stage", fake_run_stage)

    run_root = tmp_path / "run"
    run_root.mkdir()
    summary = {"stages": []}
    pipeline_pre_news._run_residual_focal_closure(
        args=_args(residual_max_rounds=1),
        run_root=run_root,
        summary=summary,
    )

    assert [stage.name for stage in calls] == [
        "residual_rerun_round_1",
        "residual_repair_all_round_1",
        "residual_schema_core_round_1",
        "residual_identity_core_round_1",
        "residual_identity_replay_round_1",
        "residual_deterministic_core_round_1",
        "residual_composition_core_round_1",
        "residual_super_auditor_round_1",
    ]
    assert summary["stages"][0]["resolved_video_ids"] == ["vid_error"]
    assert summary["stages"][0]["remaining_error_video_ids"] == []
    for stage in calls[1:]:
        video_args = [
            stage.command[index + 1]
            for index, value in enumerate(stage.command[:-1])
            if value == "--video-id"
        ]
        assert video_args == ["vid_error"]


def test_run_stage_includes_log_tail_on_failure(tmp_path) -> None:
    stage = pipeline_pre_news.PipelineStage(
        "failing_stage",
        [sys.executable, "-c", "print('linha de diagnóstico'); raise SystemExit(2)"],
    )

    result = pipeline_pre_news.run_stage(stage, tmp_path)

    assert result["status"] == "error"
    assert result["returncode"] == 2
    assert "linha de diagnóstico" in result["log_tail"]
