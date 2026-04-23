from __future__ import annotations

from argparse import Namespace
import json
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
