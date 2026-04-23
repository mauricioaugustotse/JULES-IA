from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


LOGGER = logging.getLogger("pipeline_pre_news")
ARTIFACT_ROOT = Path("artifacts/tse_youtube_notion/pipeline_pre_news")
BACKFILL_ROOT = Path("artifacts/tse_youtube_notion/backfill_2025")
DEFAULT_SUPER_MODEL = "gpt-5.4-mini"
DEFAULT_SUPER_FOCUS = "quality-core"
DEFAULT_SUPER_MIN_CONFIDENCE = "medium"
PLAYLIST_PLACEHOLDER = "https://www.youtube.com/playlist?list="


@dataclass(frozen=True)
class PipelineStage:
    name: str
    command: list[str]
    env_overrides: dict[str, str] | None = None


def discover_playlist_url_for_year(year: int) -> str:
    candidates: list[tuple[str, str, Path]] = []
    for path in sorted(BACKFILL_ROOT.glob(f"{year}_*")):
        if not path.is_dir() or "_smoke_" in path.name:
            continue
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        updated_at = str(payload.get("updated_at") or payload.get("started_at") or "")
        playlist_url = str(payload.get("playlist_url") or "").strip()
        if playlist_url:
            candidates.append((updated_at, playlist_url, path))
    if not candidates:
        raise RuntimeError(f"Não foi possível inferir a playlist de {year} a partir dos manifests locais.")
    candidates.sort(key=lambda item: (item[0], str(item[2])))
    return candidates[-1][1]


def resolve_playlist_url(playlist_url: str, year: int) -> str:
    candidate = (playlist_url or "").strip()
    if candidate and candidate != PLAYLIST_PLACEHOLDER:
        return candidate
    return discover_playlist_url_for_year(year)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encadeia o pipeline TSE do Gemini até antes da etapa de notícias."
    )
    parser.add_argument("--playlist-url", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--initial-workers", type=int, default=3)
    parser.add_argument("--auto-scale", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-initial-backfill", action="store_true")
    parser.add_argument("--skip-rerun", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument("--skip-schema-core", action="store_true")
    parser.add_argument("--skip-identity-core", action="store_true")
    parser.add_argument("--skip-identity-replay", action="store_true")
    parser.add_argument("--skip-deterministic-core", action="store_true")
    parser.add_argument("--skip-composition-core", action="store_true")
    parser.add_argument("--skip-super-auditor", action="store_true")
    parser.add_argument(
        "--repair-focus",
        default="all",
        choices=[
            "all",
            "association",
            "origem",
            "classe",
            "votacao",
            "links",
            "tipo",
            "punchline",
            "numero",
            "core-fields",
            "composition",
            "partes-advogados",
            "schema-core",
            "identity-core",
            "deterministic-core",
        ],
    )
    parser.add_argument(
        "--super-focus",
        default=DEFAULT_SUPER_FOCUS,
        choices=["all", "quality-core", "origem"],
    )
    parser.add_argument("--super-model", default=DEFAULT_SUPER_MODEL)
    parser.add_argument(
        "--super-min-confidence",
        default=DEFAULT_SUPER_MIN_CONFIDENCE,
        choices=["high", "medium", "low"],
    )
    parser.add_argument("--skip-residual-focal", action="store_true")
    parser.add_argument("--residual-max-rounds", type=int, default=3)
    parser.add_argument("--residual-video-timeout-seconds", type=int, default=2400)
    parser.add_argument("--residual-no-progress-timeout-seconds", type=int, default=900)
    parser.add_argument("--residual-retry-timeouts", type=int, default=1)
    return parser.parse_args()


def build_stage_commands(args: argparse.Namespace) -> list[PipelineStage]:
    python = sys.executable
    backfill_script = "tse_backfill_2025_notion.py"
    super_script = "super_auditor.py"
    identity_replay_script = "run_identity_replay_batch.py"
    stages: list[PipelineStage] = []

    if not args.skip_initial_backfill:
        command = [
            python,
            backfill_script,
            "--playlist-url",
            args.playlist_url,
            "--year",
            str(args.year),
            "--max-workers",
            str(args.max_workers),
            "--initial-workers",
            str(args.initial_workers),
        ]
        if args.limit > 0:
            command.extend(["--limit", str(args.limit)])
        if args.auto_scale:
            command.append("--auto-scale")
        if args.resume:
            command.append("--resume")
        stages.append(PipelineStage("initial_backfill", command))

    if not args.skip_rerun:
        stages.append(
            PipelineStage(
                "rerun_error_videos",
                [
                    python,
                    backfill_script,
                    "--playlist-url",
                    args.playlist_url,
                    "--year",
                    str(args.year),
                    "--rerun-error-videos",
                ],
            )
        )

    if not args.skip_audit:
        stages.append(
            PipelineStage(
                "audit_existing_year",
                [
                    python,
                    backfill_script,
                    "--playlist-url",
                    args.playlist_url,
                    "--year",
                    str(args.year),
                    "--audit-existing-year",
                ],
            )
        )

    if not args.skip_repair:
        stages.append(
            PipelineStage(
                "repair_existing_year",
                [
                    python,
                    backfill_script,
                    "--playlist-url",
                    args.playlist_url,
                    "--year",
                    str(args.year),
                    "--repair-existing-year",
                    "--repair-focus",
                    args.repair_focus,
                ],
            )
        )
        if not args.skip_schema_core:
            stages.append(
                PipelineStage(
                    "repair_schema_core",
                    [
                        python,
                        backfill_script,
                        "--playlist-url",
                        args.playlist_url,
                        "--year",
                        str(args.year),
                        "--repair-existing-year",
                        "--repair-focus",
                        "schema-core",
                    ],
                )
            )
        if not args.skip_deterministic_core:
            if not args.skip_identity_core:
                stages.append(
                    PipelineStage(
                        "repair_identity_core",
                        [
                            python,
                            backfill_script,
                            "--playlist-url",
                            args.playlist_url,
                            "--year",
                            str(args.year),
                            "--repair-existing-year",
                            "--repair-focus",
                            "identity-core",
                        ],
                    )
                )
                if not args.skip_identity_replay:
                    stages.append(
                        PipelineStage(
                            "repair_identity_replay",
                            [
                                python,
                                identity_replay_script,
                                "--years",
                                str(args.year),
                                "--playlist-url",
                                args.playlist_url,
                            ],
                        )
                    )
            stages.append(
                PipelineStage(
                    "repair_deterministic_core",
                    [
                        python,
                        backfill_script,
                        "--playlist-url",
                        args.playlist_url,
                        "--year",
                        str(args.year),
                        "--repair-existing-year",
                        "--repair-focus",
                        "deterministic-core",
                    ],
                )
            )
        if not args.skip_composition_core:
            stages.append(
                PipelineStage(
                    "repair_composition_core",
                    [
                        python,
                        backfill_script,
                        "--playlist-url",
                        args.playlist_url,
                        "--year",
                        str(args.year),
                        "--repair-existing-year",
                        "--repair-focus",
                        "composition",
                    ],
                )
            )

    if not args.skip_super_auditor:
        stages.append(
            PipelineStage(
                "super_auditor",
                [
                    python,
                    super_script,
                    "--years",
                    str(args.year),
                    "--playlist-override",
                    f"{args.year}={args.playlist_url}",
                    "--apply",
                    "--focus",
                    args.super_focus,
                    "--model",
                    args.super_model,
                    "--min-confidence",
                    args.super_min_confidence,
                ],
            )
        )

    return stages


def _write_summary(run_root: Path, summary: dict[str, Any]) -> None:
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_stage_result(summary: dict[str, Any], result: dict[str, Any], run_root: Path) -> None:
    summary.setdefault("stages", []).append(result)
    _write_summary(run_root, summary)


def _extract_playlist_id(playlist_url: str) -> str:
    parsed = urlparse(playlist_url)
    query_value = parse_qs(parsed.query).get("list", [""])[0].strip()
    if query_value:
        return query_value
    if parsed.fragment:
        fragment_value = parse_qs(parsed.fragment).get("list", [""])[0].strip()
        if fragment_value:
            return fragment_value
    return "playlist"


def _manifest_path_for(playlist_url: str, year: int) -> Path:
    return BACKFILL_ROOT / f"{year}_{_extract_playlist_id(playlist_url)}" / "manifest.json"


def _repair_manifest_false_errors(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    root_dir = manifest_path.parent
    videos = manifest.get("videos") or {}
    for video_id, entry in videos.items():
        status = str(entry.get("status") or "")
        last_artifact = str(entry.get("last_artifact") or "")
        if status != "error" or last_artifact != "07_backfill_summary.json":
            continue
        position = int(entry.get("position") or 0)
        if position <= 0:
            continue
        summary_path = root_dir / f"{position:03d}_{video_id}" / "07_backfill_summary.json"
        if not summary_path.exists():
            continue
        entry["status"] = "done"
        entry["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        entry["last_step"] = "done"
        entry["finished_at"] = entry.get("finished_at") or time.strftime("%Y-%m-%dT%H:%M:%S")
        entry.pop("error", None)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _load_manifest(playlist_url: str, year: int) -> tuple[Path, dict[str, Any]]:
    manifest_path = _manifest_path_for(playlist_url, year)
    if not manifest_path.exists():
        raise RuntimeError(f"Manifest do backfill não encontrado: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _repair_manifest_false_errors(manifest, manifest_path)
    return manifest_path, manifest


def _manifest_error_video_ids(manifest: dict[str, Any]) -> list[str]:
    entries = manifest.get("videos") or {}
    items: list[tuple[int, str]] = []
    for video_id, entry in entries.items():
        if str(entry.get("status") or "") != "error":
            continue
        items.append((int(entry.get("position") or 0), str(video_id)))
    items.sort()
    return [video_id for _position, video_id in items]


def _manifest_status_counts(manifest: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in (manifest.get("videos") or {}).values():
        status = str((entry or {}).get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _maybe_tolerate_rerun_failure(
    *,
    stage: PipelineStage,
    result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if stage.name != "rerun_error_videos" or result.get("status") == "done":
        return result
    try:
        manifest_path, manifest = _load_manifest(args.playlist_url, args.year)
    except Exception as exc:
        LOGGER.warning("Não foi possível inspecionar o manifest após falha no rerun: %s", exc)
        return result

    counts = _manifest_status_counts(manifest)
    done_count = int(counts.get("done", 0))
    remaining_video_ids = _manifest_error_video_ids(manifest)
    if done_count <= 0:
        return result

    tolerated = dict(result)
    tolerated["status"] = "tolerated_error"
    tolerated["tolerated"] = True
    tolerated["manifest_path"] = str(manifest_path)
    tolerated["manifest_status_counts"] = counts
    tolerated["remaining_error_video_ids"] = remaining_video_ids
    tolerated["remaining_error_count"] = len(remaining_video_ids)
    LOGGER.warning(
        "[%s] falhou, mas o manifest já tem %s vídeos concluídos e %s resíduos. "
        "O pipeline seguirá para o pós-lote.",
        stage.name,
        done_count,
        len(remaining_video_ids),
    )
    return tolerated


def _with_video_ids(command: list[str], video_ids: list[str]) -> list[str]:
    filtered = list(command)
    for video_id in video_ids:
        filtered.extend(["--video-id", video_id])
    return filtered


def _run_residual_focal_closure(
    *,
    args: argparse.Namespace,
    run_root: Path,
    summary: dict[str, Any],
) -> None:
    if args.skip_residual_focal:
        return

    manifest_path, manifest = _load_manifest(args.playlist_url, args.year)
    remaining_video_ids = _manifest_error_video_ids(manifest)
    if not remaining_video_ids:
        return

    python = sys.executable
    backfill_script = "tse_backfill_2025_notion.py"
    identity_replay_script = "run_identity_replay_batch.py"
    super_script = "super_auditor.py"

    for round_index in range(1, max(1, int(args.residual_max_rounds)) + 1):
        residual_before = list(remaining_video_ids)
        timeout_seconds = max(1, int(args.residual_video_timeout_seconds)) * (2 ** (round_index - 1))
        no_progress_seconds = max(1, int(args.residual_no_progress_timeout_seconds)) * (2 ** (round_index - 1))
        rerun_stage = PipelineStage(
            name=f"residual_rerun_round_{round_index}",
            command=_with_video_ids(
                [
                    python,
                    backfill_script,
                    "--playlist-url",
                    args.playlist_url,
                    "--year",
                    str(args.year),
                    "--rerun-error-videos",
                ],
                residual_before,
            ),
            env_overrides={
                "BACKFILL_VIDEO_TIMEOUT_SECONDS": str(timeout_seconds),
                "BACKFILL_NO_PROGRESS_TIMEOUT_SECONDS": str(no_progress_seconds),
            },
        )
        _manifest_path, manifest = _load_manifest(args.playlist_url, args.year)
        result = run_stage(rerun_stage, run_root)
        resolved_video_ids = [
            video_id
            for video_id in residual_before
            if str(((manifest.get("videos") or {}).get(video_id) or {}).get("status") or "") == "done"
        ]
        if result["status"] != "done":
            counts = _manifest_status_counts(manifest)
            if int(counts.get("done", 0)) > 0:
                tolerated = dict(result)
                tolerated["status"] = "tolerated_error"
                tolerated["tolerated"] = True
                tolerated["manifest_status_counts"] = counts
                tolerated["remaining_error_video_ids"] = _manifest_error_video_ids(manifest)
                tolerated["resolved_video_ids"] = resolved_video_ids
                result = tolerated
                LOGGER.warning(
                    "[%s] falhou, mas o manifest preserva %s vídeos feitos e %s resíduos. "
                    "O fechamento focal seguirá para os vídeos resolvidos nesta rodada.",
                    rerun_stage.name,
                    int(counts.get("done", 0)),
                    len(tolerated["remaining_error_video_ids"]),
                )
            else:
                _append_stage_result(summary, result, run_root)
                raise RuntimeError(f"Falha no residual focal ({rerun_stage.name}).")
        _append_stage_result(summary, result, run_root)

        if resolved_video_ids:
            focal_stages = [
                PipelineStage(
                    name=f"residual_repair_all_round_{round_index}",
                    command=_with_video_ids(
                        [
                            python,
                            backfill_script,
                            "--playlist-url",
                            args.playlist_url,
                            "--year",
                            str(args.year),
                            "--repair-existing-year",
                            "--repair-focus",
                            args.repair_focus,
                        ],
                        resolved_video_ids,
                    ),
                ),
                PipelineStage(
                    name=f"residual_schema_core_round_{round_index}",
                    command=_with_video_ids(
                        [
                            python,
                            backfill_script,
                            "--playlist-url",
                            args.playlist_url,
                            "--year",
                            str(args.year),
                            "--repair-existing-year",
                            "--repair-focus",
                            "schema-core",
                        ],
                        resolved_video_ids,
                    ),
                ),
                PipelineStage(
                    name=f"residual_identity_core_round_{round_index}",
                    command=_with_video_ids(
                        [
                            python,
                            backfill_script,
                            "--playlist-url",
                            args.playlist_url,
                            "--year",
                            str(args.year),
                            "--repair-existing-year",
                            "--repair-focus",
                            "identity-core",
                        ],
                        resolved_video_ids,
                    ),
                ),
                PipelineStage(
                    name=f"residual_identity_replay_round_{round_index}",
                    command=_with_video_ids(
                        [
                            python,
                            identity_replay_script,
                            "--years",
                            str(args.year),
                            "--playlist-url",
                            args.playlist_url,
                            "--max-workers",
                            str(args.max_workers),
                            "--video-timeout-seconds",
                            str(timeout_seconds),
                            "--no-progress-timeout-seconds",
                            str(no_progress_seconds),
                            "--retry-timeouts",
                            str(args.residual_retry_timeouts),
                        ],
                        resolved_video_ids,
                    ),
                ),
                PipelineStage(
                    name=f"residual_deterministic_core_round_{round_index}",
                    command=_with_video_ids(
                        [
                            python,
                            backfill_script,
                            "--playlist-url",
                            args.playlist_url,
                            "--year",
                            str(args.year),
                            "--repair-existing-year",
                            "--repair-focus",
                            "deterministic-core",
                        ],
                        resolved_video_ids,
                    ),
                ),
                PipelineStage(
                    name=f"residual_composition_core_round_{round_index}",
                    command=_with_video_ids(
                        [
                            python,
                            backfill_script,
                            "--playlist-url",
                            args.playlist_url,
                            "--year",
                            str(args.year),
                            "--repair-existing-year",
                            "--repair-focus",
                            "composition",
                        ],
                        resolved_video_ids,
                    ),
                ),
                PipelineStage(
                    name=f"residual_super_auditor_round_{round_index}",
                    command=_with_video_ids(
                        [
                            python,
                            super_script,
                            "--years",
                            str(args.year),
                            "--playlist-override",
                            f"{args.year}={args.playlist_url}",
                            "--apply",
                            "--focus",
                            args.super_focus,
                            "--model",
                            args.super_model,
                            "--min-confidence",
                            args.super_min_confidence,
                        ],
                        resolved_video_ids,
                    ),
                ),
            ]
            for stage in focal_stages:
                result = run_stage(stage, run_root)
                _append_stage_result(summary, result, run_root)
                if result["status"] != "done":
                    raise RuntimeError(f"Falha no residual focal ({stage.name}).")

        _manifest_path, manifest = _load_manifest(args.playlist_url, args.year)
        remaining_video_ids = _manifest_error_video_ids(manifest)
        if not remaining_video_ids:
            return

    summary["residual_errors"] = {
        "count": len(remaining_video_ids),
        "video_ids": remaining_video_ids,
        "manifest_path": str(_manifest_path),
    }
    LOGGER.warning(
        "Vídeos problemáticos remanescentes após o residual focal: %s. "
        "O pipeline será encerrado como concluído com resíduos.",
        ", ".join(remaining_video_ids),
    )


def ensure_python_module_available(python_executable: str, module_name: str) -> None:
    probe = subprocess.run(
        [python_executable, "-c", f"import {module_name}"],
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
    )
    if probe.returncode == 0:
        return
    message = (probe.stderr or probe.stdout or "").strip()
    raise RuntimeError(
        f"Módulo obrigatório ausente na runtime {python_executable}: {module_name}. {message}"
    )


def run_stage(stage: PipelineStage, run_root: Path) -> dict[str, Any]:
    log_path = run_root / f"{stage.name}.log"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    LOGGER.info("[%s] iniciando: %s", stage.name, shlex.join(stage.command))
    if len(stage.command) > 1 and Path(stage.command[1]).name == "super_auditor.py":
        ensure_python_module_available(stage.command[0], "openai")
    env = os.environ.copy()
    if stage.env_overrides:
        env.update(stage.env_overrides)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            stage.command,
            cwd=Path.cwd(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    LOGGER.info("[%s] concluído com código %s. Log: %s", stage.name, process.returncode, log_path)
    return {
        "name": stage.name,
        "command": stage.command,
        "started_at": started_at,
        "finished_at": finished_at,
        "returncode": process.returncode,
        "log_path": str(log_path),
        "status": "done" if process.returncode == 0 else "error",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args.playlist_url = resolve_playlist_url(args.playlist_url, args.year)
    run_root = ARTIFACT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Pipeline pré-notícias iniciado. Run root: %s", run_root)

    stages = build_stage_commands(args)
    summary: dict[str, Any] = {
        "playlist_url": args.playlist_url,
        "year": args.year,
        "run_root": str(run_root),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": [],
    }
    _write_summary(run_root, summary)

    try:
        for stage in stages:
            result = run_stage(stage, run_root)
            result = _maybe_tolerate_rerun_failure(stage=stage, result=result, args=args)
            _append_stage_result(summary, result, run_root)
            if result["status"] not in {"done", "tolerated_error"}:
                raise RuntimeError(stage.name)

        _run_residual_focal_closure(args=args, run_root=run_root, summary=summary)
    except RuntimeError as exc:
        failed_stage = str(exc) or "unknown"
        summary["status"] = "error"
        summary["failed_stage"] = failed_stage
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_summary(run_root, summary)
        raise SystemExit(1)

    summary["status"] = "done_with_residual_errors" if summary.get("residual_errors") else "done"
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_summary(run_root, summary)
    LOGGER.info("Pipeline pré-notícias concluído. Resumo: %s", run_root / "summary.json")


if __name__ == "__main__":
    main()
