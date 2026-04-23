from __future__ import annotations

import argparse
import json
import time
from collections import deque
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tse_backfill_2025_notion as backfill
from tse_backfill_2025_notion import (
    build_runtime_context,
    dump_existing_pages_snapshot,
    dump_schema_snapshot,
    find_target_video,
    iter_backfill_run_dirs,
    load_existing_pages_for_year_with_retry,
    load_repair_artifact_context,
    start_video_worker,
    poll_active_worker,
    _row_has_local_association_proof,
)
from tse_youtube_notion_core import (
    ARTIFACT_ROOT,
    NotionSessoesClient,
    PublishPreviewRow,
    RunArtifacts,
    _row_has_strong_local_judgment_evidence,
)


REPO_ROOT = Path(__file__).resolve().parent
BACKFILL_ROOT = REPO_ROOT / "artifacts" / "tse_youtube_notion" / "backfill_2025"
RUNS_ROOT = ARTIFACT_ROOT / "identity_replay_runs"


@dataclass
class ReplayTask:
    video: object
    video_id: str
    reasons: list[dict[str, str]]
    attempt: int = 0
    current_video_timeout_seconds: int = 0
    current_no_progress_timeout_seconds: int = 0


def _write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_playlist_url(year: int, explicit_playlist_url: str = "") -> str:
    explicit = str(explicit_playlist_url or "").strip()
    if explicit:
        return explicit
    manifests = sorted(BACKFILL_ROOT.glob(f"{year}_PL*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Manifesto do ano {year} não encontrado em {BACKFILL_ROOT}")
    with manifests[0].open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    playlist_url = str(payload.get("playlist_url") or payload.get("playlist") or "").strip()
    if not playlist_url:
        raise ValueError(f"Manifesto do ano {year} não contém playlist_url")
    return playlist_url


def _latest_identity_repair_summary(year: int, playlist_url: str = "") -> tuple[Path, dict] | tuple[None, None]:
    target_playlist = str(playlist_url or "").strip()
    for path in sorted(BACKFILL_ROOT.glob(f"_repair_{year}_*/summary.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload_playlist = str(payload.get("playlist_url") or "").strip()
        if target_playlist and payload_playlist and payload_playlist != target_playlist:
            continue
        videos = payload.get("videos") or []
        if any(video.get("repair_focus") == "identity-core" for video in videos if isinstance(video, dict)):
            return path, payload
    return None, None


def _row_from_summary_entry(entry: dict) -> PublishPreviewRow:
    return PublishPreviewRow(
        tema=str(entry.get("tema") or ""),
        classe_processo=str(entry.get("classe_processo") or ""),
        numero_processo=str(entry.get("numero_processo") or ""),
        youtube_link=str(entry.get("youtube_link") or ""),
        relator=str(entry.get("relator") or ""),
        resultado=str(entry.get("resultado") or ""),
        votacao=str(entry.get("votacao") or ""),
        data_sessao=str(entry.get("data_sessao") or ""),
    )


def _has_precedent_cited_marker(entry: dict) -> bool:
    texts = []
    for field in ("warnings", "errors"):
        raw_values = entry.get(field) or []
        if isinstance(raw_values, list):
            texts.extend(str(item or "") for item in raw_values)
    return any("precedente citado" in text.lower() for text in texts)


def _latest_artifact_dirs_by_video(playlist_url: str, year: int) -> dict[str, Path]:
    latest_by_video: dict[str, Path] = {}
    for run_dir in iter_backfill_run_dirs(playlist_url, year):
        for candidate_dir in sorted(path for path in run_dir.glob("*_*") if path.is_dir()):
            video_id = candidate_dir.name.split("_", 1)[-1]
            current = latest_by_video.get(video_id)
            if current is None or candidate_dir.stat().st_mtime > current.stat().st_mtime:
                latest_by_video[video_id] = candidate_dir
    return latest_by_video


def _collect_replay_candidates_from_repair_summary(playlist_url: str, year: int) -> dict[str, list[dict[str, str]]]:
    _, payload = _latest_identity_repair_summary(year, playlist_url)
    if not payload:
        return {}
    candidates: dict[str, list[dict[str, str]]] = {}
    for video_summary in payload.get("videos") or []:
        if not isinstance(video_summary, dict):
            continue
        video_id = str(video_summary.get("video_id") or "").strip()
        if not video_id:
            continue
        artifact_context = load_repair_artifact_context(playlist_url, year, video_id)
        for field_name in ("trashed_unproven", "review_pages"):
            for entry in video_summary.get(field_name) or []:
                if not isinstance(entry, dict):
                    continue
                if field_name == "review_pages" and entry.get("reason") != "identity_unproven":
                    continue
                row = _row_from_summary_entry(entry)
                if not row.numero_processo:
                    continue
                if not _row_has_local_association_proof(row, artifact_context):
                    continue
                candidates.setdefault(video_id, []).append(
                    {
                        "source": "identity-core",
                        "numero_processo": row.numero_processo,
                        "reason": str(entry.get("reason") or "identity_unproven"),
                    }
                )
    return candidates


def _collect_replay_candidates_from_backfill_summaries(playlist_url: str, year: int) -> dict[str, list[dict[str, str]]]:
    candidates: dict[str, list[dict[str, str]]] = {}
    for video_id, candidate_dir in _latest_artifact_dirs_by_video(playlist_url, year).items():
        summary_path = candidate_dir / "07_backfill_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        artifact_store = RunArtifacts(candidate_dir)
        for entry in summary_payload.get("publish_results") or []:
            if not isinstance(entry, dict) or not _has_precedent_cited_marker(entry):
                continue
            row = _row_from_summary_entry(entry)
            if not row.numero_processo:
                continue
            if not _row_has_strong_local_judgment_evidence(row, artifact_store):
                continue
            candidates.setdefault(video_id, []).append(
                {
                    "source": "publish-results",
                    "numero_processo": row.numero_processo,
                    "reason": "precedente_citado_with_local_evidence",
                }
            )
    return candidates


def _merge_candidate_maps(*candidate_maps: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
    merged: dict[str, list[dict[str, str]]] = {}
    seen: dict[str, set[tuple[str, str, str]]] = {}
    for candidate_map in candidate_maps:
        for video_id, reasons in candidate_map.items():
            video_seen = seen.setdefault(video_id, set())
            for reason in reasons:
                marker = (
                    str(reason.get("source") or ""),
                    str(reason.get("numero_processo") or ""),
                    str(reason.get("reason") or ""),
                )
                if marker in video_seen:
                    continue
                video_seen.add(marker)
                merged.setdefault(video_id, []).append(reason)
    return merged


def _filter_candidate_map(
    candidate_map: dict[str, list[dict[str, str]]],
    requested_video_ids: set[str],
) -> dict[str, list[dict[str, str]]]:
    if not requested_video_ids:
        return dict(candidate_map)
    return {
        video_id: reasons
        for video_id, reasons in candidate_map.items()
        if video_id in requested_video_ids
    }


def _is_timeout_like_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "timeout de" in text or "sem progresso real de artefatos" in text


def _find_year_entry(summary: dict[str, object], year: int) -> dict[str, object] | None:
    for entry in summary.get("years", []):  # type: ignore[assignment]
        if isinstance(entry, dict) and int(entry.get("year") or 0) == year:
            return entry
    return None


def _completed_replay_ids(year_entry: dict[str, object]) -> set[str]:
    completed: set[str] = set()
    for entry in year_entry.get("replayed_videos", []):  # type: ignore[assignment]
        if not isinstance(entry, dict):
            continue
        if entry.get("status", "done") == "done":
            video_id = str(entry.get("video_id") or "").strip()
            if video_id:
                completed.add(video_id)
    return completed


def _persisted_candidate_map(year_entry: dict[str, object] | None) -> dict[str, list[dict[str, str]]]:
    if not year_entry:
        return {}
    persisted: dict[str, list[dict[str, str]]] = {}
    for collection_name in ("candidate_videos", "failed_videos"):
        for entry in year_entry.get(collection_name, []):  # type: ignore[assignment]
            if not isinstance(entry, dict):
                continue
            video_id = str(entry.get("video_id") or "").strip()
            if not video_id:
                continue
            reasons = entry.get("reasons") or []
            if not isinstance(reasons, list):
                continue
            persisted.setdefault(video_id, []).extend(
                reason for reason in reasons if isinstance(reason, dict)
            )
    return persisted


def _ensure_year_snapshots(
    *,
    year_root: Path,
    notion_schema,
    existing_pages_by_video,
) -> None:
    if not (year_root / "_schema_snapshot.json").exists():
        dump_schema_snapshot(year_root, notion_schema)
    if not (year_root / "_existing_pages_snapshot.json").exists():
        dump_existing_pages_snapshot(year_root, existing_pages_by_video)


def _set_worker_timeouts(video_timeout_seconds: int, no_progress_timeout_seconds: int) -> None:
    backfill.VIDEO_WORKER_TIMEOUT_SECONDS = max(1, int(video_timeout_seconds))
    backfill.NO_PROGRESS_TIMEOUT_SECONDS = max(1, int(no_progress_timeout_seconds))


def _sync_active_videos(
    year_entry: dict[str, object],
    active_workers: dict[str, tuple[object, ReplayTask]],
) -> None:
    active_payload: list[dict[str, object]] = []
    for video_id, (handle, task) in active_workers.items():
        artifact_dir = getattr(handle, "artifact_dir", None)
        process = getattr(handle, "process", None)
        active_payload.append(
            {
                "video_id": video_id,
                "attempt": task.attempt,
                "worker_pid": getattr(process, "pid", None),
                "started_at": getattr(handle, "started_at", ""),
                "artifact_dir": str(artifact_dir) if artifact_dir else "",
                "last_artifact": getattr(handle, "last_seen_artifact_name", "") or "",
                "video_timeout_seconds": task.current_video_timeout_seconds,
                "no_progress_timeout_seconds": task.current_no_progress_timeout_seconds,
            }
        )
    year_entry["active_videos"] = active_payload


def _run_replay_queue(
    *,
    summary: dict[str, object],
    summary_path: Path,
    year: int,
    year_entry: dict[str, object],
    tasks: list[ReplayTask],
    worker_args: Namespace,
    year_root: Path,
    retry_timeouts: int,
    max_workers: int,
    base_video_timeout_seconds: int,
    base_no_progress_timeout_seconds: int,
    start_worker_fn=start_video_worker,
    poll_worker_fn=poll_active_worker,
    sleep_fn=time.sleep,
) -> bool:
    pending_tasks: deque[ReplayTask] = deque(tasks)
    active_workers: dict[str, tuple[object, ReplayTask]] = {}
    year_had_errors = False

    while pending_tasks or active_workers:
        summary_changed = False
        while pending_tasks and len(active_workers) < max(1, int(max_workers)):
            task = pending_tasks.popleft()
            task.attempt += 1
            task.current_video_timeout_seconds = max(1, int(base_video_timeout_seconds)) * (2 ** (task.attempt - 1))
            task.current_no_progress_timeout_seconds = max(1, int(base_no_progress_timeout_seconds)) * (2 ** (task.attempt - 1))
            _set_worker_timeouts(
                task.current_video_timeout_seconds,
                task.current_no_progress_timeout_seconds,
            )
            print(
                f"[identity-replay] year={year} starting video_id={task.video_id} "
                f"attempt={task.attempt} limits=({task.current_video_timeout_seconds}s/"
                f"{task.current_no_progress_timeout_seconds}s)",
                flush=True,
            )
            handle = start_worker_fn(task.video, worker_args, year_root)
            active_workers[task.video_id] = (handle, task)
            summary_changed = True

        completed_video_ids: list[str] = []
        for video_id, (handle, task) in list(active_workers.items()):
            _set_worker_timeouts(
                task.current_video_timeout_seconds,
                task.current_no_progress_timeout_seconds,
            )
            polled = poll_worker_fn(handle)
            if polled.get("progress_changed"):
                summary_changed = True
            status = str(polled.get("status") or "")
            if status == "running":
                continue

            completed_video_ids.append(video_id)
            if status == "done":
                replayed = list(year_entry.get("replayed_videos") or [])
                replayed.append(
                    {
                        "video_id": video_id,
                        "reasons": task.reasons,
                        "status": "done",
                        "attempts": task.attempt,
                        "summary": polled.get("summary") or {},
                    }
                )
                year_entry["replayed_videos"] = replayed
                print(
                    f"[identity-replay] year={year} done video_id={video_id} attempts={task.attempt}",
                    flush=True,
                )
                summary_changed = True
                continue

            error_text = str(polled.get("error") or "")
            if _is_timeout_like_error(RuntimeError(error_text)) and task.attempt <= max(0, int(retry_timeouts)):
                print(
                    f"[identity-replay] year={year} retrying video_id={video_id} "
                    f"after timeout; next_limits=({task.current_video_timeout_seconds * 2}s/"
                    f"{task.current_no_progress_timeout_seconds * 2}s)",
                    flush=True,
                )
                pending_tasks.appendleft(task)
                summary_changed = True
                continue

            failed = list(year_entry.get("failed_videos") or [])
            failed.append(
                {
                    "video_id": video_id,
                    "reasons": task.reasons,
                    "status": "error",
                    "attempts": task.attempt,
                    "error": error_text,
                }
            )
            year_entry["failed_videos"] = failed
            year_had_errors = True
            summary_changed = True
            print(
                f"[identity-replay] year={year} error video_id={video_id} attempts={task.attempt} "
                f"error={error_text}",
                flush=True,
            )

        for video_id in completed_video_ids:
            active_workers.pop(video_id, None)

        if summary_changed:
            _sync_active_videos(year_entry, active_workers)
            _write_summary(summary_path, summary)

        if active_workers:
            sleep_fn(backfill.WORKER_HEARTBEAT_SECONDS)

    year_entry["active_videos"] = []
    _write_summary(summary_path, summary)
    return year_had_errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reprocessa vídeos com julgados reais omitidos após o identity-core."
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2025, 2024, 2023, 2022, 2021])
    parser.add_argument("--skip-news", action="store_true", default=True)
    parser.add_argument("--resume-run-root")
    parser.add_argument("--playlist-url", default="")
    parser.add_argument("--video-timeout-seconds", type=int, default=2400)
    parser.add_argument("--no-progress-timeout-seconds", type=int, default=600)
    parser.add_argument("--retry-timeouts", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--video-id", dest="video_ids", action="append", default=[])
    args = parser.parse_args(argv)

    args.max_workers = max(1, int(args.max_workers))
    requested_video_ids = set(args.video_ids or [])
    _set_worker_timeouts(args.video_timeout_seconds, args.no_progress_timeout_seconds)

    if args.resume_run_root:
        run_root = Path(args.resume_run_root)
        summary_path = run_root / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Resumo do run não encontrado em {summary_path}")
        summary = _read_summary(summary_path)
        summary["status"] = "running"
        print(f"[identity-replay] resuming run_root={run_root}", flush=True)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = RUNS_ROOT / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        print(f"[identity-replay] run_root={run_root}", flush=True)
        summary_path = run_root / "summary.json"
        summary = {
            "run_root": str(run_root),
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "years": [],
        }
        _write_summary(summary_path, summary)

    runtime = build_runtime_context()
    notion_client = NotionSessoesClient(
        api_key=runtime["notion_api_key"],
        data_source_id=runtime["notion_data_source_id"],
    )
    notion_schema = notion_client.fetch_schema()
    if not (run_root / "_schema_snapshot.json").exists():
        dump_schema_snapshot(run_root, notion_schema)

    had_errors = False

    for year in args.years:
        year_entry = _find_year_entry(summary, year)
        if year_entry and year_entry.get("status") == "done":
            print(f"[identity-replay] year={year} already done; skipping", flush=True)
            continue

        playlist_url = _load_playlist_url(year, args.playlist_url)
        persisted_candidates = _persisted_candidate_map(year_entry)
        repair_candidates = _collect_replay_candidates_from_repair_summary(playlist_url, year)
        skipped_candidates = _collect_replay_candidates_from_backfill_summaries(playlist_url, year)
        replay_candidates = _merge_candidate_maps(
            persisted_candidates,
            repair_candidates,
            skipped_candidates,
        )
        replay_candidates = _filter_candidate_map(replay_candidates, requested_video_ids)
        existing_pages_by_video = load_existing_pages_for_year_with_retry(
            notion_client,
            notion_schema,
            year,
            playlist_url=playlist_url,
        )
        year_root = run_root / str(year)
        year_root.mkdir(parents=True, exist_ok=True)
        _ensure_year_snapshots(
            year_root=year_root,
            notion_schema=notion_schema,
            existing_pages_by_video=existing_pages_by_video,
        )
        if year_entry is None:
            year_entry = {
                "year": year,
                "playlist_url": playlist_url,
                "status": "running",
                "candidate_videos": [],
                "replayed_videos": [],
                "failed_videos": [],
                "started_at": datetime.now().isoformat(),
            }
            summary_years = list(summary["years"])  # type: ignore[arg-type]
            summary_years.append(year_entry)
            summary["years"] = summary_years
            _write_summary(summary_path, summary)
        else:
            year_entry["playlist_url"] = playlist_url
            year_entry["status"] = "running"
            year_entry.setdefault("replayed_videos", [])
            year_entry.setdefault("failed_videos", [])
            year_entry["active_videos"] = []
            year_entry.setdefault("started_at", datetime.now().isoformat())
            _write_summary(summary_path, summary)

        ordered_tasks: list[ReplayTask] = []
        for video_id, reasons in replay_candidates.items():
            try:
                video = find_target_video(playlist_url, year, video_id)
            except Exception as exc:
                print(
                    f"[identity-replay] year={year} skip video_id={video_id} (not in playlist): {exc}",
                    flush=True,
                )
                continue
            ordered_tasks.append(
                ReplayTask(
                    video=video,
                    video_id=video_id,
                    reasons=reasons,
                )
            )
        ordered_tasks.sort(key=lambda item: item.video.position)
        completed_ids = _completed_replay_ids(year_entry)
        print(
            f"[identity-replay] year={year} candidates={len(ordered_tasks)} max_workers={args.max_workers}",
            flush=True,
        )
        year_entry["candidate_videos"] = [
            {
                "video_id": task.video_id,
                "reasons": task.reasons,
            }
            for task in ordered_tasks
        ]
        _write_summary(summary_path, summary)

        worker_args = Namespace(
            playlist_url=playlist_url,
            year=year,
            skip_news=bool(args.skip_news),
            no_trash_unmatched_precedents=False,
        )
        pending_tasks = [task for task in ordered_tasks if task.video_id not in completed_ids]
        year_had_errors = _run_replay_queue(
            summary=summary,
            summary_path=summary_path,
            year=year,
            year_entry=year_entry,
            tasks=pending_tasks,
            worker_args=worker_args,
            year_root=year_root,
            retry_timeouts=args.retry_timeouts,
            max_workers=args.max_workers,
            base_video_timeout_seconds=args.video_timeout_seconds,
            base_no_progress_timeout_seconds=args.no_progress_timeout_seconds,
        )
        had_errors = had_errors or year_had_errors

        year_entry["finished_at"] = datetime.now().isoformat()
        year_entry["status"] = "done_with_errors" if year_had_errors else "done"
        print(f"[identity-replay] year={year} finished", flush=True)
        _write_summary(summary_path, summary)

    summary["status"] = "done_with_errors" if had_errors else "done"
    summary["finished_at"] = datetime.now().isoformat()
    _write_summary(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
