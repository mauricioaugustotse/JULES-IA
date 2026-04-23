from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tse_backfill_2025_notion as backfill
from tse_backfill_2025_notion import (
    build_runtime_context,
    dump_existing_pages_snapshot,
    dump_schema_snapshot,
    find_target_video,
    load_existing_pages_for_year_with_retry,
    poll_active_worker,
    start_video_worker,
)
from tse_youtube_notion_core import ARTIFACT_ROOT, NotionSessoesClient


REPO_ROOT = Path(__file__).resolve().parent
BACKFILL_ROOT = REPO_ROOT / "artifacts" / "tse_youtube_notion" / "backfill_2025"
AUDIT_RUNS_ROOT = ARTIFACT_ROOT / "semantic_bleed_audit"
RUNS_ROOT = ARTIFACT_ROOT / "semantic_bleed_corrective_runs"
RISK_PRIORITY = {"high": 2, "medium": 1, "low": 0}


@dataclass
class CorrectiveTask:
    video: object
    video_id: str
    flagged_pages: list[dict[str, Any]]
    risk_level: str
    risk_score: int
    attempt: int = 0
    current_video_timeout_seconds: int = 0
    current_no_progress_timeout_seconds: int = 0


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_playlist_url(year: int) -> str:
    manifests = sorted(BACKFILL_ROOT.glob(f"{year}_PL*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Manifesto do ano {year} não encontrado em {BACKFILL_ROOT}")
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    playlist_url = str(payload.get("playlist_url") or payload.get("playlist") or "").strip()
    if not playlist_url:
        raise ValueError(f"Manifesto do ano {year} não contém playlist_url")
    return playlist_url


def _latest_audit_run_root() -> Path:
    candidates = sorted(
        (path.parent for path in AUDIT_RUNS_ROOT.glob("*/summary.json")),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Nenhuma auditoria semantic_bleed encontrada em {AUDIT_RUNS_ROOT}")
    return candidates[0]


def _collect_candidates_from_audit_report(
    audit_run_root: Path,
    year: int,
    *,
    min_risk_level: str = "medium",
) -> dict[str, dict[str, Any]]:
    report_path = audit_run_root / f"{year}.json"
    if not report_path.exists():
        return {}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    threshold = RISK_PRIORITY.get(min_risk_level, 1)
    candidates: dict[str, dict[str, Any]] = {}
    for item in payload.get("flagged_pages") or []:
        if not isinstance(item, dict):
            continue
        risk_level = str(item.get("risk_level") or "medium").lower()
        if RISK_PRIORITY.get(risk_level, 0) < threshold:
            continue
        video_id = str(item.get("video_id") or "").strip()
        if not video_id:
            continue
        entry = candidates.setdefault(
            video_id,
            {
                "video_id": video_id,
                "video_title": str(item.get("video_title") or ""),
                "risk_level": risk_level,
                "risk_score": int(item.get("risk_score") or 0),
                "flagged_pages": [],
            },
        )
        if RISK_PRIORITY.get(risk_level, 0) > RISK_PRIORITY.get(str(entry.get("risk_level") or ""), 0):
            entry["risk_level"] = risk_level
        entry["risk_score"] = max(int(entry.get("risk_score") or 0), int(item.get("risk_score") or 0))
        flagged_page = {
            "page_id": str(item.get("page_id") or ""),
            "page_url": str(item.get("page_url") or ""),
            "numero_processo": str((item.get("row") or {}).get("numero_processo") or ""),
            "tipo_registro": str((item.get("row") or {}).get("tipo_registro") or ""),
            "risk_level": risk_level,
            "risk_score": int(item.get("risk_score") or 0),
            "reasons": [str(reason) for reason in (item.get("reasons") or [])],
        }
        dedupe_key = (
            flagged_page["page_id"],
            flagged_page["numero_processo"],
            tuple(flagged_page["reasons"]),
        )
        existing_keys = entry.setdefault("_seen_keys", set())
        if dedupe_key in existing_keys:
            continue
        existing_keys.add(dedupe_key)
        entry["flagged_pages"].append(flagged_page)

    for entry in candidates.values():
        entry.pop("_seen_keys", None)
        entry["flagged_pages"].sort(
            key=lambda item: (-int(item.get("risk_score") or 0), item.get("numero_processo") or "", item.get("page_id") or "")
        )
    return candidates


def _find_year_entry(summary: dict[str, Any], year: int) -> dict[str, Any] | None:
    for entry in summary.get("years", []):
        if isinstance(entry, dict) and int(entry.get("year") or 0) == year:
            return entry
    return None


def _completed_video_ids(year_entry: dict[str, Any]) -> set[str]:
    completed: set[str] = set()
    for entry in year_entry.get("corrected_videos", []):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status") or "done") != "done":
            continue
        video_id = str(entry.get("video_id") or "").strip()
        if video_id:
            completed.add(video_id)
    return completed


def _persisted_candidate_map(year_entry: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not year_entry:
        return {}
    persisted: dict[str, dict[str, Any]] = {}
    for collection_name in ("candidate_videos", "failed_videos"):
        for entry in year_entry.get(collection_name, []):
            if not isinstance(entry, dict):
                continue
            video_id = str(entry.get("video_id") or "").strip()
            if not video_id:
                continue
            target = persisted.setdefault(
                video_id,
                {
                    "video_id": video_id,
                    "video_title": str(entry.get("video_title") or ""),
                    "risk_level": str(entry.get("risk_level") or "medium"),
                    "risk_score": int(entry.get("risk_score") or 0),
                    "flagged_pages": [],
                },
            )
            target["risk_score"] = max(int(target.get("risk_score") or 0), int(entry.get("risk_score") or 0))
            if RISK_PRIORITY.get(str(entry.get("risk_level") or ""), 0) > RISK_PRIORITY.get(str(target.get("risk_level") or ""), 0):
                target["risk_level"] = str(entry.get("risk_level") or "medium")
            flagged_pages = entry.get("flagged_pages") or []
            if isinstance(flagged_pages, list):
                target["flagged_pages"].extend(page for page in flagged_pages if isinstance(page, dict))
    return persisted


def _merge_candidate_maps(*candidate_maps: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for candidate_map in candidate_maps:
        for video_id, entry in candidate_map.items():
            target = merged.setdefault(
                video_id,
                {
                    "video_id": video_id,
                    "video_title": str(entry.get("video_title") or ""),
                    "risk_level": str(entry.get("risk_level") or "medium"),
                    "risk_score": int(entry.get("risk_score") or 0),
                    "flagged_pages": [],
                },
            )
            target["risk_score"] = max(int(target.get("risk_score") or 0), int(entry.get("risk_score") or 0))
            if RISK_PRIORITY.get(str(entry.get("risk_level") or ""), 0) > RISK_PRIORITY.get(str(target.get("risk_level") or ""), 0):
                target["risk_level"] = str(entry.get("risk_level") or "medium")
            seen = {
                (
                    str(item.get("page_id") or ""),
                    str(item.get("numero_processo") or ""),
                    tuple(str(reason) for reason in (item.get("reasons") or [])),
                )
                for item in target["flagged_pages"]
                if isinstance(item, dict)
            }
            for page in entry.get("flagged_pages") or []:
                if not isinstance(page, dict):
                    continue
                marker = (
                    str(page.get("page_id") or ""),
                    str(page.get("numero_processo") or ""),
                    tuple(str(reason) for reason in (page.get("reasons") or [])),
                )
                if marker in seen:
                    continue
                seen.add(marker)
                target["flagged_pages"].append(page)
    for entry in merged.values():
        entry["flagged_pages"].sort(
            key=lambda item: (-int(item.get("risk_score") or 0), item.get("numero_processo") or "", item.get("page_id") or "")
        )
    return merged


def _ensure_year_snapshots(*, year_root: Path, notion_schema: Any, existing_pages_by_video: dict[str, Any]) -> None:
    if not (year_root / "_schema_snapshot.json").exists():
        dump_schema_snapshot(year_root, notion_schema)
    if not (year_root / "_existing_pages_snapshot.json").exists():
        dump_existing_pages_snapshot(year_root, existing_pages_by_video)


def _set_worker_timeouts(video_timeout_seconds: int, no_progress_timeout_seconds: int) -> None:
    backfill.VIDEO_WORKER_TIMEOUT_SECONDS = max(1, int(video_timeout_seconds))
    backfill.NO_PROGRESS_TIMEOUT_SECONDS = max(1, int(no_progress_timeout_seconds))


def _is_timeout_like_error(error_text: str) -> bool:
    text = str(error_text or "").lower()
    return "timeout de" in text or "sem progresso real de artefatos" in text


def _sync_active_videos(
    year_entry: dict[str, Any],
    active_workers: dict[str, tuple[object, CorrectiveTask]],
) -> None:
    active_payload: list[dict[str, Any]] = []
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
                "risk_level": task.risk_level,
                "risk_score": task.risk_score,
                "video_timeout_seconds": task.current_video_timeout_seconds,
                "no_progress_timeout_seconds": task.current_no_progress_timeout_seconds,
            }
        )
    year_entry["active_videos"] = active_payload


def _run_corrective_queue(
    *,
    summary: dict[str, Any],
    summary_path: Path,
    year: int,
    year_entry: dict[str, Any],
    tasks: list[CorrectiveTask],
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
    pending_tasks: deque[CorrectiveTask] = deque(tasks)
    active_workers: dict[str, tuple[object, CorrectiveTask]] = {}
    year_had_errors = False

    while pending_tasks or active_workers:
        summary_changed = False
        while pending_tasks and len(active_workers) < max(1, int(max_workers)):
            task = pending_tasks.popleft()
            task.attempt += 1
            task.current_video_timeout_seconds = max(1, int(base_video_timeout_seconds)) * (2 ** (task.attempt - 1))
            task.current_no_progress_timeout_seconds = max(1, int(base_no_progress_timeout_seconds)) * (2 ** (task.attempt - 1))
            _set_worker_timeouts(task.current_video_timeout_seconds, task.current_no_progress_timeout_seconds)
            print(
                f"[semantic-bleed-corrective] year={year} starting video_id={task.video_id} "
                f"risk={task.risk_level}/{task.risk_score} attempt={task.attempt} "
                f"limits=({task.current_video_timeout_seconds}s/{task.current_no_progress_timeout_seconds}s)",
                flush=True,
            )
            handle = start_worker_fn(task.video, worker_args, year_root)
            active_workers[task.video_id] = (handle, task)
            summary_changed = True

        completed_video_ids: list[str] = []
        for video_id, (handle, task) in list(active_workers.items()):
            _set_worker_timeouts(task.current_video_timeout_seconds, task.current_no_progress_timeout_seconds)
            polled = poll_worker_fn(handle)
            if polled.get("progress_changed"):
                summary_changed = True
            status = str(polled.get("status") or "")
            if status == "running":
                continue

            completed_video_ids.append(video_id)
            if status == "done":
                corrected = list(year_entry.get("corrected_videos") or [])
                corrected.append(
                    {
                        "video_id": video_id,
                        "video_title": getattr(task.video, "title", ""),
                        "risk_level": task.risk_level,
                        "risk_score": task.risk_score,
                        "flagged_pages": task.flagged_pages,
                        "status": "done",
                        "attempts": task.attempt,
                        "summary": polled.get("summary") or {},
                    }
                )
                year_entry["corrected_videos"] = corrected
                print(
                    f"[semantic-bleed-corrective] year={year} done video_id={video_id} attempts={task.attempt}",
                    flush=True,
                )
                summary_changed = True
                continue

            error_text = str(polled.get("error") or "")
            if _is_timeout_like_error(error_text) and task.attempt <= max(0, int(retry_timeouts)):
                print(
                    f"[semantic-bleed-corrective] year={year} retrying video_id={video_id} "
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
                    "video_title": getattr(task.video, "title", ""),
                    "risk_level": task.risk_level,
                    "risk_score": task.risk_score,
                    "flagged_pages": task.flagged_pages,
                    "status": "error",
                    "attempts": task.attempt,
                    "error": error_text,
                }
            )
            year_entry["failed_videos"] = failed
            year_had_errors = True
            summary_changed = True
            print(
                f"[semantic-bleed-corrective] year={year} error video_id={video_id} "
                f"attempts={task.attempt} error={error_text}",
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
        description="Reroda focalmente os vídeos sinalizados pela auditoria semantic_bleed."
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2025, 2024, 2023, 2022, 2021])
    parser.add_argument("--audit-run-root")
    parser.add_argument("--resume-run-root")
    parser.add_argument("--skip-news", action="store_true", default=True)
    parser.add_argument("--video-timeout-seconds", type=int, default=2400)
    parser.add_argument("--no-progress-timeout-seconds", type=int, default=600)
    parser.add_argument("--retry-timeouts", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--min-risk-level", choices=["low", "medium", "high"], default="medium")
    args = parser.parse_args(argv)

    args.max_workers = max(1, int(args.max_workers))
    _set_worker_timeouts(args.video_timeout_seconds, args.no_progress_timeout_seconds)

    audit_run_root = Path(args.audit_run_root) if args.audit_run_root else _latest_audit_run_root()

    if args.resume_run_root:
        run_root = Path(args.resume_run_root)
        summary_path = run_root / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Resumo do run não encontrado em {summary_path}")
        summary = _read_summary(summary_path)
        summary["status"] = "running"
        summary["audit_run_root"] = str(audit_run_root)
        print(f"[semantic-bleed-corrective] resuming run_root={run_root}", flush=True)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = RUNS_ROOT / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        summary_path = run_root / "summary.json"
        summary = {
            "run_root": str(run_root),
            "audit_run_root": str(audit_run_root),
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "years": [],
        }
        _write_summary(summary_path, summary)
        print(f"[semantic-bleed-corrective] run_root={run_root}", flush=True)

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
            print(f"[semantic-bleed-corrective] year={year} already done; skipping", flush=True)
            continue

        playlist_url = _load_playlist_url(year)
        persisted_candidates = _persisted_candidate_map(year_entry)
        audit_candidates = _collect_candidates_from_audit_report(
            audit_run_root,
            year,
            min_risk_level=args.min_risk_level,
        )
        corrective_candidates = _merge_candidate_maps(persisted_candidates, audit_candidates)
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
                "corrected_videos": [],
                "failed_videos": [],
                "active_videos": [],
                "started_at": datetime.now().isoformat(),
            }
            summary_years = list(summary.get("years") or [])
            summary_years.append(year_entry)
            summary["years"] = summary_years
            _write_summary(summary_path, summary)
        else:
            year_entry["playlist_url"] = playlist_url
            year_entry["status"] = "running"
            year_entry.setdefault("candidate_videos", [])
            year_entry.setdefault("corrected_videos", [])
            year_entry.setdefault("failed_videos", [])
            year_entry["active_videos"] = []
            year_entry.setdefault("started_at", datetime.now().isoformat())
            _write_summary(summary_path, summary)

        ordered_tasks: list[CorrectiveTask] = []
        for video_id, candidate in corrective_candidates.items():
            video = find_target_video(playlist_url, year, video_id)
            ordered_tasks.append(
                CorrectiveTask(
                    video=video,
                    video_id=video_id,
                    flagged_pages=list(candidate.get("flagged_pages") or []),
                    risk_level=str(candidate.get("risk_level") or "medium"),
                    risk_score=int(candidate.get("risk_score") or 0),
                )
            )
        ordered_tasks.sort(
            key=lambda item: (
                -RISK_PRIORITY.get(item.risk_level, 0),
                -item.risk_score,
                item.video.position,
            )
        )
        completed_ids = _completed_video_ids(year_entry)
        year_entry["candidate_videos"] = [
            {
                "video_id": task.video_id,
                "video_title": task.video.title,
                "risk_level": task.risk_level,
                "risk_score": task.risk_score,
                "flagged_pages": task.flagged_pages,
            }
            for task in ordered_tasks
        ]
        _write_summary(summary_path, summary)
        print(
            f"[semantic-bleed-corrective] year={year} candidates={len(ordered_tasks)} "
            f"pending={sum(1 for task in ordered_tasks if task.video_id not in completed_ids)} "
            f"max_workers={args.max_workers}",
            flush=True,
        )
        worker_args = Namespace(
            playlist_url=playlist_url,
            year=year,
            skip_news=bool(args.skip_news),
            no_trash_unmatched_precedents=False,
        )
        pending_tasks = [task for task in ordered_tasks if task.video_id not in completed_ids]
        year_had_errors = _run_corrective_queue(
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
        _write_summary(summary_path, summary)
        print(f"[semantic-bleed-corrective] year={year} finished", flush=True)

    summary["status"] = "done_with_errors" if had_errors else "done"
    summary["finished_at"] = datetime.now().isoformat()
    _write_summary(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
