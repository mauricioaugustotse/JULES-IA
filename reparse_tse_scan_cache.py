"""Recover parsed scan artifacts from retained raw responses, entirely offline.

Original artifacts are copied into an immutable-by-convention snapshot. The new
coverage report describes successful parsing/scan-window checks, never verified
judgment content. This utility does not generate publication rows or call APIs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import tse_youtube_notion_core as core


def reparse_scan_cache(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Use a new output directory outside the source run.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    original = core.RunArtifacts(source_dir).read_json("00_scan_coverage.json")
    duration = int(original["duration_seconds"])
    if duration <= 0 or not original.get("attempts"):
        raise ValueError("Source lacks a usable scan-attempt manifest.")
    output_dir.mkdir(parents=True)
    snapshot = output_dir / "original_snapshot"
    shutil.copytree(source_dir, snapshot)
    store = core.RunArtifacts(output_dir)
    for filename in ("00_scan_windows.json", "00_chapter_inventory.json"):
        if (snapshot / filename).exists():
            shutil.copy2(snapshot / filename, output_dir / filename)
    inventory = store.read_json("00_chapter_inventory.json") if store.exists("00_chapter_inventory.json") else {}
    chapters = inventory.get("chapters") or [] if inventory.get("status") == "available" else []
    attempts = []
    changes = []
    input_hashes = {}
    for old_attempt in original["attempts"]:
        if old_attempt.get("source", "video") != "video":
            continue
        attempt = deepcopy(old_attempt)
        prefix = attempt["artifact_prefix"]
        attempt["reparsed_from_raw"] = True
        for index, window in enumerate(attempt["windows"], 1):
            start, end = int(window["start_seconds"]), int(window["end_seconds"])
            stem = f"{prefix}_chunk_{index:02d}"
            raw_file = snapshot / f"{stem}.txt"
            previous_status = window.get("status")
            window.clear()
            window.update(start_seconds=start, end_seconds=end, status="pending")
            if not raw_file.exists():
                window.update(status="failed", error="Retained raw response is missing; no online request made.")
                continue
            raw_bytes = raw_file.read_bytes()
            input_hashes[raw_file.name] = hashlib.sha256(raw_bytes).hexdigest()
            shutil.copy2(raw_file, output_dir / raw_file.name)
            try:
                parsed = core._coerce_gemini_response_model(core.SessionExtraction, raw_bytes.decode("utf-8-sig"))
                degeneration = core.scan_chunk_degeneration_report(
                    parsed.judgments, window_start_seconds=start, window_end_seconds=end,
                )
                kept, discards = core.sanitize_scan_chunk_windows(
                    parsed.judgments, window_start_seconds=start, window_end_seconds=end,
                    duration_seconds=duration,
                )
                conflicts = core.scan_chunk_chapter_conflicts(
                    kept, chapters=chapters, window_start_seconds=start, window_end_seconds=end,
                )
                rejected = bool(conflicts or degeneration["suspeito"] or (
                    discards["blocos_descartados"] and not kept
                ))
                for judgment in kept:
                    judgment.scan_window = [start, end]
                parsed.judgments = kept
                store.write_json(f"{stem}.json", parsed.model_dump(mode="json"))
                store.write_json(f"{stem}.descartes.json", {
                    "plan": attempt["label"], "degeneracao": degeneration, **discards,
                })
                if conflicts:
                    store.write_json(f"{stem}.chapter_conflicts.json", {"conflicts": conflicts})
                window.update(
                    status="rejected" if rejected else "complete",
                    discarded_windows=discards["blocos_descartados"], chapter_conflicts=conflicts,
                )
            except (ValueError, TypeError, KeyError) as exc:
                window.update(status="failed", error=f"Offline parse failed: {exc}")
                store.write_json(f"{stem}.error.json", {"error": window["error"]})
            if window["status"] != previous_status:
                changes.append({
                    "plan": attempt["label"], "chunk": index, "window": [start, end],
                    "previous_status": previous_status, "new_status": window["status"],
                })
        attempts.append(attempt)
    coverage = core._scan_coverage_payload(duration, attempts)
    coverage.update(reparsed_from_raw=True, content_verified=False)
    store.write_json("00_scan_coverage.json", coverage)
    primary = [attempt for attempt in attempts if attempt["label"] == "primary"]
    primary_coverage = core._scan_coverage_payload(duration, primary)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir), "original_snapshot": str(snapshot),
        "mode": "offline_raw_response_reparse", "api_calls": 0, "notion_writes": 0,
        "content_verified": False, "publication_ready": False,
        "temporal_coverage_status": coverage["status"],
        "previous_uncovered_intervals": original.get("uncovered_intervals"),
        "uncovered_intervals": coverage["uncovered_intervals"],
        "primary_plan_temporal_coverage": primary_coverage["status"],
        "preferred_plan_for_later_review": "primary" if primary_coverage["status"] == "complete" else "combined",
        "changes": changes, "raw_input_sha256": input_hashes,
        "review_required": [
            "Verify actual judged processes and composition against official session sources and the video.",
            "Do not treat a parsed timestamp or a process reference as proof that the case was judged.",
            "Do not merge extra fallback cases when the primary plan already covers the entire video.",
        ],
    }
    store.write_json("scan_reparse_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = reparse_scan_cache(args.source, args.output)
    print(json.dumps({key: result[key] for key in (
        "mode", "api_calls", "notion_writes", "content_verified", "publication_ready",
        "temporal_coverage_status", "uncovered_intervals", "changes",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
