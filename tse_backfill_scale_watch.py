from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tse_backfill_monitor import DEFAULT_MANIFEST, load_manifest


DEFAULT_THRESHOLDS = [6, 8, 10, 12, 14, 16, 18, 20]
DEFAULT_STATE_PATH = Path("artifacts/tse_youtube_notion/backfill_2025/scale_watch_state.json")
DEFAULT_LOG_PATH = Path("artifacts/tse_youtube_notion/backfill_2025/scale_watch.log")


def parse_thresholds(value: str) -> list[int]:
    thresholds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        thresholds.append(int(part))
    unique = sorted({item for item in thresholds if item >= 1})
    return unique or DEFAULT_THRESHOLDS[:]


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"seen_thresholds": [], "last_target": None, "events": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def build_event_line(*, target: int, manifest: dict[str, Any], threshold: int) -> str:
    eta_at = manifest.get("eta_at") or "-"
    updated_at = manifest.get("updated_at") or datetime.now().isoformat(timespec="seconds")
    counts = {"done": 0, "running": 0, "pending": 0, "error": 0}
    for item in (manifest.get("videos") or {}).values():
        status = item.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return (
        f"{updated_at} | threshold={threshold} | target={target} | "
        f"done={counts['done']} running={counts['running']} pending={counts['pending']} error={counts['error']} | "
        f"eta_at={eta_at}"
    )


def process_snapshot(
    *,
    manifest: dict[str, Any],
    thresholds: list[int],
    state: dict[str, Any],
    log_path: Path,
) -> list[str]:
    target = int(manifest.get("current_target_workers") or 0)
    seen_thresholds = {int(item) for item in state.get("seen_thresholds", [])}
    emitted: list[str] = []

    for threshold in thresholds:
        if target >= threshold and threshold not in seen_thresholds:
            line = build_event_line(target=target, manifest=manifest, threshold=threshold)
            append_log(log_path, line)
            emitted.append(line)
            seen_thresholds.add(threshold)

    state["seen_thresholds"] = sorted(seen_thresholds)
    state["last_target"] = target
    state["last_updated_at"] = manifest.get("updated_at") or ""
    state["events"] = (state.get("events") or [])[-50:] + emitted
    return emitted


def main() -> None:
    parser = argparse.ArgumentParser(description="Observa marcos de autoescala do backfill 2025.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--log", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--thresholds", default="6,8,10,12,14,16,18,20")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    state_path = Path(args.state)
    log_path = Path(args.log)
    thresholds = parse_thresholds(args.thresholds)

    if not manifest_path.exists():
        raise SystemExit(f"Manifest não encontrado: {manifest_path}")

    state = load_state(state_path)

    if not args.watch:
        manifest = load_manifest(manifest_path)
        emitted = process_snapshot(manifest=manifest, thresholds=thresholds, state=state, log_path=log_path)
        save_state(state_path, state)
        print(json.dumps({"emitted": emitted, "state": state}, ensure_ascii=False, indent=2))
        return

    try:
        while True:
            manifest = load_manifest(manifest_path)
            emitted = process_snapshot(manifest=manifest, thresholds=thresholds, state=state, log_path=log_path)
            save_state(state_path, state)
            for line in emitted:
                print(line, flush=True)
            time.sleep(max(args.interval, 1.0))
    except KeyboardInterrupt:
        save_state(state_path, state)


if __name__ == "__main__":
    main()
