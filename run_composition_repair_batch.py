from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BACKFILL_ROOT = REPO_ROOT / "artifacts" / "tse_youtube_notion" / "backfill_2025"
RUNS_ROOT = REPO_ROOT / "artifacts" / "tse_youtube_notion" / "composition_repair_runs"


def _load_playlist_url(year: int) -> str:
    manifests = sorted(BACKFILL_ROOT.glob(f"{year}_PL*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Manifesto do ano {year} não encontrado em {BACKFILL_ROOT}")
    with manifests[0].open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    playlist_url = str(payload.get("playlist_url") or payload.get("playlist") or "").strip()
    if not playlist_url:
        raise ValueError(f"Manifesto do ano {year} não contém playlist_url")
    return playlist_url


def _write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Executa reparo em lote de composicao inválida (lt6/gt7).")
    parser.add_argument("--years", type=int, nargs="+", default=[2025, 2024, 2023, 2022])
    args = parser.parse_args(argv)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = RUNS_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    summary_path = run_root / "summary.json"
    summary: dict[str, object] = {
        "run_root": str(run_root),
        "started_at": datetime.now().isoformat(),
        "status": "running",
        "years": [],
    }
    _write_summary(summary_path, summary)

    for year in args.years:
        playlist_url = _load_playlist_url(year)
        year_log = run_root / f"{year}.log"
        command = [
            sys.executable,
            "tse_backfill_2025_notion.py",
            "--playlist-url",
            playlist_url,
            "--year",
            str(year),
            "--repair-existing-year",
            "--repair-focus",
            "composition",
        ]
        year_entry = {
            "year": year,
            "playlist_url": playlist_url,
            "status": "running",
            "log": str(year_log),
            "command": command,
            "started_at": datetime.now().isoformat(),
        }
        cast_years = list(summary["years"])  # type: ignore[arg-type]
        cast_years.append(year_entry)
        summary["years"] = cast_years
        _write_summary(summary_path, summary)
        with year_log.open("w", encoding="utf-8") as handle:
            handle.write(f"[command] {' '.join(command)}\n")
            handle.flush()
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        year_entry["finished_at"] = datetime.now().isoformat()
        year_entry["returncode"] = result.returncode
        year_entry["status"] = "done" if result.returncode == 0 else "error"
        _write_summary(summary_path, summary)
        if result.returncode != 0:
            summary["status"] = "error"
            summary["finished_at"] = datetime.now().isoformat()
            _write_summary(summary_path, summary)
            return result.returncode

    summary["status"] = "done"
    summary["finished_at"] = datetime.now().isoformat()
    _write_summary(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
