from __future__ import annotations

import argparse
import json
import subprocess
import time
from argparse import Namespace
from pathlib import Path

from tse_backfill_2025_notion import (
    build_runtime_context,
    dump_existing_pages_snapshot,
    dump_schema_snapshot,
    find_target_video,
    load_existing_pages_for_year_with_retry,
    run_video_worker,
)
from tse_youtube_notion_core import ARTIFACT_ROOT, NotionSessoesClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda um vídeo-alvo e, em seguida, dispara identity-core em lote.")
    parser.add_argument("--playlist-url", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--skip-news", action="store_true", default=True)
    parser.add_argument("--identity-years", nargs="+", default=["2025", "2024", "2023", "2022", "2021"])
    args = parser.parse_args()

    root_dir = ARTIFACT_ROOT / "targeted_video_then_identity" / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.video_id}"
    root_dir.mkdir(parents=True, exist_ok=True)
    print(f"[targeted-video] root_dir={root_dir}", flush=True)

    runtime = build_runtime_context()
    notion_client = NotionSessoesClient(
        api_key=runtime["notion_api_key"],
        data_source_id=runtime["notion_data_source_id"],
    )
    notion_schema = notion_client.fetch_schema()
    existing_pages_by_video = load_existing_pages_for_year_with_retry(
        notion_client,
        notion_schema,
        args.year,
        playlist_url=args.playlist_url,
    )
    dump_schema_snapshot(root_dir, notion_schema)
    dump_existing_pages_snapshot(root_dir, existing_pages_by_video)

    video = find_target_video(args.playlist_url, args.year, args.video_id)
    worker_args = Namespace(
        playlist_url=args.playlist_url,
        year=args.year,
        skip_news=args.skip_news,
        no_trash_unmatched_precedents=False,
    )
    summary = run_video_worker(
        video=video,
        args=worker_args,
        root_dir=root_dir,
        progress_heartbeat=lambda *_args, **_kwargs: None,
    )
    print(
        f"[targeted-video] finished video_id={args.video_id} "
        f"created={summary.get('created')} updated={summary.get('updated')} "
        f"blocked={summary.get('blocked')} skipped={summary.get('skipped')}",
        flush=True,
    )
    (root_dir / "video_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parent
    print(f"[identity-core] starting years={' '.join(args.identity_years)}", flush=True)
    subprocess.run(
        ["python3", "run_identity_core_batch.py", "--years", *args.identity_years],
        cwd=repo_root,
        check=True,
    )
    print(f"[identity-core] finished years={' '.join(args.identity_years)}", flush=True)
    print(f"[identity-replay] starting years={' '.join(args.identity_years)}", flush=True)
    subprocess.run(
        ["python3", "run_identity_replay_batch.py", "--years", *args.identity_years],
        cwd=repo_root,
        check=True,
    )
    print(f"[identity-replay] finished years={' '.join(args.identity_years)}", flush=True)


if __name__ == "__main__":
    main()
