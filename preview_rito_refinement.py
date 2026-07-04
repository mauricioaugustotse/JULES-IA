# -*- coding: utf-8 -*-
"""Preview OFFLINE do refinamento pelo rito sobre artifacts existentes.

Carrega o 01_session_windows.json de um vídeo já processado, obtém a
transcrição (raw_transcript_fetch.json do próprio artifact-dir quando existir;
senão baixa via youtube_transcript_api) e imprime o diff janela a janela que a
camada do rito aplicaria. Não grava nada, não chama Gemini e não toca o Notion.

Uso:
  python preview_rito_refinement.py --artifact-dir <pasta NN_videoId> [--video-id X | --url URL] [--out saida.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tse_youtube_notion_core import (
    SessionExtraction,
    TranscriptSnippet,
    count_individual_apregoamentos,
    detect_rito_events,
    extract_youtube_video_id,
    refine_session_windows_with_rito,
    require_youtube_transcript_api,
)


def load_snippets(artifact_dir: Path, video_id: str) -> list[TranscriptSnippet]:
    cached = artifact_dir / "raw_transcript_fetch.json"
    if cached.exists():
        payload = json.loads(cached.read_text(encoding="utf-8"))
        return [
            TranscriptSnippet(
                text=str(item.get("text", "")),
                start_seconds=int(item.get("start_seconds", 0)),
                end_seconds=int(item.get("end_seconds", 0)),
            )
            for item in payload.get("snippets", [])
        ]
    if not video_id:
        raise SystemExit("Sem raw_transcript_fetch.json no artifact-dir; informe --video-id ou --url.")
    api_cls = require_youtube_transcript_api()
    fetched = api_cls().fetch(video_id, languages=["pt-BR", "pt", "en"])
    snippets: list[TranscriptSnippet] = []
    for item in fetched:
        start = int(getattr(item, "start", 0))
        duration = getattr(item, "duration", 0) or 0
        snippets.append(
            TranscriptSnippet(
                text=str(getattr(item, "text", "")),
                start_seconds=start,
                end_seconds=int(start + duration),
            )
        )
    return snippets


def describe_window(window) -> str:
    flag = "IGNORADA" if window.should_ignore else "ativa"
    nums = ",".join(window.mentioned_process_numbers) or "-"
    end = window.end_seconds if window.end_seconds is not None else "?"
    reason = f" [{window.ignore_reason}]" if window.ignore_reason else ""
    return f"t={window.start_seconds:>6}-{end:<6} {flag:<8} nums={nums:<40} {window.title_hint!r}{reason}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--video-id", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    windows_path = artifact_dir / "01_session_windows.json"
    if not windows_path.exists():
        raise SystemExit(f"01_session_windows.json não encontrado em {artifact_dir}")
    session = SessionExtraction.model_validate(json.loads(windows_path.read_text(encoding="utf-8")))

    video_id = args.video_id or (extract_youtube_video_id(args.url) if args.url else "")
    if not video_id:
        # convenção NN_<videoId> do batch e NNN_<videoId> do backfill
        stem = artifact_dir.name
        video_id = stem.split("_", 1)[1] if "_" in stem else ""

    snippets = load_snippets(artifact_dir, video_id)
    events = detect_rito_events(snippets)
    refined, report = refine_session_windows_with_rito(session, events)

    print(f"Transcrição: {len(snippets)} snippets | eventos do rito: {len(events)}")
    for event in events:
        print(f"  t={event.start_seconds:>6} {event.kind:<12} {event.text[:80]!r}")
    print(f"\nApregoamentos individuais: {count_individual_apregoamentos(events)}")
    print(f"\nJanelas ANTES ({len(session.judgments)}):")
    for window in session.judgments:
        print("  " + describe_window(window))
    print(f"\nJanelas DEPOIS ({len(refined.judgments)}):")
    for window in refined.judgments:
        print("  " + describe_window(window))
    print("\nAjustes:")
    if not report.get("adjustments"):
        print("  (nenhum)")
    for adjustment in report.get("adjustments", []):
        print(f"  {adjustment['type']:<15} t={adjustment.get('start_seconds')} {adjustment.get('reason','')}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "report": report,
                    "windows_after": [w.model_dump(mode="json") for w in refined.judgments],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"\nRelatório gravado em {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
