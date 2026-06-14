"""Reprocessa video(s) especificos pelo MESMO fluxo do lote 10 (process_video_batch),
re-extraindo pela IA (via video). Util quando uma sessao foi processada cedo demais
(video recem-saido de transmissao ao vivo, que o Gemini ainda nao conseguia baixar) e
caiu no fallback de transcricao, saindo rasa. Quando o YouTube estabiliza o VOD, o
Gemini volta a processar o video e a extracao fica completa.

Uso:
    python reprocess_videos.py g_95kQsi4fQ vVi0c0lOgo4
    python reprocess_videos.py g_95kQsi4fQ --no-news
"""
from __future__ import annotations

import argparse
import logging
import queue
import threading

from tse_youtube_notion_batch_gui import BatchOptions, VideoInput, process_video_batch

LOGGER = logging.getLogger("reprocess_videos")


def drain(q, stop):
    while not stop.is_set():
        try:
            kind, *rest = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if kind == "log":
            print(rest[0], end="")


def main() -> int:
    p = argparse.ArgumentParser(description="Reprocessa videos especificos via fluxo do lote 10 (re-extrai por video).")
    p.add_argument("video_ids", nargs="+", help="video_id(s) do YouTube a reprocessar")
    p.add_argument("--no-news", action="store_true")
    p.add_argument("--no-post", action="store_true")
    p.add_argument("--allow-transcript", action="store_true",
                   help="Permite o fallback de transcricao se o video nao processar (default: EXIGE video).")
    p.add_argument("--dry-run", action="store_true", help="Processa mas NAO publica.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    videos = [
        VideoInput(position=i, url=f"https://www.youtube.com/watch?v={vid}", video_id=vid)
        for i, vid in enumerate(args.video_ids, start=1)
    ]
    options = BatchOptions(
        model="gemini-3.1-flash-lite",
        news_model="gemini-3.1-flash-lite",
        with_news=not args.no_news,
        publish=not args.dry_run,
        continue_on_error=True,
        post_publish_steps=() if (args.no_post or args.dry_run) else ("materia", "suspenso", "classe_nomes", "sanear"),
        recolor_labels=False,
        watch_dje=False,
        allow_transcript_fallback=args.allow_transcript,
    )
    LOGGER.info("Reprocessando %s video(s) via VIDEO: %s", len(videos), [v.video_id for v in videos])
    q: "queue.Queue" = queue.Queue()
    stop = threading.Event()
    t = threading.Thread(target=drain, args=(q, stop), daemon=True)
    t.start()
    try:
        result = process_video_batch(videos, options, q, threading.Event(), analysis_provider=None)
    finally:
        stop.set()
    LOGGER.info("CONCLUIDO: %s ok, %s erro | artifacts: %s",
                result.get("total_done"), result.get("total_error"), result.get("artifact_dir"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
