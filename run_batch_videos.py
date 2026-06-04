"""Runner headless do lote de videos TSE (sem GUI): chama process_video_batch e imprime o
progresso/resumo. Reutiliza EXATAMENTE o fluxo da batch GUI (client com escrita segura,
normalize_multiselect_colors_post_write=False).

Uso:
    python run_batch_videos.py URL1 URL2 ...
    python run_batch_videos.py --no-publish URL1        # so analisa (nao escreve no Notion)
    python run_batch_videos.py --no-news URL1 URL2
"""
from __future__ import annotations

import argparse
import logging
import queue
import threading

from tse_youtube_notion_batch_gui import BatchOptions, normalize_video_input, process_video_batch
from tse_youtube_notion_core import DEFAULT_GEMINI_MODEL, DEFAULT_NEWS_GEMINI_MODEL


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="+", help="Links do YouTube (live/watch).")
    ap.add_argument("--no-publish", action="store_true", help="Nao publica no Notion (so analisa).")
    ap.add_argument("--no-news", action="store_true", help="Nao enriquece com noticias.")
    ap.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    ap.add_argument("--news-model", default=DEFAULT_NEWS_GEMINI_MODEL)
    ap.add_argument("--parse-only", action="store_true", help="So valida os links e sai.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    videos = [normalize_video_input(i + 1, u) for i, u in enumerate(args.urls)]
    for v in videos:
        logging.getLogger().info("link OK: pos=%s id=%s url=%s", v.position, v.video_id, v.url)
    if args.parse_only:
        print("PARSE OK:", [v.video_id for v in videos], flush=True)
        return 0

    options = BatchOptions(
        model=args.model, news_model=args.news_model,
        with_news=not args.no_news, publish=not args.no_publish, continue_on_error=True,
    )
    q: "queue.Queue" = queue.Queue()
    stop = threading.Event()
    logging.getLogger().info("Iniciando lote: %s videos | publish=%s news=%s | modelo=%s",
                             len(videos), options.publish, options.with_news, options.model)
    result = process_video_batch(videos, options, q, stop)

    print("\n==== RESUMO DO LOTE ====", flush=True)
    print(f"solicitados={result.get('total_requested')} done={result.get('total_done')} erro={result.get('total_error')}", flush=True)
    for v in result.get("videos", []):
        if v.get("status") == "done":
            print(f"  [{v['position']:02d}] {v['video_id']} -> linhas={v['rows_extracted']} "
                  f"criadas={v['created']} atualizadas={v['updated']} bloqueadas={v['blocked']} ignoradas={v['skipped']}", flush=True)
        else:
            print(f"  [{v.get('position')}] {v.get('video_id')} -> ERRO: {str(v.get('error', ''))[:250]}", flush=True)
    print("artifacts:", result.get("artifact_dir"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
