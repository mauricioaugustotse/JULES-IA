"""Republica/reprocessa no Notion os dias de sessao que foram processados pela IA
mas nao foram parar na base, CALIBRANDO pelo MESMO fluxo do lote 10
(``process_video_batch`` da GUI): capitulos do YouTube, data pelo titulo, CNJ,
metadados, tema/punchline, noticias, publish e pos-publicacao (materia/suspenso/
classe_nomes/sanear).

- Grupo 2 ("publicou mas sumiu"): INJETA a IA bruta ja presente nos artefatos do
  backfill (judgments) -> nao re-extrai pelo Gemini, mas passa por todas as demais
  etapas de padronizacao.
- Grupo 1 ("erro"): reprocessa do zero (extrai pela IA).

Sempre verifica o Notion antes: dias cujo video ja esta publicado sao PULADOS.

Uso:
    python republish_missing_days.py --group 2 --years 2024 --limit 1 --no-news --no-post   # piloto
    python republish_missing_days.py --group 2                                              # grupo 2 todo
    python republish_missing_days.py --group 1 --skip-video-ids hr05rWalyGk                  # grupo 1
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import queue
import threading
from typing import Any, Optional

from local_secrets import get_secret
from tse_youtube_notion_core import (
    AnalysisResult,
    JudgmentBundleExtraction,
    NotionSessoesClient,
    SessionExtraction,
    build_runtime_context,
    extract_youtube_video_id,
)
from tse_youtube_notion_batch_gui import BatchOptions, VideoInput, process_video_batch

LOGGER = logging.getLogger("republish_missing_days")
BF = pathlib.Path("H:/Meu Drive/TSE_YOUTUBE_NOTION_BACKLOG/backfill_2025")
CSV_PATH = pathlib.Path("dias_faltantes_notion.csv")

GROUP_PREFIX = {1: "1_error", 2: "2_publicou", 3: "3_nada", 4: "4_sem"}


def load_analysis_from_artifacts(playlist: str, video_id: str) -> Optional[AnalysisResult]:
    """Reconstroi o AnalysisResult (IA bruta) a partir dos artefatos do backfill."""
    candidates: list[pathlib.Path] = []
    base = BF / playlist
    if base.exists():
        candidates.extend(p for p in base.glob(f"*_{video_id}") if p.is_dir())
    if not candidates:
        candidates.extend(p for p in BF.glob(f"*/*_{video_id}") if p.is_dir())
    for vdir in candidates:
        sw = vdir / "01_session_windows.json"
        if not sw.exists():
            continue
        bundles = []
        for f in sorted(vdir.glob("02_judgment_*.json")):
            try:
                bundles.append(JudgmentBundleExtraction.model_validate(json.loads(f.read_text(encoding="utf-8"))))
            except Exception:
                pass
        if not bundles:
            continue
        try:
            session = SessionExtraction.model_validate(json.loads(sw.read_text(encoding="utf-8")))
        except Exception:
            continue
        return AnalysisResult(session=session, bundles=bundles)
    return None


def video_in_notion(client: NotionSessoesClient, video_id: str) -> bool:
    try:
        hits = client.query_data_source(filter_payload={"property": "youtube_link", "url": {"contains": video_id}})
        return len(hits) > 0
    except Exception as exc:
        LOGGER.warning("Falha ao verificar %s no Notion (assumindo ausente): %s", video_id, exc)
        return False


def drain_queue(q: "queue.Queue[tuple[str, Any]]", stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            kind, *rest = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if kind == "log":
            print(rest[0], end="")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Republica/reprocessa dias faltantes calibrando pelo fluxo do lote 10.")
    p.add_argument("--group", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--years", default="all", help="'all' ou lista separada por virgula, ex.: 2024,2025")
    p.add_argument("--limit", type=int, default=0, help="Limita quantos dias processar (0 = todos).")
    p.add_argument("--skip-video-ids", default="", help="video_ids a pular (virgula).")
    p.add_argument("--no-news", action="store_true", help="Nao busca noticias (mais rapido/barato).")
    p.add_argument("--no-post", action="store_true", help="Pula a pos-publicacao (materia/suspenso/...).")
    p.add_argument("--allow-transcript", action="store_true",
                   help="Permite o fallback de transcricao se o video nao processar (default: EXIGE video).")
    p.add_argument("--dry-run", action="store_true", help="Processa e grava artefatos, mas NAO publica no Notion.")
    p.add_argument("--csv", default=str(CSV_PATH))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    years = None if args.years.strip().lower() == "all" else {y.strip() for y in args.years.split(",")}
    skip = {v.strip() for v in args.skip_video_ids.split(",") if v.strip()}
    prefix = GROUP_PREFIX[args.group]

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
    selected = []
    for r in rows:
        if not r["categoria"].startswith(prefix):
            continue
        if years is not None and (r["data_sessao"] or "")[:4] not in years:
            continue
        if r["video_id"] in skip:
            continue
        selected.append(r)
    selected.sort(key=lambda r: r["data_sessao"])
    if args.limit:
        selected = selected[: args.limit]

    LOGGER.info("Grupo %s | anos=%s | candidatos selecionados: %s", args.group, args.years, len(selected))
    if not selected:
        LOGGER.info("Nada a fazer."); return 0

    # verifica Notion (pula dias ja publicados)
    runtime = build_runtime_context()
    dsid = runtime["notion_data_source_id"]
    check_client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=dsid)

    analyses: dict[str, AnalysisResult] = {}
    videos: list[VideoInput] = []
    pulados_no_notion = 0
    sem_artefato = 0
    for i, r in enumerate(selected, start=1):
        vid = r["video_id"]
        if video_in_notion(check_client, vid):
            LOGGER.info("  PULA (ja no Notion): %s %s", r["data_sessao"], vid)
            pulados_no_notion += 1
            continue
        if args.group in (2, 3):
            analysis = load_analysis_from_artifacts(r["playlist"], vid)
            if analysis is not None:
                analyses[vid] = analysis
            elif args.group == 2:
                LOGGER.warning("  SEM artefato reconstruivel: %s %s -- pulando", r["data_sessao"], vid)
                sem_artefato += 1
                continue
            else:  # grupo 3 sem artefato: re-extrai pela IA (analysis=None)
                LOGGER.info("  sem artefato -> re-extrai pela IA: %s %s", r["data_sessao"], vid)
        videos.append(VideoInput(position=i, url=f"https://www.youtube.com/watch?v={vid}", video_id=vid))

    LOGGER.info("A processar: %s | pulados (ja no Notion): %s | sem artefato: %s",
                len(videos), pulados_no_notion, sem_artefato)
    for v in videos:
        d = next(r["data_sessao"] for r in selected if r["video_id"] == v.video_id)
        LOGGER.info("   -> %s  %s", d, v.video_id)
    if not videos:
        return 0

    options = BatchOptions(
        model=runtime.get("gemini_model", "gemini-3.1-flash-lite"),
        news_model=runtime.get("news_gemini_model", "gemini-3.1-flash-lite"),
        with_news=not args.no_news,
        publish=not args.dry_run,
        continue_on_error=True,
        post_publish_steps=() if (args.no_post or args.dry_run) else ("materia", "suspenso", "classe_nomes", "sanear"),
        recolor_labels=False,   # etiquetas Playwright exigem Edge interativo
        watch_dje=False,        # DJE exige CSVs/pasta especifica
        allow_transcript_fallback=args.allow_transcript,
    )

    provider = (lambda v: analyses.get(v.video_id)) if args.group in (2, 3) else None
    q: "queue.Queue[tuple[str, Any]]" = queue.Queue()
    stop = threading.Event()
    drainer = threading.Thread(target=drain_queue, args=(q, stop), daemon=True)
    drainer.start()
    try:
        result = process_video_batch(videos, options, q, threading.Event(), analysis_provider=provider)
    finally:
        stop.set()

    done = result.get("total_done", 0)
    err = result.get("total_error", 0)
    LOGGER.info("CONCLUIDO: %s ok, %s erro | artifacts: %s", done, err, result.get("artifact_dir"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
