# -*- coding: utf-8 -*-
"""Auditoria retroativa dos julgamentos que ficaram FORA do Notion.

Varre os artifacts históricos do fluxo TSE YouTube→Notion — lotes do batch_gui
(05_publish_results.json + 04b) e o backlog de playlists no Google Drive
(07_backfill_summary.json) — e:
  1. classifica cada item skipped/blocked por motivo (regex nos errors/warnings);
  2. marca CANDIDATOS a "julgamento real perdido" (CNJ-20 completo bloqueado,
     ou descarte pelo gate de precedente/grounding inconclusivo);
  3. opcionalmente (--with-transcript-check) baixa a transcrição de cada vídeo,
     conta os apregoamentos individuais do rito (camada determinística do core)
     e compara com o nº de linhas do Notion por vídeo/data;
  4. grava relatório CSV/JSON e alimenta a fila de vistoria (source="auditoria").

Somente leitura no Notion; nada é publicado (correções passam pela fila).
Julgamentos "em lista" são ignorados por decisão de projeto (a contagem do rito
já exclui o bloco de listas).

Uso:
  python audit_retroactive_judgments.py --years 2026 --limit 8 --with-transcript-check
  python audit_retroactive_judgments.py --years 2024,2025,2026
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from tse_normalization import infer_session_date_from_video_title
from tse_youtube_notion_core import (
    ARTIFACT_ROOT,
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NotionSessoesClient,
    TranscriptSnippet,
    count_individual_apregoamentos,
    detect_rito_events,
    extract_youtube_video_id,
    require_youtube_transcript_api,
)
from local_secrets import get_secret

import vistoria_queue

LOGGER = logging.getLogger("audit_retroactive")

DEFAULT_BATCH_ROOT = ARTIFACT_ROOT / "batch_gui"
DEFAULT_BACKLOG_ROOT = Path(r"H:\Meu Drive\TSE_YOUTUBE_NOTION_BACKLOG\backfill_2025")

REASON_CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("precedente_citado", re.compile(r"precedente citado")),
    ("grounding_inconclusivo", re.compile(r"grounding nao localizou|data tida como futura")),
    ("densidade_insuficiente", re.compile(r"densidade informacional insuficiente")),
    ("data_ausente", re.compile(r"data da sessao (?:ausente|nao identificada)")),
    ("resultado_votacao_insuficiente", re.compile(r"resultado/votacao insuficientes")),
    ("relator_composicao_insuficiente", re.compile(r"relator/composicao insuficientes")),
    ("tema_generico", re.compile(r"tema generico|tema/titulo vazio")),
    ("valor_invalido", re.compile(r"valor invalido para")),
    ("vista_sem_ministro", re.compile(r"sem pedido_vista")),
)


def _norm(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    return decomposed.encode("ascii", "ignore").decode("ascii").lower()


def classify_reason(texts: list[str]) -> str:
    joined = _norm(" | ".join(texts))
    for category, pattern in REASON_CATEGORIES:
        if pattern.search(joined):
            return category
    return "outro"


def has_full_cnj(numero: str) -> bool:
    return len(re.sub(r"\D", "", str(numero or ""))) >= 20


def is_candidate_lost_judgment(status: str, category: str, numero: str) -> bool:
    if status == "blocked" and has_full_cnj(numero):
        return True
    if status in {"skipped", "blocked"} and category in {"precedente_citado", "grounding_inconclusivo"}:
        return True
    return False


def _entry_years(data_sessao: str, fallback_year: Optional[int]) -> Optional[int]:
    if data_sessao[:4].isdigit():
        return int(data_sessao[:4])
    return fallback_year


def iter_batch_video_entries(batch_root: Path) -> Iterator[dict[str, Any]]:
    for results_file in sorted(batch_root.glob("*/*/05_publish_results.json")):
        video_dir = results_file.parent
        try:
            publish_results = json.loads(results_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows: list[dict[str, Any]] = []
        for rows_name in ("04b_enriched_preview_rows.json", "04_preview_rows.json"):
            rows_file = video_dir / rows_name
            if rows_file.exists():
                try:
                    rows = json.loads(rows_file.read_text(encoding="utf-8"))
                except Exception:
                    rows = []
                break
        url = ""
        summary_file = video_dir / "06_batch_video_summary.json"
        if summary_file.exists():
            try:
                url = str(json.loads(summary_file.read_text(encoding="utf-8")).get("url", "") or "")
            except Exception:
                url = ""
        video_id = video_dir.name.split("_", 1)[-1]
        if not url:
            url = f"https://www.youtube.com/watch?v={video_id}"
        dates = Counter(str(r.get("data_sessao", "") or "")[:10] for r in rows if isinstance(r, dict))
        dates.pop("", None)
        data_sessao = dates.most_common(1)[0][0] if dates else ""
        yield {
            "source": "batch",
            "run": video_dir.parent.name,
            "video_id": video_id,
            "video_dir": video_dir,
            "url": url,
            "title": "",
            "data_sessao": data_sessao,
            "rows": rows,
            "publish_results": publish_results,
        }


def iter_backlog_video_entries(backlog_root: Path, years: set[int]) -> Iterator[dict[str, Any]]:
    if not backlog_root.exists():
        LOGGER.warning("Backlog indisponível (%s); auditando só o batch_gui.", backlog_root)
        return
    for playlist_dir in sorted(backlog_root.glob("[0-9][0-9][0-9][0-9]_PL*")):
        year = int(playlist_dir.name[:4])
        if years and year not in years:
            continue
        for summary_file in sorted(playlist_dir.glob("*/07_backfill_summary.json")):
            video_dir = summary_file.parent
            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            title = str(summary.get("title", "") or "")
            data_sessao = infer_session_date_from_video_title(title) or ""
            video_id = str(summary.get("video_id", "") or video_dir.name.split("_", 1)[-1])
            yield {
                "source": "backlog",
                "run": playlist_dir.name,
                "video_id": video_id,
                "video_dir": video_dir,
                "url": str(summary.get("url", "") or f"https://www.youtube.com/watch?v={video_id}"),
                "title": title,
                "data_sessao": data_sessao,
                "rows": [],
                "publish_results": summary.get("publish_results") or [],
                "fallback_year": year,
            }


def transcript_apregoamento_count(video_id: str, cache_dir: Path) -> Optional[int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{video_id}.json"
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return payload.get("apregoamentos")
    try:
        api_cls = require_youtube_transcript_api()
        fetched = api_cls().fetch(video_id, languages=["pt-BR", "pt"])
        snippets = [
            TranscriptSnippet(
                text=str(getattr(item, "text", "")),
                start_seconds=int(getattr(item, "start", 0)),
                end_seconds=int(getattr(item, "start", 0) + (getattr(item, "duration", 0) or 0)),
            )
            for item in fetched
        ]
        count = count_individual_apregoamentos(detect_rito_events(snippets))
    except Exception as exc:
        LOGGER.info("Sem transcrição para %s: %s", video_id, exc)
        # Falha NÃO é cacheada: bloqueio temporário do YouTube (IP/rate limit)
        # não pode impedir o retry numa rodada futura.
        time.sleep(1.0)
        return None
    cache_file.write_text(json.dumps({"apregoamentos": count}, ensure_ascii=False), encoding="utf-8")
    time.sleep(1.0)
    return count


def notion_counts(data_source_id: str) -> tuple[dict[str, int], dict[str, int], set[tuple[str, str]]]:
    """Linhas do Notion por vídeo (video_id do youtube_link), por data_sessao e as
    chaves (digits do número, data) — usadas para não reabrir itens já corrigidos.

    Atenção: query_data_source pagina até ~10k páginas (limite silencioso da API).
    """
    client = NotionSessoesClient(
        api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=data_source_id
    )
    schema = client.fetch_schema()
    by_video: dict[str, int] = {}
    by_date: dict[str, int] = {}
    numero_date_keys: set[tuple[str, str]] = set()
    for page in client.query_data_source():
        link = client._extract_property_text(page, schema, "youtube_link")
        data = (client._extract_property_text(page, schema, "data_sessao") or "")[:10]
        numero = re.sub(r"\D", "", client._extract_property_text(page, schema, "numero_processo") or "")
        video_id = extract_youtube_video_id(link) if link else ""
        if video_id:
            by_video[video_id] = by_video.get(video_id, 0) + 1
        if data:
            by_date[data] = by_date.get(data, 0) + 1
        if numero:
            numero_date_keys.add((numero, data))
    return by_video, by_date, numero_date_keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-root", default=str(DEFAULT_BATCH_ROOT))
    parser.add_argument("--backlog-root", default=str(DEFAULT_BACKLOG_ROOT))
    parser.add_argument("--years", default="2024,2025,2026")
    parser.add_argument("--with-transcript-check", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limita o nº de vídeos auditados.")
    parser.add_argument("--no-queue", action="store_true", help="Não alimenta a fila de vistoria.")
    parser.add_argument("--out", default="")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    years = {int(y) for y in re.findall(r"\d{4}", args.years)}
    out_dir = Path(args.out) if args.out else Path("artifacts") / "auditoria_retroativa" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    seen_videos: set[str] = set()
    for entry in iter_batch_video_entries(Path(args.batch_root)):
        year = _entry_years(entry["data_sessao"], None)
        if years and (year is None or year not in years):
            continue
        entries.append(entry)
        seen_videos.add(entry["video_id"])
        if args.limit and len(entries) >= args.limit:
            break
    if not args.limit or len(entries) < args.limit:
        for entry in iter_backlog_video_entries(Path(args.backlog_root), years):
            if entry["video_id"] in seen_videos:
                continue  # batch tem mais artefatos; prevalece
            year = _entry_years(entry["data_sessao"], entry.get("fallback_year"))
            if years and (year is None or year not in years):
                continue
            entries.append(entry)
            seen_videos.add(entry["video_id"])
            if args.limit and len(entries) >= args.limit:
                break
    LOGGER.info("Vídeos auditados: %s (anos %s)", len(entries), sorted(years))

    by_video: dict[str, int] = {}
    by_date: dict[str, int] = {}
    numero_date_keys: set[tuple[str, str]] = set()
    # A consulta ao Notion serve ao transcript-check E ao cruzamento "já no
    # Notion" — sem ela, itens já corrigidos entrariam na fila como pendência.
    if args.with_transcript_check or not args.no_queue:
        by_video, by_date, numero_date_keys = notion_counts(args.data_source_id)

    report_rows: list[dict[str, Any]] = []
    queue_items: list[dict[str, Any]] = []
    category_counter: Counter = Counter()
    for entry in entries:
        rows = entry["rows"]
        results = entry["publish_results"]
        paired_rows: list[Optional[dict[str, Any]]] = list(rows[: len(results)]) + [None] * max(
            0, len(results) - len(rows)
        )
        for result, row in zip(results, paired_rows):
            status = str(result.get("status", "") or "")
            if status not in {"skipped", "blocked"}:
                continue
            reasons = list(result.get("errors") or []) + list(result.get("warnings") or [])
            category = classify_reason(reasons)
            numero = str(result.get("numero_processo", "") or "")
            candidate = is_candidate_lost_judgment(status, category, numero)
            ja_no_notion = False
            if candidate and numero_date_keys:
                numero_digits = re.sub(r"\D", "", numero)
                if numero_digits and (numero_digits, entry["data_sessao"]) in numero_date_keys:
                    ja_no_notion = True
                    candidate = False
            category_counter[f"{status}:{category}"] += 1
            report_rows.append(
                {
                    "source": entry["source"],
                    "run": entry["run"],
                    "video_id": entry["video_id"],
                    "url": entry["url"],
                    "data_sessao": entry["data_sessao"],
                    "numero_processo": numero,
                    "tema": str(result.get("tema", "") or "")[:80],
                    "status": status,
                    "categoria": category,
                    "candidato_perdido": candidate,
                    "ja_no_notion": ja_no_notion,
                    "motivo": "; ".join(reasons)[:250],
                }
            )
            if candidate:
                queue_items.append(
                    vistoria_queue.make_vistoria_item(
                        source="auditoria",
                        video_id=entry["video_id"],
                        youtube_url=entry["url"],
                        disposition=status,
                        reasons=[f"[{category}] " + "; ".join(reasons)[:300]],
                        row=row if isinstance(row, dict) and row else None,
                        artifact_dir=str(entry["video_dir"]),
                        data_sessao=entry["data_sessao"],
                        dedupe_key=f"{numero}|{str(result.get('tema', '') or '')[:60]}",
                    )
                )

    transcript_rows: list[dict[str, Any]] = []
    if args.with_transcript_check:
        cache_dir = out_dir / "transcripts"
        for entry in entries:
            count = transcript_apregoamento_count(entry["video_id"], cache_dir)
            notion_video = by_video.get(entry["video_id"], 0)
            notion_date = by_date.get(entry["data_sessao"], 0) if entry["data_sessao"] else 0
            if count is None:
                verdict = "sem_transcricao"
            elif count > max(notion_video, notion_date):
                verdict = "verificar"
            else:
                verdict = "ok"
            transcript_rows.append(
                {
                    "video_id": entry["video_id"],
                    "data_sessao": entry["data_sessao"],
                    "apregoamentos": count,
                    "linhas_notion_video": notion_video,
                    "linhas_notion_data": notion_date,
                    "verdict": verdict,
                }
            )
            if verdict == "verificar" and not args.no_queue:
                queue_items.append(
                    vistoria_queue.make_vistoria_item(
                        source="rito",
                        video_id=entry["video_id"],
                        youtube_url=entry["url"],
                        disposition="contagem_rito",
                        reasons=[
                            f"Transcrição indica {count} apregoamentos individuais; Notion tem "
                            f"{notion_video} linha(s) do vídeo / {notion_date} na data {entry['data_sessao']}."
                        ],
                        row=None,
                        artifact_dir=str(entry["video_dir"]),
                        data_sessao=entry["data_sessao"],
                    )
                )

    report_csv = out_dir / "audit_report.csv"
    with report_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source", "run", "video_id", "url", "data_sessao", "numero_processo",
            "tema", "status", "categoria", "candidato_perdido", "ja_no_notion", "motivo",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)
    if transcript_rows:
        with (out_dir / "transcript_check.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(transcript_rows[0].keys()))
            writer.writeheader()
            writer.writerows(transcript_rows)

    queued = 0
    if queue_items and not args.no_queue:
        queued = vistoria_queue.append_items(queue_items)

    summary = {
        "videos": len(entries),
        "years": sorted(years),
        "itens_skipped_blocked": len(report_rows),
        "candidatos_perdidos": sum(1 for r in report_rows if r["candidato_perdido"]),
        "por_categoria": dict(sorted(category_counter.items())),
        "transcript_check": {
            "videos": len(transcript_rows),
            "verificar": sum(1 for r in transcript_rows if r["verdict"] == "verificar"),
            "sem_transcricao": sum(1 for r in transcript_rows if r["verdict"] == "sem_transcricao"),
        }
        if transcript_rows
        else None,
        "fila_vistoria_novos": queued,
        "out": str(out_dir),
    }
    (out_dir / "audit_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
