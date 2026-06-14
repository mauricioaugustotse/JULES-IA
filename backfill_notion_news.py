"""Backfill das colunas de notícia (noticia_TSE, noticia_TRE, noticia_geral_1..9) na
base do Notion, usando o GeminiNewsEnricher com prompt reforçado (busca TSE + TRE da
UF de origem + imprensa, em UMA chamada grounded por registro) e a validação HTTP
existente (descarta link quebrado/página que não abre/irrelevante).

Econômico: 1 chamada grounded por registro, reparo institucional DESLIGADO. Use
--limit para piloto de calibração. Por padrão age só nos registros SEM nenhuma
notícia (--all para incluir os que já têm).

Uso:
    python backfill_notion_news.py --limit 50            # PILOTO (dry-run, só mede)
    python backfill_notion_news.py --limit 50 --apply    # piloto e grava
    python backfill_notion_news.py --apply               # base toda (sem notícia)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import canonicalize_numero_processo
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    GeminiNewsEnricher,
    NotionSessoesClient,
    PublishPreviewRow,
    RunArtifacts,
    dedupe_preserve_order,
    parse_multi_value_text,
)

LOGGER = logging.getLogger("backfill_notion_news")
ARTIFACT_ROOT = Path("artifacts") / "notion_news_backfill"
DONE_FILE = ARTIFACT_ROOT / "_done_ids.txt"
GERAL_COLS = [f"noticia_geral_{i}" for i in range(1, 10)]
APPLY_SLEEP_SECONDS = 0.2


def _url(page: dict[str, Any], col: str) -> str:
    value = page.get("properties", {}).get(col, {})
    return (value.get("url") or "") if value.get("type") == "url" else ""


def _row_from_page(client: NotionSessoesClient, schema: Any, page: dict[str, Any]) -> PublishPreviewRow:
    def t(field: str) -> str:
        return client._extract_property_text(page, schema, field) if field in schema.properties else ""

    return PublishPreviewRow(
        tema=t("tema"),
        punchline=t("punchline"),
        numero_processo=t("numero_processo"),
        classe_processo=t("classe_processo"),
        tribunal=t("tribunal"),
        origem=t("origem"),
        data_sessao=t("data_sessao"),
        relator=t("relator"),
        partes=parse_multi_value_text(t("partes")),
        analise_do_conteudo_juridico=t("analise_do_conteudo_juridico"),
    )


def _has_any_news(page: dict[str, Any]) -> bool:
    if _url(page, "noticia_TSE") or _url(page, "noticia_TRE"):
        return True
    return any(_url(page, col) for col in GERAL_COLS)


def _patch_news(client: NotionSessoesClient, page_id: str, tse: str, tre: str, gerais: list[str]) -> None:
    props: dict[str, Any] = {
        "noticia_TSE": {"url": tse or None},
        "noticia_TRE": {"url": tre or None},
    }
    for i, col in enumerate(GERAL_COLS):
        props[col] = {"url": gerais[i] if i < len(gerais) else None}
    notion_request_with_retry(client, "PATCH", f"/pages/{page_id}", json={"properties": props})


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill de notícias (TSE/TRE/geral) no Notion.")
    parser.add_argument("--apply", action="store_true", help="Grava no Notion (padrão: só mede/dry-run).")
    parser.add_argument("--limit", type=int, default=0, help="Limita nº de registros (piloto). 0 = todos.")
    parser.add_argument("--all", action="store_true", help="Inclui registros que já têm alguma notícia.")
    parser.add_argument("--resume", action="store_true", help="Pula registros já processados em execuções anteriores (artifacts/notion_news_backfill/_done_ids.txt).")
    parser.add_argument("--video-ids", default="", help="Restringe aos vídeos informados (ids separados por vírgula): só enriquece páginas cujo youtube_link contém um desses ids.")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    notion_key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    gemini_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not notion_key or not gemini_key:
        raise RuntimeError("Faltam chaves NOTION/GEMINI.")
    client = NotionSessoesClient(api_key=notion_key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    candidates = [p for p in pages if args.all or not _has_any_news(p)]
    candidates = [p for p in candidates if client._extract_property_text(p, schema, "tema").strip() or client._extract_property_text(p, schema, "numero_processo").strip()]
    if args.video_ids:
        wanted = {v.strip() for v in args.video_ids.split(",") if v.strip()}
        candidates = [
            p for p in candidates
            if any(v in (client._extract_property_text(p, schema, "youtube_link") or "") for v in wanted)
        ]
        LOGGER.info("Filtro --video-ids: %s vídeos -> %s páginas candidatas.", len(wanted), len(candidates))
    if args.resume and DONE_FILE.exists():
        done_ids = {line.strip() for line in DONE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()}
        before = len(candidates)
        candidates = [p for p in candidates if str(p.get("id", "")) not in done_ids]
        LOGGER.info("Resume: %s já processados, %s restantes.", before - len(candidates), len(candidates))
    if args.limit and args.limit < len(candidates):
        step = max(1, len(candidates) // args.limit)  # amostra espalhada (antigos + recentes)
        candidates = candidates[::step][: args.limit]

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    store = RunArtifacts(run_dir)
    enricher = GeminiNewsEnricher(api_key=gemini_key, artifact_store=store, logger=LOGGER, allow_institutional_repair=False)

    LOGGER.info("Registros a processar: %s (apply=%s)", len(candidates), args.apply)
    results = []
    calls = 0
    t0 = time.time()
    for idx, page in enumerate(candidates, start=1):
        row = _row_from_page(client, schema, page)
        try:
            enriched = enricher.enrich_rows([row])[0]
            calls += 1
        except Exception as exc:
            LOGGER.warning("Falha no registro %s: %s", idx, exc)
            continue
        # MERGE-SAFE: nunca apaga link existente; só preenche o que falta / acrescenta gerais.
        cur_tse = _url(page, "noticia_TSE")
        cur_tre = _url(page, "noticia_TRE")
        cur_gerais = [g for col in GERAL_COLS if (g := _url(page, col))]
        tse = cur_tse or enriched.noticia_TSE
        tre = cur_tre or enriched.noticia_TRE
        gerais = dedupe_preserve_order(cur_gerais + list(enriched.noticias_gerais))[:9]
        changed = (tse != cur_tse) or (tre != cur_tre) or (gerais != cur_gerais)
        rec = {
            "page_id": str(page.get("id", "")),
            "numero": canonicalize_numero_processo(row.numero_processo),
            "noticia_TSE": enriched.noticia_TSE,
            "noticia_TRE": enriched.noticia_TRE,
            "noticias_gerais": list(enriched.noticias_gerais),
            "changed": changed,
        }
        results.append(rec)
        if args.apply and changed:
            try:
                _patch_news(client, rec["page_id"], tse, tre, gerais)
                rec["status"] = "updated"
            except Exception as exc:
                rec["status"] = "failed"
                rec["error"] = str(exc)
            time.sleep(APPLY_SLEEP_SECONDS)
        if args.apply:
            DONE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with DONE_FILE.open("a", encoding="utf-8") as handle:
                handle.write(rec["page_id"] + "\n")
        if idx % 10 == 0:
            LOGGER.info("Processados: %s/%s", idx, len(candidates))

    elapsed = time.time() - t0
    yield_tse = sum(1 for r in results if r["noticia_TSE"])
    yield_tre = sum(1 for r in results if r["noticia_TRE"])
    yield_geral = sum(1 for r in results if r["noticias_gerais"])
    geral_links = sum(len(r["noticias_gerais"]) for r in results)
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "processed": len(results),
        "grounded_calls": calls,
        "elapsed_s": round(elapsed, 1),
        "yield_TSE": yield_tse,
        "yield_TRE": yield_tre,
        "yield_geral_pages": yield_geral,
        "geral_links_total": geral_links,
        "applied": sum(1 for r in results if r.get("status") == "updated"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    LOGGER.info("Relatórios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
