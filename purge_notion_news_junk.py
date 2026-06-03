"""Auditoria e expurgo de links-lixo nas colunas de notícia do Notion
(noticia_TSE, noticia_TRE, noticia_geral_1..9).

São lixo (NÃO são matéria de imprensa): bases de dados, visualizadores de processo
(PJe), consultas processuais e índices temáticos de jurisprudência — detectados por
``is_non_news_system_url`` — e os índices/seções institucionais genéricos detectados
por ``is_generic_institutional_news_url`` (jurisprudência, decisões por ano/assunto,
home, busca, agenda...).

Modos:
    python purge_notion_news_junk.py --report   # mapeia domínios e o que seria expurgado
    python purge_notion_news_junk.py --apply     # remove o lixo e REABRE o registro

Ao expurgar (--apply): esvazia noticia_TSE/TRE quando o link é lixo, compacta a lista
noticia_geral_1..9 removendo os lixos, e remove o page_id de
``artifacts/notion_news_backfill/_done_ids.txt`` para que o backfill volte a buscar um
link válido para aquele registro.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NotionSessoesClient,
    domain_from_url,
    is_generic_institutional_news_url,
    is_non_news_system_url,
    normalize_external_url,
)

LOGGER = logging.getLogger("purge_notion_news_junk")
ARTIFACT_ROOT = Path("artifacts") / "notion_news_purge"
DONE_FILE = Path("artifacts") / "notion_news_backfill" / "_done_ids.txt"
GERAL_COLS = [f"noticia_geral_{i}" for i in range(1, 10)]
ALL_NEWS_COLS = ["noticia_TSE", "noticia_TRE", *GERAL_COLS]
APPLY_SLEEP_SECONDS = 0.2


def _url(page: dict[str, Any], col: str) -> str:
    value = page.get("properties", {}).get(col, {})
    return (value.get("url") or "") if value.get("type") == "url" else ""


def _is_junk(url: str) -> bool:
    """Lixo = sistema/base de dados OU índice/seção institucional genérica."""
    if not url:
        return False
    return is_non_news_system_url(url) or is_generic_institutional_news_url(url)


def main() -> int:
    parser = argparse.ArgumentParser(description="Auditoria/expurgo de links-lixo nas colunas de notícia.")
    parser.add_argument("--apply", action="store_true", help="Remove o lixo e reabre o registro (padrão: só relata).")
    parser.add_argument("--report", action="store_true", help="Apenas relata domínios e o que seria expurgado.")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    client = NotionSessoesClient(api_key=key, data_source_id=args.data_source_id)
    client.fetch_schema()
    pages = client.query_data_source()

    domain_counter: Counter[str] = Counter()
    junk_domain_counter: Counter[str] = Counter()
    samples: dict[str, str] = {}
    changes: list[dict[str, Any]] = []
    reopen_ids: set[str] = set()
    total_links = 0
    total_junk = 0

    for page in pages:
        detail: dict[str, Any] = {}
        props: dict[str, Any] = {}
        # TSE / TRE: campo único -> esvazia se for lixo
        for col in ("noticia_TSE", "noticia_TRE"):
            url = _url(page, col)
            if not url:
                continue
            total_links += 1
            dom = domain_from_url(normalize_external_url(url))
            domain_counter[dom] += 1
            samples.setdefault(dom, url)
            if _is_junk(url):
                total_junk += 1
                junk_domain_counter[dom] += 1
                detail[col] = {"old": url, "new": ""}
                props[col] = {"url": None}
        # geral: compacta a lista removendo lixo
        geral_current = [(col, _url(page, col)) for col in GERAL_COLS]
        geral_links = [u for _, u in geral_current if u]
        for u in geral_links:
            total_links += 1
            dom = domain_from_url(normalize_external_url(u))
            domain_counter[dom] += 1
            samples.setdefault(dom, u)
        geral_kept = [u for u in geral_links if not _is_junk(u)]
        geral_dropped = [u for u in geral_links if _is_junk(u)]
        if geral_dropped:
            total_junk += len(geral_dropped)
            for u in geral_dropped:
                junk_domain_counter[domain_from_url(normalize_external_url(u))] += 1
            # reescreve compactando: geral_1..N com os mantidos, restante None
            for i, col in enumerate(GERAL_COLS):
                new_val = geral_kept[i] if i < len(geral_kept) else None
                old_val = dict(geral_current)[col]
                if (new_val or "") != (old_val or ""):
                    props[col] = {"url": new_val}
            detail["noticia_geral_dropped"] = geral_dropped

        if detail:
            rec = {"page_id": str(page.get("id", "")), "detail": detail}
            changes.append(rec)
            reopen_ids.add(str(page.get("id", "")))
            if args.apply:
                try:
                    notion_request_with_retry(client, "PATCH", f"/pages/{page['id']}", json={"properties": props})
                    rec["status"] = "updated"
                except Exception as exc:
                    rec["status"] = "failed"
                    rec["error"] = str(exc)
                time.sleep(APPLY_SLEEP_SECONDS)

    # Reabre registros expurgados no _done_ids do backfill (para rebuscar link válido).
    reopened = 0
    if args.apply and reopen_ids and DONE_FILE.exists():
        done = [line.strip() for line in DONE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        kept = [d for d in done if d not in reopen_ids]
        reopened = len(done) - len(kept)
        DONE_FILE.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    # relatório de domínios: ordena por contagem, marca os de sistema
    domain_report = []
    for dom, count in domain_counter.most_common():
        sample = samples.get(dom, "")
        domain_report.append({
            "domain": dom,
            "links": count,
            "junk_links": junk_domain_counter.get(dom, 0),
            "is_system": is_non_news_system_url(sample),
            "is_institutional_generic": is_generic_institutional_news_url(sample),
            "sample": sample,
        })
    (run_dir / "domain_report.json").write_text(json.dumps(domain_report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "mode": "apply" if args.apply else "report",
        "pages": len(pages),
        "total_links": total_links,
        "total_junk_links": total_junk,
        "pages_with_junk": len(changes),
        "distinct_domains": len(domain_counter),
        "junk_domains": [{"domain": d, "links": c} for d, c in junk_domain_counter.most_common()],
        "reopened_done_ids": reopened,
        "applied": sum(1 for c in changes if c.get("status") == "updated"),
        "failed": sum(1 for c in changes if c.get("status") == "failed"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    LOGGER.info("Relatórios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
