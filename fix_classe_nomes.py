"""Normaliza classe_processo escrita por EXTENSO para a sigla padrao (consistencia com o resto
da coluna). NAO e downgrade: e a MESMA classe na forma sigla. Mapa curado + confirmado por
cnj_classe_sigla (AI, RCED, RCand) + 'Conflito de Competencia'->CC. 'Apuracao de Eleicao' fica
de fora (sigla sem padrao claro -> revisao).

Uso: python fix_classe_nomes.py [--apply]
"""
from __future__ import annotations

import argparse, json, logging, time
from datetime import datetime
from pathlib import Path

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("fix_classe_nomes")
MAPA = {
    "Agravo de Instrumento": "AI",
    "Recurso contra Expedição de Diploma": "RCED",
    "Conflito de Competência": "CC",
    "Registro de Candidatura": "RCand",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    changes = []
    applied = failed = 0
    for p in pages:
        v = (t(p, "classe_processo") or "").strip()
        if v not in MAPA:
            continue
        nova = MAPA[v]
        changes.append({"page_id": p["id"], "cnj": t(p, "numero_processo"), "old": v, "new": nova})
        if args.apply:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}",
                                          json={"properties": {"classe_processo": client._build_property_value(schema, "classe_processo", nova)}})
                applied += 1
            except Exception as exc:
                failed += 1; LOGGER.warning("falha %s: %s", t(p, "numero_processo"), exc)
            time.sleep(0.15)

    run_dir = Path("artifacts") / "notion_classe_nomes" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    import collections
    por = collections.Counter((c["old"], c["new"]) for c in changes)
    LOGGER.info("RESUMO: %s | %s", json.dumps({"mode": "apply" if args.apply else "dry-run",
                "alteracoes": len(changes), "applied": applied, "failed": failed}, ensure_ascii=False), run_dir)
    for (o, n), cnt in por.most_common():
        LOGGER.info("  %sx  %s -> %s", cnt, o, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
