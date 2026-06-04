"""Corrige nomes de ministro MAL-TRANSCRITOS (audio antigo) na coluna composicao do
Notion, trocando-os pela forma canonica via escrita de PAGE-VALUE (seguro; nao mexe em
options/schema). Cirurgico: so atua nos nomes exatos do mapa CORRECTIONS.

    "Min. Luciano Nunes Maia"  -> "Min. Napoleao Nunes Maia Filho"
    "Min. Roberto Barroso"     -> "Min. Luis Roberto Barroso"
    "Min. Carmen Lossio"       -> "Min. Luciana Lossio"

Uso:
    python fix_composicao_ministro_names.py            # dry-run (relatorio)
    python fix_composicao_ministro_names.py --apply    # aplica as trocas
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
from tse_normalization import dedupe_preserve_order, parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("fix_composicao_ministro_names")
ARTIFACT_ROOT = Path("artifacts") / "notion_composicao_ministro_fix"

CORRECTIONS = {
    "Min. Luciano Nunes Maia": "Min. Napoleão Nunes Maia Filho",
    "Min. Roberto Barroso": "Min. Luís Roberto Barroso",
    "Min. Cármen Lóssio": "Min. Luciana Lóssio",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Corrige nomes de ministro mal-transcritos na composicao.")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    client = NotionSessoesClient(api_key=key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    changes: list[dict[str, Any]] = []
    stats = {"paginas": 0, "afetadas": 0, "applied": 0, "failed": 0}
    for p in pages:
        stats["paginas"] += 1
        comp = parse_multi_value_text(t(p, "composicao"))
        if not any(name in CORRECTIONS for name in comp):
            continue
        fixed = dedupe_preserve_order([CORRECTIONS.get(name, name) for name in comp])
        if fixed == comp:
            continue
        stats["afetadas"] += 1
        rec = {"page_id": p["id"], "numero": t(p, "numero_processo"),
               "old": comp, "new": fixed}
        if args.apply:
            props = {"composicao": {"multi_select": [{"name": n} for n in fixed]}}
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                rec["status"] = "updated"; stats["applied"] += 1
            except Exception as exc:
                rec["status"] = "failed"; rec["error"] = str(exc); stats["failed"] += 1
            time.sleep(0.2)
        changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:30]:
        trocas = [f"{o}->{CORRECTIONS[o]}" for o in c["old"] if o in CORRECTIONS]
        LOGGER.info("  [%s] %s | %s", c["numero"], c["page_id"][:8], "; ".join(trocas))
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
