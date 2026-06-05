"""Re-limpa a coluna partes de TODA a base aplicando o normalize_partes_list canonico
(split conjunto + remove lixo + ESTIRPA PAPEIS/posicoes + 'e outros' + dedup fuzzy).
Idempotente; escrita page-value (rich_text).

Uso:
  python reclean_partes_base.py            # dry-run
  python reclean_partes_base.py --apply
"""
from __future__ import annotations

import argparse, json, logging, time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import clean_partes_list, parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("reclean_partes_base")
ARTIFACT_ROOT = Path("artifacts") / "notion_partes_reclean"


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

    changes: list[dict[str, Any]] = []
    stats = {"paginas": 0, "afetadas": 0, "applied": 0, "failed": 0}
    for p in pages:
        cur = parse_multi_value_text(client._extract_property_text(p, schema, "partes"))
        if not cur:
            continue
        stats["paginas"] += 1
        new = clean_partes_list(cur)
        if new == cur:
            continue
        stats["afetadas"] += 1
        rec = {"page_id": p["id"], "numero": client._extract_property_text(p, schema, "numero_processo"),
               "old": cur, "new": new}
        if args.apply:
            built = client._build_property_value(schema, "partes", new) or client._build_empty_property_value(schema, "partes")
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": {"partes": built}})
                rec["status"] = "updated"; stats["applied"] += 1
            except Exception as exc:
                rec["status"] = "failed"; rec["error"] = str(exc); stats["failed"] += 1
            time.sleep(0.12)
        changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
