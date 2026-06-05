"""Re-normaliza a coluna pedido_vista (select) na base: aplica normalize_pedido_vista_value,
que agora extrai SO o ministro de frases com texto de acao colado (ex.: 'Min. Nunes Marques
suspendendo a conclusao do julgamento' -> 'Min. Nunes Marques') e esvazia lixo ('Min. que').
Escrita page-value (select) segura.

Uso:
  python fix_pedido_vista.py            # dry-run
  python fix_pedido_vista.py --apply
"""
from __future__ import annotations

import argparse, json, logging, time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import normalize_pedido_vista_value
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("fix_pedido_vista")
ARTIFACT_ROOT = Path("artifacts") / "notion_pedido_vista_fix"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    client = NotionSessoesClient(api_key=key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    changes: list[dict[str, Any]] = []
    stats = {"paginas": 0, "afetadas": 0, "esvaziadas": 0, "applied": 0, "failed": 0}
    for p in pages:
        stats["paginas"] += 1
        cur = (client._extract_property_text(p, schema, "pedido_vista") or "").strip()
        if not cur:
            continue
        norm = normalize_pedido_vista_value(cur)
        if norm == cur:
            continue
        stats["afetadas"] += 1
        if not norm:
            stats["esvaziadas"] += 1
        rec = {"page_id": p["id"], "numero": client._extract_property_text(p, schema, "numero_processo"),
               "old": cur, "new": norm}
        if args.apply:
            props = {"pedido_vista": {"select": {"name": norm}} if norm else {"select": None}}
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
        LOGGER.info("  [%s] %r -> %r", c["numero"], c["old"], c["new"])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
