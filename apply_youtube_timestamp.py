"""Aplica os timestamps da Tarefa 4 a partir do changes.json salvo (sem re-indexar o backlog).
Idempotente: so altera se o link atual ainda for EXATAMENTE o 'old' (sem t=).

Uso:
  python apply_youtube_timestamp.py [--json <changes.json>] --apply
"""
from __future__ import annotations

import argparse, glob, json, logging, time

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("apply_youtube_timestamp")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    path = args.json or sorted(glob.glob("artifacts/notion_youtube_timestamp/*/changes.json"))[-1]
    recs = json.load(open(path, encoding="utf-8"))
    by_id = {r["page_id"]: r for r in recs}
    LOGGER.info("changes.json: %s | timestamps: %s", path, len(by_id))

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    upd = skip = applied = failed = 0
    for p in pages:
        r = by_id.get(p["id"])
        if not r:
            continue
        cur = (t(p, "youtube_link") or "").strip()
        if cur != r["old"] or "t=" in cur:  # estado mudou / ja tem timestamp
            skip += 1
            continue
        upd += 1
        if args.apply:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}",
                                          json={"properties": {"youtube_link": client._build_property_value(schema, "youtube_link", r["new"])}})
                applied += 1
            except Exception as exc:
                failed += 1; LOGGER.warning("falha %s: %s", r.get("numero"), exc)
            time.sleep(0.15)
    LOGGER.info("RESUMO: %s", json.dumps({"mode": "apply" if args.apply else "dry-run",
                "no_json": len(by_id), "aplicaveis": upd, "skip": skip, "applied": applied, "failed": failed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
