"""Aplica as correcoes de origem da validacao SADP (sadp_validate_origem) a partir do
divergencias.json salvo. Idempotente: so corrige se a origem ATUAL ainda for a (errada)
origem_base; grava a origem_sadp (município oficial). Re-consulta o estado vivo.

Uso:
  python apply_origem_corrections.py [--json <divergencias.json>] --apply
"""
from __future__ import annotations

import argparse, glob, json, logging, re, time, unicodedata

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("apply_origem_corrections")


def fold(x):
    x = unicodedata.normalize("NFKD", str(x or "").lower())
    return re.sub(r"\s+", " ", "".join(c for c in x if not unicodedata.combining(c))).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    path = args.json or sorted(glob.glob("artifacts/notion_sadp_origem/*/divergencias.json"))[-1]
    recs = {r["page_id"]: r for r in json.load(open(path, encoding="utf-8"))}
    LOGGER.info("divergencias.json: %s | correcoes: %s", path, len(recs))

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    corr = skip = applied = failed = 0
    for p in pages:
        r = recs.get(p["id"])
        if not r:
            continue
        cur = (t(p, "origem") or "").strip()
        if fold(cur) != fold(r["origem_base"]):  # estado vivo mudou / ja corrigido
            skip += 1
            continue
        corr += 1
        if args.apply:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}",
                                          json={"properties": {"origem": client._build_property_value(schema, "origem", r["origem_sadp"])}})
                applied += 1
            except Exception as exc:
                failed += 1; LOGGER.warning("falha %s: %s", r.get("cnj"), exc)
            time.sleep(0.15)
    LOGGER.info("RESUMO: %s", json.dumps({"mode": "apply" if args.apply else "dry-run",
                "no_json": len(recs), "corrigiveis": corr, "skip": skip, "applied": applied, "failed": failed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
