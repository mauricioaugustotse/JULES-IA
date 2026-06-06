"""Aplica os flips 'resolvido' da pesquisa Gemini (suspenso_research_gemini) a partir do
research.json salvo (deterministico, sem re-rodar o Gemini). So toca paginas que AINDA estao
'Suspenso por vista' (re-consulta o estado vivo). Flip: resultado->'Suspenso mas julgado
depois', votacao->'Suspenso*'.

Uso:
  python apply_suspenso_resolvido.py --json <research.json>            # dry-run
  python apply_suspenso_resolvido.py --json <research.json> --apply
"""
from __future__ import annotations

import argparse, glob, json, logging, time

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("apply_suspenso_resolvido")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    path = args.json or sorted(glob.glob("artifacts/notion_suspenso_research/*/research.json"))[-1]
    data = json.load(open(path, encoding="utf-8"))
    resolvidos = {r["page_id"]: r for r in data if str(r.get("situacao", "")).lower() == "resolvido"}
    LOGGER.info("research.json: %s | resolvidos: %s", path, len(resolvidos))

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    flip = skip = applied = failed = 0
    for p in pages:
        if p["id"] not in resolvidos:
            continue
        if (t(p, "resultado") or "").strip() != "Suspenso por vista":  # estado vivo mudou
            skip += 1
            continue
        flip += 1
        if args.apply:
            props = {"resultado": client._build_property_value(schema, "resultado", "Suspenso mas julgado depois"),
                     "votacao": client._build_property_value(schema, "votacao", "Suspenso*")}
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                applied += 1
            except Exception as exc:
                failed += 1; LOGGER.warning("falha %s: %s", t(p, "numero_processo"), exc)
            time.sleep(0.15)
    LOGGER.info("RESUMO: %s", json.dumps({"mode": "apply" if args.apply else "dry-run",
                "resolvidos_no_json": len(resolvidos), "flip": flip, "skip_estado_mudou": skip,
                "applied": applied, "failed": failed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
