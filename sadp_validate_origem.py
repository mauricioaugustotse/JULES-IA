"""Valida a CORRETUDE da coluna origem de TODOS os casos com CNJ-20, contra o MUNICIPIO
oficial do SADP (busca exata por numUnico). Diferente dos tools de completar/corrigir
(que so tocam casos que precisavam), aqui conferimos todo CNJ-20 ja existente. Reporta
divergencias (origem da base != municipio oficial); com --apply, corrige p/ o oficial
(SADP autoritativo). Compara por fold (ignora acento/caixa) p/ nao marcar diff trivial.

Uso:
  python sadp_validate_origem.py [--limit N]            # dry-run
  python sadp_validate_origem.py [--limit N] --apply
"""
from __future__ import annotations

import argparse, json, logging, re, time, unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from sadp_correct_cnj20 import _fmt_origem
from sadp_lookup import fetch_detail, make_session, search_numunico
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("sadp_validate_origem")
ARTIFACT_ROOT = Path("artifacts") / "notion_sadp_origem"


def fold(x: str) -> str:
    x = unicodedata.normalize("NFKD", str(x or "").lower())
    return re.sub(r"\s+", " ", "".join(c for c in x if not unicodedata.combining(c))).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.7)
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    alvos = [p for p in pages if len(re.sub(r"\D", "", t(p, "numero_processo") or "")) >= 20]
    if args.limit and args.limit < len(alvos):
        alvos = alvos[::max(1, len(alvos) // args.limit)][:args.limit]
    LOGGER.info("CNJ-20 a validar origem: %s", len(alvos))

    sess = make_session()
    diverg: list[dict[str, Any]] = []
    stats = {"alvos": len(alvos), "achou": 0, "nao_achou": 0, "ok": 0, "vazia_preenche": 0,
             "divergente": 0, "applied": 0, "falhas": 0}
    for i, p in enumerate(alvos, 1):
        cnj = (t(p, "numero_processo") or "").strip()
        try:
            cands = search_numunico(sess, cnj)
        except Exception:
            stats["falhas"] += 1; time.sleep(args.sleep); continue
        match = next((c for c in cands if re.sub(r"\D", "", c.get("cnj", "")) == re.sub(r"\D", "", cnj)), None)
        if not match or not match.get("nprot"):
            stats["nao_achou"] += 1; time.sleep(args.sleep); continue
        detail = fetch_detail(sess, match["nprot"], "tse")
        oficial = _fmt_origem(detail.get("municipio", "")) if detail else ""
        if not oficial:
            stats["nao_achou"] += 1; time.sleep(args.sleep); continue
        stats["achou"] += 1
        atual = (t(p, "origem") or "").strip()
        if fold(atual) == fold(oficial):
            stats["ok"] += 1; time.sleep(args.sleep); continue
        tipo = "vazia_preenche" if not atual else "divergente"
        stats[tipo] += 1
        rec = {"page_id": p["id"], "cnj": cnj, "origem_base": atual, "origem_sadp": oficial, "tipo": tipo}
        if args.apply:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}",
                                          json={"properties": {"origem": client._build_property_value(schema, "origem", oficial)}})
                stats["applied"] += 1
            except Exception as exc:
                stats["falhas"] += 1; rec["erro"] = str(exc)
            time.sleep(0.15)
        diverg.append(rec)
        if i % 200 == 0:
            LOGGER.info("  ... %s/%s | ok=%s diverg=%s vazia=%s nao_achou=%s", i, len(alvos), stats["ok"], stats["divergente"], stats["vazia_preenche"], stats["nao_achou"])
        time.sleep(args.sleep)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "divergencias.json").write_text(json.dumps(diverg, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s | %s", json.dumps(summary, ensure_ascii=False), run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
