"""Valida classe_processo dos CNJ-20 contra o DataJud (classe oficial), com NO-DOWNGRADE:
- base vazia -> preenche com a classe do DataJud;
- base MAIS especifica (ex.: 'AgRg-REspe' termina em 'REspe' do DataJud) -> MANTEM (no-downgrade);
- FAMILIA diferente (ex.: base 'AIJE' vs DataJud 'REspe') -> provavel erro do Gemini -> FLAG
  (com --apply, corrige p/ a classe do DataJud, pois familia divergente = classe errada).
So conta como divergencia real quando as familias-base diferem (prefixo de recurso AgRg-/ED- nao conta).

Uso:
  python classe_validate_datajud.py [--limit N]            # dry-run
  python classe_validate_datajud.py [--limit N] --apply
"""
from __future__ import annotations

import argparse, json, logging, re, time
from datetime import datetime
from pathlib import Path

from audit_notion_sessoes_round2 import notion_request_with_retry
from cnj_datajud import lookup_process
from local_secrets import get_secret
from tse_normalization import normalize_classe_processo
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("classe_validate_datajud")
ARTIFACT_ROOT = Path("artifacts") / "notion_classe_datajud"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.2)
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
    LOGGER.info("CNJ-20 a validar classe: %s", len(alvos))

    preenche, divergente, review = [], [], []
    stats = {"alvos": len(alvos), "achou": 0, "nao_achou": 0, "ok": 0, "preenche_vazia": 0,
             "no_downgrade_mantem": 0, "divergente_familia": 0, "applied": 0, "falhas": 0}
    for i, p in enumerate(alvos, 1):
        cnj = t(p, "numero_processo")
        try:
            proc = lookup_process(cnj, tribunal=t(p, "tribunal") or "", year=(t(p, "data_sessao") or "")[:4])
        except Exception:
            stats["falhas"] += 1; time.sleep(args.sleep); continue
        dj = normalize_classe_processo(proc.classe_sigla) if proc and proc.classe_sigla else ""
        if not dj:
            stats["nao_achou"] += 1; time.sleep(args.sleep); continue
        stats["achou"] += 1
        base = normalize_classe_processo(t(p, "classe_processo"))
        props, tipo = {}, ""
        if not base:
            tipo = "preenche_vazia"; props["classe_processo"] = client._build_property_value(schema, "classe_processo", dj)
            preenche.append({"page_id": p["id"], "cnj": cnj, "nova": dj})
        elif base == dj or base.endswith(dj):  # igual OU base mais especifica (AgRg-REspe ~ REspe)
            stats["ok" if base == dj else "no_downgrade_mantem"] += 1
            time.sleep(args.sleep); continue
        elif dj.endswith(base):  # DataJud mais especifico que a base -> upgrade
            tipo = "divergente_familia"; props["classe_processo"] = client._build_property_value(schema, "classe_processo", dj)
            divergente.append({"page_id": p["id"], "cnj": cnj, "base": base, "datajud": dj, "acao": "upgrade"})
        else:  # familias diferentes -> erro do Gemini
            tipo = "divergente_familia"; props["classe_processo"] = client._build_property_value(schema, "classe_processo", dj)
            divergente.append({"page_id": p["id"], "cnj": cnj, "base": base, "datajud": dj, "acao": "corrige_familia"})
        stats[tipo] += 1
        if args.apply and props:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                stats["applied"] += 1
            except Exception as exc:
                stats["falhas"] += 1
            time.sleep(0.15)
        if i % 300 == 0:
            LOGGER.info("  ... %s/%s | achou=%s preenche=%s diverg=%s", i, len(alvos), stats["achou"], stats["preenche_vazia"], stats["divergente_familia"])
        time.sleep(args.sleep)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "preenche.json").write_text(json.dumps(preenche, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "divergente.json").write_text(json.dumps(divergente, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s | %s", json.dumps(summary, ensure_ascii=False), run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
