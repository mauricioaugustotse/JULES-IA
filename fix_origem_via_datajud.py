"""Corrige a origem dos casos FLAGADOS pelo origem_uf_check (UF da origem != UF do TR do CNJ).
A UF certa vem do TR (autoritativa). O MUNICIPIO vem da ZONA do DataJud no indice TRE forcado
(orgaoJulgador 'Nª ZONA ELEITORAL - MUNICIPIO'). E zona-seat (ressalva conhecida), mas como a
base estava com UF errada (ex.: 'Brasilia/DF' default do Gemini), a correcao p/ 'Municipio/UF'
correto e um ganho claro. So corrige quando consegue o municipio da zona; senao -> revisao.

Uso:
  python fix_origem_via_datajud.py            # dry-run
  python fix_origem_via_datajud.py --apply
"""
from __future__ import annotations

import argparse, glob, json, logging, re, time
from datetime import datetime
from pathlib import Path

from audit_notion_sessoes_round2 import notion_request_with_retry
from cnj_datajud import _search
from fill_partes_advogados_from_jurisprudencia import proper_case
from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("fix_origem_via_datajud")
ARTIFACT_ROOT = Path("artifacts") / "notion_origem_datajud_fix"
_ZONA_RE = re.compile(r"ZONA ELEITORAL\s*-\s*(.+)", re.IGNORECASE)


def zona_municipio(orgao: str) -> str:
    m = _ZONA_RE.search(str(orgao or ""))
    if not m:
        return ""
    s = re.split(r"\s+-\s+", m.group(1).strip())[0].strip()  # tira bairro apos ' - '
    s = re.sub(r"\s*[-/]\s*[A-Za-zÀ-ÿ]{2}\.?\s*$", "", s).strip()  # tira sufixo /UF ou -UF
    return s if s and "/" not in s and len(s) > 2 else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--flag-json", default="")
    ap.add_argument("--tr-json", default="")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    flag_path = args.flag_json or sorted(glob.glob("artifacts/notion_origem_uf_check/*/flagados.json"))[-1]
    tr_path = args.tr_json or sorted(glob.glob("artifacts/notion_origem_uf_check/*/tr_uf.json"))[-1]
    flag = json.load(open(flag_path, encoding="utf-8"))
    tr_uf = json.load(open(tr_path, encoding="utf-8"))
    LOGGER.info("flagados: %s | TRs: %s", len(flag), len(tr_uf))

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = {p["id"]: p for p in client.query_data_source()}

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    changes, review = [], []
    stats = {"flag": len(flag), "achou_zona": 0, "corrige": 0, "sem_zona": 0, "estado_mudou": 0, "applied": 0, "falhas": 0}
    for r in flag:
        p = pages.get(r["page_id"])
        if not p or (t(p, "origem") or "").strip() != r["origem"]:
            stats["estado_mudou"] += 1
            continue
        uf = tr_uf.get(r["tr"], "")
        if not uf:
            continue
        d = re.sub(r"\D", "", r["cnj"])
        try:
            hits = _search(f"api_publica_tre-{uf.lower()}", {"query": {"match": {"numeroProcesso": d[:20]}}, "size": 2})
        except Exception:
            stats["falhas"] += 1; time.sleep(0.3); continue
        org = (hits[0].get("_source", {}).get("orgaoJulgador", {}) or {}).get("nome", "") if hits else ""
        muni = zona_municipio(org)
        if not muni:
            stats["sem_zona"] += 1
            review.append({"page_id": r["page_id"], "cnj": r["cnj"], "origem": r["origem"], "tr_uf": uf, "orgao": org})
            time.sleep(0.2); continue
        stats["achou_zona"] += 1
        nova = f"{proper_case(muni)}/{uf.upper()}"
        stats["corrige"] += 1
        rec = {"page_id": r["page_id"], "cnj": r["cnj"], "origem_base": r["origem"], "origem_nova": nova, "orgao": org}
        if args.apply:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{r['page_id']}",
                                          json={"properties": {"origem": client._build_property_value(schema, "origem", nova)}})
                stats["applied"] += 1
            except Exception as exc:
                stats["falhas"] += 1; rec["erro"] = str(exc)
            time.sleep(0.15)
        changes.append(rec)
        time.sleep(0.2)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:12]:
        LOGGER.info("  %s | %s -> %s", c["cnj"], c["origem_base"], c["origem_nova"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
