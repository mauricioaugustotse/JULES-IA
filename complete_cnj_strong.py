"""Completa o numero_processo (CNJ-20) das paginas INCOMPLETAS usando o CSV CONSOLIDADO
COMPLETO do DJE, com chaves fortes e match UNICO, em cascata:
  1) numeroProcesso(base) + dataDecisao == data_sessao  -> CNJ unico   (mesma regra do complete_cnj)
  2) numeroProcesso UNICO em TODO o CSV                  -> CNJ unico   (data da sessao != data da decisao)
  3) numeroProcesso + UF (da origem) UNICO                              (UF inferida do CNJ via base)
  4) numeroProcesso + >=1 parte em comum UNICO
Em TODOS os casos exige CNJ UNICO + SANIDADE de prefixo (cnj[:9] == numero curto da base).

IMPORTANTE: passe o CSV COMPLETO em --input (NAO o reduzido), senao a unicidade global (passos 2-4)
fica falsa. dry-run por padrao.

Uso:
  python complete_cnj_strong.py --input "C:\\Users\\mauri\\Downloads\\TSE_decisoes_consolidado_2026-06-28.csv"          # dry-run
  python complete_cnj_strong.py --input "C:\\Users\\mauri\\Downloads\\TSE_decisoes_consolidado_2026-06-28.csv" --apply
"""
from __future__ import annotations

import argparse, csv, collections, json, logging, re, time, unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

csv.field_size_limit(50 * 1024 * 1024)
LOGGER = logging.getLogger("complete_cnj_strong")
ARTIFACT_ROOT = Path("artifacts") / "notion_cnj_complete_strong"


def digits(s):
    return re.sub(r"\D", "", str(s or ""))


def iso(d):
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", str(d or ""))
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else ""


def fold(s):
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(c for c in s if not unicodedata.combining(c)).strip()


def format_cnj(d):
    return f"{d[0:7]}-{d[7:9]}.{d[9:13]}.{d[13:14]}.{d[14:16]}.{d[16:20]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV CONSOLIDADO COMPLETO do DJE (nao o reduzido).")
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

    # TR code (CNJ[14:16]) -> UF, aprendido das paginas que JA tem CNJ-20 + origem
    tr_counter = collections.Counter()
    for p in pages:
        u = digits(t(p, "numero_processo"))
        if len(u) >= 20:
            m = re.search(r"/([A-Z]{2})$", t(p, "origem"))
            if m:
                tr_counter[(u[14:16], m.group(1))] += 1
    TR2UF: dict[str, str] = {}
    for (tr, uf), _n in tr_counter.most_common():
        TR2UF.setdefault(tr, uf)

    inc = []
    for p in pages:
        bd = digits(t(p, "numero_processo"))
        if len(bd) >= 20:
            continue
        m = re.search(r"/([A-Z]{2})$", t(p, "origem"))
        inc.append({
            "id": p["id"],
            "cur": t(p, "numero_processo"),
            "proc": bd,
            "data": (t(p, "data_sessao") or "")[:10],
            "uf": m.group(1) if m else "",
            "partes": {fold(x) for x in parse_multi_value_text(t(p, "partes"))},
        })
    procs = {x["proc"] for x in inc if x["proc"]}
    LOGGER.info("incompletas: %s | com proc: %s | procs distintos: %s", len(inc), sum(1 for x in inc if x["proc"]), len(procs))

    by_proc: dict[str, set] = collections.defaultdict(set)
    by_proc_date: dict[tuple, set] = collections.defaultdict(set)
    by_proc_partes: dict[str, dict] = collections.defaultdict(lambda: collections.defaultdict(set))
    with open(args.input, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            proc = digits(row.get("numeroProcesso"))
            if proc not in procs:
                continue
            uni = digits(row.get("numeroUnico"))[:20]
            if len(uni) < 20:
                continue
            by_proc[proc].add(uni)
            by_proc_date[(proc, iso(row.get("dataDecisao")))].add(uni)
            for pt in parse_multi_value_text(row.get("partes", "")):
                by_proc_partes[proc][uni].add(fold(pt))
    LOGGER.info("indexados %s numeros do CSV completo", len(by_proc))

    changes: list[dict[str, Any]] = []
    stats = collections.Counter()
    for x in inc:
        proc = x["proc"]
        if not proc:
            stats["sem_proc"] += 1
            continue
        sane = {u for u in by_proc.get(proc, set()) if u[:9].lstrip("0") == proc.lstrip("0")}
        if not sane:
            stats["sem_candidato"] += 1
            continue
        via = ""
        hit = None
        # 1) proc + data exata
        bdt = sane & by_proc_date.get((proc, x["data"]), set()) if x["data"] else set()
        if len(bdt) == 1:
            hit, via = next(iter(bdt)), "proc+data"
        # 2) proc unico global
        elif len(sane) == 1:
            hit, via = next(iter(sane)), "proc_unico"
        # 3) proc + UF
        elif x["uf"]:
            byuf = {u for u in sane if TR2UF.get(u[14:16]) == x["uf"]}
            if len(byuf) == 1:
                hit, via = next(iter(byuf)), "proc+uf"
        # 4) proc + partes
        if hit is None and x["partes"]:
            byp = {u for u in sane if x["partes"] & by_proc_partes[proc].get(u, set())}
            if len(byp) == 1:
                hit, via = next(iter(byp)), "proc+partes"
        if hit is None:
            stats["ambiguo"] += 1
            continue
        stats["completados"] += 1
        stats[via] += 1
        formatted = format_cnj(hit)
        rec = {"page_id": x["id"], "data": x["data"], "uf": x["uf"], "old": x["cur"], "new": formatted, "via": via}
        if args.apply:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{x['id']}",
                                          json={"properties": {"numero_processo": client._build_property_value(schema, "numero_processo", formatted)}})
                rec["status"] = "updated"; stats["applied"] += 1
            except Exception as exc:
                rec["status"] = "failed"; rec["error"] = str(exc); stats["failed"] += 1
            time.sleep(0.12)
        changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **dict(stats)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:15]:
        LOGGER.info("  [%s] %s -> %s (%s)", c["data"], c["old"], c["new"], c["via"])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
