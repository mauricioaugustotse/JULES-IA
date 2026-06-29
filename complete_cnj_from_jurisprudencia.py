"""Completa o numero_processo (CNJ-20) das paginas com numero INCOMPLETO, casando com os
CSVs de jurisprudencia por sinais FORTES e CONCORDANTES:
  (A) numeroProcesso(digitos da base) + dataDecisao == data_sessao -> CNJ unico, OU
  (B) dataDecisao == data_sessao + >=1 parte em comum -> CNJ unico.
So preenche quando ha CNJ UNICO e os sinais que disparam concordam. numeroProcesso sozinho
(sem data) e EXCLUIDO (risco de coincidencia entre anos/tribunais).

Uso:
  python complete_cnj_from_jurisprudencia.py --input-dir "C:\\Users\\mauri\\Downloads"          # dry-run
  python complete_cnj_from_jurisprudencia.py --input-dir "C:\\Users\\mauri\\Downloads" --apply
"""
from __future__ import annotations

import argparse, csv, glob, json, logging, re, time, unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

csv.field_size_limit(50 * 1024 * 1024)  # textoDecisao pode passar do default (128 KB) e abortar o DictReader
LOGGER = logging.getLogger("complete_cnj_from_jurisprudencia")
ARTIFACT_ROOT = Path("artifacts") / "notion_cnj_complete"


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--input-dir", action="append", default=None)
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    input_dirs = args.input_dir or ["C:\\Users\\mauri\\Downloads"]
    by_proc_date: dict[tuple, set] = defaultdict(set)
    by_date: dict[str, list] = defaultdict(list)
    files = []
    for d in input_dirs:
        files.extend(glob.glob(str(Path(d) / "*.csv")))
    for path in files:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                uni = digits(row.get("numeroUnico"))
                if len(uni) < 20:
                    continue
                dt = iso(row.get("dataDecisao"))
                by_proc_date[(digits(row.get("numeroProcesso")), dt)].add(uni)
                by_date[dt].append((uni, {fold(x) for x in parse_multi_value_text(row.get("partes", ""))}))
    LOGGER.info("CSVs: %s | (proc,data) chaves: %s", len(files), len(by_proc_date))

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    changes: list[dict[str, Any]] = []
    stats = {"incompletos": 0, "completados": 0, "por_proc_data": 0, "por_data_partes": 0, "applied": 0, "failed": 0}
    for p in pages:
        cur = t(p, "numero_processo")
        bd = digits(cur)
        if len(bd) >= 20:
            continue
        stats["incompletos"] += 1
        data = (t(p, "data_sessao") or "")[:10]
        if not data:
            continue
        partes = {fold(x) for x in parse_multi_value_text(t(p, "partes"))}
        # SO numeroProcesso + DATA (sinal que verifica o NUMERO). data+partes sozinho e
        # arriscado (casa outro processo com mesmas partes/dia) -> excluido.
        pd = by_proc_date.get((bd, data), set()) if bd else set()
        if len(pd) != 1:
            continue
        cnj20 = next(iter(pd))
        # SANIDADE: o numero da base tem que bater com o prefixo NNNNNNN-DD do CNJ (sem zeros a esq.)
        if cnj20[:9].lstrip("0") != bd.lstrip("0"):
            stats["sanidade_pulada"] = stats.get("sanidade_pulada", 0) + 1
            continue
        # corroboracao opcional: data+partes concorda?
        dp = {uni for uni, pt in by_date.get(data, []) if partes and len(partes & pt) >= 1}
        corrob = "+partes" if dp == pd else ""
        formatted = format_cnj(cnj20)
        stats["completados"] += 1
        stats["por_proc_data"] += 1
        rec = {"page_id": p["id"], "data": data, "old": cur, "new": formatted, "via": "proc+data" + corrob}
        if args.apply:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}",
                                          json={"properties": {"numero_processo": client._build_property_value(schema, "numero_processo", formatted)}})
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
    for c in changes[:10]:
        LOGGER.info("  [%s] %s -> %s (%s)", c["data"], c["old"], c["new"], c["via"])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
