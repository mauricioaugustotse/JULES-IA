"""Harmonizacao Suspenso (camada CSV): para cada pagina 'Suspenso por vista', procura nos
CSVs de jurisprudencia uma DECISAO POSTERIOR a sessao de suspensao (= o caso voltou e foi
julgado). CNJ-20 casa por numeroUnico (confiavel); curto casa por numeroProcesso+UF com
janela de ate WINDOW anos (evita falso match com processo homonimo distante). Se achar ->
flip resultado 'Suspenso por vista'->'Suspenso mas julgado depois' e votacao 'Suspenso'->'Suspenso*'.

Uso:
  python suspenso_crosscheck_csv.py --input-dir "C:\\Users\\mauri\\Downloads"            # dry-run
  python suspenso_crosscheck_csv.py --input-dir "C:\\Users\\mauri\\Downloads" --apply
"""
from __future__ import annotations

import argparse, csv, datetime, glob, json, logging, re, time
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("suspenso_crosscheck_csv")
ARTIFACT_ROOT = Path("artifacts") / "notion_suspenso_crosscheck"
WINDOW_YEARS = 3
_UF = "AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO"
UF_RE = re.compile(rf"\b({_UF})\b")


def digits(s):
    return re.sub(r"\D", "", str(s or ""))


def parse_br(s):
    try:
        return datetime.datetime.strptime(str(s or "").strip(), "%d/%m/%Y").date()
    except Exception:
        return None


def parse_iso(s):
    try:
        return datetime.date.fromisoformat(str(s or "")[:10])
    except Exception:
        return None


def extract_uf(origem):
    m = UF_RE.search(str(origem or "").upper())
    return m.group(1) if m else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--input-dir", default=r"C:\Users\mauri\Downloads")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    cnj_latest: dict[str, datetime.date] = {}
    proc_latest: dict[tuple, datetime.date] = {}
    files = glob.glob(str(Path(args.input_dir) / "*.csv"))
    for path in files:
        try:
            with open(path, encoding="utf-8-sig", newline="") as h:
                for row in csv.DictReader(h):
                    d = parse_br(row.get("dataDecisao"))
                    if not d:
                        continue
                    cnj = digits(row.get("numeroUnico"))[:20]
                    if len(cnj) >= 20:
                        if d > cnj_latest.get(cnj, datetime.date.min):
                            cnj_latest[cnj] = d
                    proc = digits(row.get("numeroProcesso"))
                    uf = (row.get("siglaUF") or "").strip().upper()
                    if proc:
                        k = (proc, uf)
                        if d > proc_latest.get(k, datetime.date.min):
                            proc_latest[k] = d
        except Exception as exc:
            LOGGER.warning("falha lendo %s: %s", Path(path).name, exc)
    LOGGER.info("CSVs=%s | CNJ com decisao=%s | proc+uf com decisao=%s", len(files), len(cnj_latest), len(proc_latest))

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    changes: list[dict[str, Any]] = []
    stats = {"suspenso": 0, "flip_cnj": 0, "flip_curto": 0, "sem_evidencia": 0, "applied": 0, "failed": 0}
    for p in pages:
        if (t(p, "resultado") or "").strip() != "Suspenso por vista":
            continue
        stats["suspenso"] += 1
        ds = parse_iso(t(p, "data_sessao"))
        num = digits(t(p, "numero_processo"))
        later, via = None, ""
        if len(num) >= 20:
            later = cnj_latest.get(num[:20]); via = "cnj"
        elif num:
            uf = extract_uf(t(p, "origem"))
            if uf:
                later = proc_latest.get((num, uf)); via = "curto"
        ok = bool(later and ds and later > ds)
        if ok and via == "curto" and (later - ds).days > WINDOW_YEARS * 366:
            ok = False  # decisao muito distante: provavel homonimo
        if not ok:
            stats["sem_evidencia"] += 1
            continue
        stats["flip_cnj" if via == "cnj" else "flip_curto"] += 1
        rec = {"page_id": p["id"], "numero": t(p, "numero_processo"), "data_sessao": str(ds),
               "decisao_posterior": str(later), "via": via,
               "old": {"resultado": "Suspenso por vista", "votacao": t(p, "votacao")}}
        if args.apply:
            props = {"resultado": client._build_property_value(schema, "resultado", "Suspenso mas julgado depois"),
                     "votacao": client._build_property_value(schema, "votacao", "Suspenso*")}
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                rec["status"] = "updated"; stats["applied"] += 1
            except Exception as exc:
                rec["status"] = "failed"; rec["error"] = str(exc); stats["failed"] += 1
            time.sleep(0.15)
        changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
