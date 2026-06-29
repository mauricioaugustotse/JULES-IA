"""Confronta/corrige metadados ESTRUTURAIS da base de sessoes a partir do CSV oficial do DJE,
casando por CNJ-20 (estes campos sao atributos do PROCESSO, independem da data da decisao):

  - eleicao  <- anoEleicao            (normalize_eleicao_value; so quando ha 20\\d{2})
  - origem   <- nomeMunicipio/siglaUF (normalize_origem_value; so quando ha municipio E uf)
  - tribunal <- siglaTribunalJE       (normalize_tre; o CSV de jurisprudencia e do TSE -> "TSE",
                                       util para corrigir paginas marcadas erroneamente como TRE-XX)

Espelha classe_from_jurisprudencia.py: dry-run por padrao, escreve via page-value PATCH
(client._build_property_value) com seguranca, gera artifacts/notion_metadata_juris/<ts>/.

Uso:
  python fill_metadata_from_jurisprudencia.py --input-dir "C:\\Users\\mauri\\ProjetoConversor\\dje_consolidado"          # dry-run
  python fill_metadata_from_jurisprudencia.py --input-dir "C:\\Users\\mauri\\ProjetoConversor\\dje_consolidado" --apply
"""
from __future__ import annotations

import argparse, csv, glob, json, logging, re, time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from fill_partes_advogados_from_jurisprudencia import proper_case
from local_secrets import get_secret
from tse_normalization import normalize_eleicao_value, normalize_origem_value
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

csv.field_size_limit(50 * 1024 * 1024)  # textoDecisao pode passar do default (128 KB) e abortar o DictReader
LOGGER = logging.getLogger("fill_metadata_from_jurisprudencia")
ARTIFACT_ROOT = Path("artifacts") / "notion_metadata_juris"
# 'tribunal' fica FORA: a base usa esse campo como tribunal de ORIGEM (TRE-XX bate a UF da origem
# em 2563 de 2614 casos); o CSV de jurisprudencia e do TSE (siglaTribunalJE=TSE p/ tudo), entao
# sobrescrever apagaria a origem. Confrontamos apenas eleicao e origem.
COLS = ("eleicao", "origem")


def digits(s) -> str:
    return re.sub(r"\D", "", str(s or ""))


def cf(s) -> str:
    return str(s or "").strip().casefold()


def csv_origem(row) -> str:
    mun = str(row.get("nomeMunicipio") or "").strip()
    uf = str(row.get("siglaUF") or "").strip()
    if not mun or len(uf) != 2:
        return ""
    return normalize_origem_value(f"{proper_case(mun)}/{uf.upper()}")


def load_meta(input_dirs: list[str]) -> dict[str, dict[str, str]]:
    """CNJ-20 -> {eleicao, origem, tribunal} (primeiro valor NAO-vazio visto por campo)."""
    meta: dict[str, dict[str, str]] = {}
    files: list[str] = []
    for d in input_dirs:
        files.extend(glob.glob(str(Path(d) / "*.csv")))
    for path in files:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                uni = digits(row.get("numeroUnico"))[:20]
                if len(uni) < 20:
                    continue
                rec = meta.setdefault(uni, {})
                if "eleicao" not in rec:
                    e = normalize_eleicao_value(row.get("anoEleicao", ""))
                    if e and re.fullmatch(r"20\d{2}", e):
                        rec["eleicao"] = e
                if "origem" not in rec:
                    o = csv_origem(row)
                    if o:
                        rec["origem"] = o
    LOGGER.info("CSVs: %s | processos (CNJ-20) com metadados: %s", len(files), len(meta))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--input-dir", action="append", default=None, help="Pasta(s) com CSVs do DJE.")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    input_dirs = args.input_dir or [r"C:\Users\mauri\ProjetoConversor\dje_consolidado"]
    meta = load_meta(input_dirs)

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    def normalized_current(col: str, raw: str) -> str:
        if col == "eleicao":
            return normalize_eleicao_value(raw)
        if col == "origem":
            return normalize_origem_value(raw)
        return str(raw or "").strip()

    changes: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"paginas": 0, "match": 0, "applied": 0, "failed": 0}
    for col in COLS:
        stats[col] = {"iguais": 0, "vazio_preenchido": 0, "diferente": 0}

    for p in pages:
        stats["paginas"] += 1
        cnj = digits(t(p, "numero_processo"))[:20]
        if len(cnj) < 20:
            continue
        rec = meta.get(cnj)
        if not rec:
            continue
        stats["match"] += 1
        props: dict[str, Any] = {}
        detail: dict[str, Any] = {}
        for col in COLS:
            new = rec.get(col)
            if not new:
                continue
            cur_raw = t(p, col)
            cur = normalized_current(col, cur_raw)
            if cf(cur) == cf(new):
                stats[col]["iguais"] += 1
                continue
            kind = "vazio_preenchido" if not cur_raw.strip() else "diferente"
            stats[col][kind] += 1
            props[col] = client._build_property_value(schema, col, new)
            detail[col] = {"old": cur_raw, "new": new, "kind": kind}
        if not props:
            continue
        rec_out = {"page_id": p["id"], "numero": t(p, "numero_processo"), "cnj": cnj, "detail": detail}
        if args.apply:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                rec_out["status"] = "updated"; stats["applied"] += 1
            except Exception as exc:
                rec_out["status"] = "failed"; rec_out["error"] = str(exc); stats["failed"] += 1
            time.sleep(0.12)
        changes.append(rec_out)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats, "paginas_com_mudanca": len(changes)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:15]:
        for col, d in c["detail"].items():
            LOGGER.info("  [%s] %s: %r -> %r (%s)", c["numero"], col, d["old"], d["new"], d["kind"])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
