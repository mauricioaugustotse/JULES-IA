"""Completa/UPGRADE a coluna classe_processo via CSV de jurisprudencia (descricaoClasse ->
forma canonica), casando por CNJ-20 + DATA EXATA. PROTECAO CONTRA DOWNGRADE: so troca quando
a classe do CSV e MAIS especifica que a base (mesma raiz + prefixo a mais, ex.: REspe ->
AgRg-REspe) ou a base esta vazia. Nunca rebaixa AgRg-REspe -> REspe nem troca por classe de
raiz diferente.

Uso:
  python classe_from_jurisprudencia.py --input-dir "C:\\Users\\mauri\\Downloads"          # dry-run
  python classe_from_jurisprudencia.py --input-dir "C:\\Users\\mauri\\Downloads" --apply
"""
from __future__ import annotations

import argparse, csv, glob, json, logging, re, time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import normalize_classe_processo
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, SAFE_DYNAMIC_SELECT_OPTIONS, NotionSessoesClient

csv.field_size_limit(50 * 1024 * 1024)  # textoDecisao pode passar do default (128 KB) e abortar o DictReader
LOGGER = logging.getLogger("classe_from_jurisprudencia")
ARTIFACT_ROOT = Path("artifacts") / "notion_classe_juris"


def digits(s):
    return re.sub(r"\D", "", str(s or ""))


def iso(d):
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", str(d or ""))
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else ""


def csv_classe(row, known):
    # sigla PRIMEIRO (AI/HC/AIJE/AC mapeiam melhor pela sigla que pela descricao por extenso);
    # so aceita se o resultado for uma classe CANONICA ja conhecida da base (evita gravar por
    # extenso, ex.: "Agravo de Instrumento", ou siglas-variantes do CSV como "REspEl"/"AREspEl").
    for field in ("siglaClasse", "descricaoClasse"):
        c = normalize_classe_processo(row.get(field, ""))
        if c and c in known:
            return c
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--input-dir", action="append", default=None)
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    # vocabulario canonico = classes que a base JA usa + as opcoes seguras conhecidas.
    known = {c for c in (normalize_classe_processo(t(p, "classe_processo")) for p in pages) if c}
    known |= set(SAFE_DYNAMIC_SELECT_OPTIONS.get("classe_processo", set()))

    input_dirs = args.input_dir or ["C:\\Users\\mauri\\Downloads"]
    juris: dict[tuple, str] = {}
    files = []
    for d in input_dirs:
        files.extend(glob.glob(str(Path(d) / "*.csv")))
    for path in files:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                uni = digits(row.get("numeroUnico"))[:20]
                dt = iso(row.get("dataDecisao"))
                if len(uni) < 20 or not dt:
                    continue
                c = csv_classe(row, known)
                if c:
                    juris[(uni, dt)] = c
    LOGGER.info("CSVs: %s | (cnj,data) com classe canonica: %s | vocab: %s", len(files), len(juris), len(known))

    changes: list[dict[str, Any]] = []
    stats = {"match": 0, "iguais": 0, "upgrade": 0, "vazio_preenchido": 0, "downgrade_evitado": 0, "diferente_mantido": 0, "applied": 0, "failed": 0}
    for p in pages:
        cnj = digits(t(p, "numero_processo"))[:20]
        date = (t(p, "data_sessao") or "")[:10]
        if len(cnj) < 20 or not date:
            continue
        nc = juris.get((cnj, date))
        if not nc:
            continue
        stats["match"] += 1
        base = t(p, "classe_processo")
        nb = normalize_classe_processo(base)
        if nc == nb:
            stats["iguais"] += 1
            continue
        if not nb:
            new = nc; stats["vazio_preenchido"] += 1
        elif nc.endswith(nb) and len(nc) > len(nb):
            new = nc; stats["upgrade"] += 1
        elif nb.endswith(nc) and len(nb) > len(nc):
            stats["downgrade_evitado"] += 1; continue
        else:
            stats["diferente_mantido"] += 1; continue
        rec = {"page_id": p["id"], "numero": t(p, "numero_processo"), "data": date, "old": base, "new": new}
        if args.apply:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}",
                                          json={"properties": {"classe_processo": client._build_property_value(schema, "classe_processo", new)}})
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
    for c in changes[:15]:
        LOGGER.info("  [%s] %r -> %r", c["numero"], c["old"], c["new"])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
