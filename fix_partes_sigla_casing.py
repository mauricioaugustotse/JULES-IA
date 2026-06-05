"""Padroniza a CAIXA das siglas de partido/federacao entre parenteses na coluna partes
(ex.: '(Pl)'->'(PL)', '(Pdt)'->'(PDT)'). SEGURO: so altera quando o token entre parenteses
e uma SIGLA conhecida (dicionario), preservando casing especial ('(PCdoB)') e nao tocando
em parenteses que nao sao sigla ('(filho)', '(Missão)'). Escrita page-value.

Uso:
  python fix_partes_sigla_casing.py            # dry-run
  python fix_partes_sigla_casing.py --apply
"""
from __future__ import annotations

import argparse, json, logging, re, time, unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import dedupe_preserve_order, parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("fix_partes_sigla_casing")
ARTIFACT_ROOT = Path("artifacts") / "notion_partes_sigla_casing"

# lowercase -> sigla canonica (so siglas REAIS de partido/federacao)
PARTY_SIGLAS = {
    "pt": "PT", "pl": "PL", "pdt": "PDT", "mdb": "MDB", "psdb": "PSDB", "pp": "PP",
    "psd": "PSD", "pcdob": "PCdoB", "psol": "PSOL", "pv": "PV", "rede": "REDE",
    "novo": "NOVO", "podemos": "PODEMOS", "psb": "PSB", "ptb": "PTB", "pros": "PROS",
    "solidariedade": "SOLIDARIEDADE", "avante": "AVANTE", "patriota": "PATRIOTA",
    "pmn": "PMN", "prtb": "PRTB", "dc": "DC", "pcb": "PCB", "pco": "PCO", "pstu": "PSTU",
    "up": "UP", "agir": "AGIR", "dem": "DEM", "psc": "PSC", "phs": "PHS", "prb": "PRB",
    "pps": "PPS", "psl": "PSL", "pmb": "PMB", "ptc": "PTC", "psdc": "PSDC", "prd": "PRD",
    "uniao": "UNIÃO", "cidadania": "CIDADANIA", "republicanos": "REPUBLICANOS",
    "upb": "UPB", "ppl": "PPL", "pen": "PEN", "phs": "PHS", "pmdb": "PMDB",
}
PAREN_RE = re.compile(r"\(([^()/]+)\)")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").strip().lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def correct_one(value: str) -> str:
    def repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        canon = PARTY_SIGLAS.get(_norm(inner))
        if canon and inner != canon:
            return f"({canon})"
        return m.group(0)
    return PAREN_RE.sub(repl, value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    client = NotionSessoesClient(api_key=key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    changes: list[dict[str, Any]] = []
    stats = {"paginas": 0, "afetadas": 0, "trocas": 0, "applied": 0, "failed": 0}
    for p in pages:
        stats["paginas"] += 1
        cur = parse_multi_value_text(client._extract_property_text(p, schema, "partes"))
        if not cur:
            continue
        fixed = dedupe_preserve_order([correct_one(v) for v in cur])
        if fixed == cur:
            continue
        diffs = [(a, correct_one(a)) for a in cur if a != correct_one(a)]
        stats["afetadas"] += 1
        stats["trocas"] += len(diffs)
        rec = {"page_id": p["id"], "numero": client._extract_property_text(p, schema, "numero_processo"),
               "trocas": [f"{a} -> {b}" for a, b in diffs]}
        if args.apply:
            props = {"partes": client._build_property_value(schema, "partes", fixed) or client._build_empty_property_value(schema, "partes")}
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                rec["status"] = "updated"; stats["applied"] += 1
            except Exception as exc:
                rec["status"] = "failed"; rec["error"] = str(exc); stats["failed"] += 1
            time.sleep(0.2)
        changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:30]:
        LOGGER.info("  [%s] %s", c["numero"], "; ".join(c["trocas"]))
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
