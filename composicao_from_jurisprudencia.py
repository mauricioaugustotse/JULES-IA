"""Corrige composicao pela fonte AUTORITATIVA do CSV (trecho 'Composicao: Ministros ...' do
textoDecisao), casando por CNJ-20 + DATA EXATA (data_sessao == dataDecisao). So assim o CSV
e a composicao DAQUELA MESMA sessao (evita misturar decisoes/sessoes diferentes do processo).
Quando casa e difere, o CSV prevalece (e o registro oficial do acordao).

Uso:
  python composicao_from_jurisprudencia.py --input-dir "C:\\Users\\mauri\\Downloads"          # dry-run
  python composicao_from_jurisprudencia.py --input-dir "C:\\Users\\mauri\\Downloads" --apply
"""
from __future__ import annotations

import argparse, csv, glob, json, logging, re, time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import dedupe_preserve_order, normalize_ministro_name, parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("composicao_from_jurisprudencia")
ARTIFACT_ROOT = Path("artifacts") / "notion_composicao_juris"
COMP_RE = re.compile(r"Composi[çc][ãa]o:\s*(.+?)\.(?:\s|$)", re.IGNORECASE | re.DOTALL)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", str(s or ""))


def _iso(d: str) -> str:
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", str(d or ""))
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else ""


def parse_composicao(texto: str) -> list[str]:
    m = COMP_RE.search(texto or "")
    if not m:
        return []
    blob = re.sub(r"(?i)^\s*ministr[oa]s?\s+", "", m.group(1).strip())
    out: list[str] = []
    for piece in re.split(r",|\s+e\s+", blob):
        piece = re.sub(r"\([^)]*\)", "", piece).strip()
        if not piece:
            continue
        n = normalize_ministro_name(piece)
        if n and len(n.split()) <= 5:  # evita 2 ministros colados (ex.: 'Floriano ... Andre Ramos Tavares')
            out.append(n)
    return dedupe_preserve_order(out)


def load_composicoes(input_dirs: list[str]) -> dict[tuple[str, str], list[str]]:
    data: dict[tuple[str, str], list[str]] = {}
    files: list[str] = []
    for d in input_dirs:
        files.extend(glob.glob(str(Path(d) / "*.csv")))
    LOGGER.info("CSVs: %s", len(files))
    for path in files:
        try:
            with open(path, encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames or "numeroUnico" not in reader.fieldnames:
                    continue
                for row in reader:
                    num = (_digits(row.get("numeroUnico")) or _digits(row.get("numeroProcesso")))[:20]
                    date = _iso(row.get("dataDecisao"))
                    if len(num) < 20 or not date:
                        continue
                    comp = parse_composicao(row.get("textoDecisao", ""))
                    if not (6 <= len(comp) <= 8):
                        continue
                    key = (num, date)
                    prev = data.get(key)
                    if prev is None or len(comp) > len(prev):
                        data[key] = comp
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Falha %s: %s", Path(path).name, exc)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--input-dir", action="append", default=["C:\\Users\\mauri\\Downloads"])
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    juris = load_composicoes(args.input_dir)
    LOGGER.info("(CNJ, data) com composicao oficial: %s", len(juris))

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    changes: list[dict[str, Any]] = []
    stats = {"match_cnj_data": 0, "iguais": 0, "corrige": 0, "applied": 0, "failed": 0}
    for p in pages:
        num = _digits(t(p, "numero_processo"))[:20]
        date = (t(p, "data_sessao") or "")[:10]
        if len(num) < 20 or not date:
            continue
        official = juris.get((num, date))
        if not official:
            continue
        stats["match_cnj_data"] += 1
        cur = dedupe_preserve_order([normalize_ministro_name(x) or x for x in parse_multi_value_text(t(p, "composicao"))])
        if set(cur) == set(official):
            stats["iguais"] += 1
            continue
        stats["corrige"] += 1
        rec = {"page_id": p["id"], "numero": t(p, "numero_processo"), "data": date, "old": cur, "new": official}
        if args.apply:
            props = {"composicao": {"multi_select": [{"name": n} for n in official]}}
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
    for c in changes[:8]:
        LOGGER.info("  [%s %s] %s -> %s", c["data"], c["numero"], c["old"], c["new"])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
