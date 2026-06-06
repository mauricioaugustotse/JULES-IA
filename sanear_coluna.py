"""Saneamento DEFINITIVO de uma coluna multi_select (advogados / composicao):
- funde GEMEOS de acento/caixa (mesmo fold) -> canonico = forma mais acentuada/completa;
- aplica correcoes ESPECIFICAS (typos, prefixo junk, nomes colados que viram split);
- remove fragmentos junk; reescreve cada pagina (page-value) + dedup preservando ordem.

Uso:
  python sanear_coluna.py --column composicao            # dry-run
  python sanear_coluna.py --column advogados --apply
"""
from __future__ import annotations

import argparse, collections, json, logging, re, time, unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import dedupe_preserve_order, parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("sanear_coluna")
ARTIFACT_ROOT = Path("artifacts") / "notion_sanear_coluna"

# correcoes especificas por coluna: valor -> str (renomeia) | list (split) | "" (remove)
ESPECIFICO: dict[str, dict[str, Any]] = {
    "composicao": {
        "Min. AIexandre de Moraes": "Min. Alexandre de Moraes",
        "Min. Luis Fetipe Salomão": "Min. Luís Felipe Salomão",
        "Min. Ministra: Rosa Weber": "Min. Rosa Weber",
        "Min. s Luís Roberto Barroso": "Min. Luís Roberto Barroso",
        "Min. Carlos Mário da Silva Velloso Filho": "Min. Carlos Mário Velloso Filho",
        "Min. Edson Fachin Og Fernandes": ["Min. Edson Fachin", "Min. Og Fernandes"],
    },
    "advogados": {
        "Albuquerque": "", "abreu Lemos": "", "agostini Boari": "",
        "ana Tamasaukas": "Dra. Ana Tamasaukas", "inti Ali Miranda Faiad": "Dr. Inti Ali Miranda Faiad",
    },
}


_ADV_BOUND = re.compile(r"\s+(?=Dra?\.\s)")


def split_multi_adv(v: str) -> list[str]:
    """'Dr. X e Dr. Y' / 'Dr. X Dra. Y' -> ['Dr. X','Dr. Y'] (so quando ha 2+ prefixos Dr./Dra.)."""
    if len(re.findall(r"(?i)\bDra?\.", v)) < 2:
        return [v]
    parts = [re.sub(r"(?i)[\s,]+e$", "", p.strip()).strip(" ,") for p in _ADV_BOUND.split(v) if p.strip()]
    return [p for p in parts if p] or [v]


def fold(x: str) -> str:
    x = unicodedata.normalize("NFKD", str(x or "").lower())
    return re.sub(r"\s+", " ", "".join(ch for ch in x if not unicodedata.combining(ch))).strip()


def diac(x: str) -> int:
    return sum(1 for ch in unicodedata.normalize("NFKD", str(x)) if unicodedata.combining(ch))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--column", required=True, choices=["advogados", "composicao"])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    col = args.column

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    distinct = collections.Counter()
    for p in pages:
        for v in parse_multi_value_text(t(p, col)):
            distinct[v] += 1

    # 1) gemeos de acento/caixa -> canonico (mais diacritico, depois mais comum, depois maior)
    byf: dict[str, list[str]] = collections.defaultdict(list)
    for v in distinct:
        byf[fold(v)].append(v)
    canon_fold: dict[str, str] = {}
    for k, vs in byf.items():
        vs = list(dict.fromkeys(vs))
        if len(vs) > 1:
            best = max(vs, key=lambda v: (diac(v), distinct[v], len(v)))
            for v in vs:
                if v != best:
                    canon_fold[v] = best

    espec = ESPECIFICO.get(col, {})

    def mapv(v: str):
        if v in espec:
            return espec[v]  # str | list | ""
        return canon_fold.get(v, v)

    changes: list[dict[str, Any]] = []
    stats = {"distintos": len(distinct), "gemeos": len(canon_fold), "especificos": len(espec),
             "paginas_alteradas": 0, "applied": 0, "falhas": 0}
    for p in pages:
        cur = parse_multi_value_text(t(p, col))
        if not cur:
            continue
        new: list[str] = []
        for v in cur:
            mv = mapv(v)
            if mv == "":
                continue
            for piece in (mv if isinstance(mv, list) else [mv]):
                if col == "advogados":
                    new.extend(split_multi_adv(piece))
                else:
                    new.append(piece)
        new = dedupe_preserve_order(new)
        if new == cur:
            continue
        stats["paginas_alteradas"] += 1
        rec = {"page_id": p["id"], "old": cur, "new": new}
        if args.apply:
            props = {col: client._build_property_value(schema, col, new) or client._build_empty_property_value(schema, col)}
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                stats["applied"] += 1
            except Exception as exc:
                stats["falhas"] += 1; rec["erro"] = str(exc)
            time.sleep(0.15)
        changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "canon_fold.json").write_text(json.dumps(canon_fold, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", "column": col, **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for v, c in list(canon_fold.items())[:20]:
        LOGGER.info("  gemeo: %r -> %r", v, c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
