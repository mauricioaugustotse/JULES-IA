"""Consolida valores da coluna partes que sao a MESMA entidade escrita de formas
diferentes. CONSERVADOR p/ nao fundir homonimos:
  - AUTO (--apply): funde apenas variantes de ACENTO/CAIXA (mesma string sem acento/caixa)
    -> escolhe a forma mais acentuada/completa como canonica.
  - REVISAO (review_subset.json): candidatos por token-subset (um nome contido no outro,
    mesmo 1o+ultimo token) sao SO REPORTADOS, nao aplicados (risco de homonimo).

Uso:
  python consolidate_partes_pessoa.py            # dry-run
  python consolidate_partes_pessoa.py --apply    # aplica SO os merges de acento/caixa
"""
from __future__ import annotations

import argparse, json, logging, re, time, unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import dedupe_preserve_order, parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("consolidate_partes_pessoa")
ARTIFACT_ROOT = Path("artifacts") / "notion_partes_consolida"
_PUNCT = re.compile(r"[^0-9a-zà-ÿ ]")
_LEGAL = re.compile(r"(?i)\b(partido|coliga|federa|diret|tribunal|minist|associa|sindicato|instituto|federacao|junta|c[aâ]mara|prefeitura|uni[aã]o)\b")


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", s)).strip()


def diacritics(s: str) -> int:
    return sum(1 for c in unicodedata.normalize("NFKD", str(s)) if unicodedata.combining(c))


def pick_canonical(variants: list[str]) -> str:
    return max(variants, key=lambda v: (diacritics(v), sum(1 for c in v if c.isupper()), len(v)))


def person_tokens(v: str) -> list[str]:
    base = re.sub(r"\([^)]*\)", " ", v)  # tira papel entre parenteses
    return [t for t in fold(base).split() if len(t) > 1 and t not in {"de", "da", "do", "dos", "das", "e"}]


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

    distinct: dict[str, int] = defaultdict(int)
    for p in pages:
        for v in parse_multi_value_text(client._extract_property_text(p, schema, "partes")):
            distinct[v] += 1

    # 1) merges de acento/caixa (SEGURO)
    by_fold: dict[str, list[str]] = defaultdict(list)
    for v in distinct:
        by_fold[fold(v)].append(v)
    canon_map: dict[str, str] = {}
    for k, vs in by_fold.items():
        vs = list(dict.fromkeys(vs))
        if len(vs) > 1:
            best = pick_canonical(vs)
            for v in vs:
                if v != best:
                    canon_map[v] = best

    # 2) candidatos por token-subset (SO REVISAO) — apenas PESSOAS (exclui PJ)
    persons = [v for v in distinct if not _LEGAL.search(v)]
    tok = {v: set(person_tokens(v)) for v in persons}
    review_subset = []
    for a in persons:
        ta = tok[a]
        if len(ta) < 2:
            continue
        for b in persons:
            if a is b:
                continue
            tb = tok[b]
            if len(tb) > len(ta) and ta < tb:  # a estritamente contido em b
                pa, pb = person_tokens(a), person_tokens(b)
                if pa and pb and pa[0] == pb[0] and pa[-1] == pb[-1]:
                    review_subset.append({"curto": a, "completo": b, "n_curto": distinct[a], "n_completo": distinct[b]})
                    break

    # aplica SO os merges de acento/caixa
    changes: list[dict[str, Any]] = []
    stats = {"distintos": len(distinct), "merges_acento_caixa": len(canon_map),
             "candidatos_subset_revisao": len(review_subset), "paginas_alteradas": 0, "applied": 0, "failed": 0}
    if canon_map:
        for p in pages:
            cur = parse_multi_value_text(client._extract_property_text(p, schema, "partes"))
            if not cur:
                continue
            new = dedupe_preserve_order([canon_map.get(v, v) for v in cur])
            if new == cur:
                continue
            stats["paginas_alteradas"] += 1
            rec = {"page_id": p["id"], "numero": client._extract_property_text(p, schema, "numero_processo"), "old": cur, "new": new}
            if args.apply:
                props = {"partes": {"multi_select": [{"name": n} for n in new]}}
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
    (run_dir / "merges_acento_caixa.json").write_text(json.dumps(canon_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "review_subset.json").write_text(json.dumps(review_subset, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for v, c in list(canon_map.items())[:20]:
        LOGGER.info("  merge: %r -> %r", v, c)
    LOGGER.info("Candidatos subset (revisao manual) em %s", run_dir / "review_subset.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
