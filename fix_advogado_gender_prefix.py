"""Corrige o tratamento Dr./Dra. dos advogados na base do Notion conforme o genero
inferido do PRIMEIRO nome (padronizacao pedida). CONSERVADOR:
  - "Dr. <nome feminino>"  -> "Dra. <nome>"   (erro comum; nome em FEMALE_NAME_HINTS ou
                                               terminado em 'a' fora das excecoes masculinas)
  - "Dra. <nome masculino explicito>" -> "Dr. <nome>"  (so se o 1o nome esta em MALE_NAME_HINTS)
Nao mexe em entradas sem prefixo (pessoas juridicas/instituicoes). Escrita page-value segura.

Uso:
  python fix_advogado_gender_prefix.py            # dry-run
  python fix_advogado_gender_prefix.py --apply
"""
from __future__ import annotations

import argparse, json, logging, re, time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import (
    FEMALE_NAME_HINTS, MALE_NAME_HINTS, dedupe_preserve_order,
    infer_advogado_prefix, normalize_token, parse_multi_value_text,
)
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("fix_advogado_gender_prefix")
ARTIFACT_ROOT = Path("artifacts") / "notion_advogado_gender_fix"
PREFIX_RE = re.compile(r"^(Dr\.|Dra\.)\s+", re.IGNORECASE)


def correct_one(value: str) -> str:
    m = PREFIX_RE.match(value or "")
    if not m:
        return value
    cur = m.group(1).lower()
    name = value[m.end():]
    first = normalize_token(name.split()[0]) if name.split() else ""
    inferred = infer_advogado_prefix(name)
    if cur == "dr." and inferred == "Dra.":
        return f"Dra. {name}"
    if cur == "dra." and first in MALE_NAME_HINTS:
        return f"Dr. {name}"
    return value


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
        cur = parse_multi_value_text(client._extract_property_text(p, schema, "advogados"))
        if not cur:
            continue
        fixed = dedupe_preserve_order([correct_one(v) for v in cur])
        if fixed == cur:
            continue
        diffs = [(a, b) for a, b in zip(cur, [correct_one(v) for v in cur]) if a != b]
        stats["afetadas"] += 1
        stats["trocas"] += len(diffs)
        rec = {"page_id": p["id"], "numero": client._extract_property_text(p, schema, "numero_processo"),
               "trocas": [f"{a} -> {b}" for a, b in diffs]}
        if args.apply:
            props = {"advogados": {"multi_select": [{"name": n} for n in fixed]}}
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
    for c in changes[:25]:
        LOGGER.info("  [%s] %s", c["numero"], "; ".join(c["trocas"]))
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
