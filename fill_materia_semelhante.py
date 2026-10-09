"""Atualiza a relation `materia_semelhante` vinculando entre si os varios julgamentos do MESMO
processo (mesmo CNJ canonico) em sessoes/datas distintas (ex.: Julgamento 1 suspenso por vista +
Julgamento 2 conclusivo). Mesma logica de `audit_notion_sessoes_round2.build_relation_targets`:
agrupa por CNJ completo com DV valido e exclui pares da mesma sessao. Preserva vinculos
manuais fora do escopo e identidades incertas; corrige colisoes do antigo numero curto.
dry-run por padrao.

Uso:
  python fill_materia_semelhante.py            # dry-run
  python fill_materia_semelhante.py --apply
"""
from __future__ import annotations

import argparse, collections, json, logging, time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from notion_process_relations import (
    ProcessRelationRecord,
    build_full_cnj_relation_targets,
    full_cnj_relation_key,
    index_relation_records,
    reconcile_process_relations,
)
from tse_normalization import extract_youtube_video_id
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("fill_materia_semelhante")
ARTIFACT_ROOT = Path("artifacts") / "notion_materia_semelhante"
RELATION_FIELD = "materia_semelhante"


def relation_ids(page: dict) -> list[str]:
    return [it.get("id", "") for it in page.get("properties", {}).get(RELATION_FIELD, {}).get("relation", []) if it.get("id")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    if RELATION_FIELD not in schema.properties:
        LOGGER.error("Propriedade %r nao existe na base.", RELATION_FIELD)
        return 1
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    # A identidade completa evita colisoes entre anos, tribunais e zonas.
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for p in pages:
        canon = full_cnj_relation_key(t(p, "numero_processo"))
        if canon:
            groups[canon].append(p)

    meta = {p["id"]: (extract_youtube_video_id(t(p, "youtube_link")), (t(p, "data_sessao") or "")[:10]) for p in pages}
    indexed = index_relation_records(
        ProcessRelationRecord(p["id"], t(p, "numero_processo"), *meta[p["id"]]) for p in pages
    )
    proposed = build_full_cnj_relation_targets(indexed.values())

    changes: list[dict[str, Any]] = []
    stats = collections.Counter()
    stats["grupos_multi"] = sum(1 for g in groups.values() if len(g) >= 2)
    for canon, grp in groups.items():
        for p in grp:
            atual = relation_ids(p)
            novo = reconcile_process_relations(indexed[p["id"]], atual, proposed.get(p["id"], []), indexed)
            if set(novo) == set(atual):
                stats["ja_ok"] += 1
                continue
            stats["atualizadas"] += 1
            rec = {"page_id": p["id"], "cnj": canon, "numero": t(p, "numero_processo"),
                   "antes": len(atual), "depois": len(novo), "adicionados": sorted(set(novo) - set(atual)),
                   "removidos": sorted(set(atual) - set(novo))}
            if args.apply:
                try:
                    built = client._build_property_value(schema, RELATION_FIELD, novo)
                    notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": {RELATION_FIELD: built}})
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
        LOGGER.info("  [%s] %s: %s -> %s vinculos", c["cnj"], c["numero"], c["antes"], c["depois"])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
