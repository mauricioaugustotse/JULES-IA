"""#5: re-computa a relation 'materia_semelhante' (liga registros do MESMO processo = mesmo
CNJ-20 canonico, em video/data diferentes). Como o SADP completou muitos CNJ-20, mais registros
agora podem ser ligados. Usa CNJ completo com DV valido e preserva vinculos manuais fora
do escopo e identidades incertas. Corrige colisoes confirmadas do antigo numero curto.

Uso:
  python materia_semelhante_update.py            # dry-run
  python materia_semelhante_update.py --apply
"""
from __future__ import annotations

import argparse, json, logging, time
from collections import defaultdict
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

LOGGER = logging.getLogger("materia_semelhante_update")
ARTIFACT_ROOT = Path("artifacts") / "notion_materia_semelhante"


def _rel_ids(page: dict, field: str) -> list[str]:
    val = page.get("properties", {}).get(field, {})
    return [r.get("id") for r in (val.get("relation") or []) if r.get("id")] if val.get("type") == "relation" else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    if "materia_semelhante" not in schema.properties:
        LOGGER.error("coluna materia_semelhante nao existe"); return 1
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    recs = []
    for p in pages:
        number = t(p, "numero_processo")
        canonical = full_cnj_relation_key(number)
        recs.append({"page_id": p["id"], "canonical": canonical, "numero": number,
                     "video_id": extract_youtube_video_id(t(p, "youtube_link") or ""),
                     "data": t(p, "data_sessao"), "cur": _rel_ids(p, "materia_semelhante")})

    by_proc: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if r["canonical"]:
            by_proc[r["canonical"]].append(r)

    indexed = index_relation_records(
        ProcessRelationRecord(r["page_id"], r["numero"], r["video_id"], r["data"]) for r in recs
    )
    proposed = build_full_cnj_relation_targets(indexed.values())
    desired = {
        r["page_id"]: reconcile_process_relations(indexed[r["page_id"]], r["cur"], proposed.get(r["page_id"], []), indexed)
        for r in recs
    }

    changes, stats = [], {"com_cnj": sum(bool(r["canonical"]) for r in recs), "grupos_multi": sum(1 for g in by_proc.values() if len(g) >= 2),
                          "paginas_com_relacao": sum(bool(v) for v in desired.values()), "a_atualizar": 0, "applied": 0, "falhas": 0}
    cur_by_id = {r["page_id"]: r["cur"] for r in recs}
    for page_id, related in desired.items():
        if set(related) == set(cur_by_id.get(page_id, [])):
            continue
        stats["a_atualizar"] += 1
        current = cur_by_id.get(page_id, [])
        changes.append({"page_id": page_id, "old_n": len(current), "new_n": len(related),
                        "before": current, "after": related,
                        "added": sorted(set(related) - set(current)),
                        "removed": sorted(set(current) - set(related))})
        if args.apply:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{page_id}",
                                          json={"properties": {"materia_semelhante": {"relation": [{"id": i} for i in related]}}})
                stats["applied"] += 1
            except Exception as exc:
                stats["falhas"] += 1; LOGGER.warning("falha %s: %s", page_id, exc)
            time.sleep(0.15)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
