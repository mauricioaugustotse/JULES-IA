"""#5: re-computa a relation 'materia_semelhante' (liga registros do MESMO processo = mesmo
CNJ-20 canonico, em video/data diferentes). Como o SADP completou muitos CNJ-20, mais registros
agora podem ser ligados. Conservador: so liga quando ha >=2 registros do mesmo processo em
contextos distintos (video ou data diferente). Atualiza so onde a relation difere.

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
from tse_normalization import canonicalize_numero_processo, dedupe_preserve_order, extract_youtube_video_id
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
        canonical = canonicalize_numero_processo(t(p, "numero_processo"))
        if not canonical:
            continue
        recs.append({"page_id": p["id"], "canonical": canonical,
                     "video_id": extract_youtube_video_id(t(p, "youtube_link") or ""),
                     "data": t(p, "data_sessao"), "cur": _rel_ids(p, "materia_semelhante")})

    by_proc: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_proc[r["canonical"]].append(r)

    desired: dict[str, list[str]] = {}
    for proc, group in by_proc.items():
        uniq, seen = [], set()
        for r in group:
            if r["page_id"] not in seen:
                seen.add(r["page_id"]); uniq.append(r)
        if len(uniq) < 2:
            continue
        for r in uniq:
            related = [o["page_id"] for o in uniq if o["page_id"] != r["page_id"]
                       and (o["video_id"] != r["video_id"] or o["data"] != r["data"])]
            if related:
                desired[r["page_id"]] = dedupe_preserve_order(related)

    changes, stats = [], {"com_cnj": len(recs), "grupos_multi": sum(1 for g in by_proc.values() if len(g) >= 2),
                          "paginas_com_relacao": len(desired), "a_atualizar": 0, "applied": 0, "falhas": 0}
    cur_by_id = {r["page_id"]: r["cur"] for r in recs}
    for page_id, related in desired.items():
        if set(related) == set(cur_by_id.get(page_id, [])):
            continue
        stats["a_atualizar"] += 1
        changes.append({"page_id": page_id, "old_n": len(cur_by_id.get(page_id, [])), "new_n": len(related)})
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
