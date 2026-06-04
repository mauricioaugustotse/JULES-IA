"""Auditor/corretor da coluna composicao no Notion segundo a regra 3+2+2:
3 ministros do STF + 2 do STJ + 2 juristas/advogados (=7). Pode haver AUSENCIA de
julgador, mas NUNCA deve constar quem so esteve PRESENTE (nao julgou).

AUTO (seguro, com --apply): canonicaliza cada nome (normalize_ministro_name) e remove
duplicatas. Com --prune-unknowns, remove tambem entradas que NAO resolvem para um
ministro conhecido (lixo/ mal-transcrito/ provavel mero presente).
NAO faz poda automatica de EXCEDENTE por categoria (qual remover exige saber quem
julgou) -> apenas REPORTA essas paginas para revisao manual.

Uso:
  python audit_composicao_322.py                      # dry-run (so relatorio)
  python audit_composicao_322.py --apply              # canonicaliza + dedup
  python audit_composicao_322.py --apply --prune-unknowns   # + remove desconhecidos
"""
from __future__ import annotations

import argparse, json, logging, time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import (
    MINISTROS_JURISTAS, MINISTROS_STF, MINISTROS_STJ,
    dedupe_preserve_order, normalize_ministro_name, parse_multi_value_text,
)
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("audit_composicao_322")
ARTIFACT_ROOT = Path("artifacts") / "notion_composicao_322"


def classify(name: str) -> tuple[str, str]:
    canon = normalize_ministro_name(name) or name
    if canon in MINISTROS_STF:
        return canon, "STF"
    if canon in MINISTROS_STJ:
        return canon, "STJ"
    if canon in MINISTROS_JURISTAS:
        return canon, "jurista"
    return canon, "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--prune-unknowns", action="store_true", help="Remove entradas que nao sao ministro conhecido.")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    client = NotionSessoesClient(api_key=key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    changes: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    unknown_names: Counter = Counter()
    stats = Counter()
    for p in pages:
        comp = parse_multi_value_text(client._extract_property_text(p, schema, "composicao"))
        if not comp:
            continue
        stats["com_composicao"] += 1
        pairs = [classify(v) for v in comp]
        # canonicaliza + dedup (mantendo ordem); opcionalmente remove desconhecidos
        kept = []
        for canon, cat in pairs:
            if args.prune_unknowns and cat == "?":
                continue
            kept.append(canon)
        kept = dedupe_preserve_order(kept)
        c = Counter(cat for _, cat in pairs)
        # AUSENCIA (total<7) e permitida; violacao real = EXCESSO por categoria OU DESCONHECIDO
        excesso = c["STF"] > 3 or c["STJ"] > 2 or c["jurista"] > 2
        unknown = c["?"]
        viola = excesso or unknown > 0
        if viola:
            stats["violam_322"] += 1
            if excesso:
                stats["excesso_categoria"] += 1
            if unknown:
                stats["tem_desconhecido"] += 1
            for canon, cat in pairs:
                if cat == "?":
                    unknown_names[canon] += 1
            review.append({"page_id": p["id"], "numero": client._extract_property_text(p, schema, "numero_processo"),
                           "contagem": {"STF": c["STF"], "STJ": c["STJ"], "jurista": c["jurista"], "desconhecido": c["?"]},
                           "composicao": comp})
        # escrita: so se canonicalizacao/dedup/prune mudou algo
        if kept and kept != comp:
            stats["paginas_alteradas"] += 1
            rec = {"page_id": p["id"], "numero": client._extract_property_text(p, schema, "numero_processo"),
                   "old": comp, "new": kept}
            if args.apply:
                props = {"composicao": {"multi_select": [{"name": n} for n in kept]}}
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
    (run_dir / "review_322.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "desconhecidos.json").write_text(
        json.dumps(unknown_names.most_common(), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", "prune_unknowns": args.prune_unknowns,
               "desconhecidos_distintos": len(unknown_names), **dict(stats)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    LOGGER.info("Paginas que VIOLAM 3+2+2 (revisao manual): %s | em %s", stats["violam_322"], run_dir / "review_322.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
