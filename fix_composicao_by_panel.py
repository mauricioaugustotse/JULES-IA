"""Corrige a coluna composicao usando o PADRAO/QUORUM da SESSAO (insight do usuario:
o painel julgador e estavel; o mesmo time julga TODOS os casos do dia e repete em datas
proximas). Para cada data_sessao, computa o PAINEL CONSENSO = ministros que aparecem na
maioria dos casos daquele dia, classificados em STF/STJ/jurista e limitados a 3/2/2.
Quando o consenso e um 3+2+2 COMPLETO (=7), substitui a composicao dos casos daquela data
que VIOLAM (excesso de categoria / mero presente) pelo painel. Conservador: so atua com
painel completo (7); datas com poucos casos ou consenso incompleto sao puladas (revisao).

Uso:
  python fix_composicao_by_panel.py            # dry-run
  python fix_composicao_by_panel.py --apply
"""
from __future__ import annotations

import argparse, json, logging, time
from collections import Counter, defaultdict
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

LOGGER = logging.getLogger("fix_composicao_by_panel")
ARTIFACT_ROOT = Path("artifacts") / "notion_composicao_panel"
CAPS = {"STF": 3, "STJ": 2, "jurista": 2}
MIN_CASES = 3  # data precisa de >=3 casos p/ estabelecer padrao


def cat(canon: str) -> str:
    if canon in MINISTROS_STF:
        return "STF"
    if canon in MINISTROS_STJ:
        return "STJ"
    if canon in MINISTROS_JURISTAS:
        return "jurista"
    return "?"


def canon_list(values: list[str]) -> list[str]:
    return dedupe_preserve_order([(normalize_ministro_name(v) or v) for v in values])


def violates(comp: list[str]) -> bool:
    c = Counter(cat(m) for m in comp)
    return c["STF"] > 3 or c["STJ"] > 2 or c["jurista"] > 2 or c["?"] > 0


def panel_for(cases: list[list[str]]):
    """Painel consenso da data: por categoria, os mais frequentes (>= 60% dos casos) ate o cap.
    Retorna o painel SO se for um 3+2+2 completo (=7); senao None (baixa confianca)."""
    n = len(cases)
    if n < MIN_CASES:
        return None
    freq: Counter = Counter()
    for comp in cases:
        for m in set(comp):
            freq[m] += 1
    thr = max(2, (n * 6 + 9) // 10)  # ~ceil(0.6n)
    panel: list[str] = []
    for category, capn in CAPS.items():
        cand = sorted(((freq[m], m) for m in freq if cat(m) == category and freq[m] >= thr), reverse=True)
        panel += [m for _, m in cand[:capn]]
    counts = Counter(cat(m) for m in panel)
    if counts["STF"] == 3 and counts["STJ"] == 2 and counts["jurista"] == 2 and len(panel) == 7:
        return panel
    return None


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

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    by_date: dict[str, list[tuple[Any, list[str]]]] = defaultdict(list)
    for p in pages:
        comp = canon_list(parse_multi_value_text(t(p, "composicao")))
        if not comp:
            continue
        date = (t(p, "data_sessao") or "")[:10]
        if not date:
            continue
        by_date[date].append((p, comp))

    changes: list[dict[str, Any]] = []
    stats = Counter()
    panels_ok = 0
    for date, cases in by_date.items():
        panel = panel_for([c for _, c in cases])
        if panel is None:
            stats["datas_sem_painel"] += 1
            continue
        panels_ok += 1
        panel_set = set(panel)
        for p, comp in cases:
            cs = set(comp)
            if cs == panel_set:
                continue  # ja e exatamente o painel (qualquer ordem)
            subset = cs <= panel_set
            # A data so chega aqui se o consenso e um 3+2+2 COMPLETO (=7) -> a sessao TEVE
            # os 7 (ausencia real deixaria o consenso <7 e a data seria pulada). Logo,
            # corrige EXCESSO (violates) e qualquer SUBCONJUNTO (sub-extracao). Deixa de
            # fora so VALIDO-mas-DIFERENTE do painel (ministro fora do painel = troca) -> revisao.
            if not (violates(comp) or subset):
                stats["divergentes_revisao"] += 1
                continue
            stats["paginas_corrigidas"] += 1
            rec = {"page_id": p["id"], "numero": t(p, "numero_processo"), "data": date,
                   "old": comp, "new": panel}
            if args.apply:
                props = {"composicao": {"multi_select": [{"name": n} for n in panel]}}
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
    summary = {"mode": "apply" if args.apply else "dry-run", "datas": len(by_date),
               "datas_com_painel_322": panels_ok, **dict(stats)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:12]:
        LOGGER.info("  [%s %s] %s -> painel %s", c["data"], c["numero"], len(c["old"]), len(c["new"]))
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
