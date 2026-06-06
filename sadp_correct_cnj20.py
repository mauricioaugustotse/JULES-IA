"""TAREFA 2: corrige/enriquece via SADP os processos que JA TEM CNJ-20 mas com partes/origem
INCOMPLETAS ou advogados vazios (ex.: parte 'Claudionor' em 0000510-82). Como o CNJ identifica
o processo de forma EXATA, busca por numero -> casa o candidato com CNJ identico -> abre o
detalhe -> corrige (merge) partes, enriquece advogados, preenche relator, padroniza origem.
Conservador: so atua quando acha o CNJ EXATO no SADP.

Uso:
  python sadp_correct_cnj20.py [--limit N]            # dry-run
  python sadp_correct_cnj20.py [--limit N] --apply
"""
from __future__ import annotations

import argparse, json, logging, re, time, unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from fill_partes_advogados_from_jurisprudencia import merge_names, proper_case
from local_secrets import get_secret
from sadp_lookup import fetch_detail, make_session, search_numunico
from tse_normalization import (
    dedupe_preserve_order, normalize_advogado_name, normalize_ministro_name,
    normalize_partes_list, parse_multi_value_text,
)
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("sadp_correct_cnj20")
ARTIFACT_ROOT = Path("artifacts") / "notion_sadp_correct_cnj20"
_CONN = {"de", "da", "do", "dos", "das", "e"}


def _sig_tokens(v: str) -> list[str]:
    base = re.sub(r"\([^)]*\)", " ", str(v or ""))
    base = unicodedata.normalize("NFKD", base.lower())
    base = "".join(c for c in base if not unicodedata.combining(c))
    return [t for t in base.split() if len(t) > 1 and t not in _CONN]


def _parte_incompleta(partes: list[str]) -> bool:
    # alguma parte-PESSOA com 1 token significativo (nome solto, ex.: 'Claudionor')
    return any(len(_sig_tokens(p)) <= 1 and not re.search(r"(?i)partido|coliga|federa|tribunal|minist", p) for p in partes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.8)
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    alvos = []
    for p in pages:
        d = re.sub(r"\D", "", t(p, "numero_processo") or "")
        if len(d) < 20:
            continue
        base_partes = parse_multi_value_text(t(p, "partes"))
        if _parte_incompleta(base_partes) or not base_partes:  # foco: partes incompleta/vazia
            alvos.append(p)
    if args.limit and args.limit < len(alvos):
        alvos = alvos[::max(1, len(alvos) // args.limit)][:args.limit]
    LOGGER.info("CNJ-20 a corrigir (partes incompletas / advogados vazios): %s", len(alvos))

    sess = make_session()
    changes: list[dict[str, Any]] = []
    stats = {"alvos": len(alvos), "achou_cnj": 0, "corrige_partes": 0, "grava_partes": 0, "corrige_adv": 0,
             "grava_adv": 0, "grava_relator": 0, "grava_origem": 0, "nao_achou": 0, "applied": 0, "falhas": 0}
    for p in alvos:
        cnj = (t(p, "numero_processo") or "").strip()
        try:
            cands = search_numunico(sess, cnj)
        except Exception:
            stats["falhas"] += 1; time.sleep(args.sleep); continue
        match = next((c for c in cands if re.sub(r"\D", "", c.get("cnj", "")) == re.sub(r"\D", "", cnj)), None)
        if not match or not match.get("nprot"):
            stats["nao_achou"] += 1; time.sleep(args.sleep); continue
        stats["achou_cnj"] += 1
        detail = fetch_detail(sess, match["nprot"], "tse")
        if not detail:
            time.sleep(args.sleep); continue
        base_partes = parse_multi_value_text(t(p, "partes"))
        base_advs = parse_multi_value_text(t(p, "advogados"))
        sadp_partes = parse_multi_value_text(normalize_partes_list(
            [proper_case(v.split(",")[0].strip()) for v in detail.get("partes", []) if v.split(",")[0].strip()]))
        sadp_advs = dedupe_preserve_order([a for a in (normalize_advogado_name(proper_case(v)) for v in detail.get("advogados", [])) if a])
        props: dict[str, Any] = {}
        rec = {"page_id": p["id"], "cnj": cnj}
        if sadp_partes:
            if not base_partes:
                props["partes"] = client._build_property_value(schema, "partes", sadp_partes); stats["grava_partes"] += 1
                rec["partes"] = sadp_partes
            else:
                merged = merge_names(base_partes, sadp_partes)
                if merged != base_partes:
                    props["partes"] = client._build_property_value(schema, "partes", merged); stats["corrige_partes"] += 1
                    rec["partes_corrige"] = f"{base_partes} -> {merged}"
        if sadp_advs:
            if not base_advs:
                props["advogados"] = client._build_property_value(schema, "advogados", sadp_advs); stats["grava_adv"] += 1
            else:
                merged_adv = merge_names(base_advs, sadp_advs)
                if merged_adv != base_advs:
                    props["advogados"] = client._build_property_value(schema, "advogados", merged_adv); stats["corrige_adv"] += 1
        if not (t(p, "relator") or "").strip() and detail.get("relator"):
            rel = normalize_ministro_name(detail["relator"])
            if rel:
                props["relator"] = client._build_property_value(schema, "relator", rel); stats["grava_relator"] += 1
        som = _fmt_origem(detail.get("municipio", ""))
        if som and som != (t(p, "origem") or "").strip():
            props["origem"] = client._build_property_value(schema, "origem", som); stats["grava_origem"] += 1
            rec["origem"] = f"{t(p,'origem')} -> {som}"
        if props:
            if args.apply:
                try:
                    notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                    stats["applied"] += 1
                except Exception as exc:
                    stats["falhas"] += 1; rec["erro"] = str(exc)
                time.sleep(0.15)
            changes.append(rec)
        time.sleep(args.sleep)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s | %s", json.dumps(summary, ensure_ascii=False), run_dir)
    return 0


def _fmt_origem(municipio: str) -> str:
    m = re.match(r"(.+?)\s*-\s*([A-Za-z]{2})\s*$", str(municipio or "").strip())
    return proper_case(m.group(1).strip()) + "/" + m.group(2).upper() if m else ""


if __name__ == "__main__":
    raise SystemExit(main())
