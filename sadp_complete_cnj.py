"""BACKFILL via SADP Push (TSE, sem captcha): completa numero_processo (CNJ-20) dos casos
incompletos do Notion e, para os 'Suspenso por vista', usa a SITUACAO do SADP para harmonizar
(Baixado/Decidido/etc. = resolvido -> flip 'Suspenso mas julgado depois' + 'Suspenso*').
Match conservador (sadp_lookup.best_match): exige municipio+UF (ou municipio+ano+classe);
ambiguidade/origem-vazia/numero-formato-novo -> lista de revisao, nao grava.

Uso:
  python sadp_complete_cnj.py [--limit N]            # dry-run
  python sadp_complete_cnj.py [--limit N] --apply
"""
from __future__ import annotations

import argparse, json, logging, re, time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from fill_partes_advogados_from_jurisprudencia import merge_names, proper_case
from sadp_lookup import best_match, fetch_detail, make_session, search_number, situacao_resolvido
from tse_normalization import (
    dedupe_preserve_order, normalize_advogado_name, normalize_classe_processo,
    normalize_ministro_name, normalize_partes_list, parse_multi_value_text,
)
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("sadp_complete_cnj")
ARTIFACT_ROOT = Path("artifacts") / "notion_sadp_complete"
UF_RE = re.compile(r"\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b")


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

    inc = [p for p in pages if 0 < len(re.sub(r"\D", "", t(p, "numero_processo") or "")) < 20]
    if args.limit and args.limit < len(inc):
        inc = inc[::max(1, len(inc) // args.limit)][:args.limit]

    sess = make_session()
    matches: list[dict] = []
    review: list[dict] = []
    stats = {"casos": len(inc), "match": 0, "grava_numero": 0, "grava_classe": 0, "classe_divergente": 0,
             "grava_partes": 0, "grava_advogados": 0, "grava_relator": 0, "grava_origem": 0,
             "classe_corrigida": 0, "corrige_partes": 0, "corrige_advogados": 0, "confirmado_por_parte": 0,
             "flip_suspenso": 0, "sem_candidato": 0, "ambiguo_revisao": 0, "applied_num": 0, "applied_flip": 0, "falhas": 0}
    for i, p in enumerate(inc, 1):
        num = re.sub(r"\D", "", t(p, "numero_processo"))
        origem = t(p, "origem") or ""
        ufm = UF_RE.search(origem.upper())
        uf = ufm.group(1) if ufm else ""
        muni = re.sub(r"[-/].*$", "", origem).strip()
        year = (t(p, "data_sessao") or "")[:4]
        classe = t(p, "classe_processo")
        try:
            cands = search_number(sess, num)
        except Exception as exc:
            stats["falhas"] += 1
            time.sleep(args.sleep)
            continue
        m = best_match(cands, muni, uf, year, classe)
        # auto-grava so com municipio batido (score>=5) -> confianca alta
        municipio_ok = bool(m and muni and len(muni) > 2 and _fold_in(muni, m.get("origem", "")))
        if not m or m.get("score", 0) < 5 or not municipio_ok:
            if cands:
                stats["ambiguo_revisao"] += 1
                review.append({"page_id": p["id"], "numero": t(p, "numero_processo"), "origem": origem,
                               "ano": year, "classe": classe, "n_cand": len(cands),
                               "candidatos": [{"cnj": c.get("cnj"), "origem": c.get("origem"), "sit": c.get("situacao")} for c in cands[:5]]})
            else:
                stats["sem_candidato"] += 1
            time.sleep(args.sleep)
            continue
        # DETALHE do SADP PRIMEIRO: precisamos das partes/advogados p/ CONFIRMAR o processo.
        detail = fetch_detail(sess, m.get("nprot", ""), "tse")
        base_partes = parse_multi_value_text(t(p, "partes"))
        base_advs = parse_multi_value_text(t(p, "advogados"))
        sadp_partes_raw = [v.split(",")[0].strip() for v in (detail.get("partes", []) if detail else []) if v.split(",")[0].strip()]
        sadp_advs_raw = [v for v in (detail.get("advogados", []) if detail else []) if v]
        confirmado = _confirma_processo(base_partes, base_advs, sadp_partes_raw, sadp_advs_raw)
        if confirmado:
            stats["confirmado_por_parte"] += 1
        ident = m.get("identificacao", "")
        code = re.split(r"[-_]+\d", ident)[0].strip("_- ") if ident else ""
        sadp_classe = normalize_classe_processo(code) if code else ""
        base_classe = normalize_classe_processo(classe)
        classe_diverge = bool(sadp_classe and base_classe and sadp_classe != base_classe and not base_classe.endswith(sadp_classe))
        # classe diverge SEM confirmacao por parte/advogado = provavel processo DIFERENTE -> revisao
        if classe_diverge and not confirmado:
            stats["classe_divergente"] += 1
            review.append({"page_id": p["id"], "numero": t(p, "numero_processo"), "motivo": "classe_divergente_nao_confirmado",
                           "cnj": m["cnj"], "base_classe": base_classe, "sadp_classe": sadp_classe, "origem": origem})
            time.sleep(args.sleep)
            continue
        stats["match"] += 1
        resolvido = situacao_resolvido(m.get("situacao", ""))
        is_susp = (t(p, "resultado") or "").strip() == "Suspenso por vista"
        props: dict[str, Any] = {"numero_processo": client._build_property_value(schema, "numero_processo", m["cnj"])}
        stats["grava_numero"] += 1
        rec = {"page_id": p["id"], "old_num": t(p, "numero_processo"), "cnj": m["cnj"], "situacao": m.get("situacao"),
               "origem": origem, "score": m["score"], "confirmado": confirmado}
        # CLASSE: se divergente-mas-CONFIRMADO -> CORRIGE (SADP autoritativo); se vazia -> preenche
        if classe_diverge and confirmado:
            props["classe_processo"] = client._build_property_value(schema, "classe_processo", sadp_classe)
            stats["classe_corrigida"] += 1; rec["classe_corrigida"] = f"{base_classe} -> {sadp_classe}"
        elif sadp_classe and not base_classe:
            props["classe_processo"] = client._build_property_value(schema, "classe_processo", sadp_classe)
            stats["grava_classe"] += 1; rec["classe_preenchida"] = sadp_classe
        if is_susp and resolvido is True:
            props["resultado"] = client._build_property_value(schema, "resultado", "Suspenso mas julgado depois")
            props["votacao"] = client._build_property_value(schema, "votacao", "Suspenso*")
            stats["flip_suspenso"] += 1
            rec["flip"] = "Suspenso por vista -> Suspenso mas julgado depois"
        if detail:
            sadp_partes = parse_multi_value_text(normalize_partes_list([proper_case(x) for x in sadp_partes_raw]))
            sadp_advs = dedupe_preserve_order([a for a in (normalize_advogado_name(proper_case(v)) for v in sadp_advs_raw) if a])
            # PARTES: preenche se vazia; se CONFIRMADO, corrige incompleto (merge prefere a forma completa)
            if not base_partes and sadp_partes:
                props["partes"] = client._build_property_value(schema, "partes", sadp_partes)
                stats["grava_partes"] += 1; rec["partes"] = sadp_partes
            elif confirmado and sadp_partes:
                merged = merge_names(base_partes, sadp_partes)
                if merged != base_partes:
                    props["partes"] = client._build_property_value(schema, "partes", merged)
                    stats["corrige_partes"] += 1; rec["partes_corrige"] = f"{base_partes} -> {merged}"
            # ADVOGADOS: preenche se vazio; se CONFIRMADO, enriquece (merge)
            if not base_advs and sadp_advs:
                props["advogados"] = client._build_property_value(schema, "advogados", sadp_advs)
                stats["grava_advogados"] += 1; rec["advogados"] = sadp_advs
            elif confirmado and sadp_advs:
                merged_adv = merge_names(base_advs, sadp_advs)
                if merged_adv != base_advs:
                    props["advogados"] = client._build_property_value(schema, "advogados", merged_adv)
                    stats["corrige_advogados"] += 1
            if not (t(p, "relator") or "").strip() and detail.get("relator"):
                rel = normalize_ministro_name(detail["relator"])
                if rel:
                    props["relator"] = client._build_property_value(schema, "relator", rel)
                    stats["grava_relator"] += 1; rec["relator"] = rel
            # ORIGEM oficial 'Municipio/UF' -> corrige se difere (match confirma o municipio)
            sadp_origem = _fmt_origem(detail.get("municipio", ""))
            cur_origem = (t(p, "origem") or "").strip()
            if sadp_origem and sadp_origem != cur_origem:
                props["origem"] = client._build_property_value(schema, "origem", sadp_origem)
                stats["grava_origem"] += 1; rec["origem"] = f"{cur_origem} -> {sadp_origem}"
            time.sleep(args.sleep * 0.5)
        if args.apply:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                stats["applied_num"] += 1
                if "resultado" in props:
                    stats["applied_flip"] += 1
            except Exception as exc:
                stats["falhas"] += 1; rec["erro"] = str(exc)
            time.sleep(0.15)
        matches.append(rec)
        if i % 50 == 0:
            LOGGER.info("  ... %s/%s | match=%s revisao=%s sem_cand=%s", i, len(inc), stats["match"], stats["ambiguo_revisao"], stats["sem_candidato"])
        time.sleep(args.sleep)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "matches.json").write_text(json.dumps(matches, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s | %s", json.dumps(summary, ensure_ascii=False), run_dir)
    return 0


def _foldtxt(x: str) -> str:
    import unicodedata
    x = unicodedata.normalize("NFKD", str(x or "").lower())
    return re.sub(r"\s+", " ", "".join(c for c in x if not unicodedata.combining(c))).strip()


def _confirma_processo(base_partes, base_advs, sadp_partes_raw, sadp_advs_raw) -> bool:
    """Confirma ser O MESMO processo: um nome de PARTE/ADVOGADO da base aparece nos do SADP.
    >=2 tokens batendo = forte; 1 token (>=4 letras, ex.: 'Claudionor') tambem confirma pois o
    match ja e por numero+municipio+ano."""
    blob = _foldtxt(" | ".join(list(sadp_partes_raw) + list(sadp_advs_raw)))
    blob_toks = set(blob.split())
    for name in list(base_partes) + list(base_advs):
        toks = [tk for tk in _foldtxt(name).split() if len(tk) > 2]
        if len(toks) >= 2 and sum(1 for tk in toks if tk in blob) >= 2:
            return True
        if len(toks) == 1 and len(toks[0]) >= 4 and toks[0] in blob_toks:
            return True
    return False


def _fmt_origem(municipio: str) -> str:
    """'SÃO PAULO - SP' -> 'São Paulo/SP' (convencao do projeto)."""
    m = re.match(r"(.+?)\s*-\s*([A-Za-z]{2})\s*$", str(municipio or "").strip())
    return proper_case(m.group(1).strip()) + "/" + m.group(2).upper() if m else ""


def _fold_in(needle: str, haystack: str) -> bool:
    import unicodedata
    def f(x):
        x = unicodedata.normalize("NFKD", str(x or "").lower())
        return "".join(c for c in x if not unicodedata.combining(c))
    return f(needle) in f(haystack)


if __name__ == "__main__":
    raise SystemExit(main())
