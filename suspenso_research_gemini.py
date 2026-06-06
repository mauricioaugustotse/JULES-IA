"""Harmonizacao Suspenso (camada Gemini+busca): para cada 'Suspenso por vista' SEM evidencia,
faz 1 chamada grounded (Google Search, econômica) pedindo ao Gemini para localizar o processo
no site do TSE/jurisprudencia/acompanhamento processual e dizer: (a) o numero CNJ-20 (se houver),
(b) se ja foi RESOLVIDO (retomado e concluido apos a vista) ou segue PENDENTE.
Acoes: se achar CNJ-20 -> grava numero_processo (e o caso fica pronto p/ DataJud); se 'resolvido'
-> flip resultado->'Suspenso mas julgado depois' e votacao->'Suspenso*'. 'pendente'/'indeterminado'
nao mexem nas etiquetas.

Uso:
  python suspenso_research_gemini.py --limit 5            # PILOTO (dry-run)
  python suspenso_research_gemini.py --limit 5 --apply    # piloto e grava
  python suspenso_research_gemini.py --apply              # todos os 'Suspenso por vista'
"""
from __future__ import annotations

import argparse, json, logging, re, time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import (
    DEFAULT_GEMINI_MODEL, DEFAULT_NOTION_DATA_SOURCE_ID, NewsEnrichmentResult, NotionSessoesClient,
    _build_gemini_rest_part, _extract_generate_content_grounding_urls,
    call_gemini_generate_content_rest, parse_multi_value_text,
)
from tse_normalization import canonicalize_numero_processo

LOGGER = logging.getLogger("suspenso_research_gemini")
ARTIFACT_ROOT = Path("artifacts") / "notion_suspenso_research"

SYSTEM = (
    "Voce e um assistente de ACOMPANHAMENTO PROCESSUAL do TSE. Use a busca (site oficial do TSE, "
    "jurisprudencia.tse.jus.br, consultas processuais e imprensa juridica) para localizar O MESMO "
    "processo julgado pelo Plenario do TSE e descobrir seu andamento ATUAL. Responda SOMENTE com um "
    "objeto JSON valido, sem texto fora dele."
)
PROMPT = (
    "Em {data} o Plenario do TSE iniciou o julgamento abaixo e houve PEDIDO DE VISTA (julgamento "
    "SUSPENSO). Localize o MESMO processo e diga o andamento de hoje.\n"
    "Dados:\n- classe: {classe}\n- numero (parcial/curto): {numero}\n- partes: {partes}\n"
    "- tema: {tema}\n- relator: {relator}\n- origem: {origem}\n\n"
    "Responda em JSON: {{\"numero_cnj\": \"<CNJ 20 digitos no formato NNNNNNN-DD.AAAA.6.UF.OOOO, ou \\\"\\\" se nao achar>\", "
    "\"situacao\": \"resolvido|pendente|indeterminado\", \"data_resolucao\": \"<AAAA-MM-DD ou \\\"\\\">\", "
    "\"resumo\": \"<1 frase curta>\"}}. "
    "'resolvido' = o julgamento foi retomado e CONCLUIDO depois da vista (ha decisao final). "
    "'pendente' = ainda aguarda retorno ao Plenario. 'indeterminado' = nao foi possivel confirmar. "
    "Seja economico: priorize fontes oficiais; nao invente numero."
)


def _json_from_text(text: str) -> dict:
    m = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    notion_key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    gemini_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not gemini_key:
        raise RuntimeError("Falta GEMINI_API_KEY/GOOGLE_API_KEY.")
    client = NotionSessoesClient(api_key=notion_key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    cands = [p for p in pages if (t(p, "resultado") or "").strip() == "Suspenso por vista"]
    if args.limit and args.limit < len(cands):
        step = max(1, len(cands) // args.limit)
        cands = cands[::step][:args.limit]

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    stats = {"casos": 0, "achou_cnj": 0, "resolvido": 0, "pendente": 0, "indeterminado": 0,
             "grava_numero": 0, "flip_resolvido": 0, "applied_num": 0, "applied_flip": 0, "falhas": 0}
    LOGGER.info("Casos 'Suspenso por vista' a pesquisar: %s (apply=%s)", len(cands), args.apply)
    for i, p in enumerate(cands, 1):
        stats["casos"] += 1
        prompt = PROMPT.format(
            data=t(p, "data_sessao"), classe=t(p, "classe_processo") or "-", numero=t(p, "numero_processo") or "-",
            partes=", ".join(parse_multi_value_text(t(p, "partes"))[:4]) or "-", tema=t(p, "tema") or "-",
            relator=t(p, "relator") or "-", origem=t(p, "origem") or "-")
        try:
            _parsed, text, payload = call_gemini_generate_content_rest(
                api_key=gemini_key, model_name=DEFAULT_GEMINI_MODEL,
                contents=[{"parts": [_build_gemini_rest_part(text=prompt)]}],
                system_instruction=SYSTEM, response_model=NewsEnrichmentResult,
                temperature=0.1, use_google_search=True)
        except Exception as exc:
            stats["falhas"] += 1
            LOGGER.warning("falha caso %s: %s", i, exc)
            continue
        data = _json_from_text(text)
        cnj_raw = str(data.get("numero_cnj") or "").strip()
        cnj = canonicalize_numero_processo(cnj_raw)
        if len(re.sub(r"\D", "", cnj)) != 20:  # so aceita CNJ-20 REAL (evita numero parcial/alucinado)
            cnj = ""
        situacao = str(data.get("situacao") or "indeterminado").strip().lower()
        urls = _extract_generate_content_grounding_urls(payload)
        rec = {"page_id": p["id"], "numero_atual": t(p, "numero_processo"), "data": t(p, "data_sessao"),
               "tema": (t(p, "tema") or "")[:50], "numero_cnj": cnj, "situacao": situacao,
               "data_resolucao": data.get("data_resolucao"), "resumo": data.get("resumo"), "fontes": urls[:3]}
        stats["achou_cnj" if cnj else "indeterminado" if not cnj and situacao not in {"resolvido", "pendente"} else "casos"] += 0
        if cnj:
            stats["achou_cnj"] += 1
        stats[situacao if situacao in {"resolvido", "pendente", "indeterminado"} else "indeterminado"] += 1

        props: dict[str, Any] = {}
        cur_num = canonicalize_numero_processo(t(p, "numero_processo"))
        if cnj and cnj != cur_num:
            props["numero_processo"] = client._build_property_value(schema, "numero_processo", cnj)
            stats["grava_numero"] += 1; rec["acao_numero"] = f"{t(p,'numero_processo')!r} -> {cnj}"
        if situacao == "resolvido":
            props["resultado"] = client._build_property_value(schema, "resultado", "Suspenso mas julgado depois")
            props["votacao"] = client._build_property_value(schema, "votacao", "Suspenso*")
            stats["flip_resolvido"] += 1; rec["acao_flip"] = "Suspenso por vista -> Suspenso mas julgado depois"
        if args.apply and props:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                if "numero_processo" in props:
                    stats["applied_num"] += 1
                if "resultado" in props:
                    stats["applied_flip"] += 1
            except Exception as exc:
                stats["falhas"] += 1; rec["erro"] = str(exc)
            time.sleep(0.2)
        out.append(rec)
        LOGGER.info("  [%s/%s] %s | cnj=%s situacao=%s | %s", i, len(cands), (t(p, "numero_processo") or "-")[:18],
                    cnj or "-", situacao, rec["tema"])

    (run_dir / "research.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s | %s", json.dumps(summary, ensure_ascii=False), run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
