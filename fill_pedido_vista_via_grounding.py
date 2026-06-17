"""Preenche pedido_vista (QUEM pediu vista) nos processos suspensos por vista cujo campo esta
VAZIO, via Grounding with Google Search (noticia oficial do TSE). O Gemini-video NAO captura
essa fala breve do ministro apos o voto do relator; a fonte eficaz e a noticia/ata do TSE.
Passo GOING-FORWARD, plugavel no post_publish_orchestrator. Modelo default flash-lite (grounding
nao precisa de modelo caro; ver memoria de modelos por etapa).

Guards conservadores (so grava quando TODOS passam):
  (a) nome e ministro plausivel; (b) NAO e o relator; (c) confianca nao-baixa;
  (d) o nome casa (fuzzy) com um ministro JA conhecido na base (anti-alucinacao).
Escrita page-value (select) segura. Dry-run padrao.

Uso:
  python fill_pedido_vista_via_grounding.py                # dry-run
  python fill_pedido_vista_via_grounding.py --apply
  python fill_pedido_vista_via_grounding.py --limit 5 --apply
"""
from __future__ import annotations

import argparse, difflib, json, logging, re, time, unicodedata
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import (is_plausible_ministro_name, normalize_ministro_name,
                               normalize_pedido_vista_value, parse_multi_value_text)
from tse_youtube_notion_core import (DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS,
                                     DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient,
                                     _build_gemini_rest_part, call_gemini_generate_content_rest)

LOGGER = logging.getLogger("fill_pedido_vista_via_grounding")
ARTIFACT_ROOT = Path("artifacts") / "notion_pedido_vista_grounding"

SYS = ("Voce e pesquisador juridico-eleitoral. Use EXCLUSIVAMENTE o Grounding with Google Search. "
       "Em sessoes do TSE, quando um julgamento e SUSPENSO por PEDIDO DE VISTA, um ministro (que NAO "
       "e o relator) pede vista para examinar melhor os autos. Encontre, em fonte oficial do TSE "
       "(tse.jus.br/comunicacao/noticias) ou imprensa juridica confiavel, QUAL ministro pediu vista "
       "no processo ESPECIFICO informado. So responda se a fonte se referir claramente a ESTE processo "
       "(mesmo numero/partes/relator/data). Nunca devolva o relator. Se nao localizar com seguranca, "
       "deixe vazio. Nao invente nomes.")

EXTRACT = ("Converta o texto pesquisado abaixo em JSON com EXATAMENTE estas chaves: "
           "ministro_pediu_vista (string 'Min. Nome' do ministro que pediu vista, ou vazio), "
           "confidence (alta, media ou baixa), evidencia (a frase da fonte que sustenta, citando o "
           "numero do processo se houver). Use EXCLUSIVAMENTE o que esta no texto; nao invente.\n\nTEXTO:\n")


class PedidoVistaResult(BaseModel):
    ministro_pediu_vista: str = ""
    confidence: str = ""
    evidencia: str = ""


def fold(x: str) -> str:
    x = re.sub(r"^min\.?\s*", "", str(x or "").lower())
    x = unicodedata.normalize("NFKD", x)
    return re.sub(r"[^a-z ]", " ", "".join(c for c in x if not unicodedata.combining(c))).strip()


_STOP_TOKENS = {"de", "da", "do", "dos", "das", "e"}


def _ministro_tokens(nome: str) -> set:
    base = fold(normalize_ministro_name(nome) or nome)  # normaliza alias + tira "Min."
    return {tok for tok in base.split() if tok and tok not in _STOP_TOKENS}


def _mesmo_ministro(a: str, b: str) -> bool:
    """True se a e b sao o MESMO ministro, tolerando nome do meio omitido/expandido (ex.: 'Carlos
    Horbach' vs 'Carlos Bastide Horbach') — tokens significativos de um sao subconjunto do outro.
    Compara ambos NORMALIZADOS (alias), nao o valor bruto."""
    ta, tb = _ministro_tokens(a), _ministro_tokens(b)
    return bool(ta and tb and (ta <= tb or tb <= ta))


def _grounded_text(key: str, model: str, prompt: str) -> str:
    """1a passada: Google Search via REST direto, devolvendo o TEXTO cru. Direto (nao via coercer)
    p/ SEMPRE obter o texto da busca — modelos nao-especiais levantariam no coercer em prosa."""
    import requests
    from tse_youtube_notion_core import GEMINI_REST_BASE_URL, _extract_generate_content_text
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "systemInstruction": {"parts": [{"text": SYS}]},
               "generationConfig": {"temperature": 0.1},
               "tools": [{"googleSearch": {}}]}
    r = requests.post(f"{GEMINI_REST_BASE_URL}/models/{model}:generateContent?key={key}",
                      json=payload, timeout=(10, DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS))
    if r.status_code >= 400:
        raise RuntimeError(f"REST {r.status_code}: {r.text[:200]}")
    return _extract_generate_content_text(r.json())


def ground_pedido_vista(key: str, model: str, contexto: str, retries: int = 3) -> tuple[PedidoVistaResult, str]:
    """Ground-then-extract com RETRIES: a busca de evento recente e NAO-deterministica (as vezes a
    busca do Google nao traz a noticia); re-tenta enquanto vier vazio. 1a passada com Google Search
    (texto cru, REST direto); 2a SEM busca estrutura no schema. Devolve (parsed, texto_da_busca) — o
    texto serve p/ o guard anti-cross-process (conferir o numero do processo na fonte)."""
    prompt = ("Identifique qual ministro pediu vista no processo abaixo (suspenso por pedido de "
              "vista). Cite o numero do processo na evidencia.\n\n" + contexto)
    last_text = ""
    for attempt in range(max(1, retries)):
        text = _grounded_text(key, model, prompt)
        if (text or "").strip():
            last_text = text
            parsed, _, _ = call_gemini_generate_content_rest(
                api_key=key, model_name=model,
                contents=[{"parts": [_build_gemini_rest_part(text=EXTRACT + text)]}],
                system_instruction="Voce converte um texto factual ja pesquisado em JSON, sem inventar.",
                response_model=PedidoVistaResult, temperature=0.0, use_google_search=False,
                timeout_seconds=DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS)
            if (parsed.ministro_pediu_vista or "").strip():
                return parsed, text
        if attempt < retries - 1:
            time.sleep(0.5)
    return PedidoVistaResult(), last_text


def _numero_na_fonte(numero: str, *textos: str) -> bool:
    """A fonte (evidencia + texto da busca) menciona ESTE processo? Confere o nucleo NNNNNNN-DD
    (9 digitos) do CNJ nos digitos do texto — anti-cross-process. Numero curto (<9): nao verificavel,
    deixa passar (os outros guards seguram)."""
    dig = re.sub(r"\D", "", numero or "")
    if len(dig) < 9:
        return True
    alvo = dig[:9]
    blob_dig = re.sub(r"\D", "", " ".join(textos))
    return alvo in blob_dig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    # 2.5-flash (não flash-lite) por exceção JUSTIFICADA: testado jun/2026, o flash-lite NÃO
    # encontra a notícia do pedido de vista (retorna "sem registro"); o 2.5-flash acha. Volume
    # baixíssimo (só "Suspenso por vista" com campo vazio) torna o custo extra irrelevante.
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retries", type=int, default=3, help="re-tenta a busca grounded no vazio (nao-determinismo)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    gkey = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    # canon: ministros conhecidos (composicao + relator + pedido_vista ja na base) -> anti-alucinacao
    canon: dict[str, str] = {}
    for p in pages:
        for nm in parse_multi_value_text(t(p, "composicao")) + [t(p, "relator"), t(p, "pedido_vista")]:
            nm = (nm or "").strip()
            cf = fold(nm)
            if len(cf) >= 4 and cf not in canon:
                canon[cf] = nm

    def known(v: str) -> bool:
        vf = fold(v)
        return vf in canon or any(
            difflib.SequenceMatcher(None, vf, cf).ratio() >= 0.9 or (len(vf) > 4 and (vf in cf or cf in vf))
            for cf in canon)

    targets = [p for p in pages
               if (t(p, "resultado") or "").strip() == "Suspenso por vista"
               and not (t(p, "pedido_vista") or "").strip()
               and (t(p, "numero_processo") or "").strip()]
    if args.limit:
        targets = targets[:args.limit]
    LOGGER.info("alvos (Suspenso por vista + pedido_vista vazio + tem numero): %d", len(targets))

    changes: list[dict] = []
    stats = {"alvos": len(targets), "aprovados": 0, "gravados": 0, "flag_vazio": 0,
             "flag_relator": 0, "flag_confianca": 0, "flag_cross_process": 0,
             "flag_desconhecido": 0, "falhas": 0}
    for p in targets:
        numero, relator = t(p, "numero_processo"), t(p, "relator")
        contexto = "\n".join(f"{k}: {v}" for k, v in [
            ("numero_processo", numero), ("classe_processo", t(p, "classe_processo")),
            ("relator", relator), ("partes", t(p, "partes")), ("origem", t(p, "origem")),
            ("data_sessao", t(p, "data_sessao")), ("tema", t(p, "tema"))] if v)
        try:
            res, gtext = ground_pedido_vista(gkey, args.model, contexto, args.retries)
        except Exception as exc:
            stats["falhas"] += 1
            LOGGER.warning("grounding falhou %s: %s", numero, str(exc)[:120])
            continue
        v = normalize_pedido_vista_value(res.ministro_pediu_vista)
        rec = {"page_id": p["id"], "numero": numero, "relator": relator, "proposto": v,
               "confidence": res.confidence, "evidencia": (res.evidencia or "")[:220]}
        if not v or not is_plausible_ministro_name(v):
            rec["status"] = "vazio/implausivel"; stats["flag_vazio"] += 1; changes.append(rec); continue
        if relator and _mesmo_ministro(v, relator):
            rec["status"] = "e_o_relator"; stats["flag_relator"] += 1; changes.append(rec); continue
        if fold(res.confidence) not in {"alta", "media", "high", "medium"}:  # ALLOWLIST: vazio/baixa nao passa
            rec["status"] = "confianca_insuficiente"; stats["flag_confianca"] += 1; changes.append(rec); continue
        if not _numero_na_fonte(numero, res.evidencia or "", gtext):  # anti-cross-process
            rec["status"] = "numero_ausente_na_fonte"; stats["flag_cross_process"] += 1; changes.append(rec); continue
        if not known(v):
            rec["status"] = "ministro_desconhecido"; stats["flag_desconhecido"] += 1; changes.append(rec); continue
        stats["aprovados"] += 1
        rec["status"] = "aprovado"
        if args.apply:
            built = client._build_property_value(schema, "pedido_vista", v)
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": {"pedido_vista": built}})
                rec["status"] = "gravado"; stats["gravados"] += 1; time.sleep(0.2)
            except Exception as exc:
                rec["status"] = "falha_gravacao"; rec["error"] = str(exc)[:120]; stats["falhas"] += 1
        changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:25]:
        LOGGER.info("  [%s] %s -> %r (%s) %s", c.get("status"), c["numero"], c["proposto"],
                    c.get("confidence"), (c.get("evidencia") or "")[:80])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
