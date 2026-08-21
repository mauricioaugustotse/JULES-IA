"""Consulta o SADP do TSE — processos FISICOS/legado — para DESCOBRIR o CNJ-20 e a SITUACAO
de processos cujo numero no Notion esta incompleto.

MIGRACAO 20/08/2026: o SADP Push antigo (sadppush.tse.jus.br/sadpPush, HTML server-side)
foi DESATIVADO pelo TSE (NXDOMAIN). O substituto e a SPA "SADP Consulta"
(https://sadp-consulta.tse.jus.br/consulta, embutida na pagina "Processos Fisicos" do
portal) cuja API REST e publica, sem captcha e sem WAF:
  GET {API}/v1/{tribunal}/processos/listar/numeroProcesso/{digitos<=13}?size=200
  GET {API}/v1/{tribunal}/processos/listar/numeroUnico/{20 digitos}
  GET {API}/v1/{tribunal}/processos/consultar/numeroProtocolo/{nprot}   (detalhe: partes,
      advogados, relator, assunto, numeroUnico mascarado)
  GET {API}/v1/{tribunal}/andamentos/listar/numeroProtocolo/{nprot}     (publicacoes DJe)
As assinaturas publicas deste modulo (make_session, search_number, search_numunico,
fetch_detail, fetch_detail_e_publicacoes, best_match, situacao_resolvido) foram MANTIDAS —
GUI e scripts irmaos nao precisam mudar. O formato de cada resultado tambem: dict com
{cnj, cells, nprot, origem, situacao, identificacao} (cells nas 7 posicoes da tabela velha).

Rode `python sadp_lookup.py --pilot N` para um piloto read-only em N casos incompletos do Notion.
"""
from __future__ import annotations

import argparse, re, time, unicodedata
import requests

SADP_BASE = "https://sadp-consulta.tse.jus.br/consulta"  # SPA (link para navegador)
SADP_API = "https://sadp-consulta-api.tse.jus.br/sadp-consulta/rest/v1"
SITUACOES_RESOLVIDO = ("decidido", "transitado", "baixado", "arquivado", "julgado")
SITUACOES_PENDENTE = ("distribuido", "concluso", "vista", "pauta", "redistribu", "suspenso", "diligencia")
_CNJ_RE = re.compile(r"^\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}$")


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                      "Accept": "application/json"})
    return s


def _fold(x: str) -> str:
    x = unicodedata.normalize("NFKD", str(x or "").lower())
    return re.sub(r"\s+", " ", "".join(c for c in x if not unicodedata.combining(c))).strip()


def _rec_from_item(it: dict) -> dict:
    """Converte um item de listaProcessos da API nova para o formato de registro que a GUI e
    os scripts sempre consumiram (cells = 7 colunas da tabela do SADP velho)."""
    num_unico_raw = str(it.get("numeroUnico") or "").strip()
    cnj = num_unico_raw if _CNJ_RE.match(num_unico_raw) else ""  # sigiloso/corregedoria vem "-..."
    cidade, uf = str(it.get("nomeCidade") or "").strip(), str(it.get("siglaEstado") or "").strip()
    origem = f"{cidade}-{uf}" if cidade and uf else (cidade or "")
    ident = f"{str(it.get('siglaClasse') or '').strip()}-{str(it.get('numeroProcesso') or '').strip()}"
    natureza = str(it.get("identificacaoProcesso") or "").strip()
    cells = ["", str(it.get("numeroProtocoloFormatado") or ""), origem,
             str(it.get("ultimaSituacao") or ""), ident, cnj or num_unico_raw, natureza]
    return {"cnj": cnj, "cells": cells, "nprot": str(it.get("numeroProtocolo") or ""),
            "origem": origem, "situacao": str(it.get("ultimaSituacao") or ""),
            "identificacao": ident, "classe": str(it.get("siglaClasse") or ""),
            "eletronico": bool(it.get("eletronico")), "raw": it}


def _listar(session: requests.Session, url: str) -> list[dict]:
    try:
        r = session.get(url, params={"size": 200}, timeout=45)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    try:
        payload = r.json()
    except Exception:
        return []
    return [_rec_from_item(it) for it in payload.get("listaProcessos") or []]


def search_number(session: requests.Session, num: str, tribunal: str = "tse") -> list[dict]:
    """Busca por numero de processo (curto, digitos crus; a API limita a 13 caracteres)."""
    d = re.sub(r"\D", "", str(num or ""))[:13]
    if not d:
        return []
    return _listar(session, f"{SADP_API}/{tribunal}/processos/listar/numeroProcesso/{d}")


def search_numunico(session: requests.Session, cnj: str, tribunal: str = "tse") -> list[dict]:
    """Busca EXATA por numero unico (CNJ-20, digitos crus)."""
    d = re.sub(r"\D", "", str(cnj or ""))
    if len(d) < 20:
        return []
    return _listar(session, f"{SADP_API}/{tribunal}/processos/listar/numeroUnico/{d[:20]}")


def best_match(candidates: list[dict], municipio: str, uf: str, year: str, classe: str) -> dict | None:
    # dedup/merge por CNJ (linhas repetidas do mesmo processo nao sao ambiguidade)
    by_cnj: dict[str, dict] = {}
    for c in candidates:
        cnj = c.get("cnj")
        if not cnj:
            continue
        m = by_cnj.setdefault(cnj, {"cnj": cnj})
        for k in ("origem", "situacao", "identificacao", "nprot"):
            if c.get(k) and not m.get(k):
                m[k] = c[k]
    muni_f, uf_f, classe_f = _fold(municipio), _fold(uf), _fold(classe)
    scored = []
    for cnj, c in by_cnj.items():
        og = _fold(c.get("origem", ""))
        cnj_year = c["cnj"][11:15]
        score = 0
        if muni_f and len(muni_f) > 2 and muni_f in og:
            score += 3
        if uf_f and og.endswith("-" + uf_f):
            score += 2
        try:
            cyi, yi = int(cnj_year), int(year or 0)
            if cyi == yi:
                score += 2
            elif 0 <= yi - cyi <= 12:  # autuacao costuma ser ANTERIOR a sessao
                score += 1
        except Exception:
            pass
        ident = _fold(c.get("identificacao", "")).split("-")[0]
        if classe_f and ident and (ident in classe_f or any(tok in ident for tok in classe_f.split("-") if tok)):
            score += 1
        scored.append((score, c))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    if scored[0][0] < 4:  # exige >= municipio+uf (3+2) ou equivalente
        return None
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        return None  # empate real -> ambiguo
    return {**scored[0][1], "score": scored[0][0]}


def situacao_resolvido(situacao: str) -> bool | None:
    sf = _fold(situacao)
    if any(k in sf for k in SITUACOES_RESOLVIDO):
        return True
    if any(k in sf for k in SITUACOES_PENDENTE):
        return False
    return None


def fetch_detail(session: requests.Session, nprot: str, tribunal: str = "tse") -> dict:
    """Detalhe por protocolo: partes/advogados (com papel na API, achatado para o formato
    antigo), relator, assunto, fase (ultima situacao), cnj mascarado e municipio."""
    if not nprot:
        return {}
    try:
        r = session.get(f"{SADP_API}/{tribunal}/processos/consultar/numeroProtocolo/{nprot}", timeout=45)
    except Exception:
        return {}
    if r.status_code != 200:
        return {}
    try:
        dados = (r.json() or {}).get("dadosProcesso") or {}
    except Exception:
        return {}
    if not dados:
        return {}
    out = {"partes": [], "advogados": [], "relator": str(dados.get("nomeRelator") or ""),
           "assunto": str(dados.get("assunto") or ""), "fase": str(dados.get("ultimaSituacao") or ""),
           "cnj": "", "municipio": str(dados.get("nomeMunicipioFormatado") or "")}
    num_unico = str(dados.get("numeroUnico") or "").strip()
    if _CNJ_RE.match(num_unico):
        out["cnj"] = num_unico
    for parte in dados.get("partes") or []:
        nome = str(parte.get("nomeParte") or "").strip()
        if not nome:
            continue
        if parte.get("advogado") or str(parte.get("descricaoParte") or "").upper().startswith("ADVOGAD"):
            out["advogados"].append(nome)
        else:
            out["partes"].append(nome)
    out["raw"] = dados
    return out


# ---------------------------------------------------------------------------
# Publicacoes no DJe (Diario da Justica Eletronico), extraidas dos ANDAMENTOS.
# A API devolve os mesmos textos que o SADP velho mostrava no HTML, ex.:
#   "Publicacao em 27/10/2017 Diario de justica eletronico N. 209 Pag. 74/75. Acordao de 01/08/2017"
DJE_CONSULTA_URL = "https://dje-consulta.tse.jus.br/"  # sistema oficial de consulta do DJe do TSE (SPA)

_PUB_RE = re.compile(
    r"Publica[çc][aã]o em (\d{2}/\d{2}/\d{4})\s+Di[aá]rio de justi[çc]a eletr[oô]nico"
    r"(?:\s*N\.?\s*0*(\d+))?"            # numero da edicao (sem zeros a esquerda)
    r"(?:\s*Pag\.?\s*([\d/\-]+))?"        # pagina(s)
    r"\s*\.?\s*"
    # ato (Acordao, Decisao Monocratica, Intimacao...): 1 palavra + ate 3 seguintes. Os lookaheads
    # (?!Publica)(?!Disponibiliza) impedem que o ato comece OU atravesse o inicio do PROXIMO evento
    # (senao o quantificador engole "Publicacao em..." da entrada seguinte e a perde/funde).
    r"((?!Publica)(?!Disponibiliza)[A-Za-zÀ-ÿ./()]+"
    r"(?:\s+(?!de\s+\d{2}/\d{2}/\d{4})(?!d[oa]\b)(?!no\b)(?!Publica)(?!Disponibiliza)[A-Za-zÀ-ÿ./()]+){0,3})?"
    r"(?:\s+de\s+(\d{2}/\d{2}/\d{4}))?", re.I)
_DISP_RE = re.compile(
    r"Disponibiliza[çc][aã]o no Di[aá]rio da Justi[çc]a Eletr[oô]nico em (\d{2}/\d{2}/\d{4})", re.I)


def _data_key(ddmmaaaa: str) -> tuple:
    d = (ddmmaaaa or "").split("/")
    return (d[2], d[1], d[0]) if len(d) == 3 else ("", "", "")


def parse_publicacoes_dje(texto: str) -> list[dict]:
    """Lista as publicacoes/disponibilizacoes no DJe achadas no texto dos andamentos, mais
    recentes primeiro. Cada item: {evento, data, edicao, pagina, ato, data_ato}."""
    txt = re.sub(r"\s+", " ", str(texto or ""))
    out: list[dict] = []
    for m in _PUB_RE.finditer(txt):
        ato = re.sub(r"\s+", " ", (m.group(4) or "")).strip(" .-")
        ato = re.split(r"\s+(?:Publica|Disponibiliza)", ato)[0].strip(" .-")  # defesa: nunca colar o proximo evento
        out.append({"evento": "publicacao", "data": m.group(1), "edicao": m.group(2) or "",
                    "pagina": m.group(3) or "", "ato": ato, "data_ato": m.group(5) or ""})
    for m in _DISP_RE.finditer(txt):
        out.append({"evento": "disponibilizacao", "data": m.group(1), "edicao": "",
                    "pagina": "", "ato": "", "data_ato": ""})
    seen, uniq = set(), []
    for p in sorted(out, key=lambda p: _data_key(p["data"]), reverse=True):
        k = (p["evento"], p["data"], p["edicao"], p["pagina"], p["ato"], p["data_ato"])
        if k not in seen:
            seen.add(k); uniq.append(p)
    return uniq


def fetch_detail_e_publicacoes(session: requests.Session, nprot: str, tribunal: str = "tse") -> dict:
    """Detalhe (partes/advogados/relator) + publicacoes no DJe (dos andamentos)."""
    det = fetch_detail(session, nprot, tribunal)
    if not det:
        return {}  # falha de rede/protocolo -> falsy, para a GUI sinalizar e permitir retry
    try:
        r = session.get(f"{SADP_API}/{tribunal}/andamentos/listar/numeroProtocolo/{nprot}", timeout=45)
        andamentos = (r.json() or {}).get("listaAndamentos") or [] if r.status_code == 200 else []
    except Exception:
        andamentos = []
    texto = " ".join(
        f"{a.get('descricaoAndamento') or ''} {a.get('complemento') or ''}" for a in andamentos
    )
    det["publicacoes_dje"] = parse_publicacoes_dje(texto)
    det["ok"] = True
    return det


def _pilot(n: int) -> None:
    import re as _re
    from local_secrets import get_secret
    from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient
    UF_RE = _re.compile(r"\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b")
    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=DEFAULT_NOTION_DATA_SOURCE_ID)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    inc = [p for p in pages if 0 < len(_re.sub(r"\D", "", t(p, "numero_processo") or "")) < 20]
    if n and n < len(inc):
        inc = inc[::max(1, len(inc) // n)][:n]
    sess = make_session()
    for p in inc:
        num = _re.sub(r"\D", "", t(p, "numero_processo"))
        origem = t(p, "origem"); ufm = UF_RE.search(origem.upper())
        uf = ufm.group(1) if ufm else ""
        muni = _re.sub(r"[-/].*$", "", origem).strip() if origem else ""
        year = (t(p, "data_sessao") or "")[:4]
        cands = search_number(sess, num)
        match = best_match(cands, muni, uf, year, t(p, "classe_processo"))
        print(f"\n[{t(p,'numero_processo')}] origem={origem!r} ano={year} classe={t(p,'classe_processo')!r} | candidatos={len(cands)}")
        if match:
            print(f"   MATCH cnj={match['cnj']} situacao={match.get('situacao','?')!r} origem={match.get('origem','?')!r} score={match['score']}")
        else:
            for c in cands[:4]:
                print(f"   cand: cnj={c.get('cnj','-')} origem={c.get('origem','?')!r} sit={c.get('situacao','?')!r} id={c.get('identificacao','?')!r}")
        time.sleep(1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=8)
    args = ap.parse_args()
    _pilot(args.pilot)
