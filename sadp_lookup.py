"""Consulta o SADP Push do TSE (https://sadppush.tse.jus.br/sadpPush/) — publico, SEM captcha
(Captcha Ativado: false) — para DESCOBRIR o CNJ-20 e a SITUACAO de processos cujo numero no
Notion esta incompleto. Busca por numero (acao=pesquisarNumProcesso) no TSE; parseia a tabela
de resultados (Protocolo | Origem | Situacao | Identificacao | Numeracao Unica/Processo) e CASA
o candidato certo por MUNICIPIO/UF + ANO + familia de CLASSE. Conservador: na ambiguidade, nao
decide (vai para revisao).

Funcoes reutilizaveis: make_session, search_number, parse_results, best_match.
Rode `python sadp_lookup.py --pilot N` para um piloto read-only em N casos incompletos do Notion.
"""
from __future__ import annotations

import argparse, re, time, unicodedata
import requests

SADP_BASE = "https://sadppush.tse.jus.br/sadpPush/"
SITUACOES_RESOLVIDO = ("decidido", "transitado", "baixado", "arquivado", "julgado")
SITUACOES_PENDENTE = ("distribuido", "concluso", "vista", "pauta", "redistribu", "suspenso", "diligencia")


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    s.get(SADP_BASE, timeout=30)  # estabelece JSESSIONID
    return s


def _fold(x: str) -> str:
    x = unicodedata.normalize("NFKD", str(x or "").lower())
    return re.sub(r"\s+", " ", "".join(c for c in x if not unicodedata.combining(c))).strip()


def search_number(session: requests.Session, num: str, tribunal: str = "tse") -> list[dict]:
    params = {"acao": "pesquisarNumProcesso", "comboTribunal": tribunal, "siglaTribunal": tribunal,
              "numProcesso": re.sub(r"\D", "", str(num or ""))}
    r = session.get(SADP_BASE + "Pesquisa.do", params=params, timeout=45)
    if r.status_code != 200:
        return []
    return parse_results(r.text)


def search_numunico(session: requests.Session, cnj: str, tribunal: str = "tse") -> list[dict]:
    """Busca EXATA por numero unico (CNJ-20). numUnicoSequencial=NNNNNNN+DD (9), Ano=AAAA, Origem=OOOO."""
    d = re.sub(r"\D", "", str(cnj or ""))
    if len(d) < 20:
        return []
    params = {"acao": "pesquisarNumUnico", "comboTribunal": tribunal, "siglaTribunal": tribunal,
              "numUnicoSequencial": d[0:9], "numUnicoAno": d[9:13], "numUnicoOrigem": d[13:20]}
    try:
        r = session.get(SADP_BASE + "Pesquisa.do", params=params, timeout=45)
    except Exception:
        return []
    return parse_results(r.text) if r.status_code == 200 else []


def parse_results(html: str) -> list[dict]:
    """Extrai uma linha por processo: {cnj, origem, situacao, identificacao, protocolo}."""
    out: list[dict] = []
    # cada linha de dados contem um CNJ-20 (ou um numero curto) na coluna Numeracao Unica.
    for m in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.I | re.S):
        rowhtml = m.group(1)
        cells = [re.sub(r"<[^>]+>", " ", c) for c in re.findall(r"<td[^>]*>(.*?)</td>", rowhtml, re.I | re.S)]
        cells = [re.sub(r"\s+", " ", c).strip() for c in cells]
        joined = " ".join(cells)
        cnjm = re.search(r"\d{7}-\d{2}\.\d{4}\.6\.\d{2}\.\d{4}", joined)
        if not cnjm and not re.search(r"\b\d{2,}/\d{4}\b", joined):
            continue
        nprm = re.search(r"nprot=(\d+)", rowhtml)
        rec = {"cnj": cnjm.group(0) if cnjm else "", "cells": cells, "nprot": nprm.group(1) if nprm else ""}
        for c in cells:
            cf = _fold(c)
            if re.match(r"^[a-zà-ÿ ]+-[a-z]{2}$", cf) and "origem" not in rec:
                rec["origem"] = c  # MUNICIPIO-UF
            if any(k in cf for k in SITUACOES_RESOLVIDO + SITUACOES_PENDENTE) and "situacao" not in rec:
                rec["situacao"] = c
            if re.match(r"^[A-Z][A-Za-z_]*-\d+$", c.strip()) and "identificacao" not in rec:
                rec["identificacao"] = c.strip()
        if cnjm or rec.get("origem"):
            out.append(rec)
    return out


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


_DETAIL_LABEL = re.compile(
    r"((?:AGRAVANTE|AGRAVAD[OA]|RECORRENTE|RECORRID[OA]|EMBARGANTE|EMBARGAD[OA]|REQUERENTE|REQUERID[OA]|"
    r"IMPETRANTE|IMPETRAD[OA]|INTERESSAD[OA]|REPRESENTANTE|REPRESENTAD[OA]|RECLAMANTE|RECLAMAD[OA]|"
    r"AUTOR[AES]*|R[EÉ]US?|APELANTE|APELAD[OA]|LITISCONSORTE|ASSISTENTE|EXEQUENTE|EXECUTAD[OA]|DENUNCIANTE|"
    r"DENUNCIAD[OA]|PACIENTE|CONSULENTE|CONSULTANTE|TERCEIR[OA])S?|ADVOGAD[OA](?:\s+INDICAD[OA])?|"
    r"RELATOR\(A\)|RELATOR|ASSUNTO|FASE ATUAL|LOCALIZA[CÇ][AÃ]O|MUNIC[IÍ]PIO|N[º°.\s]*[UÚ]NICO|PROTOCOLO|"
    r"PROCESSO|CONTROLE|N\.?\s*[º°]?\s*Origem|UF)\s*:", re.I)
_PARTY_LABELS = {"agravante", "agravado", "recorrente", "recorrido", "embargante", "embargado", "requerente",
                 "requerido", "impetrante", "impetrado", "interessado", "representante", "representado",
                 "reclamante", "reclamado", "autor", "reu", "apelante", "apelado", "litisconsorte", "assistente",
                 "exequente", "executado", "denunciante", "denunciado", "paciente", "consulente", "consultante", "terceiro"}


def parse_detail(html: str) -> dict:
    """Extrai do detalhe (ExibirDadosProcesso.do): partes (com papel), advogados, relator,
    assunto, fase atual, cnj, municipio. Valores crus (caixa-alta) — o consumidor normaliza."""
    import html as _h  # decodifica TODAS as entidades (&#39; apostrofo, &quot;, &amp;, &nbsp;...)
    txt = re.sub(r"\s+", " ", _h.unescape(re.sub(r"<[^>]+>", " ", html)))
    labs = list(_DETAIL_LABEL.finditer(txt))
    out = {"partes": [], "advogados": [], "relator": "", "assunto": "", "fase": "", "cnj": "", "municipio": ""}
    for i, m in enumerate(labs):
        label = _fold(m.group(1))
        value = txt[m.end():(labs[i + 1].start() if i + 1 < len(labs) else len(txt))].strip(" :.-")
        if not value:
            continue
        head = label.replace("(a)", "").split()[0]
        if head.startswith("advogad"):
            out["advogados"].append(value)
        elif head.startswith("relator"):
            out["relator"] = value
        elif label == "assunto":
            out["assunto"] = value
        elif "fase" in label:
            out["fase"] = value
        elif head.startswith("munic"):
            out["municipio"] = value
        elif head in _PARTY_LABELS:
            out["partes"].append(value)
    cnjm = re.search(r"\d{7}-\d{2}\.\d{4}\.6\.\d{2}\.\d{4}", txt)
    out["cnj"] = cnjm.group(0) if cnjm else ""
    return out


def fetch_detail(session: requests.Session, nprot: str, tribunal: str = "tse") -> dict:
    if not nprot:
        return {}
    try:
        r = session.get(SADP_BASE + "ExibirDadosProcesso.do", params={"nprot": nprot, "comboTribunal": tribunal}, timeout=45)
    except Exception:
        return {}
    return parse_detail(r.text) if r.status_code == 200 else {}


# ---------------------------------------------------------------------------
# Publicacoes no DJe (Diario da Justica Eletronico) extraidas do ANDAMENTO.
# A aba "Andamento/Decisao" do SADP (POST em ExibirPartesProcessoJudDocRec.do,
# processo fixado antes por um GET em ExibirDadosProcesso.do) traz linhas como:
#   "Publicacao em 27/10/2017 Diario de justica eletronico N. 209 Pag. 74/75. Acordao de 01/08/2017"
#   "Disponibilizacao no Diario da Justica Eletronico em 26/10/2017 ..."
# Dai extraimos data de publicacao, numero da edicao, pagina(s), ato e data do ato.
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


def parse_publicacoes_dje(html: str) -> list[dict]:
    """Lista as publicacoes/disponibilizacoes no DJe achadas no andamento, mais recentes primeiro.
    Cada item: {evento, data, edicao, pagina, ato, data_ato}."""
    import html as _h
    txt = re.sub(r"\s+", " ", _h.unescape(re.sub(r"<[^>]+>", " ", html or "")))
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
    """Detalhe (partes/advogados/relator) + publicacoes no DJe, numa unica passada na mesma sessao.
    O GET fixa o processo na sessao e da as partes; o POST traz o andamento (de onde saem as publicacoes)."""
    if not nprot:
        return {}
    try:
        rget = session.get(SADP_BASE + "ExibirDadosProcesso.do",
                           params={"nprot": nprot, "comboTribunal": tribunal}, timeout=45)
        if rget.status_code != 200:
            return {}  # sem o GET ok o processo nao fica fixado na sessao -> o POST seria nao-confiavel
        rget.encoding = "iso-8859-1"
        det = parse_detail(rget.text)
        # Apenas "Andamento" + "Decisão": incluir "Despachos"/"Documentos Juntados" faz o SADP
        # devolver uma pagina vazia (sem andamento) em parte dos processos.
        rpost = session.post(SADP_BASE + "ExibirPartesProcessoJudDocRec.do",
                            data=[("partesSelecionadas", "Andamento"), ("partesSelecionadas", "Decisão")], timeout=45)
        rpost.encoding = "iso-8859-1"
        det["publicacoes_dje"] = parse_publicacoes_dje(rpost.text) if rpost.status_code == 200 else []
        det["ok"] = True  # GET 200: detalhe válido (mesmo sem CNJ-20, caso de processos antigos)
        return det
    except Exception:
        return {}  # falha de rede -> falsy, para a GUI sinalizar "detalhe indisponivel" e permitir retry


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
