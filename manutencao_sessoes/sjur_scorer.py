# -*- coding: utf-8 -*-
"""Scorer do casamento núcleo duro × hits do SJUR.

Para cada caso, pontua os hits: data exata da sessão, UF da origem do vídeo,
classe, semelhança do núcleo com o número corrompido da base, nome da parte.
Aplica guardas: DV mod 97 do candidato, whitelists da campanha, armadilha
conhecida (0600808-20/Boulos).

Saída: nucleo_duro_sjur_propostas_20260821.json com veredito por caso:
  FORTE (aplicável mediante conferência), CANDIDATO (revisão), SEM_MATCH.
"""
import json, os, re, collections, datetime as dt

ART = r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria"
HITS_ARQS = [os.path.join(ART, "nucleo_duro_sjur_hits_20260821.jsonl"),
             os.path.join(ART, "nucleo_duro_sjur_hits_fase2_20260821.jsonl")]
OUT = os.path.join(ART, "nucleo_duro_sjur_propostas_20260821.json")

# guardas da campanha
WL_LEGADO = os.path.join(ART, "base_full_whitelist_dv_legado_20260821.json")
ARMADILHA_NUCLEOS = {"0600808"}  # RCAND Boulos: matching ja desmascarado 2x

wl_numeros = set()
if os.path.exists(WL_LEGADO):
    wl = json.load(open(WL_LEGADO, encoding="utf-8"))
    def _extrai(o):
        if isinstance(o, str):
            wl_numeros.add(re.sub(r"\D", "", o))
        elif isinstance(o, dict):
            for v in o.values():
                _extrai(v)
        elif isinstance(o, list):
            for v in o:
                _extrai(v)
    _extrai(wl)


def dv_mod97_ok(cnj20: str) -> bool:
    d = re.sub(r"\D", "", cnj20)
    if len(d) != 20:
        return False
    seq, dv, resto = d[:7], d[7:9], d[9:]
    calc = 98 - (int(seq + resto + "00") % 97)
    return f"{calc:02d}" == dv


def uf_da_origem(origem: str) -> str:
    m = re.search(r"/([A-Z]{2})\s*$", origem or "")
    return m.group(1) if m else ""


def norm_classe(c: str) -> str:
    c = (c or "").upper()
    c = re.sub(r"[^A-Z]", "", c.split(" ")[0]) if c else ""
    return c


def data_br_iso(d: str) -> str:
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", d or "")
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else ""


import unicodedata

def _norm_nome(s: str) -> set:
    s = re.sub(r"<[^>]+>", "", s or "")          # tags de highlight do SJUR
    s = re.sub(r"\(.*?\)", " ", s)               # apelidos/siglas entre parenteses
    s = re.sub(r"^[^:]{0,30}:\s*", "", s)        # prefixo de papel ("PARTE:", "Recorrente:")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    toks = {t for t in re.findall(r"[a-z]{3,}", s)
            if t not in {"dos", "das", "der", "van", "von", "junior", "filho", "neto",
                         "sobrinho", "nacional", "estadual", "municipal", "partido",
                         "coligacao", "diretorio", "eleitoral", "ministerio", "publico"}}
    return toks


VOCAB_PARTIDARIO = {
    "social", "democratico", "democracia", "liberal", "trabalhadores", "trabalhista",
    "progressista", "progressistas", "republicano", "republicanos", "movimento",
    "brasileiro", "brasileira", "socialista", "socialismo", "unificado", "verde",
    "rede", "sustentabilidade", "cidadania", "podemos", "patriota", "uniao", "brasil",
    "avante", "solidariedade", "novo", "liberdade", "comunista", "cristao", "crista",
    "popular", "mobilizacao", "renovacao", "ordem", "pais", "frente", "unidade"}


def partes_casam(partes_video, partes_sjur):
    """'pleno' se um nome próprio casa; 'generico' se só vocabulário partidário casa."""
    resultado = None
    for pv in partes_video or []:
        tv = _norm_nome(pv)
        if len(tv) < 2:
            continue
        for ps in partes_sjur or []:
            ts = _norm_nome(ps)
            if not ts:
                continue
            inter = tv & ts
            if len(inter) >= 2 and len(inter) / len(tv) >= 0.6:
                if inter - VOCAB_PARTIDARIO:
                    return "pleno"
                resultado = resultado or "generico"
    return resultado


registros = []
vistos = set()
for arq in HITS_ARQS:
    if os.path.exists(arq):
        for ln in open(arq, encoding="utf-8"):
            reg = json.loads(ln)
            if reg["page_id"] in vistos:
                continue
            vistos.add(reg["page_id"])
            registros.append(reg)
propostas = []
placar = collections.Counter()

for reg in registros:
    caso_uf = uf_da_origem(reg.get("origem_video") or "")
    base_num = re.sub(r"\D", "", reg.get("numero_base") or "")
    base_nucleo = base_num[:7] if len(base_num) >= 7 else ""
    base_classe = norm_classe(reg.get("classe_base") or "")
    data_sessao = reg.get("data")

    candidatos = {}
    for cons in reg.get("consultas", []):
        for h in cons.get("hits", []):
            cnj = re.sub(r"\D", "", h.get("numeroUnico") or "")
            if not cnj or cnj in candidatos:
                continue
            score, motivos = 0, []
            d_iso = data_br_iso(h.get("dataDecisao"))
            if d_iso == data_sessao:
                score += 4; motivos.append("data_exata")
            elif d_iso and data_sessao and abs(
                    (dt.date.fromisoformat(d_iso) - dt.date.fromisoformat(data_sessao)).days) <= 10:
                score += 1; motivos.append("data_ate10d")
            if caso_uf and h.get("siglaUF") == caso_uf:
                score += 2; motivos.append("uf")
            hclasse = norm_classe(h.get("siglaClasse") or "")
            if base_classe and hclasse == base_classe:
                score += 1; motivos.append("classe")
            if base_nucleo and cnj[:7] == base_nucleo:
                score += 3; motivos.append("nucleo_igual")
            elif base_nucleo and cnj[:5] == base_num[:5]:
                score += 1; motivos.append("nucleo_5digitos")
            if h.get("siglaTribunalJE") == "TSE":
                score += 1; motivos.append("tse")
            pc = partes_casam(reg.get("partes_uteis"), h.get("partes"))
            if pc == "pleno":
                score += 4; motivos.append("partes_casam")
            elif pc == "generico":
                score += 2; motivos.append("partes_casam_generico")
            candidatos[cnj] = {"hit": h, "score": score, "motivos": motivos,
                               "consulta": cons.get("termo"),
                               "com_data": cons.get("com_data")}

    ordenados = sorted(candidatos.values(), key=lambda c: -c["score"])
    top = ordenados[0] if ordenados else None
    veredito = "SEM_MATCH"
    guardas = []   # bloqueiam FORTE automático
    flags = []     # informativos p/ o verificador
    if top:
        cnj = re.sub(r"\D", "", top["hit"].get("numeroUnico") or "")
        if cnj[:7] in ARMADILHA_NUCLEOS:
            guardas.append("ARMADILHA_BOULOS")
        if cnj in wl_numeros:
            flags.append("cnj_ja_usado_em_outra_pagina")  # normal: mesmo processo, sessões distintas
        if not dv_mod97_ok(cnj):
            flags.append("dv_fora_mod97_legado")  # numero OFICIAL do SJUR: transição, não erro
        segundo = ordenados[1]["score"] if len(ordenados) > 1 else 0
        modo_tema = reg.get("modo") == "tema"
        tem_partes = "partes_casam" in top["motivos"]
        tem_data = "data_exata" in top["motivos"] or "data_ate10d" in top["motivos"]
        tem_ancora = any(m in top["motivos"] for m in
                         ("nucleo_igual", "nucleo_5digitos", "uf", "classe"))
        if guardas:
            veredito = "BLOQUEADO_GUARDA"
        elif not modo_tema and tem_partes and tem_data and \
                ("data_exata" in top["motivos"] or tem_ancora) and top["score"] - segundo >= 3:
            veredito = "FORTE"  # partes casando + data é a dupla evidência
        elif top["score"] >= 4:
            veredito = "CANDIDATO"
        else:
            veredito = "FRACO"
    placar[veredito] += 1
    propostas.append({
        "page_id": reg["page_id"], "data": data_sessao, "numero_base": reg.get("numero_base"),
        "classe_base": reg.get("classe_base"), "origem_video": reg.get("origem_video"),
        "partes_uteis": reg.get("partes_uteis"), "tema": reg.get("tema"),
        "modo": reg.get("modo", "partes"), "veredito": veredito, "guardas": guardas,
        "flags": flags,
        "top3": [{"cnj": c["hit"].get("numeroUnicoFormatado"),
                  "dataDecisao": c["hit"].get("dataDecisao"),
                  "classe": c["hit"].get("siglaClasse"), "uf": c["hit"].get("siglaUF"),
                  "municipio": c["hit"].get("nomeMunicipio"),
                  "partes": c["hit"].get("partes"), "relatores": c["hit"].get("relatores"),
                  "score": c["score"], "motivos": c["motivos"],
                  "consulta": c["consulta"]} for c in ordenados[:3]],
    })

json.dump(propostas, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("PLACAR:", dict(placar))
print("Salvo em", OUT)
