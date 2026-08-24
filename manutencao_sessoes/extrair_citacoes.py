# -*- coding: utf-8 -*-
"""Preenche fundamentacao_normativa, precedentes_citados e resoluções_citadas
VAZIOS extraindo as citações do inteiro teor gravado na própria página.
Uso: extrair_citacoes.py [--apply] [--max N]
"""
import glob
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402
import fill_inteiro_teor as fit  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
STAMP = time.strftime("%Y%m%d_%H%M%S")
APPLY = "--apply" in sys.argv
MAXN = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 0

censo = json.load(open(sorted(glob.glob(str(ART / "censo_merito_*.json")))[-1], encoding="utf-8"))
alvos = {}
for campo in ("fundamentacao_normativa", "precedentes_citados", "resoluções_citadas"):
    for x in censo[campo]:
        if x["teor"]:
            alvos.setdefault(x["page_id"], []).append(campo)
print(f"{len(alvos)} páginas com campo(s) de citação vazio(s) e teor", flush=True)

RES_RE = re.compile(r"Resolu[çc][ãa]o(?:[- ]TSE)?(?:\s*/?\s*TSE)?\s*n?[.º°]*\s*(\d{2}\.\d{3}|\d{5})(?:/(\d{4}))?", re.I)
NORMA_RE = re.compile(
    r"\b(?:art(?:igo)?s?\.?\s*\d+[ºo°]?(?:[-–]\w)?(?:\s*,?\s*(?:caput|§+\s*\d+[ºo°]?|inciso[s]?\s+[IVXLC]+|[IVXLC]+|al[íi]nea[s]?\s+\w))*)"
    r"\s*(?:,?\s*(?:d[aoe]|c/c)\s*)+"
    r"((?:Constitui[çc][ãa]o Federal|CF|C[óo]digo Eleitoral|CE\b|C[óo]digo de Processo Civil|CPC|"
    r"C[óo]digo Penal|CP\b|Lei(?:\s+Complementar)?\s*n?[.º°]*\s*[\d.]+(?:/\d{2,4})?|LC\s*n?[.º°]*\s*\d+(?:/\d{2,4})?|"
    r"Lei das Elei[çc][õo]es|EC\s*n?[.º°]*\s*\d+(?:/\d{4})?|Emenda Constitucional\s*n?[.º°]*\s*\d+(?:/\d{4})?))", re.I)
PREC_RE = re.compile(
    r"\b((?:AgR[-–]?|ED[-–]?|ED[-–]cl[-–]?)*(?:REspe?|RO(?:[-–]El)?|AI|AE|RCED|MS|HC|RHC|Rp|Pet|AIJE|RCAND|CTA|PA|PC|DJ|Cta|AgRg[-–]\w+)"
    r"\s*n?[.º°]*\s*[\d][\d.,/\-–]{2,25})", re.I)
SUM_RE = re.compile(r"S[úu]mula(?:[- ]TSE|[- ]STF|[- ]STJ)?\s*n?[.º°]*\s*(\d+)(?:\s*d[oe]\s*(TSE|STF|STJ))?", re.I)


def extrai(corpo):
    res = []
    for m in RES_RE.finditer(corpo):
        num = m.group(1)
        if not num.count(".") and len(num) == 5:
            num = num[:2] + "." + num[2:]
        item = f"Resolução-TSE nº {num}" + (f"/{m.group(2)}" if m.group(2) else "")
        if item not in res:
            res.append(item)
    normas = []
    chaves = set()
    for m in NORMA_RE.finditer(corpo):
        item = re.sub(r"\s+", " ", m.group(0)).strip(" ,.;")
        if item.isupper():
            item = item.lower().replace("constituição federal", "Constituição Federal")
            item = re.sub(r"\blei\b", "Lei", item)
            item = re.sub(r"\bcódigo", "Código", item)
        chave = re.sub(r"\W", "", item).casefold()
        if chave not in chaves and len(normas) < 12:
            chaves.add(chave)
            normas.append(item)
    precs = []
    for m in PREC_RE.finditer(corpo):
        item = re.sub(r"\s+", " ", m.group(1)).strip(" ,.;-–")
        if len(re.sub(r"\D", "", item)) >= 3 and item not in precs and len(precs) < 10:
            precs.append(item)
    for m in SUM_RE.finditer(corpo):
        org = m.group(2) or "TSE"
        item = f"Súmula nº {m.group(1)} do {org}"
        if item not in precs and len(precs) < 12:
            precs.append(item)
    return normas, precs, res


client = core.NotionSessoesClient(core.get_notion_api_key())
itens = list(alvos.items())
if MAXN:
    itens = itens[:MAXN]
log, stats = [], {"preenchidos": 0, "sem_citacao": 0, "sem_secao": 0, "erro": 0}
for i, (pid, campos) in enumerate(itens):
    try:
        children = fit.get_all_children(client, pid)
        idx = fit.marker_index(children)
        if idx is None:
            stats["sem_secao"] += 1
            continue
        corpo = fit.norm_ws(" ".join(fit._paragrafos_da_secao(children, idx)))
        normas, precs, res = extrai(corpo)
        props = {}
        reg = {"page_id": pid, "campos": {}}
        if "fundamentacao_normativa" in campos and normas:
            v = "; ".join(normas)
            props["fundamentacao_normativa"] = {"rich_text": core.chunk_rich_text(v)}
            reg["campos"]["fundamentacao_normativa"] = v[:150]
        if "precedentes_citados" in campos and precs:
            v = "; ".join(precs)
            props["precedentes_citados"] = {"rich_text": core.chunk_rich_text(v)}
            reg["campos"]["precedentes_citados"] = v[:150]
        if "resoluções_citadas" in campos and res:
            v = "; ".join(res)
            props["resoluções_citadas"] = {"rich_text": core.chunk_rich_text(v)}
            reg["campos"]["resoluções_citadas"] = v[:150]
        if not props:
            stats["sem_citacao"] += 1
            continue
        if APPLY:
            notion_request_with_retry(client, "PATCH", f"/pages/{pid}", json={"properties": props})
        log.append(reg)
        stats["preenchidos"] += 1
    except Exception as exc:
        stats["erro"] += 1
        log.append({"page_id": pid, "erro": str(exc)[:120]})
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(itens)} {stats}", flush=True)

out = ART / f"citacoes_{'apply' if APPLY else 'dry'}_{STAMP}.json"
out.write_text(json.dumps({"stats": stats, "log": log}, ensure_ascii=False, indent=1), encoding="utf-8")
print("stats:", stats, flush=True)
print("salvo:", out, flush=True)
