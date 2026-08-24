# -*- coding: utf-8 -*-
"""Busca no SJUR o inteiro teor (textoDecisao/textoEmenta) dos julgados sem
teor/sem fonte no CSV, por CNJ + data da sessão.

Consulta 1: "CNJ formatado completo" entre aspas; fallback: 20 dígitos sem
pontuação. Aceita hit com os MESMOS 20 dígitos e dataDecisao em [-5,+60]d da
sessão (preferindo Acórdão e a data mais próxima). JSONL retomável.
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.parse

from playwright.sync_api import sync_playwright

ART = r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria"
FILA = os.path.join(ART, "fila_teor_sjur_20260822.json")
OUT = os.path.join(ART, "fila_teor_sjur_docs_20260822.jsonl")
# --fila/--out permitem rodadas de RE-BUSCA (timeouts, docs truncados) sem
# poluir o jsonl da rodada original; --timeout sobe o teto por consulta
# (a rodada de 23/08 perdeu 199 páginas por timeout de 25s, todas recuperáveis).
TIMEOUT_MS = 25000
_args = sys.argv[1:]
for _i, _a in enumerate(_args):
    if _a == "--fila":
        FILA = _args[_i + 1] if os.path.isabs(_args[_i + 1]) else os.path.join(ART, _args[_i + 1])
    elif _a == "--out":
        OUT = _args[_i + 1] if os.path.isabs(_args[_i + 1]) else os.path.join(ART, _args[_i + 1])
    elif _a == "--timeout":
        TIMEOUT_MS = int(_args[_i + 1])

fila = json.load(open(FILA, encoding="utf-8"))
feitos = set()
if os.path.exists(OUT):
    for ln in open(OUT, encoding="utf-8"):
        try:
            feitos.add(json.loads(ln)["page_id"])
        except Exception:
            pass
fila = [c for c in fila if c["page_id"] not in feitos and re.sub(r"\D", "", c.get("numero") or "")]
print(f"fila desta rodada: {len(fila)} (feitos: {len(feitos)})", flush=True)
if not fila:
    print("FIM", flush=True)
    sys.exit(0)


def hit_compativel(content, digitos20, data_sessao):
    cands = []
    for c in content or []:
        cnj = re.sub(r"\D", "", c.get("numeroUnico") or c.get("numeroProcesso") or "")
        if cnj != digitos20:
            continue
        m = re.match(r"(\d{2})/(\d{2})/(\d{4})", c.get("dataDecisao") or "")
        if not m:
            continue
        d = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        try:
            delta = (dt.date.fromisoformat(d) - dt.date.fromisoformat(data_sessao)).days
        except ValueError:
            continue
        if not (-5 <= delta <= 60):
            continue
        is_ac = 1 if re.search(r"ac[óo]rd[ãa]o", c.get("descricaoTipoDecisao") or "", re.I) else 0
        cands.append(((is_ac, -abs(delta), len(c.get("textoDecisao") or "")), c))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, channel="msedge")
    ctx = browser.new_context(locale="pt-BR")
    page = ctx.new_page()
    page.goto("https://jurisprudencia.tse.jus.br/", wait_until="networkidle", timeout=90000)
    time.sleep(2)

    fout = open(OUT, "a", encoding="utf-8")
    antirrobo_seguidos = 0
    for idx, caso in enumerate(fila):
        numero = (caso.get("numero") or "").strip()
        digitos = re.sub(r"\D", "", numero)[:20]
        data_sessao = (caso.get("data") or "")[:10]
        reg = {"page_id": caso["page_id"], "numero": numero, "data": data_sessao,
               "status_fila": caso.get("status"), "doc": None, "consultas": []}
        termos = [f'"{numero}"', digitos] if len(digitos) == 20 else [f'"{numero}"']
        for termo in termos:
            url = (f"https://jurisprudencia.tse.jus.br/#/jurisprudencia/pesquisa?"
                   f"expressaoLivre={urllib.parse.quote(termo)}&params=s&cb=t{idx}")
            try:
                with page.expect_response(
                        lambda r: "public/pesquisa" in r.url and r.request.method == "POST",
                        timeout=TIMEOUT_MS) as ev:
                    page.goto(url, timeout=TIMEOUT_MS + 20000)
                resposta = ev.value.json()
            except Exception as e:
                reg["consultas"].append({"termo": termo, "erro": str(e)[:150]})
                continue
            msg = resposta.get("mensagem")
            if msg and "antirrob" in str(msg).lower():
                antirrobo_seguidos += 1
                reg["consultas"].append({"termo": termo, "erro": "antirrobo"})
                if antirrobo_seguidos >= 3:
                    print("3 antirrobo seguidos — abortando (retomável).", flush=True)
                    fout.write(json.dumps(reg, ensure_ascii=False) + "\n")
                    fout.close()
                    browser.close()
                    sys.exit(5)
                time.sleep(20)
                continue
            antirrobo_seguidos = 0
            total = resposta.get("totalRegistros")
            reg["consultas"].append({"termo": termo, "total": total})
            hit = hit_compativel(resposta.get("content"), digitos, data_sessao)
            if hit:
                reg["doc"] = {
                    "dataDecisao": hit.get("dataDecisao"),
                    "tipoDecisao": hit.get("descricaoTipoDecisao"),
                    "classe": hit.get("siglaClasse"),
                    "textoDecisao": hit.get("textoDecisao") or "",
                    "textoEmenta": hit.get("textoEmenta") or "",
                    "partes": [f"{p.get('tipoParte','')}: {p.get('nomeParte','')}"
                               for p in (hit.get("partes") or [])[:10]],
                }
                break
            time.sleep(0.6)
        fout.write(json.dumps(reg, ensure_ascii=False) + "\n")
        fout.flush()
        ok = "DOC" if reg["doc"] else "sem_doc"
        print(f"[{idx+1}/{len(fila)}] {numero} ({data_sessao}): {ok}", flush=True)
        time.sleep(0.5)
    fout.close()
    browser.close()
print("FIM", flush=True)

