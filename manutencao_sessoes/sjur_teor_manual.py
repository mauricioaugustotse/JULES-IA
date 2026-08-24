# -*- coding: utf-8 -*-
"""Resgate PONTUAL de teor no SJUR, com inspeção dos hits CRUS (sem o filtro de
data do motor em lote) e retry agressivo.

Existe porque o motor em lote descarta silenciosamente dois casos legítimos:
(a) timeout repetido; (b) acórdão cuja dataDecisao cai fora da janela [-5,+60]d
da sessão (julgamento estendido, publicação tardia). Aqui os hits aparecem para
decisão, e --gravar aceita o hit escolhido.

Uso:
  sjur_teor_manual.py "0600994-58.2020.6.26.0094"              # lista os hits
  sjur_teor_manual.py "<CNJ>" --gravar <page_id> [--hit N]     # grava o hit N (default 0)
"""
import json
import os
import re
import sys
import time
import urllib.parse

from playwright.sync_api import sync_playwright

ART = r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria"
NUMERO = sys.argv[1]
GRAVAR = sys.argv[sys.argv.index("--gravar") + 1] if "--gravar" in sys.argv else None
HIT_N = int(sys.argv[sys.argv.index("--hit") + 1]) if "--hit" in sys.argv else 0
TENTATIVAS = 4
digitos = re.sub(r"\D", "", NUMERO)[:20]

hits = []
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, channel="msedge")
    page = browser.new_context(locale="pt-BR").new_page()
    page.goto("https://jurisprudencia.tse.jus.br/", wait_until="networkidle", timeout=90000)
    time.sleep(2)
    for termo in (f'"{NUMERO}"', digitos):
        for tent in range(TENTATIVAS):
            url = ("https://jurisprudencia.tse.jus.br/#/jurisprudencia/pesquisa?"
                   f"expressaoLivre={urllib.parse.quote(termo)}&params=s&cb=m{tent}{int(time.time())}")
            try:
                with page.expect_response(
                        lambda r: "public/pesquisa" in r.url and r.request.method == "POST",
                        timeout=90000) as ev:
                    page.goto(url, timeout=120000)
                resp = ev.value.json()
            except Exception as e:
                print(f"  tentativa {tent+1} ({termo}): {str(e)[:80]}", flush=True)
                time.sleep(5)
                continue
            if (resp.get("mensagem") or "").lower().count("antirrob"):
                print("  antirrobô — aguardando 30s", flush=True)
                time.sleep(30)
                continue
            for c in resp.get("content") or []:
                cnj = re.sub(r"\D", "", c.get("numeroUnico") or c.get("numeroProcesso") or "")
                hits.append({
                    "cnj": cnj, "igual": cnj == digitos,
                    "dataDecisao": c.get("dataDecisao"), "tipo": c.get("descricaoTipoDecisao"),
                    "classe": c.get("siglaClasse"),
                    "textoDecisao": c.get("textoDecisao") or "", "textoEmenta": c.get("textoEmenta") or "",
                })
            break
        if hits:
            break
    browser.close()

print(f"\n{len(hits)} hit(s) para {NUMERO}:")
for i, h in enumerate(hits):
    print(f"  [{i}] {h['dataDecisao']} | {h['classe']} | {h['tipo']} | cnj_igual={h['igual']} | "
          f"decisão {len(h['textoDecisao'])} chars")
    print(f"      {re.sub(r'<[^>]+>', ' ', h['textoDecisao'])[:150]!r}")

if GRAVAR and hits:
    import html
    sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
    import tse_youtube_notion_core as core
    from audit_notion_sessoes_round2 import notion_request_with_retry
    import fill_inteiro_teor as fit

    h = hits[HIT_N]
    limpa = lambda s: fit.norm_ws(html.unescape(re.sub(r"<[^>]+>", " ", s or "")))
    ementa, decisao = limpa(h["textoEmenta"]), limpa(h["textoDecisao"])
    if ementa and ementa in decisao:
        ementa = ""
    client = core.NotionSessoesClient(core.get_notion_api_key())
    children = fit.get_all_children(client, GRAVAR)
    idx = fit.marker_index(children)
    marker = fit._heading_text(children[idx]) if idx is not None else "Inteiro teor (acórdão — SJUR/TSE)"
    blocks = fit.build_blocks(ementa, decisao, marker=marker or "Inteiro teor (acórdão — SJUR/TSE)")
    if idx is not None:
        for b in children[idx:]:
            notion_request_with_retry(client, "DELETE", f"/blocks/{b['id']}")
    fit.append_blocks(client, GRAVAR, blocks)
    print(f"\nGRAVADO em {GRAVAR}: hit[{HIT_N}] {h['dataDecisao']}, {len(blocks)} blocos")
