# -*- coding: utf-8 -*-
"""Grava no corpo das páginas os teores recuperados do SJUR (fila_teor_sjur_docs).

Retomável: pula page_ids já gravados (log acumulado aplicar_teor_feitos.json).
Substitui seção "inteiro teor" existente; marker SJUR.
Uso: aplicar_teor.py [--apply]
"""
import html
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
REFAZER = "--refazer" in sys.argv  # regrava TODOS (novo formato de parágrafos)
MARKER_SJUR = "Inteiro teor (acórdão — SJUR/TSE)"
FEITOS = ART / "aplicar_teor_feitos.json"

feitos = set() if REFAZER else (
    set(json.loads(FEITOS.read_text(encoding="utf-8"))) if FEITOS.exists() else set())


def limpa_html(s):
    return html.unescape(re.sub(r"<[^>]+>", " ", str(s or "")))


fila = []
for arq in sorted(ART.glob("fila_teor_sjur_docs_*.jsonl")):
    for ln in arq.read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        if d.get("doc") and d["page_id"] not in feitos:
            fila.append(d)
print(f"{len(fila)} teores a gravar ({len(feitos)} já feitos); APPLY={APPLY}", flush=True)

client = core.NotionSessoesClient(core.get_notion_api_key())

# GUARDA DA CADEIA (24/08/2026): linha de VISTA não recebe o acórdão da sessão de
# CONCLUSÃO. A janela [-5,+60]d do motor casa o acórdão final com a linha anterior
# (ex.: vista de 26/09/2024 recebendo o acórdão de 21/11/2024, +56d) — foi assim que
# um teor de outra sessão entrou numa linha suspensa. O acórdão pertence à página que
# registra a proclamação; a linha interrompida fica sem teor, por definição.
import datetime as _dt  # noqa: E402

_schema = client.fetch_schema()
_pages = {p["id"]: p for p in client.query_data_source()}
SUSP_VOT = {"Suspenso", "Suspenso*"}
SUSP_RES = {"Suspenso", "Suspenso mas julgado depois", "Sobrestado"}


def eh_linha_de_vista_com_acordao_alheio(pid: str, doc: dict) -> bool:
    p = _pages.get(pid)
    if p is None:
        return False
    vot = client._extract_property_text(p, _schema, "votacao")
    res = client._extract_property_text(p, _schema, "resultado")
    if vot not in SUSP_VOT and res not in SUSP_RES:
        return False
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", doc.get("dataDecisao") or "")
    ds = client._extract_property_text(p, _schema, "data_sessao")[:10]
    if not m or not ds:
        return True  # linha de vista sem data confiável do doc: não arriscar
    try:
        delta = (_dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                 - _dt.date.fromisoformat(ds)).days
    except ValueError:
        return True
    return delta > 5  # acórdão de sessão POSTERIOR: pertence à linha conclusiva
ok, falhas = [], []
for i, d in enumerate(fila):
    pid = d["page_id"]
    doc = d["doc"]
    try:
        if eh_linha_de_vista_com_acordao_alheio(pid, doc):
            falhas.append({"page_id": pid, "erro": "pulado: linha de vista (acórdão é da sessão de conclusão)"})
            continue
        decisao = fit.norm_ws(limpa_html(doc.get("textoDecisao") or ""))
        ementa = fit.norm_ws(limpa_html(doc.get("textoEmenta") or ""))
        if not decisao and not ementa:
            continue
        if ementa and ementa in decisao:
            ementa = ""
        blocks = fit.build_blocks(ementa, decisao, marker=MARKER_SJUR)
        if APPLY:
            children = fit.get_all_children(client, pid)
            idx = fit.marker_index(children)
            if idx is not None:
                for b in children[idx:]:
                    notion_request_with_retry(client, "DELETE", f"/blocks/{b['id']}")
            fit.append_blocks(client, pid, blocks)
            feitos.add(pid)
        ok.append(pid)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(fila)}", flush=True)
            if APPLY:
                FEITOS.write_text(json.dumps(sorted(feitos)), encoding="utf-8")
    except Exception as exc:
        falhas.append({"page_id": pid, "erro": str(exc)[:200]})

if APPLY:
    FEITOS.write_text(json.dumps(sorted(feitos)), encoding="utf-8")
print(f"gravados: {len(ok)} | falhas: {len(falhas)}", flush=True)
for f in falhas[:5]:
    print("  FALHA", f, flush=True)
