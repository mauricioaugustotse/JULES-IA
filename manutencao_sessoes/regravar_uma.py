# -*- coding: utf-8 -*-
"""Regrava o teor de UMA página (por número) com o cortador atual.
Fonte: doc SJUR do jsonl se houver; senão reconstrói dos parágrafos gravados."""
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402
import fill_inteiro_teor as fit  # noqa: E402

NUMERO = sys.argv[1]
ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
alvo = None
for p in client.query_data_source():
    if NUMERO in client._extract_property_text(p, schema, "numero_processo"):
        alvo = p
        break
assert alvo, "página não encontrada"
pid = alvo["id"]
print("page:", pid)

doc = None
for arq in list(ART.glob("fila_teor_sjur_docs_*.jsonl")):
    for ln in arq.read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        if d["page_id"].replace("-", "") == pid.replace("-", "") and d.get("doc"):
            doc = d["doc"]
if doc:
    ementa = fit.norm_ws(html.unescape(re.sub(r"<[^>]+>", " ", doc.get("textoEmenta") or "")))
    decisao = fit.norm_ws(html.unescape(re.sub(r"<[^>]+>", " ", doc.get("textoDecisao") or "")))
    marker = "Inteiro teor (acórdão — SJUR/TSE)"
    fonte = "doc SJUR"
else:
    children = fit.get_all_children(client, pid)
    idx = fit.marker_index(children)
    assert idx is not None, "sem seção de teor"
    corpo = fit.norm_ws(" ".join(fit._paragrafos_da_secao(children, idx)))
    ementa, decisao = "", corpo
    marker = fit._heading_text(children[idx]) or fit.MARKER
    fonte = "parágrafos existentes"
if ementa and ementa in decisao:
    ementa = ""
blocks = fit.build_blocks(ementa, decisao, marker=marker)
children = fit.get_all_children(client, pid)
idx = fit.marker_index(children)
if idx is not None:
    for b in children[idx:]:
        notion_request_with_retry(client, "DELETE", f"/blocks/{b['id']}")
fit.append_blocks(client, pid, blocks)
print(f"regravado ({fonte}): {len(blocks)} blocos")
