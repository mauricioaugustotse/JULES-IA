# -*- coding: utf-8 -*-
"""Regrava os teores com formato defasado (lista teor_reformatar_20260823.json).

Fonte: doc SJUR (jsonl) quando houver; senão os próprios parágrafos re-segmentados.
Retomável via regravar_defasados_feitos.json.
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
FEITOS = ART / "regravar_defasados_feitos.json"

alvo = json.loads((ART / "teor_reformatar_20260823.json").read_text(encoding="utf-8"))
feitos = set(json.loads(FEITOS.read_text(encoding="utf-8"))) if FEITOS.exists() else set()
fila = [p for p in alvo if p not in feitos]
print(f"{len(alvo)} alvo, {len(feitos)} feitos, fila {len(fila)}", flush=True)

docs = {}
for arq in list(ART.glob("fila_teor_sjur_docs_*.jsonl")):
    for ln in arq.read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        if d.get("doc"):
            docs[d["page_id"].replace("-", "")] = d["doc"]
for ln in (ART / "nucleo_duro_sjur_enriquecido_20260821.jsonl").read_text(encoding="utf-8").splitlines():
    d = json.loads(ln)
    if d.get("doc"):
        docs.setdefault(d["page_id"].replace("-", ""), d["doc"])

client = core.NotionSessoesClient(core.get_notion_api_key())
ok = erros = 0
for i, pid in enumerate(fila):
    try:
        children = fit.get_all_children(client, pid)
        idx = fit.marker_index(children)
        if idx is None:
            feitos.add(pid)
            continue
        marker = fit._heading_text(children[idx]) or fit.MARKER
        doc = docs.get(pid.replace("-", ""))
        if doc:
            ementa = fit.norm_ws(html.unescape(re.sub(r"<[^>]+>", " ", doc.get("textoEmenta") or "")))
            decisao = fit.norm_ws(html.unescape(re.sub(r"<[^>]+>", " ", doc.get("textoDecisao") or "")))
        else:
            corpo = fit.norm_ws(" ".join(fit._paragrafos_da_secao(children, idx)))
            ementa, decisao = "", corpo
        if ementa and ementa in decisao:
            ementa = ""
        blocks = fit.build_blocks(ementa, decisao, marker=marker)
        for b in children[idx:]:
            notion_request_with_retry(client, "DELETE", f"/blocks/{b['id']}")
        fit.append_blocks(client, pid, blocks)
        feitos.add(pid)
        ok += 1
    except Exception as exc:
        erros += 1
        print(f"  ERRO {pid}: {str(exc)[:100]}", flush=True)
    if (i + 1) % 50 == 0:
        FEITOS.write_text(json.dumps(sorted(feitos)), encoding="utf-8")
        print(f"  {i+1}/{len(fila)} ok={ok} erros={erros}", flush=True)

FEITOS.write_text(json.dumps(sorted(feitos)), encoding="utf-8")
print(f"FIM ok={ok} erros={erros}", flush=True)
