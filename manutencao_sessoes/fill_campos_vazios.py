# -*- coding: utf-8 -*-
"""Preenche campos VAZIOS (partes, origem, relator) com dados oficiais dos docs
SJUR já baixados (fila_teor_sjur_docs + nucleo_duro_sjur_enriquecido).

Conservador: só preenche vazio; relator apenas com match único em option existente.
Uso: fill_campos_vazios.py [--apply]
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
STAMP = time.strftime("%Y%m%d_%H%M%S")
APPLY = "--apply" in sys.argv

docs = {}
for arq in list(ART.glob("fila_teor_sjur_docs_*.jsonl")):
    for ln in arq.read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        if d.get("doc"):
            docs[d["page_id"]] = d["doc"]
for ln in (ART / "nucleo_duro_sjur_enriquecido_20260821.jsonl").read_text(encoding="utf-8").splitlines():
    d = json.loads(ln)
    if d.get("doc"):
        docs.setdefault(d["page_id"], d["doc"])
print(f"{len(docs)} docs oficiais disponíveis", flush=True)

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
pages = {p["id"]: p for p in client.query_data_source()}

relator_prop = (schema.raw_payload.get("properties") or {}).get("relator", {})
opcoes_relator = [o.get("name", "") for o in relator_prop.get("select", {}).get("options", [])]
print(f"{len(opcoes_relator)} options de relator", flush=True)


def t(p, campo):
    return client._extract_property_text(p, schema, campo)


MIN = {"de", "da", "do", "das", "dos", "e", "em"}


def title_pt(s):
    out = []
    for i, w in enumerate(str(s or "").lower().split()):
        out.append(w if (i and w in MIN) else w.capitalize())
    return " ".join(out)


def acha_relator(nome_sjur):
    """Match único por sobrenome(s) contra options existentes."""
    toks = [x for x in re.findall(r"\w+", nome_sjur.lower()) if len(x) > 2 and x not in MIN]
    if not toks:
        return None
    cand = [o for o in opcoes_relator if all(tk in o.lower() for tk in toks[-2:])]
    if len(cand) == 1:
        return cand[0]
    cand = [o for o in opcoes_relator if toks[-1] in o.lower()]
    return cand[0] if len(cand) == 1 else None


log, stats = [], {"partes": 0, "origem": 0, "relator": 0, "sem_pagina": 0, "nada": 0}
for pid, doc in docs.items():
    p = pages.get(pid) or pages.get(pid.replace("-", "")) or None
    if p is None:
        pid_fmt = re.sub(r"^(.{8})(.{4})(.{4})(.{4})(.{12})$", r"\1-\2-\3-\4-\5", pid.replace("-", ""))
        p = pages.get(pid_fmt)
    if p is None:
        stats["sem_pagina"] += 1
        continue
    props = {}
    acoes = []
    if not t(p, "partes").strip() and doc.get("partes"):
        nomes = []
        for x in doc["partes"]:
            nome = re.sub(r"^[^:]{1,30}:\s*", "", x).strip()
            if nome and nome.upper() not in {n.upper() for n in nomes}:
                nomes.append(title_pt(nome))
        if nomes:
            props["partes"] = {"rich_text": core.chunk_rich_text(", ".join(nomes))}
            acoes.append("partes")
            stats["partes"] += 1
    if not t(p, "origem").strip() and doc.get("municipio"):
        origem = f"{title_pt(doc['municipio'])}/{doc.get('uf') or ''}".rstrip("/")
        props["origem"] = {"rich_text": core.chunk_rich_text(origem)}
        acoes.append("origem")
        stats["origem"] += 1
    if not t(p, "relator").strip() and doc.get("relatores"):
        op = acha_relator(doc["relatores"][0])
        if op:
            props["relator"] = {"select": {"name": op}}
            acoes.append(f"relator={op}")
            stats["relator"] += 1
    if not props:
        stats["nada"] += 1
        continue
    if APPLY:
        try:
            notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
        except Exception as exc:
            log.append({"page_id": pid, "erro": str(exc)[:150]})
            continue
    log.append({"page_id": pid, "acoes": acoes})

out = ART / f"fill_campos_vazios_{'apply' if APPLY else 'dry'}_{STAMP}.json"
out.write_text(json.dumps({"stats": stats, "log": log}, ensure_ascii=False, indent=1), encoding="utf-8")
print("stats:", stats, flush=True)
print("salvo:", out, flush=True)
