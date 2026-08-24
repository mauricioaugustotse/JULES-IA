# -*- coding: utf-8 -*-
"""Varredura full-base das properties: advogados (pontuação sobrando), partes
(entradas repetidas), composicao (duplicatas e >7 via consenso do dia).

Uso: varrer_props.py [--apply]
Fixes aplicados são mecânicos e restauráveis (log com valor anterior).
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
STAMP = time.strftime("%Y%m%d_%H%M%S")
APPLY = "--apply" in sys.argv

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
pages = client.query_data_source()
print(f"{len(pages)} páginas; APPLY={APPLY}", flush=True)


def t(p, campo):
    return client._extract_property_text(p, schema, campo)


def multi(p, campo):
    prop = p.get("properties", {}).get(campo, {})
    return [o.get("name", "") for o in prop.get("multi_select", [])]


# ---------- advogados: pontuação/travessão sobrando ----------
SOBRA = re.compile(r"^[\s\-–—;,.]+|[\s\-–—;,]+$")


def limpa_advogados(valor: str) -> str:
    itens = [x.strip() for x in re.split(r"[;,\n]| e (?=Dr)", valor) if x.strip()]
    if not itens:
        itens = [valor.strip()] if valor.strip() else []
    out = []
    for it in itens:
        novo = SOBRA.sub("", it)
        novo = re.sub(r"[\s\-–—;,.]+$", "", novo).strip()
        novo = re.sub(r"\s{2,}", " ", novo)
        if novo and novo not in out:
            out.append(novo)
    return ", ".join(out)


# ---------- partes: entradas repetidas ----------
def dedup_partes(valor: str) -> str:
    itens = [x.strip() for x in re.split(r"[;,]\s*", valor) if x.strip()]
    vistos, out = set(), []
    for it in itens:
        chave = re.sub(r"\s+", " ", it).upper()
        if chave not in vistos:
            vistos.add(chave)
            out.append(re.sub(r"\s+", " ", it))
    return ", ".join(out)


# ---------- composicao: consenso do dia p/ >7 ----------
comp_por_dia = defaultdict(Counter)
paginas_info = []
for p in pages:
    data = t(p, "data_sessao")[:10]
    comp = multi(p, "composicao")
    paginas_info.append((p, data, comp))
    if data and comp and len(set(comp)) <= 7:
        comp_por_dia[data].update(set(comp))


def consenso_do_dia(data: str) -> set:
    cont = comp_por_dia.get(data)
    if not cont:
        return set()
    # ministros presentes em >= 30% das páginas do dia (aparições)
    total = max(cont.values())
    return {m for m, c in cont.items() if c >= max(2, total * 0.3)}


achados = {"advogados": [], "partes": [], "composicao_dup": [], "composicao_gt7": []}
fixes = []

for p, data, comp in paginas_info:
    pid = p["id"]
    adv = t(p, "advogados")
    if adv.strip():
        novo = limpa_advogados(adv)
        if novo != adv.strip():
            achados["advogados"].append({"page_id": pid, "de": adv, "para": novo})
            fixes.append((pid, "advogados", adv, novo, "rich_text"))
    partes = t(p, "partes")
    if partes.strip():
        novo = dedup_partes(partes)
        if novo != re.sub(r"\s+", " ", partes.strip()):
            achados["partes"].append({"page_id": pid, "de": partes[:200], "para": novo[:200]})
            fixes.append((pid, "partes", partes, novo, "rich_text"))
    if comp:
        unico = list(dict.fromkeys(comp))
        if len(unico) != len(comp):
            achados["composicao_dup"].append({"page_id": pid, "de": comp, "para": unico})
            fixes.append((pid, "composicao", comp, unico, "multi_select"))
            comp = unico
        if len(comp) > 7:
            cons = consenso_do_dia(data)
            filtrado = [m for m in comp if m in cons] if cons else comp
            item = {"page_id": pid, "data": data, "n": len(comp), "comp": comp,
                    "consenso": sorted(cons), "filtrado": filtrado}
            achados["composicao_gt7"].append(item)
            if cons and 0 < len(filtrado) <= 7 and len(filtrado) < len(comp):
                fixes.append((pid, "composicao", comp, filtrado, "multi_select"))

print({k: len(v) for k, v in achados.items()}, flush=True)
print(f"fixes aplicáveis: {len(fixes)}", flush=True)

log = []
if APPLY:
    for i, (pid, campo, de, para, tipo) in enumerate(fixes):
        if tipo == "rich_text":
            payload = {campo: {"rich_text": core.chunk_rich_text(para)}}
        else:
            payload = {campo: {"multi_select": [{"name": m} for m in para]}}
        try:
            notion_request_with_retry(client, "PATCH", f"/pages/{pid}", json={"properties": payload})
            log.append({"page_id": pid, "campo": campo, "de": de, "para": para})
        except Exception as exc:
            log.append({"page_id": pid, "campo": campo, "erro": str(exc)[:200]})
        if (i + 1) % 50 == 0:
            print(f"  aplicados {i+1}/{len(fixes)}", flush=True)

out = ART / f"varredura_props_{'apply' if APPLY else 'scan'}_{STAMP}.json"
out.write_text(json.dumps({"achados": achados, "aplicados": log}, ensure_ascii=False, indent=1),
               encoding="utf-8")
print(f"salvo: {out}", flush=True)
