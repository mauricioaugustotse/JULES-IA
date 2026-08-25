# -*- coding: utf-8 -*-
"""Monta os lotes da revisão Sonnet a partir das suspeitas da re-triagem."""
import json
import sys
from pathlib import Path

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
SRC = Path(sys.argv[1])

raiz = json.loads(SRC.read_text(encoding="utf-8"))
obj = None


def acha(o):
    global obj
    if obj is not None:
        return
    if isinstance(o, dict):
        if "suspeitas" in o and isinstance(o.get("suspeitas"), list):
            obj = o
            return
        for v in o.values():
            acha(v)
    elif isinstance(o, list):
        for v in o:
            acha(v)
    elif isinstance(o, str) and '"suspeitas"' in o:
        try:
            acha(json.loads(o))
        except Exception:
            pass


acha(raiz)
assert obj is not None, "não achei as suspeitas"
(ART / "retriagem_veredito_20260824.json").write_text(
    json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
print("total:", obj.get("total"), "| por campo:", obj.get("por_campo"))

full = {}
for f in sorted((ART / "reteor_lotes_20260824").glob("lote_0*.json")):
    if "_formatted" in f.name:
        continue
    for x in json.loads(f.read_text(encoding="utf-8")):
        full[x["page_id"]] = x

comb, faltando = [], 0
for s in obj["suspeitas"]:
    item = full.get(s["page_id"])
    if not item:
        faltando += 1
        continue
    comb.append({"suspeita": s, "pagina": item})

lotes = ART / "revisao_reteor_lotes_20260824"
lotes.mkdir(exist_ok=True)
for f in lotes.glob("*.json"):
    f.unlink()
n = 0
for i in range(0, len(comb), 12):
    n += 1
    (lotes / f"lote_{n:02d}.json").write_text(
        json.dumps(comb[i:i + 12], ensure_ascii=False), encoding="utf-8")
print(f"{len(comb)} itens ({faltando} sem lote) em {n} lotes: {lotes}")
