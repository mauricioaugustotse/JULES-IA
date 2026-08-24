# -*- coding: utf-8 -*-
"""Renumera "Julgamento N" por dia usando o t= do youtube_link como fonte da ordem.

Só mexe em dias PROBLEMÁTICOS (números duplicados, buracos ou ordem divergente do t=)
onde TODAS as linhas "Julgamento N" têm t= no link — senão pula (sem fonte).
Uso: renumerar_por_t.py [--apply]
"""
import json
import re
import sys
import time
from collections import defaultdict
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


def t(p, campo):
    return client._extract_property_text(p, schema, campo)


def tsec(link):
    m = re.search(r"[?&]t=(\d+)", link or "")
    return int(m.group(1)) if m else None


por_dia = defaultdict(list)
for p in pages:
    tr = t(p, "tipo_registro")
    m = re.match(r"Julgamento (\d+)$", tr or "")
    if not m:
        continue
    por_dia[t(p, "data_sessao")[:10]].append({
        "page_id": p["id"], "n": int(m.group(1)), "t": tsec(t(p, "youtube_link")),
        "video": (re.search(r"v=([\w-]{6,})", t(p, "youtube_link") or "") or [None]) and
                 (re.search(r"v=([\w-]{6,})", t(p, "youtube_link") or "").group(1)
                  if re.search(r"v=([\w-]{6,})", t(p, "youtube_link") or "") else None),
    })

problemas, sem_fonte, planos = [], [], []
for dia, grp in sorted(por_dia.items()):
    ns = [x["n"] for x in grp]
    duplicado = len(ns) != len(set(ns))
    buraco = sorted(ns) != list(range(1, len(ns) + 1))
    # ordem divergente: comparar ordem por n com ordem por (video, t)
    com_t = [x for x in grp if x["t"] is not None]
    ordem_ok = True
    if len(com_t) == len(grp) and len(grp) > 1:
        por_n = sorted(grp, key=lambda x: x["n"])
        por_vt = sorted(grp, key=lambda x: (x["video"] or "", x["t"]))
        ordem_ok = [x["page_id"] for x in por_n] == [x["page_id"] for x in por_vt]
    if not (duplicado or buraco or not ordem_ok):
        continue
    if len(com_t) != len(grp):
        sem_fonte.append({"dia": dia, "linhas": len(grp), "com_t": len(com_t),
                          "motivo": ("duplicado" if duplicado else "") + ("+buraco" if buraco else "")})
        continue
    # plano: renumerar por (video, t)
    novo = sorted(grp, key=lambda x: (x["video"] or "", x["t"]))
    plano = [{"page_id": x["page_id"], "de": x["n"], "para": i + 1}
             for i, x in enumerate(novo) if x["n"] != i + 1]
    if plano:
        planos.append({"dia": dia, "mudancas": plano})
        problemas.append(dia)

print(f"dias problemáticos renumeráveis: {len(planos)} | sem fonte (t= incompleto): {len(sem_fonte)}",
      flush=True)
total_mud = sum(len(pl["mudancas"]) for pl in planos)
print(f"páginas a renumerar: {total_mud}", flush=True)

log = []
if APPLY:
    for pl in planos:
        for m in pl["mudancas"]:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{m['page_id']}", json={
                    "properties": {"tipo_registro": {"select": {"name": f"Julgamento {m['para']}"}}}})
                log.append({**m, "dia": pl["dia"]})
            except Exception as exc:
                log.append({**m, "dia": pl["dia"], "erro": str(exc)[:150]})

out = ART / f"renumeracao_t_{'apply' if APPLY else 'dry'}_{STAMP}.json"
out.write_text(json.dumps({"planos": planos, "sem_fonte": sem_fonte, "aplicados": log},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print("salvo:", out, flush=True)
