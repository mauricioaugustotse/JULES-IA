# -*- coding: utf-8 -*-
"""Detecta punchlines/análises que narram SUSPENSÃO em linhas com desfecho
CONCLUSIVO (e teor oficial no corpo) — contradição vídeo × oficial."""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
import fill_inteiro_teor as fit  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
STAMP = time.strftime("%Y%m%d_%H%M%S")

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
pages = client.query_data_source()


def t(p, campo):
    return client._extract_property_text(p, schema, campo)


SUSP_TXT = re.compile(
    r"foi suspenso|julgamento (?:foi )?adiado|pedido de vista|aguarda.{0,20}voto[- ]vista|"
    r"ser[áa] retomado|suspens[ãa]o do julgamento", re.I)
SUSP_ROTULOS = {"Suspenso", "Suspenso*"}
SUSP_RES = {"Suspenso", "Suspenso mas julgado depois", "Sobrestado"}

candidatos = []
for p in pages:
    res, vot = t(p, "resultado"), t(p, "votacao")
    if vot in SUSP_ROTULOS or res in SUSP_RES or not (res or vot):
        continue
    punch = t(p, "punchline")
    anal = t(p, "analise_do_conteudo_juridico")
    alvo = ""
    if SUSP_TXT.search(punch or ""):
        alvo = "punchline"
    elif SUSP_TXT.search((anal or "")[-400:]):
        alvo = "analise_fim"
    if not alvo:
        continue
    candidatos.append({"page_id": p["id"], "data": t(p, "data_sessao")[:10],
                       "numero": t(p, "numero_processo"), "classe": t(p, "classe_processo"),
                       "resultado": res, "votacao": vot, "campo_suspeito": alvo,
                       "punchline": punch, "analise_fim": (anal or "")[-500:]})

print(f"{len(candidatos)} candidatos por texto; buscando teor...", flush=True)
for i, c in enumerate(candidatos):
    try:
        children = fit.get_all_children(client, c["page_id"])
        idx = fit.marker_index(children)
        if idx is not None:
            corpo = fit.norm_ws(" ".join(fit._paragrafos_da_secao(children, idx)))
            m = re.search(r"(?:o Tribunal|Acordam)[^.]{0,80}?por\s+(maioria|unanimidade)[^.]{0,300}",
                          corpo, re.I)
            c["dispositivo_no_teor"] = (m.group(0) if m else corpo[:300])[:400]
        else:
            c["dispositivo_no_teor"] = ""
    except Exception:
        c["dispositivo_no_teor"] = ""
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(candidatos)}", flush=True)

out = ART / f"punchline_vs_teor_{STAMP}.json"
out.write_text(json.dumps(candidatos, ensure_ascii=False, indent=1), encoding="utf-8")
com_teor = sum(1 for c in candidatos if c["dispositivo_no_teor"])
print(f"candidatos: {len(candidatos)} ({com_teor} com dispositivo no teor)", flush=True)
print("salvo:", out, flush=True)
