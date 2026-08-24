# -*- coding: utf-8 -*-
"""Gera os lotes da triagem meritória full-base (camada 3).

Por página: prosa (punchline/análise/raciocínio), rótulos (resultado/votacao/classe/
tema/data) e o teor comprimido (início com ementa + fim com dispositivo).
Lotes de 40 páginas em ART/triagem_lotes_20260823/.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
import fill_inteiro_teor as fit  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
LOTES = ART / "triagem_lotes_20260823"
LOTES.mkdir(exist_ok=True)

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
pages = client.query_data_source()
print(f"{len(pages)} páginas", flush=True)


def t(p, campo):
    return client._extract_property_text(p, schema, campo)


itens = []
reformatar = []
t0 = time.time()
for i, p in enumerate(pages):
    teor = ""
    try:
        children = fit.get_all_children(client, p["id"])
        idx = fit.marker_index(children)
        if idx is not None:
            paras = fit._paragrafos_da_secao(children, idx)
            corpo = fit.norm_ws(" ".join(paras))
            teor = corpo[:800] + (" [...] " + corpo[-2500:] if len(corpo) > 3300 else corpo[800:])
            # formato defasado (ementas estruturadas grudadas etc.)?
            if [x.strip() for x in paras if x.strip()] != fit._paragrafos_finais(corpo):
                reformatar.append(p["id"])
    except Exception:
        pass
    itens.append({
        "page_id": p["id"], "data": t(p, "data_sessao")[:10], "numero": t(p, "numero_processo"),
        "classe": t(p, "classe_processo"), "tema": t(p, "tema"),
        "resultado": t(p, "resultado"), "votacao": t(p, "votacao"),
        "pedido_vista": t(p, "pedido_vista"),
        "punchline": t(p, "punchline"), "analise": t(p, "analise_do_conteudo_juridico")[:2500],
        "raciocinio": t(p, "raciocinio_juridico")[:1500], "teor": teor,
    })
    if (i + 1) % 300 == 0:
        print(f"  {i+1}/{len(pages)} ({time.time()-t0:.0f}s)", flush=True)

# cadeia: irmãs do mesmo CNJ (p/ harmonizar prosa de julgamentos interrompidos/retomados)
import re
from collections import defaultdict

por_cnj = defaultdict(list)
for x in itens:
    cnj = re.sub(r"\D", "", x["numero"])[:20]
    if len(cnj) >= 20:
        por_cnj[cnj].append(x)
for cnj, grp in por_cnj.items():
    if len(grp) < 2:
        continue
    grp.sort(key=lambda x: x["data"])
    for x in grp:
        x["cadeia"] = [{"data": y["data"], "resultado": y["resultado"], "votacao": y["votacao"],
                        "punchline": (y["punchline"] or "")[:160]}
                       for y in grp if y["page_id"] != x["page_id"]]

n = 0
for i in range(0, len(itens), 40):
    n += 1
    (LOTES / f"lote_{n:03d}.json").write_text(
        json.dumps(itens[i:i + 40], ensure_ascii=False), encoding="utf-8")
print(f"{len(itens)} itens em {n} lotes: {LOTES}", flush=True)
(ART / "teor_reformatar_20260823.json").write_text(json.dumps(reformatar), encoding="utf-8")
print(f"teores com formato defasado a regravar: {len(reformatar)}", flush=True)
