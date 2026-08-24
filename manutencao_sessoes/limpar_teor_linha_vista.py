# -*- coding: utf-8 -*-
"""Remove teor gravado em LINHA DE VISTA quando o acórdão pertence à sessão de
CONCLUSÃO (outra página do mesmo processo).

Convenção da base (firmada pelo painel da campanha): a linha interrompida por vista
registra "Suspenso mas julgado depois"/"Suspenso*" e NÃO carrega o acórdão — ele
pertence à página que registra a proclamação. A janela [-5,+60]d do motor de teor
casava o acórdão final com a linha anterior, colando o documento na página errada.

Só remove quando existe OUTRA página do mesmo CNJ, conclusiva e posterior — assim
o acórdão não se perde da base. Uso: limpar_teor_linha_vista.py [--apply]
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
import fill_inteiro_teor as fit  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
APPLY = "--apply" in sys.argv
SUSP_VOT = {"Suspenso", "Suspenso*"}
SUSP_RES = {"Suspenso", "Suspenso mas julgado depois", "Sobrestado"}

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
pages = client.query_data_source()


def t(p, campo):
    return client._extract_property_text(p, schema, campo)


por_cnj = defaultdict(list)
linhas = []
for p in pages:
    cnj = re.sub(r"\D", "", t(p, "numero_processo"))[:20]
    reg = {"page_id": p["id"], "cnj": cnj, "data": t(p, "data_sessao")[:10],
           "votacao": t(p, "votacao"), "resultado": t(p, "resultado"),
           "numero": t(p, "numero_processo")}
    linhas.append(reg)
    if len(cnj) >= 20:
        por_cnj[cnj].append(reg)

alvos = []
for x in linhas:
    if x["votacao"] not in SUSP_VOT and x["resultado"] not in SUSP_RES:
        continue
    irmas = por_cnj.get(x["cnj"]) or []
    conclusiva_depois = [y for y in irmas
                         if y["page_id"] != x["page_id"] and y["data"] > x["data"]
                         and y["votacao"] not in SUSP_VOT and y["resultado"] not in SUSP_RES]
    if conclusiva_depois:
        alvos.append((x, conclusiva_depois[0]))

print(f"{len(alvos)} linhas de vista com irmã conclusiva posterior", flush=True)

log, stats = [], {"removido": 0, "sem_teor": 0, "erro": 0}
for i, (x, final) in enumerate(alvos):
    try:
        children = fit.get_all_children(client, x["page_id"])
        idx = fit.marker_index(children)
        if idx is None:
            stats["sem_teor"] += 1
            continue
        corpo = fit.norm_ws(" ".join(fit._paragrafos_da_secao(children, idx)))
        if APPLY:
            for b in children[idx:]:
                notion_request_with_retry(client, "DELETE", f"/blocks/{b['id']}")
        stats["removido"] += 1
        log.append({"page_id": x["page_id"], "numero": x["numero"], "data_vista": x["data"],
                    "data_conclusao": final["data"], "backup": corpo[:800]})
    except Exception as exc:
        stats["erro"] += 1
        log.append({"page_id": x["page_id"], "erro": str(exc)[:120]})
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(alvos)} {stats}", flush=True)

out = ART / f"limpar_teor_vista_{'apply' if APPLY else 'dry'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
out.write_text(json.dumps({"stats": stats, "log": log}, ensure_ascii=False, indent=1), encoding="utf-8")
print("stats:", stats)
print("salvo:", out)
