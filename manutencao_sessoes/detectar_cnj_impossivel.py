# -*- coding: utf-8 -*-
"""Detecta números CNJ IMPOSSÍVEIS para a data da sessão.

Um processo cujo ano do CNJ é POSTERIOR ao ano da sessão não pode ter sido julgado
nela (o processo ainda não existia). É prova de número corrompido pelo ASR — não
exige juízo, só aritmética. Também reporta o caso limítrofe (mesmo ano) quando a
sessão é anterior a fevereiro, e o número com ano absurdamente antigo (>12 anos).

Uso: detectar_cnj_impossivel.py  (só relatório; a correção exige fonte externa)
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
pages = client.query_data_source()


def t(p, campo):
    return client._extract_property_text(p, schema, campo)


impossiveis, muito_antigos = [], []
for p in pages:
    num = t(p, "numero_processo")
    data = t(p, "data_sessao")[:10]
    m = re.search(r"\.(\d{4})\.\d\.\d{2}\.", num)
    if not m or not data:
        continue
    ano_cnj, ano_sessao = int(m.group(1)), int(data[:4])
    reg = {"page_id": p["id"], "numero": num, "data_sessao": data,
           "ano_cnj": ano_cnj, "delta_anos": ano_cnj - ano_sessao,
           "classe": t(p, "classe_processo"), "tema": t(p, "tema")[:90],
           "resultado": t(p, "resultado"), "punchline": t(p, "punchline")[:200],
           "youtube": t(p, "youtube_link")}
    if ano_cnj > ano_sessao:
        impossiveis.append(reg)
    elif ano_sessao - ano_cnj > 12:
        muito_antigos.append(reg)

print(f"{len(pages)} páginas verificadas")
print(f"IMPOSSÍVEIS (CNJ nasceu depois da sessão): {len(impossiveis)}")
from collections import Counter
print("  por delta de anos:", dict(Counter(x["delta_anos"] for x in impossiveis)))
print(f"suspeitos por idade (>12 anos entre CNJ e sessão): {len(muito_antigos)}")
for x in impossiveis[:10]:
    print(f"   {x['data_sessao']} | {x['numero']} (+{x['delta_anos']}a) | {x['classe']} | {x['tema'][:50]}")

out = ART / f"cnj_impossiveis_{time.strftime('%Y%m%d_%H%M%S')}.json"
out.write_text(json.dumps({"impossiveis": impossiveis, "muito_antigos": muito_antigos},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print("salvo:", out)
