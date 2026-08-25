# -*- coding: utf-8 -*-
"""Detecta linhas que provavelmente tomaram o número de um PRECEDENTE citado.

Sintoma (caso 0600994-58 em 20/08/2024): duas linhas da MESMA sessão tratam do
mesmo assunto com números DIFERENTES — uma é o processo realmente julgado e a
outra herdou o CNJ de um precedente mencionado no voto. O extrator de vídeo capta
o número que ouviu, não o que está sendo julgado.

Sinais somados: mesma data + tema semelhante + números distintos; agrava se a
suspeita tem campo vazio (votacao/resultado) ou não aparece no DJe/SJUR.
Só relatório — a atribuição exige juízo (painel).
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
STOP = {"de", "da", "do", "das", "dos", "e", "em", "a", "o", "por", "para", "no", "na", "com"}

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
pages = client.query_data_source()


def t(p, campo):
    return client._extract_property_text(p, schema, campo)


def toks(s):
    return {w for w in re.findall(r"\w{4,}", (s or "").lower()) if w not in STOP}


por_dia = defaultdict(list)
for p in pages:
    data = t(p, "data_sessao")[:10]
    if not data:
        continue
    por_dia[data].append({
        "page_id": p["id"], "numero": t(p, "numero_processo"),
        "cnj": re.sub(r"\D", "", t(p, "numero_processo"))[:20],
        "classe": t(p, "classe_processo"), "tema": t(p, "tema"),
        "resultado": t(p, "resultado"), "votacao": t(p, "votacao"),
        "tipo_registro": t(p, "tipo_registro"), "punchline": t(p, "punchline")[:160],
    })

suspeitos = []
for data, grp in por_dia.items():
    for i, a in enumerate(grp):
        for b in grp[i + 1:]:
            if a["cnj"] == b["cnj"] or not a["cnj"] or not b["cnj"]:
                continue
            ta, tb = toks(a["tema"]), toks(b["tema"])
            if not ta or not tb:
                continue
            sim = len(ta & tb) / max(1, min(len(ta), len(tb)))
            if sim < 0.5:
                continue
            # o mais frágil é o candidato a "número de precedente"
            def fragilidade(x):
                return (0 if x["resultado"] else 1) + (0 if x["votacao"] else 1)
            frag_a, frag_b = fragilidade(a), fragilidade(b)
            if frag_a == frag_b == 0:
                grau = "ambos_completos"
            else:
                grau = "um_incompleto"
            suspeito, ancora = (a, b) if frag_a >= frag_b else (b, a)
            suspeitos.append({"data": data, "similaridade": round(sim, 2), "grau": grau,
                              "suspeito": suspeito, "ancora": ancora})

suspeitos.sort(key=lambda x: (-x["similaridade"], x["data"]))
print(f"{len(suspeitos)} par(es) de linhas gêmeas por tema na mesma sessão")
incompletos = [x for x in suspeitos if x["grau"] == "um_incompleto"]
print(f"  com campo vazio no suspeito (sinal forte): {len(incompletos)}")
for x in suspeitos[:10]:
    s, a = x["suspeito"], x["ancora"]
    print(f"  {x['data']} sim={x['similaridade']} | suspeito {s['numero']} [{s['resultado']}/{s['votacao']}]"
          f" x âncora {a['numero']} [{a['resultado']}/{a['votacao']}]")

out = ART / f"numero_de_precedente_{time.strftime('%Y%m%d_%H%M%S')}.json"
out.write_text(json.dumps(suspeitos, ensure_ascii=False, indent=1), encoding="utf-8")
print("salvo:", out)
