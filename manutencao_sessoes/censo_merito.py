# -*- coding: utf-8 -*-
"""Censo dos campos meritórios: vazios/curtos por campo, cruzado com ter teor."""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
pages = client.query_data_source()

CAMPOS = ["punchline", "analise_do_conteudo_juridico", "raciocinio_juridico",
          "fundamentacao_normativa", "precedentes_citados", "resoluções_citadas"]

# quem tem teor: feitos SJUR + changes do sanear
tem_teor = set()
try:
    tem_teor |= {x.replace("-", "") for x in json.loads((ART / "aplicar_teor_feitos.json").read_text(encoding="utf-8"))}
except Exception:
    pass
import glob
for run in sorted(glob.glob(r"C:\Users\mauri\JULES-IA\artifacts\notion_inteiro_teor\*\changes.json")):
    for ch in json.load(open(run, encoding="utf-8")):
        if str(ch.get("status", "")).startswith(("gravado", "regravado", "ja_")):
            tem_teor.add(ch["page_id"].replace("-", ""))

stats = {c: Counter() for c in CAMPOS}
listas = {c: [] for c in CAMPOS}
for p in pages:
    pid = p["id"].replace("-", "")
    teor = pid in tem_teor
    for c in CAMPOS:
        v = client._extract_property_text(p, schema, c).strip()
        if not v:
            k = "vazio_com_teor" if teor else "vazio_sem_teor"
            stats[c][k] += 1
            listas[c].append({"page_id": p["id"], "teor": teor})
        elif len(v) < 40:
            stats[c]["curto"] += 1
        else:
            stats[c]["ok"] += 1

print(f"{len(pages)} páginas; {len(tem_teor)} com teor conhecido\n")
for c in CAMPOS:
    print(f"{c}: {dict(stats[c])}")
out = ART / f"censo_merito_{time.strftime('%Y%m%d_%H%M%S')}.json"
out.write_text(json.dumps({c: listas[c] for c in CAMPOS}, ensure_ascii=False), encoding="utf-8")
print("\nlistas salvas:", out)
