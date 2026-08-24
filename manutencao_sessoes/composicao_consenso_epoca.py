# -*- coding: utf-8 -*-
"""Fallback p/ composição >7 sem lista no acórdão: filtra pelo elenco da ÉPOCA
(7 ministros mais frequentes nas composições ≤7 da base em ±45 dias da sessão).
Só aplica quando o filtrado fica com 3-7 nomes. Uso: [--apply]
"""
import datetime as dt
import json
import sys
import time
from collections import Counter
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


def multi(p, campo):
    return [o.get("name", "") for o in p.get("properties", {}).get(campo, {}).get("multi_select", [])]


linhas = []
for p in pages:
    data = t(p, "data_sessao")[:10]
    comp = multi(p, "composicao")
    if data and comp:
        linhas.append((p["id"], data, comp, t(p, "numero_processo")))

boas = [(d, set(c)) for _, d, c, _ in linhas if len(set(c)) <= 7]


def elenco_epoca(data_iso):
    base = dt.date.fromisoformat(data_iso)
    cont = Counter()
    n = 0
    for d, c in boas:
        try:
            delta = abs((dt.date.fromisoformat(d) - base).days)
        except ValueError:
            continue
        if delta <= 45:
            cont.update(c)
            n += 1
    if n < 5:
        return None
    return {m for m, _ in cont.most_common(7)}


log, stats = [], {"corrigidas": 0, "sem_epoca": 0, "filtrado_invalido": 0}
for pid, data, comp, numero in linhas:
    unico = list(dict.fromkeys(comp))
    if len(unico) <= 7:
        continue
    elenco = elenco_epoca(data)
    if not elenco:
        stats["sem_epoca"] += 1
        continue
    filtrado = [m for m in unico if m in elenco]
    if not (3 <= len(filtrado) <= 7) or len(filtrado) == len(unico):
        stats["filtrado_invalido"] += 1
        log.append({"page_id": pid, "numero": numero, "data": data, "n": len(unico),
                    "status": "sem_fix", "comp": unico, "elenco_epoca": sorted(elenco)})
        continue
    if APPLY:
        notion_request_with_retry(client, "PATCH", f"/pages/{pid}", json={
            "properties": {"composicao": {"multi_select": [{"name": m} for m in filtrado]}}})
    log.append({"page_id": pid, "numero": numero, "data": data, "de": unico, "para": filtrado,
                "removidos": [m for m in unico if m not in filtrado], "status": "corrigida"})
    stats["corrigidas"] += 1

out = ART / f"composicao_epoca_{'apply' if APPLY else 'dry'}_{STAMP}.json"
out.write_text(json.dumps({"stats": stats, "log": log}, ensure_ascii=False, indent=1), encoding="utf-8")
print("stats:", stats)
print("salvo:", out)
