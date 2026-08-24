# -*- coding: utf-8 -*-
"""Audita os teores gravados a partir do sjur_enriquecer (campanha do núcleo duro).

Dois defeitos conhecidos daquela fonte:
  (a) TRUNCAMENTO em 2500 chars (limpa(n=2500));
  (b) tipo ERRADO: "Decisão monocrática" colada como se fosse o acórdão da sessão
      — monocrática não é julgamento colegiado, e a linha de VISTA não deve receber
      o acórdão da sessão de conclusão.

Reporta (e com --apply REMOVE) as seções de teor cujo doc de origem é monocrático
ou cuja dataDecisao está fora de [-5,+60]d da sessão. Backup do texto no log.
Uso: auditar_teor_enriquecido.py [--apply]
"""
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402
import fill_inteiro_teor as fit  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
APPLY = "--apply" in sys.argv

enr = {}
for ln in (ART / "nucleo_duro_sjur_enriquecido_20260821.jsonl").read_text(encoding="utf-8").splitlines():
    d = json.loads(ln)
    if d.get("doc"):
        enr[d["page_id"]] = d

# docs bons (motor, sem truncamento) para saber quem já foi substituído
bons = set()
for arq in ART.glob("fila_teor_sjur_docs_*.jsonl"):
    for ln in arq.read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        if d.get("doc"):
            bons.add(d["page_id"])

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
por_id = {p["id"]: p for p in client.query_data_source()}
print(f"{len(enr)} páginas com doc do enriquecedor; {len(bons)} já têm doc do motor", flush=True)

problemas, log = [], []
for pid, d in enr.items():
    p = por_id.get(pid)
    if p is None:
        continue
    doc = d["doc"]
    data_sessao = client._extract_property_text(p, schema, "data_sessao")[:10]
    tipo = (doc.get("tipoDecisao") or doc.get("descricaoTipoDecisao") or "")
    dec = doc.get("textoDecisao") or ""
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", doc.get("dataDecisao") or "")
    delta = None
    if m and data_sessao:
        d_doc = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        try:
            delta = (dt.date.fromisoformat(d_doc) - dt.date.fromisoformat(data_sessao)).days
        except ValueError:
            pass
    motivos = []
    if re.search(r"monocr", tipo, re.I):
        motivos.append(f"tipo={tipo}")
    if delta is not None and not (-5 <= delta <= 60):
        motivos.append(f"data {doc.get('dataDecisao')} a {delta}d da sessão {data_sessao}")
    if len(dec) >= 2490:
        motivos.append("truncado_2500")
    if not motivos:
        continue
    substituido = pid in bons
    problemas.append({"page_id": pid, "numero": client._extract_property_text(p, schema, "numero_processo"),
                      "data_sessao": data_sessao, "motivos": motivos, "ja_substituido": substituido})

print(f"\n{len(problemas)} páginas com teor suspeito do enriquecedor:")
from collections import Counter
print(" motivos:", Counter(m.split("=")[0].split(" ")[0] for x in problemas for m in x["motivos"]))
print(" já substituídas por doc bom:", sum(1 for x in problemas if x["ja_substituido"]))

# Três baldes por FORÇA da evidência. A data POSTERIOR sozinha não condena: o painel
# do núcleo duro já assentou que acórdão meses após a sessão pode ser julgamento
# estendido legítimo. Já um documento ANTERIOR à sessão nunca pode ser o resultado dela.
def _delta(x):
    for m in x["motivos"]:
        mm = re.search(r"a (-?\d+)d da sessão", m)
        if mm:
            return int(mm.group(1))
    return None


remover, rebuscar, reportar = [], [], []
for x in problemas:
    if x["ja_substituido"]:
        continue
    d = _delta(x)
    trunc = any("truncado" in m for m in x["motivos"])
    mono = any("monocr" in m for m in x["motivos"])
    if mono or (d is not None and d < -5):
        remover.append(x)          # documento anterior à sessão ou monocrático: não é o acórdão
    elif trunc:
        rebuscar.append(x)         # texto cortado, mas plausível: buscar íntegra
    else:
        reportar.append(x)         # só data posterior: pode ser julgamento estendido
print(f" REMOVER (anterior à sessão/monocrática): {len(remover)}")
print(f" RE-BUSCAR (truncado, data plausível): {len(rebuscar)}")
print(f" reportar (só data posterior — pode ser estendido): {len(reportar)}")
for x in remover[:6]:
    print(f"   rm {x['data_sessao']} {x['numero'][:24]} | {x['motivos']}")
fila_rb = [{"page_id": x["page_id"], "numero": x["numero"], "data": x["data_sessao"],
            "motivo": "truncado"} for x in rebuscar]
(ART / "fila_teor_truncados_rebuscar.json").write_text(
    json.dumps(fila_rb, ensure_ascii=False, indent=1), encoding="utf-8")

if APPLY:
    for x in remover:
        try:
            children = fit.get_all_children(client, x["page_id"])
            idx = fit.marker_index(children)
            if idx is None:
                continue
            x["backup"] = fit.norm_ws(" ".join(fit._paragrafos_da_secao(children, idx)))[:600]
            for b in children[idx:]:
                notion_request_with_retry(client, "DELETE", f"/blocks/{b['id']}")
            x["status"] = "removido"
        except Exception as exc:
            x["status"] = f"erro: {str(exc)[:120]}"
        log.append(x)

out = ART / f"auditar_teor_enriquecido_{'apply' if APPLY else 'dry'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
out.write_text(json.dumps({"problemas": problemas, "removidos": log}, ensure_ascii=False, indent=1),
               encoding="utf-8")
print("salvo:", out)
