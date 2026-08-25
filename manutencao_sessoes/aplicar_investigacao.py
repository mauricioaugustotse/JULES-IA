# -*- coding: utf-8 -*-
"""Aplica o veredito do painel de investigação (cadeia / cnj_impossivel /
numero_precedente / sem_teor).

Guardas: vocabulário fechado para resultado/votacao; CNJ só no formato completo;
colisão (mesmo CNJ proposto para 2+ páginas) bloqueia ambas; arquivamento com
backup COMPLETO das properties. Uso: aplicar_investigacao.py <output.json> [--apply]
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
SRC = Path(sys.argv[1])
APPLY = "--apply" in sys.argv
STAMP = time.strftime("%Y%m%d_%H%M%S")
CNJ_RE = re.compile(r"^\d{7}-\d{2}\.\d{4}\.6\.\d{2}\.\d{4}$")
VOCAB_VOT = {"Unânime", "Por maioria", "Suspenso", "Suspenso*"}
CAMPOS = {"numero_processo": "rich_text", "resultado": "select", "votacao": "select",
          "tema": "rich_text", "punchline": "rich_text"}

raiz = json.loads(SRC.read_text(encoding="utf-8"))
obj = None


def acha(o):
    global obj
    if obj is not None:
        return
    if isinstance(o, dict):
        if "placar" in o and "itens" in o:
            obj = o
            return
        for v in o.values():
            acha(v)
    elif isinstance(o, list):
        for v in o:
            acha(v)
    elif isinstance(o, str) and '"placar"' in o:
        try:
            acha(json.loads(o))
        except Exception:
            pass


acha(raiz)
assert obj is not None, "não achei o veredito no output"
(ART / f"investigacao_veredito_{STAMP}.json").write_text(
    json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
print("placar:", obj["placar"])

itens = obj["itens"]
# colisão de CNJ proposto
propostos = defaultdict(list)
for x in itens:
    n = (x.get("correcoes") or {}).get("numero_processo")
    if n:
        propostos[n.strip()].append(x["page_id"])
colide = {pid for n, ps in propostos.items() if len(ps) > 1 for pid in ps}
if colide:
    print(f"  {len(colide)} páginas bloqueadas por colisão de CNJ proposto")

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
log = []
for x in itens:
    pid, dec = x["page_id"], x.get("decisao")
    reg = {"page_id": pid, "frente": x.get("frente"), "decisao": dec,
           "justificativa": (x.get("justificativa") or "")[:200],
           "evidencia": (x.get("evidencia") or "")[:200]}
    try:
        if dec == "CORRIGIR":
            if pid in colide:
                reg["status"] = "pulado_colisao"
                log.append(reg)
                continue
            page = notion_request_with_retry(client, "GET", f"/pages/{pid}")
            props, backup, pulados = {}, {}, []
            for campo, valor in (x.get("correcoes") or {}).items():
                tipo = CAMPOS.get(campo)
                if not tipo or not str(valor or "").strip():
                    pulados.append(campo)
                    continue
                valor = str(valor).strip()
                if campo == "numero_processo" and not CNJ_RE.match(valor):
                    pulados.append(f"{campo}(formato)")
                    continue
                if campo == "votacao" and valor not in VOCAB_VOT:
                    pulados.append(f"{campo}(vocab)")
                    continue
                backup[campo] = client._extract_property_text(page, schema, campo)
                props[campo] = ({"select": {"name": valor}} if tipo == "select"
                                else {"rich_text": core.chunk_rich_text(valor)})
            if not props:
                reg["status"] = "nada_a_aplicar"
                reg["pulados"] = pulados
            else:
                if APPLY:
                    notion_request_with_retry(client, "PATCH", f"/pages/{pid}",
                                              json={"properties": props})
                reg.update(status="corrigido", campos=sorted(props), backup=backup, pulados=pulados)
        elif dec == "ARQUIVAR":
            page = notion_request_with_retry(client, "GET", f"/pages/{pid}")
            if APPLY:
                notion_request_with_retry(client, "PATCH", f"/pages/{pid}", json={"archived": True})
            reg.update(status="arquivado", properties_backup=page.get("properties"))
        else:
            reg["status"] = f"sem_acao ({dec})"
    except Exception as exc:
        reg["status"] = "erro"
        reg["erro"] = str(exc)[:200]
    log.append(reg)

out = ART / f"investigacao_aplicacao_{'apply' if APPLY else 'dry'}_{STAMP}.json"
out.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
print(Counter(x["status"] for x in log))
print("log:", out)
