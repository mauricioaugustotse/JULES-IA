# -*- coding: utf-8 -*-
"""Aplica as 154 correções confirmadas pela revisão meritória (etiquetas + prosa).
Backup dos valores atuais no log. Uso: aplicar_revisao.py [--apply]
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
SRC = Path([a for a in sys.argv[1:] if not a.startswith("--")][0])
STAMP = time.strftime("%Y%m%d_%H%M%S")
APPLY = "--apply" in sys.argv

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
assert obj is not None
(ART / "revisao_reteor_veredito_20260824.json").write_text(
    json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
print("placar:", obj["placar"])

MAPA = {"analise": "analise_do_conteudo_juridico", "raciocinio": "raciocinio_juridico",
        "punchline": "punchline", "tema": "tema",
        "resultado": "resultado", "votacao": "votacao",
        "classe": "classe_processo", "classe_processo": "classe_processo"}
SELECTS = {"resultado", "votacao", "classe_processo"}
# classe só entra se a option JÁ existir — nunca criar variante nova no schema
CLASSES_VALIDAS = None
VOCAB_VOT = {"Unânime", "Por maioria", "Suspenso", "Suspenso*"}

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
props_schema = schema.raw_payload.get("properties") or {}
res_opts = {o.get("name") for o in props_schema.get("resultado", {}).get("select", {}).get("options", [])}
tema_tipo = props_schema.get("tema", {}).get("type", "rich_text")
CLASSES_VALIDAS = {o.get("name") for o in
                   props_schema.get("classe_processo", {}).get("select", {}).get("options", [])}
print(f"{len(CLASSES_VALIDAS)} options de classe_processo no schema")

corrigir = [x for x in obj["itens"] if x.get("decisao") == "CORRIGIR" and x.get("correcoes")]
log = []
for x in corrigir:
    pid = x["page_id"]
    props = {}
    backup = {}
    pulados = []
    try:
        page = notion_request_with_retry(client, "GET", f"/pages/{pid}")
        for campo, valor in x["correcoes"].items():
            prop = MAPA.get(campo)
            if not prop or not valor or not str(valor).strip():
                pulados.append(campo)
                continue
            valor = str(valor).strip()
            backup[prop] = client._extract_property_text(page, schema, prop)
            if prop in SELECTS:
                if prop == "votacao" and valor not in VOCAB_VOT:
                    pulados.append(f"{campo}={valor}(fora_vocab)")
                    continue
                if prop == "classe_processo" and CLASSES_VALIDAS and valor not in CLASSES_VALIDAS:
                    # option nova em classe fragmenta o vocabulário da base — nunca criar
                    pulados.append(f"{campo}={valor}(option_inexistente)")
                    continue
                if prop == "resultado" and res_opts and valor not in res_opts:
                    # opção nova legítima do vocabulário do prompt — o Notion cria
                    pass
                props[prop] = {"select": {"name": valor}}
            elif prop == "tema" and tema_tipo == "title":
                props[prop] = {"title": core.chunk_rich_text(valor)}
            else:
                props[prop] = {"rich_text": core.chunk_rich_text(valor)}
        if not props:
            log.append({"page_id": pid, "status": "nada", "pulados": pulados})
            continue
        if APPLY:
            notion_request_with_retry(client, "PATCH", f"/pages/{pid}", json={"properties": props})
        log.append({"page_id": pid, "status": "aplicada", "campos": sorted(props),
                    "backup": backup, "pulados": pulados,
                    "justificativa": (x.get("justificativa") or "")[:150]})
    except Exception as exc:
        log.append({"page_id": pid, "status": "erro", "erro": str(exc)[:200]})

out = ART / f"revisao_reteor_aplicacao_{'apply' if APPLY else 'dry'}_{STAMP}.json"
out.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
print(Counter(x["status"] for x in log))
campos_ct = Counter(c for x in log if x["status"] == "aplicada" for c in x["campos"])
print("por campo:", dict(campos_ct))
print("log:", out)

