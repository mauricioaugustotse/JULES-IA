# -*- coding: utf-8 -*-
"""Extrai o veredito do painel da cadeia e aplica as correções (resultado/votacao)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
SRC = Path(r"C:\Users\mauri\AppData\Local\Temp\claude\c--Users-mauri-ProjetoConversor\d48cbf3a-d817-49bc-903a-c46866886b18\tasks\w95jcgazp.output")
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
(ART / "cadeia_julgado_veredito_20260823.json").write_text(
    json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
print("placar:", obj["placar"])

corrigir = [x for x in obj["itens"] if x.get("decisao") == "CORRIGIR"
            and x.get("resultado_novo") and x.get("votacao_novo")]
print(f"a corrigir: {len(corrigir)}; APPLY={APPLY}")

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
log = []
for x in corrigir:
    pid = x["page_id"]
    try:
        page = notion_request_with_retry(client, "GET", f"/pages/{pid}")
        antes = {c: client._extract_property_text(page, schema, c) for c in ("resultado", "votacao")}
        if APPLY:
            notion_request_with_retry(client, "PATCH", f"/pages/{pid}", json={"properties": {
                "resultado": {"select": {"name": x["resultado_novo"]}},
                "votacao": {"select": {"name": x["votacao_novo"]}},
            }})
        log.append({"page_id": pid, "direcao": x.get("direcao"), "antes": antes,
                    "depois": {"resultado": x["resultado_novo"], "votacao": x["votacao_novo"]},
                    "justificativa": x.get("justificativa", "")[:200]})
    except Exception as exc:
        log.append({"page_id": pid, "erro": str(exc)[:200]})

erros = [x for x in log if x.get("erro")]
out = ART / f"cadeia_aplicacao_{'apply' if APPLY else 'dry'}_{STAMP}.json"
out.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"ok: {len(log) - len(erros)} | erros: {len(erros)}")
print("salvo:", out)
