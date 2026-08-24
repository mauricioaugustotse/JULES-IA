# -*- coding: utf-8 -*-
"""Extrai o veredito do painel de punchlines e aplica as reescritas."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
SRC = Path(r"C:\Users\mauri\AppData\Local\Temp\claude\c--Users-mauri-ProjetoConversor\d48cbf3a-d817-49bc-903a-c46866886b18\tasks\w4uwgvbis.output")
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
(ART / "punchline_veredito_20260823.json").write_text(
    json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
print("placar:", obj["placar"])

reescrever = [x for x in obj["itens"] if x.get("decisao") == "REESCREVER" and x.get("punchline_nova")]
client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
log = []
for x in reescrever:
    pid = x["page_id"]
    try:
        page = notion_request_with_retry(client, "GET", f"/pages/{pid}")
        antes = client._extract_property_text(page, schema, "punchline")
        if APPLY:
            notion_request_with_retry(client, "PATCH", f"/pages/{pid}", json={
                "properties": {"punchline": {"rich_text": core.chunk_rich_text(x["punchline_nova"])}}})
        log.append({"page_id": pid, "antes": antes[:200], "depois": x["punchline_nova"][:200]})
    except Exception as exc:
        log.append({"page_id": pid, "erro": str(exc)[:150]})

erros = [x for x in log if x.get("erro")]
anal = [x["page_id"] for x in obj["itens"] if x.get("analise_contraditoria")]
out = ART / f"punchline_aplicacao_{'apply' if APPLY else 'dry'}_{STAMP}.json"
out.write_text(json.dumps({"aplicadas": log, "analises_contraditorias_pendentes": anal},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print(f"reescritas: {len(log) - len(erros)} | erros: {len(erros)} | análises flagadas: {len(anal)}")
print("log:", out)
