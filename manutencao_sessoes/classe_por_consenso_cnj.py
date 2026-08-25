# -*- coding: utf-8 -*-
"""Propaga a classe OFICIAL para as páginas do mesmo CNJ que não têm teor.

Um processo tem UMA classe. Quando ao menos uma página do CNJ traz o cabeçalho
formal do acórdão (fonte oficial), as demais páginas do mesmo número devem estar
na mesma FAMÍLIA. Foi assim com 0600749-95.2019.6.00.0000: quatro páginas, uma com
teor "INSTRUÇÃO (11544) N. 0600749-95..." e três rotuladas "PC" porque o tema
falava de prestação de contas — mas o processo é a Instrução que EDITA as normas
sobre o assunto.

Guardas: só propaga quando a família diverge (REspe→ED-REspe é fase, não erro);
só para options existentes; e NÃO propaga para páginas cujo `resultado` é
incompatível com a classe oficial (sinal de que o número é que está errado) —
essas são reportadas para revisão.

Uso: classe_por_consenso_cnj.py [--apply]
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
import fill_inteiro_teor as fit  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
APPLY = "--apply" in sys.argv

src = (Path(r"C:\Users\mauri\JULES-IA\manutencao_sessoes\classe_pelo_cabecalho.py")
       .read_text(encoding="utf-8"))
ns = {"re": re}
exec(src[src.index("MAPA = ["):src.index("client = core.NotionSessoesClient")], ns)
classe_do_cabecalho, familia = ns["classe_do_cabecalho"], ns["familia"]

# resultados típicos por classe — incompatibilidade sugere número errado, não classe errada
RESULT_RECURSAL = {"Provido", "Desprovido", "Provido em parte", "Não conhecido", "Acolhidos",
                   "Rejeitados", "Prejudicado"}
INCOMPATIVEL = {
    "Instrução": RESULT_RECURSAL - {"Prejudicado"},
    "PA": RESULT_RECURSAL - {"Prejudicado"},
    "Lista Tríplice": RESULT_RECURSAL - {"Prejudicado"},
}

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
CLASSES = {o.get("name") for o in (schema.raw_payload.get("properties") or {})
           .get("classe_processo", {}).get("select", {}).get("options", [])}
pages = client.query_data_source()

por_cnj = defaultdict(list)
for p in pages:
    cnj = re.sub(r"\D", "", client._extract_property_text(p, schema, "numero_processo"))[:20]
    if len(cnj) == 20:
        por_cnj[cnj].append(p)
multi = {c: g for c, g in por_cnj.items() if len(g) > 1}
print(f"{len(multi)} CNJs com mais de uma página", flush=True)

propagar, revisar, stats = [], [], Counter()
for cnj, grp in multi.items():
    oficial = None
    for p in grp:
        try:
            ch = fit.get_all_children(client, p["id"])
            idx = fit.marker_index(ch)
            if idx is None:
                continue
            corpo = fit.norm_ws(" ".join(fit._paragrafos_da_secao(ch, idx)))
        except Exception:
            continue
        # o teor precisa ser DESTE processo
        m = re.search(r"[\d.\-–]{15,}", corpo[:400])
        if m and re.sub(r"\D", "", m.group(0))[:20] != cnj:
            continue
        c = classe_do_cabecalho(corpo)
        if c and c in CLASSES:
            oficial = c
            break
    if not oficial:
        stats["sem_classe_oficial"] += 1
        continue
    stats["cnj_com_classe_oficial"] += 1
    for p in grp:
        def t(campo):
            return client._extract_property_text(p, schema, campo)
        atual = t("classe_processo")
        if atual and familia(atual) == familia(oficial):
            stats["ok"] += 1
            continue
        reg = {"page_id": p["id"], "cnj": cnj, "numero": t("numero_processo"),
               "data": t("data_sessao")[:10], "atual": atual, "oficial": oficial,
               "resultado": t("resultado"), "tema": t("tema")[:70]}
        if t("resultado") in INCOMPATIVEL.get(oficial, set()):
            stats["resultado_incompativel"] += 1
            revisar.append({**reg, "alerta": "resultado incompatível com a classe oficial — "
                                             "possível número errado nesta linha"})
            continue
        stats["propagar"] += 1
        propagar.append(reg)

print("stats:", dict(stats), flush=True)
print(f"\nPROPAGAR: {len(propagar)} | REVISAR (resultado incompatível): {len(revisar)}")
for (a, o), n in Counter((x["atual"], x["oficial"]) for x in propagar).most_common(10):
    print(f"  {n:3d}  {a!r:16} -> {o!r}")
for x in propagar[:8]:
    print(f"   {x['data']} {x['numero'][:24]} {x['atual']!r}->{x['oficial']!r} | {x['tema'][:40]}")
for x in revisar[:5]:
    print(f"   [REVISAR] {x['data']} {x['numero'][:24]} {x['atual']!r} res={x['resultado']!r}")

log = []
if APPLY:
    for x in propagar:
        try:
            notion_request_with_retry(client, "PATCH", f"/pages/{x['page_id']}", json={
                "properties": {"classe_processo": {"select": {"name": x["oficial"]}}}})
            log.append({**x, "status": "aplicado"})
        except Exception as exc:
            log.append({**x, "status": "erro", "erro": str(exc)[:150]})
    print("aplicados:", Counter(x["status"] for x in log))

out = ART / f"classe_consenso_{'apply' if APPLY else 'dry'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
out.write_text(json.dumps({"stats": dict(stats), "propagar": propagar, "revisar": revisar,
                           "aplicados": log}, ensure_ascii=False, indent=1), encoding="utf-8")
print("salvo:", out)
