# -*- coding: utf-8 -*-
"""Revalida as divergências de classe com a função COM guardas.

A primeira varredura rodou uma versão permissiva que lia a EMENTA como cabeçalho
("ELEIÇÕES 2016. REGISTRO DE CANDIDATURA. RECURSO ESPECIAL..." → RCand, errado).
A versão atual exige cabeçalho FORMAL (classe seguida do número). Como ela é
estritamente mais restritiva, o conjunto certo é subconjunto do já detectado:
revalidar as 806 basta e evita reler as 3.388 páginas.

Uso: revalidar_classe.py [--apply]
"""
import glob
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402
import fill_inteiro_teor as fit  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
APPLY = "--apply" in sys.argv

# importa MAPA/FAMILIA/classe_do_cabecalho da versão atual, sem rodar o main
src = (Path(r"C:\Users\mauri\JULES-IA\manutencao_sessoes\classe_pelo_cabecalho.py")
       .read_text(encoding="utf-8"))
ns = {"re": re}
exec(src[src.index("MAPA = ["):src.index("client = core.NotionSessoesClient")], ns)
classe_do_cabecalho, familia = ns["classe_do_cabecalho"], ns["familia"]

anterior = json.loads(Path(sorted(glob.glob(str(ART / "classe_cabecalho_dry_*.json")))[-1])
                      .read_text(encoding="utf-8"))
cand = anterior["divergentes"]
print(f"{len(cand)} candidatos da varredura permissiva; revalidando com guardas", flush=True)

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
CLASSES = {o.get("name") for o in (schema.raw_payload.get("properties") or {})
           .get("classe_processo", {}).get("select", {}).get("options", [])}

confirmados, stats = [], Counter()
for i, d in enumerate(cand):
    try:
        ch = fit.get_all_children(client, d["page_id"])
        idx = fit.marker_index(ch)
        if idx is None:
            stats["sem_teor_agora"] += 1
            continue
        corpo = fit.norm_ws(" ".join(fit._paragrafos_da_secao(ch, idx)))
    except Exception:
        stats["erro"] += 1
        continue
    oficial = classe_do_cabecalho(corpo)
    if not oficial:
        stats["falso_positivo_ementa"] += 1
        continue
    if oficial not in CLASSES:
        stats["option_inexistente"] += 1
        continue
    atual = d["atual"]
    if atual and familia(atual) == familia(oficial):
        stats["mesma_familia"] += 1
        continue
    dig_pag = re.sub(r"\D", "", d["numero"])[:20]
    m = re.search(r"[\d.\-–]{15,}", corpo[:400])
    dig_cab = re.sub(r"\D", "", m.group(0))[:20] if m else ""
    if dig_pag and dig_cab and dig_pag != dig_cab:
        stats["teor_de_outro_processo"] += 1
        continue
    stats["confirmado"] += 1
    confirmados.append({**d, "oficial_revalidado": oficial, "cabecalho": corpo[:150]})
    if (i + 1) % 150 == 0:
        print(f"  {i+1}/{len(cand)} {dict(stats)}", flush=True)

print("stats:", dict(stats), flush=True)
print(f"\nCONFIRMADOS: {len(confirmados)} (de {len(cand)})")
for (a, o), n in Counter((c["atual"], c["oficial_revalidado"]) for c in confirmados).most_common(12):
    print(f"  {n:3d}  {a!r:16} -> {o!r}")
for c in confirmados[:8]:
    print(f"   {c['data']} {c['numero'][:24]} {c['atual']!r}->{c['oficial_revalidado']!r}")
    print(f"      {c['cabecalho'][:100]}")

log = []
if APPLY:
    for c in confirmados:
        try:
            notion_request_with_retry(client, "PATCH", f"/pages/{c['page_id']}", json={
                "properties": {"classe_processo": {"select": {"name": c["oficial_revalidado"]}}}})
            log.append({**c, "status": "aplicado"})
        except Exception as exc:
            log.append({**c, "status": "erro", "erro": str(exc)[:150]})
    print("aplicados:", Counter(x["status"] for x in log))

out = ART / f"classe_revalidada_{'apply' if APPLY else 'dry'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
out.write_text(json.dumps({"stats": dict(stats), "confirmados": confirmados, "aplicados": log},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print("salvo:", out)
