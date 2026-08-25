# -*- coding: utf-8 -*-
"""Monta os lotes de RE-TRIAGEM para as páginas cujo TEOR MUDOU.

A triagem meritória (Haiku→Sonnet) avaliou a prosa contra o teor vigente na época.
Quando o teor é substituído (fonte melhor, texto destruncado, documento errado
removido), a avaliação precisa ser refeita SÓ nessas páginas — não na base inteira.

Fontes do escopo (logs da campanha):
  - artifacts/notion_inteiro_teor/*/changes.json  → status regravado_fonte|gravado_faltante
  - aplicar_teor_feitos + fila_teor_sjur_docs_*   → teores gravados do SJUR
  - limpar_teor_vista_apply_*                     → teor REMOVIDO (prosa pode ter sobrado)
  - auditar_teor_enriquecido_apply_*              → teor removido por ser doc errado
  - investigacao_aplicacao_apply_*                → número/etiquetas corrigidos
"""
import glob
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
import fill_inteiro_teor as fit  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
TEOR_RUNS = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_inteiro_teor")
LOTES = ART / f"reteor_lotes_{time.strftime('%Y%m%d')}"

alvos = {}


def marca(pid, motivo):
    if pid:
        alvos.setdefault(pid, set()).add(motivo)


# 1) teor substituído/gravado pela fonte CSV (rodada com o consolidado TOTAL)
for f in sorted(TEOR_RUNS.glob("*/changes.json"))[-3:]:
    for c in json.loads(Path(f).read_text(encoding="utf-8")):
        if c.get("status") in ("regravado_fonte", "gravado_faltante", "regravado"):
            marca(c.get("page_id"), c["status"])

# 2) teores gravados a partir do SJUR (timeouts recuperados + truncados)
for arq in ART.glob("fila_teor_sjur_docs_*.jsonl"):
    if "20260822" in arq.name:      # rodada original já coberta pela triagem anterior
        continue
    for ln in arq.read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        if d.get("doc"):
            marca(d["page_id"], "sjur_novo")

# 3) teor REMOVIDO (linha de vista / documento errado): a prosa pode ter ficado órfã
for pat, motivo in (("limpar_teor_vista_apply_*.json", "teor_removido_vista"),
                    ("auditar_teor_enriquecido_apply_*.json", "teor_removido_errado")):
    fs = sorted(glob.glob(str(ART / pat)))
    if not fs:
        continue
    d = json.loads(Path(fs[-1]).read_text(encoding="utf-8"))
    itens = d.get("log") or d.get("removidos") or []
    for x in itens:
        if isinstance(x, dict) and x.get("page_id"):
            marca(x["page_id"], motivo)

# 4) número/etiquetas corrigidos na investigação
for pat in ("investigacao_aplicacao_apply_*.json", "aplicacao_pontual_apply_*.json"):
    fs = sorted(glob.glob(str(ART / pat)))
    if not fs:
        continue
    for x in json.loads(Path(fs[-1]).read_text(encoding="utf-8")):
        if x.get("status") in ("corrigido", "aplicado"):
            marca(x["page_id"], "numero_corrigido")

# 5) ESTREOU COM TEOR: estava na fila dos "sem teor" e agora tem seção gravada.
# Estas são as mais importantes da re-triagem: na passagem anterior o triador julgou
# a prosa SEM fonte de confronto — só dava para checar coerência interna.
fila_sem_teor = set()
f = ART / "fila_teor_sjur_20260822.json"
if f.exists():
    for x in json.loads(f.read_text(encoding="utf-8")):
        fila_sem_teor.add(x["page_id"])
print(f"fila histórica de 'sem teor': {len(fila_sem_teor)} páginas (serão checadas)", flush=True)

print(f"{len(alvos)} páginas com teor/identidade alterados desde a triagem", flush=True)
from collections import Counter
print("motivos:", Counter(m for ms in alvos.values() for m in ms), flush=True)

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
pages = {p["id"]: p for p in client.query_data_source()}


def t(p, campo):
    return client._extract_property_text(p, schema, campo)


por_cnj = {}
for p in pages.values():
    cnj = re.sub(r"\D", "", t(p, "numero_processo"))[:20]
    if len(cnj) >= 20:
        por_cnj.setdefault(cnj, []).append(p)

# varre a fila histórica: quem tem teor AGORA estreou desde a triagem
for pid in fila_sem_teor:
    if pid in alvos or pid not in pages:
        continue
    try:
        ch = fit.get_all_children(client, pid)
        if fit.marker_index(ch) is not None:
            marca(pid, "estreou_com_teor")
    except Exception:
        pass
print(f"escopo final: {len(alvos)} páginas", flush=True)
print("motivos:", Counter(m for ms in alvos.values() for m in ms), flush=True)

itens = []
for i, (pid, motivos) in enumerate(alvos.items()):
    p = pages.get(pid)
    if p is None:
        continue          # arquivada
    teor = ""
    try:
        ch = fit.get_all_children(client, pid)
        idx = fit.marker_index(ch)
        if idx is not None:
            corpo = fit.norm_ws(" ".join(fit._paragrafos_da_secao(ch, idx)))
            teor = corpo[:800] + (" [...] " + corpo[-2500:] if len(corpo) > 3300 else corpo[800:])
    except Exception:
        pass
    cnj = re.sub(r"\D", "", t(p, "numero_processo"))[:20]
    cadeia = [{"data": t(q, "data_sessao")[:10], "resultado": t(q, "resultado"),
               "votacao": t(q, "votacao"), "punchline": t(q, "punchline")[:160]}
              for q in por_cnj.get(cnj, []) if q["id"] != pid]
    itens.append({
        "page_id": pid, "motivos": sorted(motivos), "data": t(p, "data_sessao")[:10],
        "numero": t(p, "numero_processo"), "classe": t(p, "classe_processo"),
        "tema": t(p, "tema"), "resultado": t(p, "resultado"), "votacao": t(p, "votacao"),
        "pedido_vista": t(p, "pedido_vista"),
        "punchline": t(p, "punchline"), "analise": t(p, "analise_do_conteudo_juridico")[:2500],
        "raciocinio": t(p, "raciocinio_juridico")[:1500], "teor": teor, "cadeia": cadeia,
    })
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(alvos)}", flush=True)

LOTES.mkdir(exist_ok=True)
n = 0
for i in range(0, len(itens), 40):
    n += 1
    (LOTES / f"lote_{n:03d}.json").write_text(json.dumps(itens[i:i + 40], ensure_ascii=False),
                                              encoding="utf-8")
print(f"{len(itens)} itens em {n} lotes: {LOTES}", flush=True)
