# -*- coding: utf-8 -*-
"""Detector da CADEIA DO JULGADO (bidirecional).

Direção A: processo com linha FINAL conclusiva e linha ANTERIOR também conclusiva
da mesma cadeia → anterior deveria ser Suspenso*/"Suspenso mas julgado depois".
Exceção: ED/reconsideração são transparentes (não invalidam desfecho anterior).

Direção B: linha marcada Suspenso*/"Suspenso mas julgado depois" (ou Suspenso)
QUE TEM inteiro teor gravado no corpo → o dispositivo pode provar desfecho real
(ex.: 0600188-95 CTA "Respondida"/"Por maioria").

Gera candidatos ricos p/ painel; não aplica nada.

Modo watcher (pós-processamento automático): --so-a --fila
  Roda só a direção A (por properties, sem ler corpos), SUPRIME os pares já
  julgados (vereditos cadeia_julgado_veredito_*.json — LEGITIMO/INCERTO — e os
  já corrigidos), e ACRESCENTA só os pares NOVOS à fila persistente
  cadeia_fila_pendente.json. Nunca aplica nada: a correção continua exigindo
  painel/revisão (rotina do kit).
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402
import fill_inteiro_teor as fit  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
STAMP = time.strftime("%Y%m%d_%H%M%S")
SO_A = "--so-a" in sys.argv
FILA = "--fila" in sys.argv
FILA_PATH = ART / "cadeia_fila_pendente.json"

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
pages = client.query_data_source()
print(f"{len(pages)} páginas", flush=True)


def t(p, campo):
    return client._extract_property_text(p, schema, campo)


SUSP = {"Suspenso", "Suspenso*"}
SUSP_RES = {"Suspenso", "Suspenso mas julgado depois", "Sobrestado"}
ED_RE = re.compile(r"\bED\b|EMB|embargos de declara|reconsidera", re.I)


def eh_ed(classe, tema, punch):
    return bool(ED_RE.search(f"{classe} {tema} {punch}"))


def norm_classe(c):
    return re.sub(r"[^A-Z]", "", (c or "").upper().split(" ")[0])


linhas = []
for p in pages:
    cnj = re.sub(r"\D", "", t(p, "numero_processo"))[:20]
    if len(cnj) < 20:
        continue
    linhas.append({
        "page_id": p["id"], "cnj": cnj, "data": t(p, "data_sessao")[:10],
        "classe": t(p, "classe_processo"), "resultado": t(p, "resultado"),
        "votacao": t(p, "votacao"), "pedido_vista": t(p, "pedido_vista"),
        "tema": t(p, "tema")[:120], "punchline": t(p, "punchline")[:200],
        "tipo_registro": t(p, "tipo_registro"),
    })

por_cnj = defaultdict(list)
for ln in linhas:
    por_cnj[ln["cnj"]].append(ln)

# --- Direção A ---
dir_a = []
for cnj, grp in por_cnj.items():
    if len(grp) < 2:
        continue
    grp.sort(key=lambda x: x["data"])
    concl = [x for x in grp if x["votacao"] not in SUSP and x["resultado"] not in SUSP_RES
             and (x["resultado"] or x["votacao"])]
    if len(concl) < 2:
        continue
    final = concl[-1]
    for ant in concl[:-1]:
        if ant["data"] >= final["data"]:
            continue
        # ED/reconsideração transparentes: se a final é ED, anterior mantém; se anterior é ED, idem
        if eh_ed(final["classe"], final["tema"], final["punchline"]):
            continue
        if eh_ed(ant["classe"], ant["tema"], ant["punchline"]):
            continue
        if norm_classe(ant["classe"]) != norm_classe(final["classe"]):
            continue
        dir_a.append({"cnj": cnj, "anterior": ant, "final": final})

if FILA:
    # modo watcher: acrescenta só pares NOVOS à fila persistente, suprimindo julgados
    julgados = set()
    for vf in ART.glob("cadeia_julgado_veredito_*.json"):
        try:
            v = json.loads(vf.read_text(encoding="utf-8"))
            julgados.update(x["page_id"].replace("-", "") for x in v.get("itens", []))
        except Exception:
            pass
    for af in ART.glob("cadeia_aplicacao_apply_*.json"):
        try:
            julgados.update(x["page_id"].replace("-", "") for x in json.loads(af.read_text(encoding="utf-8"))
                            if isinstance(x, dict) and x.get("page_id"))
        except Exception:
            pass
    fila = []
    chaves = set()
    if FILA_PATH.exists():
        fila = json.loads(FILA_PATH.read_text(encoding="utf-8"))
        chaves = {(x["anterior"]["page_id"], x["final"]["data"]) for x in fila}
    novos = 0
    for par in dir_a:
        pid_ant = par["anterior"]["page_id"]
        chave = (pid_ant, par["final"]["data"])
        if pid_ant.replace("-", "") in julgados or chave in chaves:
            continue
        fila.append(par)
        chaves.add(chave)
        novos += 1
    FILA_PATH.write_text(json.dumps(fila, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"cadeia: {len(dir_a)} pares na base; {novos} NOVOS na fila (total pendente: {len(fila)})",
          flush=True)
    print(f"fila: {FILA_PATH}", flush=True)
    raise SystemExit(0)

# --- Direção B ---
dir_b = []
if not SO_A:
    suspensas = [x for x in linhas if x["votacao"] in SUSP or x["resultado"] in SUSP_RES]
    print(f"direção A: {len(dir_a)} pares | suspensas p/ checar teor: {len(suspensas)}", flush=True)
    for i, x in enumerate(suspensas):
        try:
            children = fit.get_all_children(client, x["page_id"])
        except Exception:
            continue
        idx = fit.marker_index(children)
        if idx is None:
            continue
        corpo = fit.norm_ws(" ".join(fit._paragrafos_da_secao(children, idx)))
        m = re.search(r"(?:o Tribunal|Acordam)[^.]{0,80}?por\s+(maioria|unanimidade)[^.]{0,300}", corpo, re.I)
        disp = m.group(0) if m else ""
        if disp:
            dir_b.append({**x, "dispositivo_no_teor": disp[:400]})
        if (i + 1) % 50 == 0:
            print(f"  teor {i+1}/{len(suspensas)}", flush=True)

out = ART / f"cadeia_julgado_candidatos_{STAMP}.json"
out.write_text(json.dumps({"direcao_a": dir_a, "direcao_b": dir_b}, ensure_ascii=False, indent=1),
               encoding="utf-8")
print(f"A: {len(dir_a)} pares | B: {len(dir_b)} suspensas com dispositivo no teor", flush=True)
print("salvo:", out, flush=True)
