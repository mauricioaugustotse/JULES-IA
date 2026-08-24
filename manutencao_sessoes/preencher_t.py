# -*- coding: utf-8 -*-
"""Preenche t= faltante no youtube_link das linhas dos dias sem fonte de ordem,
usando a janela (start_seconds) do 02_judgment casado nos artifacts.
Uso: preencher_t.py [--apply]
"""
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
BATCH = r"C:\Users\mauri\JULES-IA\artifacts\tse_youtube_notion\batch_gui"
BACKFILL = r"H:\Meu Drive\TSE_YOUTUBE_NOTION_BACKLOG\backfill_2025"
STAMP = time.strftime("%Y%m%d_%H%M%S")
APPLY = "--apply" in sys.argv

FULL = "--full" in sys.argv
if FULL:
    dias_alvo = None  # base inteira
    print("modo FULL: todas as linhas sem t=", flush=True)
else:
    ren = json.load(open(sorted(glob.glob(str(ART / "renumeracao_t_dry_*.json")))[-1], encoding="utf-8"))
    dias_alvo = {x["dia"] for x in ren["sem_fonte"]}
    print(f"{len(dias_alvo)} dias sem fonte de ordem", flush=True)


def norm_pid(p):
    return re.sub(r"[^0-9a-f]", "", (p or "").lower())


# índice page_id -> (artifact_dir, video_id, url)
loc = {}


def registra(pr_list, artifact_dir, video_id, url=None):
    for r in pr_list or []:
        pid = norm_pid(r.get("page_id"))
        if pid and pid not in loc:
            loc[pid] = (artifact_dir, video_id, url, r.get("numero_processo"), r.get("tema"))


for lote in sorted(os.listdir(BATCH)):
    lp = os.path.join(BATCH, lote)
    if not os.path.isdir(lp):
        continue
    for sub in sorted(os.listdir(lp)):
        f = os.path.join(lp, sub, "05_publish_results.json")
        if os.path.isfile(f):
            try:
                pr = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            m = re.match(r"\d+_(.+)$", sub)
            registra(pr, os.path.join(lp, sub), m.group(1) if m else sub)
for plist in sorted(os.listdir(BACKFILL)):
    pp = os.path.join(BACKFILL, plist)
    if not os.path.isdir(pp):
        continue
    for sub in sorted(os.listdir(pp)):
        f = os.path.join(pp, sub, "07_backfill_summary.json")
        if os.path.isfile(f):
            try:
                s7 = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            registra(s7.get("publish_results"), os.path.join(pp, sub), s7.get("video_id"), s7.get("url"))
print(f"índice: {len(loc)} page_ids", flush=True)


def norm_txt(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def acha_start(video_dir, numero, tema):
    alvo_t = norm_txt(tema)
    alvo_n = re.sub(r"\D", "", numero or "")
    best = None
    try:
        arquivos = sorted(os.listdir(video_dir))
    except OSError:
        return None
    for fn in arquivos:
        if not (fn.startswith("02_judgment") and fn.endswith(".json")):
            continue
        try:
            jd = json.load(open(os.path.join(video_dir, fn), encoding="utf-8"))
        except Exception:
            continue
        for it in jd.get("items") or []:
            score = 0
            if alvo_t and norm_txt(it.get("tema")) == alvo_t:
                score += 2
            n_it = re.sub(r"\D", "", it.get("numero_processo") or "")
            if alvo_n and n_it and (alvo_n.startswith(n_it) or n_it.startswith(alvo_n)):
                score += 1
            if score and (best is None or score > best[0]):
                best = (score, jd.get("start_seconds"))
    return best[1] if best else None


client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
pages = client.query_data_source()


def t(p, campo):
    return client._extract_property_text(p, schema, campo)


log, stats = [], {"preenchidos": 0, "sem_artifact": 0, "sem_janela": 0, "ja_tem": 0}
for p in pages:
    dia = t(p, "data_sessao")[:10]
    if dias_alvo is not None and dia not in dias_alvo:
        continue
    if not re.match(r"Julgamento \d+$", t(p, "tipo_registro") or ""):
        continue
    link = t(p, "youtube_link")
    if re.search(r"[?&]t=\d+", link or ""):
        stats["ja_tem"] += 1
        continue
    ent = loc.get(norm_pid(p["id"]))
    if not ent:
        stats["sem_artifact"] += 1
        continue
    vdir, vid, url, numero, tema = ent
    start = acha_start(vdir, numero, tema)
    if start is None:
        stats["sem_janela"] += 1
        continue
    novo = f"https://www.youtube.com/watch?v={vid}&t={int(start)}"
    if APPLY:
        try:
            notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={
                "properties": {"youtube_link": {"url": novo}}})
        except Exception as exc:
            log.append({"page_id": p["id"], "erro": str(exc)[:150]})
            continue
    log.append({"page_id": p["id"], "dia": dia, "de": link, "para": novo})
    stats["preenchidos"] += 1

out = ART / f"preencher_t_{'apply' if APPLY else 'dry'}_{STAMP}.json"
out.write_text(json.dumps({"stats": stats, "log": log}, ensure_ascii=False, indent=1), encoding="utf-8")
print("stats:", stats, flush=True)
print("salvo:", out, flush=True)
