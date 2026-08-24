# -*- coding: utf-8 -*-
"""Corrige a composição pela lista oficial do ACÓRDÃO gravado no corpo.

Extrai o bloco "Presidência do(a) Ministro(a) X. Presentes o(a)s Ministro(a)s ..."
do teor; mapeia para as options existentes; substitui a property composicao quando
a atual diverge (sobra gente fora da lista oficial ou >7).
Uso: composicao_por_acordao.py [--apply] [--max N]
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402
from tse_normalization import normalize_ministro_name  # noqa: E402
import fill_inteiro_teor as fit  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
STAMP = time.strftime("%Y%m%d_%H%M%S")
APPLY = "--apply" in sys.argv
MAXN = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 0

client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
pages = client.query_data_source()

comp_prop = (schema.raw_payload.get("properties") or {}).get("composicao", {})
# Canoniza o vocabulário lido do schema: este script mapeia o nome do acórdão para uma
# OPÇÃO EXISTENTE, então uma opção divergente que sobreviva no select (ou seja recriada
# por engano) se espalharia por todas as páginas que ele tocasse. Ver `_ministros_canonico`.
OPCOES = list(dict.fromkeys(
    normalize_ministro_name(o.get("name", "")) or o.get("name", "")
    for o in comp_prop.get("multi_select", {}).get("options", [])))
print(f"{len(pages)} páginas; {len(OPCOES)} options de composicao; APPLY={APPLY}", flush=True)

COMP_RE = re.compile(
    r"Composi[çc][ãa]o do julgamento:\s*Ministr[oa]s?\s*(?:\(as?\))?\s*(.{10,400}?)(?:\.\s|\.$|;|\bProcurador)",
    re.I | re.S)
PRES_RE = re.compile(
    r"Presid[êe]ncia d[oa] Ministr[oa]\s+([A-ZÀ-Ú][^.]{2,60}?)\.\s*"
    r"Presentes?\s+(.{10,600}?)(?:,?\s*e\s+[oa]\s+(?:Vice-)?Procurador|\.\s+SESS|\.\s*$|;\s)",
    re.I | re.S)
SPLIT_NOMES = re.compile(
    r",\s*|\s+e\s+|\b[oa]s?\s+Ministr[oa]s?\s+|\b[oa]\s+Ministr[oa]\s+", re.I)
PAREN_RE = re.compile(r"\s*\((?:Vice-)?Presidente[^)]*\)|\s*\(as?\)", re.I)


def norm(s):
    import unicodedata
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()


def mapeia(nome):
    nome = re.sub(r"\s+", " ", nome).strip(" ,.;")
    if len(nome) < 4:
        return None
    toks = [x for x in re.findall(r"\w+", norm(nome)) if len(x) > 2 and x not in
            ("dos", "das", "min", "ministro", "ministra")]
    if not toks:
        return None
    cand = [o for o in OPCOES if all(tk in norm(o) for tk in toks[-2:])]
    if len(cand) == 1:
        return cand[0]
    cand = [o for o in OPCOES if toks[-1] in norm(o)]
    return cand[0] if len(cand) == 1 else None


def t(p, campo):
    return client._extract_property_text(p, schema, campo)


def multi(p, campo):
    return [o.get("name", "") for o in p.get("properties", {}).get(campo, {}).get("multi_select", [])]


alvos = []
for p in pages:
    comp = multi(p, "composicao")
    if comp:
        alvos.append((p, comp))
print(f"{len(alvos)} páginas com composicao", flush=True)
if MAXN:
    alvos = alvos[:MAXN]

log, stats = [], {"corrigidas": 0, "ja_ok": 0, "sem_teor": 0, "sem_padrao": 0,
                  "mapeamento_incompleto": 0, "oficial_suspeita": 0}
for i, (p, comp) in enumerate(alvos):
    pid = p["id"]
    try:
        children = fit.get_all_children(client, pid)
        idx = fit.marker_index(children)
        if idx is None:
            stats["sem_teor"] += 1
            continue
        corpo = fit.norm_ws(" ".join(fit._paragrafos_da_secao(children, idx)))
        mc = COMP_RE.search(corpo)
        if mc:
            lista_txt = PAREN_RE.sub("", mc.group(1))
            presidente, nomes = "", [x for x in SPLIT_NOMES.split(lista_txt) if x.strip()]
        else:
            m = PRES_RE.search(corpo)
            if not m:
                stats["sem_padrao"] += 1
                continue
            presidente, presentes_txt = m.group(1), m.group(2)
            nomes = [x for x in SPLIT_NOMES.split(presentes_txt) if x.strip()]
        oficial = []
        falhou = False
        for n in [presidente] + nomes:
            op = mapeia(n)
            if op is None:
                if re.search(r"Ministr|[A-ZÀ-Ú]\w+ [A-ZÀ-Ú]", n):
                    falhou = True
                continue
            if op not in oficial:
                oficial.append(op)
        if falhou or not (3 <= len(oficial) <= 7):
            stats["mapeamento_incompleto" if falhou else "oficial_suspeita"] += 1
            continue
        atual = list(dict.fromkeys(comp))
        fora = [x for x in atual if x not in oficial]
        if not fora and len(atual) <= 7 and set(atual) == set(oficial):
            stats["ja_ok"] += 1
            continue
        if not fora and len(atual) <= 7:
            # subconjunto da oficial (faltam presentes) — completar pela oficial
            pass
        if APPLY:
            notion_request_with_retry(client, "PATCH", f"/pages/{pid}", json={
                "properties": {"composicao": {"multi_select": [{"name": x} for x in oficial]}}})
        log.append({"page_id": pid, "numero": t(p, "numero_processo"), "data": t(p, "data_sessao")[:10],
                    "de": atual, "para": oficial, "removidos": fora})
        stats["corrigidas"] += 1
    except Exception as exc:
        log.append({"page_id": pid, "erro": str(exc)[:150]})
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(alvos)} {stats}", flush=True)

out = ART / f"composicao_acordao_{'apply' if APPLY else 'dry'}_{STAMP}.json"
out.write_text(json.dumps({"stats": stats, "log": log}, ensure_ascii=False, indent=1), encoding="utf-8")
print("stats:", stats, flush=True)
print("salvo:", out, flush=True)
