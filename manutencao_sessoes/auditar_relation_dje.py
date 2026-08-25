# -*- coding: utf-8 -*-
"""Audita (e repara) a relation `sessões` <-> `DJe`, partindo do lado das sessões.

POR QUE NAO USAR O `DJE_relations.py --modo cross`: aquele le a base DJe INTEIRA
(188 mil paginas, ~30 min de leitura) e regrava do lado DJe sem comparar com o
estado atual. Para MANUTENCAO isso e caro e cego. Aqui a varredura e ao contrario:
parte das poucas centenas de paginas de sessoes SEM relation e faz uma consulta por
processo — minutos, e diz POR QUE cada uma ficou sem par.

O DIAGNOSTICO E O PRODUTO. Uma sessao sem relation cai em um de quatro casos:
  · par_existe_*      -> o acordao esta no DJe e a ligacao e que falhou (reparavel aqui);
  · ato administrativo -> Instrucao/PA/"Aprovada" nao geram acordao no acervo de
                          jurisprudencia; a ausencia e correta;
  · sessao recente     -> o acordao ainda nao foi publicado/coletado;
  · linha SUSPENSA     -> o acordao pertence a linha conclusiva, nao a esta;
  · ausente com numero implausivel (ano do CNJ POSTERIOR a sessao) -> forte indicio de
    NUMERO ERRADO na pagina. Confirmado na campanha: as paginas sem par sao as mesmas
    que nao tem inteiro teor — falta de acordao e falta de teor tem a mesma causa.

Uso: auditar_relation_dje.py [--apply] [--limite N]
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\ProjetoConversor")
import NOTION_relatoriodeIA_v2 as report  # noqa: E402

import requests  # noqa: E402

DS_DJE = "32872195-5c64-8093-ab9b-000b8a94e7dd"
DS_SESSOES = "2eb72195-5c64-80ea-9cd5-000b0e01745d"
REL_SES = "Related to DJe (sess\u00e3o de julgamento)"
ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
# classes/resultados que produzem ato normativo ou administrativo, nao acordao
ADM_CLASSES = {"Instrução", "PA", "Lista Tríplice"}
ADM_RESULTADOS = {"Aprovada", "Homologada"}

APPLY = "--apply" in sys.argv
LIMITE = int(sys.argv[sys.argv.index("--limite") + 1]) if "--limite" in sys.argv else 0

_S = requests.Session()
_S.headers.update({"Authorization": f"Bearer {report.resolve_notion_key()}",
                   "Notion-Version": "2025-09-03", "Content-Type": "application/json"})


def req(method, path, body=None):
    for i in range(5):
        r = _S.request(method, f"https://api.notion.com{path}", json=body, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 * (i + 1))
            continue
        if not r.ok:
            raise RuntimeError(f"{r.status_code} {r.text[:200]}")
        return r.json()
    raise RuntimeError("esgotou retries")


def txt(prop):
    if not prop:
        return ""
    t = prop.get("type")
    if t in ("rich_text", "title"):
        return "".join(x.get("plain_text", "") for x in (prop.get(t) or []))
    if t == "select":
        return ((prop.get("select") or {}) or {}).get("name") or ""
    if t == "date":
        return ((prop.get("date") or {}) or {}).get("start") or ""
    if t == "relation":
        return [x.get("id") for x in (prop.get("relation") or [])]
    return ""


def d20(v):
    return re.sub(r"\D", "", str(v or ""))[:20]


def ano_do_cnj(numero):
    m = re.match(r"\D*\d{7}-?\d{2}\.?(\d{4})", (numero or "").replace(" ", ""))
    return int(m.group(1)) if m else None


def query_all(ds, filtro=None):
    out, cur = [], ""
    while True:
        body = {"page_size": 100}
        if filtro:
            body["filter"] = filtro
        if cur:
            body["start_cursor"] = cur
        q = req("POST", f"/v1/data_sources/{ds}/query", body)
        out += q["results"]
        if not q.get("has_more"):
            break
        cur = q.get("next_cursor") or ""
        if not cur:
            break
    return out


def diagnostico(s, hits):
    """Classifica a AUSENCIA de par. Ordem importa: o motivo mais especifico ganha."""
    if hits:
        tipos = [h["tipo"] for h in hits]
        return "par_existe_acordao" if any((t or "").lower().startswith("ac") for t in tipos) \
            else "par_existe_so_monocratica"
    if s["classe"] in ADM_CLASSES or s["resultado"] in ADM_RESULTADOS:
        return "ato_administrativo"
    if (s["votacao"] or "").startswith("Suspenso"):
        return "linha_suspensa"
    ano = ano_do_cnj(s["numero"])
    if ano and ano > int(s["data"][:4] or 0):
        return "numero_implausivel"      # ano do CNJ posterior a sessao
    if s["data"] >= time.strftime("%Y-%m-%d", time.gmtime(time.time() - 180 * 86400)):
        return "sessao_recente"
    return "sem_acordao_no_acervo"


def main():
    print("lendo a base sessoes...", flush=True)
    ses = query_all(DS_SESSOES)
    sem = []
    for r in ses:
        p = r["properties"]
        if txt(p.get(REL_SES)):
            continue
        num = txt(p.get("numero_processo"))
        sem.append({"page_id": r["id"], "url": r.get("url"), "numero": num, "cnj": d20(num),
                    "data": txt(p.get("data_sessao"))[:10],
                    "classe": txt(p.get("classe_processo")),
                    "resultado": txt(p.get("resultado")), "votacao": txt(p.get("votacao")),
                    "tema": txt(p.get("tema"))[:70]})
    print(f"  {len(ses)} paginas | {len(sem)} sem relation para o DJe", flush=True)
    if LIMITE:
        sem = sem[:LIMITE]

    stats, res = Counter(), []
    for i, s in enumerate(sem, 1):
        hits = []
        if len(s["cnj"]) == 20:
            # consulta pelo NUCLEO (7 digitos + DV) e confere o CNJ-20 inteiro:
            # `contains` do numero formatado falha quando as bases divergem na grafia
            nucleo = s["numero"].split(".")[0].strip()
            try:
                rr = query_all(DS_DJE, {"property": "numeroUnico",
                                        "rich_text": {"contains": nucleo}})
                hits = [{"id": h["id"],
                         "data": txt(h["properties"].get("dataDecisao"))[:10],
                         "tipo": txt(h["properties"].get("descricaoTipoDecisao")),
                         "classe": txt(h["properties"].get("siglaClasse"))}
                        for h in rr
                        if d20(txt(h["properties"].get("numeroUnico"))) == s["cnj"]]
            except Exception as exc:
                s["erro"] = str(exc)[:150]
        else:
            s["diag"] = "cnj_invalido"
            stats["cnj_invalido"] += 1
            res.append(s)
            continue
        s["dje"] = hits
        s["diag"] = diagnostico(s, hits)
        stats[s["diag"]] += 1
        if APPLY and hits:
            try:
                req("PATCH", f"/v1/pages/{s['page_id']}",
                    {"properties": {REL_SES: {"relation": [{"id": h["id"]} for h in hits[:25]]}}})
                stats["gravadas"] += 1
            except Exception as exc:
                s["erro_gravacao"] = str(exc)[:150]
                stats["erro_gravacao"] += 1
        res.append(s)
        if i % 100 == 0:
            print(f"  {i}/{len(sem)} {dict(stats)}", flush=True)

    print("\nRESUMO:", dict(stats))
    arq = ART / f"relation_dje_{'apply' if APPLY else 'dry'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    ART.mkdir(parents=True, exist_ok=True)
    arq.write_text(json.dumps({"stats": dict(stats), "itens": res}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("salvo:", arq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
