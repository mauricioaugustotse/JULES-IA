# -*- coding: utf-8 -*-
"""Preenche o inteiro teor das páginas de sessões a partir do ACÓRDÃO já pareado no DJe.

Depois que `auditar_relation_dje.py` liga a sessão às decisões do mesmo processo, o
acórdão vira uma fonte local: `textoEmenta` + `textoDecisao` da página do DJe são o
mesmo material que o `..\\fill_inteiro_teor.py` extrai do CSV — sem SJUR, sem CSV de
1,2 GB, sem captcha. Gravamos com o cortador vigente, então o formato já sai novo.

QUAL ACÓRDÃO GRAVAR — a data é o juiz:
  1. `dataDecisao` == data da sessão -> é o acórdão DAQUELE julgamento. Critério
     inequívoco (mesmo processo, mesmo dia) e o único aceito por padrão;
  2. janela [-5,+60]d -> SÓ com `--janela`, e ainda assim exige que o RESULTADO do
     dispositivo bata com a etiqueta da página e que não seja acórdão de embargos;
  3. qualquer outra distância -> nunca.

LIÇÃO QUE CUSTOU A PRIMEIRA VERSÃO (25/08): o gate original aceitava até 180d quando o
dispositivo concordava com a etiqueta `votacao` (unanimidade x maioria). Isso NÃO separa
nada — a maioria dos julgamentos é unânime, então a concordância acontece por acaso. Dos
30 casos aprovados assim, praticamente todos eram acórdãos de EMBARGOS DE DECLARAÇÃO
julgados meses depois ("Desprovido" na página x "rejeitou os embargos" no acórdão). Só o
RESULTADO separa fases do processo; a votação, nunca. Na janela [-5,+60]d o mesmo teste
reprovou 25 de 33.

LINHA DE VISTA NÃO RECEBE TEOR (regra da campanha, ver README): numa página
"Suspenso"/"Suspenso*" o acórdão pertence à irmã conclusiva. Elas são puladas mesmo
quando há acórdão pareado — foi exatamente assim que 29 páginas ganharam documento de
outra sessão na rodada de 23/08.

Uso: preencher_teor_do_dje.py [--apply] [--limite N]
"""
import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
sys.path.insert(0, r"C:\Users\mauri\ProjetoConversor")
import tse_youtube_notion_core as core  # noqa: E402
import fill_inteiro_teor as fit  # noqa: E402
from import_dje_faltantes import resultado_votacao_from_dispositivo  # noqa: E402
import NOTION_relatoriodeIA_v2 as report  # noqa: E402
import requests  # noqa: E402

DS_SESSOES = "2eb72195-5c64-80ea-9cd5-000b0e01745d"
REL_SES = "Related to DJe (sess\u00e3o de julgamento)"
ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
APPLY = "--apply" in sys.argv
LIMITE = int(sys.argv[sys.argv.index("--limite") + 1]) if "--limite" in sys.argv else 0
JANELA = "--janela" in sys.argv   # aceita [-5,+60]d com confirmacao pelo RESULTADO

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


def dt(s):
    try:
        return date(*map(int, (s or "")[:10].split("-")))
    except Exception:
        return None


def eh_embargos(dispositivo):
    return "embargos" in (dispositivo or "").lower()[:150]


def escolher(acordaos, data_sessao, resultado_pagina, classe_pagina=""):
    """Devolve (acordao, motivo) ou (None, motivo_da_recusa)."""
    ds = dt(data_sessao)
    if not ds or not acordaos:
        return None, "sem_data_ou_sem_acordao"
    cand = [(a, (dt(a["data"]) - ds).days) for a in acordaos if dt(a["data"])]
    if not cand:
        return None, "acordao_sem_data"
    exata = [a for a, d in cand if d == 0]
    if exata:
        return exata[0], "data_exata"
    if not JANELA:
        return None, "sem_acordao_no_dia_{:+d}d".format(min((d for _, d in cand), key=abs))
    # com --janela: o RESULTADO tem de bater; a votacao nao separa fase nenhuma
    for a, d in sorted([(a, d) for a, d in cand if -5 <= d <= 60], key=lambda x: abs(x[1])):
        res, _ = resultado_votacao_from_dispositivo(a.get("decisao") or "")
        pagina_e_ed = classe_pagina.upper().startswith("ED")
        if res and resultado_pagina and res == resultado_pagina and \
                (pagina_e_ed or not eh_embargos(a.get("decisao"))):
            return a, "janela_{:+d}d_resultado_confere".format(d)
    return None, "janela_sem_confirmacao"


def main():
    print("lendo a base sessoes...", flush=True)
    pages, cur = [], ""
    while True:
        b = {"page_size": 100}
        if cur:
            b["start_cursor"] = cur
        q = req("POST", "/v1/data_sources/{}/query".format(DS_SESSOES), b)
        pages += q["results"]
        if not q.get("has_more"):
            break
        cur = q.get("next_cursor") or ""
        if not cur:
            break

    alvos = []
    for r in pages:
        p = r["properties"]
        rel = txt(p.get(REL_SES)) or []
        if not rel:
            continue
        alvos.append({"page_id": r["id"], "url": r.get("url"),
                      "numero": txt(p.get("numero_processo")),
                      "data": txt(p.get("data_sessao"))[:10],
                      "votacao": txt(p.get("votacao")), "resultado": txt(p.get("resultado")),
                      "classe": txt(p.get("classe_processo")),
                      "tema": txt(p.get("tema"))[:60], "rel": rel})
    print("  {} paginas | {} com relation para o DJe".format(len(pages), len(alvos)), flush=True)

    client = core.NotionSessoesClient(core.get_notion_api_key())
    stats, log = Counter(), []
    vistos = 0
    for a in alvos:
        # linha de vista nao carrega acordao (regra da campanha)
        if (a["votacao"] or "").startswith("Suspenso"):
            stats["pulada_linha_suspensa"] += 1
            continue
        try:
            ch = fit.get_all_children(client, a["page_id"])
        except Exception:
            stats["erro_leitura"] += 1
            continue
        if fit.marker_index(ch) is not None:
            stats["ja_tem_teor"] += 1
            continue
        vistos += 1
        if LIMITE and vistos > LIMITE:
            break
        acordaos = []
        for pid in a["rel"]:
            try:
                pg = req("GET", "/v1/pages/{}".format(pid))
            except Exception:
                continue
            pp = pg["properties"]
            tipo = txt(pp.get("descricaoTipoDecisao"))
            if not (tipo or "").lower().startswith("ac"):
                continue
            acordaos.append({"id": pid, "data": txt(pp.get("dataDecisao"))[:10],
                             "ementa": txt(pp.get("textoEmenta")),
                             "decisao": txt(pp.get("textoDecisao"))})
        if not acordaos:
            stats["sem_acordao_pareado"] += 1
            continue
        esc, motivo = escolher(acordaos, a["data"], a["resultado"], a.get("classe") or "")
        base = {k: v for k, v in a.items() if k != "rel"}
        if not esc:
            stats["recusado_" + motivo.split("_")[0]] += 1
            log.append(dict(base, status="recusado", motivo=motivo,
                            datas_acordaos=[x["data"] for x in acordaos]))
            continue
        ementa, decisao = fit.montar(esc["ementa"], esc["decisao"])
        if len((ementa + decisao).strip()) < 80:
            stats["texto_curto_demais"] += 1
            log.append(dict(base, status="texto_curto", n=len((ementa + decisao).strip())))
            continue
        blocos = fit.build_blocks(ementa, decisao)
        stats["gravar_" + motivo.split("_")[0]] += 1
        item = dict(base, motivo=motivo, acordao=esc["id"], data_acordao=esc["data"],
                    chars=len(ementa) + len(decisao), blocos=len(blocos))
        if APPLY:
            try:
                fit.append_blocks(client, a["page_id"], blocos)
                item["status"] = "gravado"
                stats["gravadas"] += 1
            except Exception as exc:
                item["status"] = "erro"
                item["erro"] = str(exc)[:150]
                stats["erro_gravacao"] += 1
        else:
            item["status"] = "gravaria"
        log.append(item)
        if len(log) % 20 == 0:
            print("  {} avaliadas {}".format(len(log), dict(stats)), flush=True)

    print("\nRESUMO: {}".format(dict(stats)))
    ART.mkdir(parents=True, exist_ok=True)
    arq = ART / "teor_do_dje_{}_{}.json".format("apply" if APPLY else "dry",
                                                time.strftime("%Y%m%d_%H%M%S"))
    arq.write_text(json.dumps({"stats": dict(stats), "itens": log}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("salvo: {}".format(arq))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
