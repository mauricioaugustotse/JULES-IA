# -*- coding: utf-8 -*-
"""Audita a coluna `tema` da base sessões com o MESMO gate que o fluxo usa ao publicar.

Por que reusar `core._tema_looks_generic` em vez de reescrever os padrões aqui: se o
auditor tivesse regra própria, retroativo e futuro divergiriam — a base ficaria limpa
segundo um critério e suja segundo o outro. Toda regra nova entra no core (onde há teste
que a trava) e este script a herda de graça.

O QUE O GATE PEGA (validado contra a base em 25/08/2026):
  · nome de autoridade no lugar da matéria — "Ministro Sérgio Banhos". Nasce quando o
    vídeo traz só a proclamação ("pedido de vista do Ministro X") e o modelo toma quem
    falou pelo assunto julgado. Exige CARGO + NOME PRÓPRIO, então "nulidade por
    impedimento de Ministro" e "competência monocrática do relator" seguem válidos:
    ali a autoridade integra a TESE;
  · rótulo processual puro — "Embargos de declaração", "Agravo regimental" dizem o RITO,
    não a matéria (eram o tema de 6 páginas);
  · frase narrativa, número de processo, resultado e classe como tema; tema vazio.

Uso: auditar_tema.py            # relatório
     auditar_tema.py --csv      # saída colável para revisão
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
sys.path.insert(0, r"C:\Users\mauri\ProjetoConversor")
import tse_youtube_notion_core as core  # noqa: E402
import NOTION_relatoriodeIA_v2 as report  # noqa: E402
import requests  # noqa: E402

DS_SESSOES = "2eb72195-5c64-80ea-9cd5-000b0e01745d"
ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
CSV = "--csv" in sys.argv

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
    return ""


def main():
    pages, cur = [], ""
    while True:
        b = {"page_size": 100}
        if cur:
            b["start_cursor"] = cur
        q = req("POST", f"/v1/data_sources/{DS_SESSOES}/query", b)
        pages += q["results"]
        if not q.get("has_more"):
            break
        cur = q.get("next_cursor") or ""
        if not cur:
            break

    ruins, ct = [], Counter()
    for r in pages:
        p = r["properties"]
        tema = (txt(p.get("tema")) or "").strip()
        # PublishPreviewRow e' o contrato do gate: sem os campos da linha ele nao consegue
        # comparar o tema com classe/resultado/votacao (um tema que so repete a etiqueta
        # tambem e' generico).
        row = core.PublishPreviewRow(
            numero_processo=txt(p.get("numero_processo")),
            classe_processo=txt(p.get("classe_processo")),
            resultado=txt(p.get("resultado")), votacao=txt(p.get("votacao")),
            tema=tema, punchline=txt(p.get("punchline")))
        if not core.tema_looks_generic(tema, row):
            continue
        motivo = ("vazio" if not tema else
                  "nomeia_autoridade" if core._tema_nomeia_autoridade(tema) else
                  "rotulo_processual"
                  if core.normalize_class_text(tema) in core._TEMA_ROTULOS_PROCESSUAIS
                  else "generico")
        ct[motivo] += 1
        ruins.append({"page_id": r["id"], "url": r.get("url"), "tema": tema, "motivo": motivo,
                      "numero": row.numero_processo, "data": txt(p.get("data_sessao"))[:10],
                      "classe": row.classe_processo, "punchline": row.punchline[:120]})

    print(f"{len(pages)} paginas | {len(ruins)} com tema recusado pelo gate do fluxo")
    for k, v in ct.most_common():
        print(f"  {v:5d}  {k}")
    for x in ruins if CSV else ruins[:30]:
        print(f"  {x['data']} {x['numero'][:24]:24s} {x['tema'][:46]:46s} <- {x['motivo']}")
    ART.mkdir(parents=True, exist_ok=True)
    arq = ART / f"auditoria_tema_{time.strftime('%Y%m%d_%H%M%S')}.json"
    arq.write_text(json.dumps({"stats": dict(ct), "itens": ruins}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("salvo:", arq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
