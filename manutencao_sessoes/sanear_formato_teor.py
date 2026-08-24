# -*- coding: utf-8 -*-
"""Detecta E regrava, numa única passada, os teores cuja segmentação difere da
que o cortador ATUAL (fill_inteiro_teor._paragrafos_finais) produziria.

Use sempre que o cortador evoluir (novos padrões de ementa/itens). Fonte do texto:
doc SJUR salvo quando houver; senão os próprios parágrafos gravados (o conteúdo é
o mesmo — muda só a quebra). Preserva o marker (SJUR/DJE) da página.

Retomável: manutencao_sessoes\\.estado\\sanear_formato_feitos.json (por page_id +
assinatura do cortador). Uso: sanear_formato_teor.py [--apply] [--max N]
"""
import hashlib
import html
import inspect
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402
import fill_inteiro_teor as fit  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
ESTADO = Path(__file__).parent / ".estado"
ESTADO.mkdir(exist_ok=True)
APPLY = "--apply" in sys.argv
MAXN = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 0

# assinatura do cortador: muda quando segmenta_semantico/to_paragraphs mudam,
# invalidando o "já feito" das rodadas anteriores.
_src = inspect.getsource(fit.segmenta_semantico) + inspect.getsource(fit.to_paragraphs)
VERSAO = hashlib.sha1(_src.encode("utf-8")).hexdigest()[:10]
FEITOS = ESTADO / f"sanear_formato_feitos_{VERSAO}.json"
feitos = set(json.loads(FEITOS.read_text(encoding="utf-8"))) if FEITOS.exists() else set()
print(f"cortador v{VERSAO}; {len(feitos)} páginas já saneadas nesta versão; APPLY={APPLY}", flush=True)

docs = {}
for arq in list(ART.glob("fila_teor_sjur_docs_*.jsonl")) + [ART / "nucleo_duro_sjur_enriquecido_20260821.jsonl"]:
    if not arq.exists():
        continue
    for ln in arq.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if d.get("doc"):
            docs.setdefault(d["page_id"].replace("-", ""), d["doc"])

client = core.NotionSessoesClient(core.get_notion_api_key())
pages = [p for p in client.query_data_source() if p["id"] not in feitos]
if MAXN:
    pages = pages[:MAXN]
print(f"{len(pages)} páginas a verificar", flush=True)

stats = {"regravadas": 0, "ja_ok": 0, "sem_teor": 0, "erro": 0}
mudancas = []
for i, p in enumerate(pages):
    pid = p["id"]
    try:
        children = fit.get_all_children(client, pid)
        idx = fit.marker_index(children)
        if idx is None:
            stats["sem_teor"] += 1
            feitos.add(pid)
            continue
        paras = [x.strip() for x in fit._paragrafos_da_secao(children, idx) if x.strip()]
        corpo = fit.norm_ws(" ".join(paras))
        if paras == fit._paragrafos_finais(corpo):
            stats["ja_ok"] += 1
            feitos.add(pid)
            continue
        marker = fit._heading_text(children[idx]) or fit.MARKER
        doc = docs.get(pid.replace("-", ""))
        if doc:
            ementa = fit.norm_ws(html.unescape(re.sub(r"<[^>]+>", " ", doc.get("textoEmenta") or "")))
            decisao = fit.norm_ws(html.unescape(re.sub(r"<[^>]+>", " ", doc.get("textoDecisao") or "")))
        else:
            ementa, decisao = "", corpo
        if ementa and ementa in decisao:
            ementa = ""
        blocks = fit.build_blocks(ementa, decisao, marker=marker)
        if APPLY:
            for b in children[idx:]:
                notion_request_with_retry(client, "DELETE", f"/blocks/{b['id']}")
            fit.append_blocks(client, pid, blocks)
            feitos.add(pid)
        stats["regravadas"] += 1
        mudancas.append({"page_id": pid, "de_paragrafos": len(paras), "para_blocos": len(blocks)})
    except Exception as exc:
        stats["erro"] += 1
        print(f"  ERRO {pid}: {str(exc)[:120]}", flush=True)
    if (i + 1) % 100 == 0:
        if APPLY:
            FEITOS.write_text(json.dumps(sorted(feitos)), encoding="utf-8")
        print(f"  {i+1}/{len(pages)} {stats}", flush=True)

if APPLY:
    FEITOS.write_text(json.dumps(sorted(feitos)), encoding="utf-8")
out = ART / f"sanear_formato_{'apply' if APPLY else 'dry'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
out.write_text(json.dumps({"cortador": VERSAO, "stats": stats, "mudancas": mudancas},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print(f"FIM {stats}", flush=True)
print("salvo:", out, flush=True)
