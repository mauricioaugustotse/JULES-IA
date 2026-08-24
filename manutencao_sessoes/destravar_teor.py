# -*- coding: utf-8 -*-
"""Tira do controle de 'já feito' as páginas de um jsonl de docs, para que
recebam o teor NOVO (caso típico: texto que estava truncado ou de fonte errada).

Sem isso, uma página marcada como pronta nunca receberia o texto completo — e o
sanear_formato tampouco a corrigiria, porque o formato do texto truncado pode
estar correto: o que falta é conteúdo.

Uso: destravar_teor.py <arquivo_docs.jsonl>
"""
import json
import sys
from pathlib import Path

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
ESTADO = Path(__file__).parent / ".estado"
alvo = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if alvo and not alvo.is_absolute():
    alvo = ART / alvo
if not alvo or not alvo.exists():
    sys.exit(f"jsonl não encontrado: {alvo}")

com_doc = set()
for ln in alvo.read_text(encoding="utf-8").splitlines():
    d = json.loads(ln)
    if d.get("doc"):
        com_doc.add(d["page_id"])
print(f"{alvo.name}: {len(com_doc)} páginas com doc")

FEITOS = ART / "aplicar_teor_feitos.json"
if FEITOS.exists():
    feitos = set(json.loads(FEITOS.read_text(encoding="utf-8")))
    antes = len(feitos)
    feitos -= com_doc
    FEITOS.write_text(json.dumps(sorted(feitos)), encoding="utf-8")
    print(f"  aplicar_teor: {antes} -> {len(feitos)}")

for f in ESTADO.glob("sanear_formato_feitos_*.json"):
    try:
        s = set(json.loads(f.read_text(encoding="utf-8")))
    except Exception:
        continue
    novo = s - com_doc
    if len(novo) != len(s):
        f.write_text(json.dumps(sorted(novo)), encoding="utf-8")
        print(f"  {f.name}: {len(s)} -> {len(novo)}")
