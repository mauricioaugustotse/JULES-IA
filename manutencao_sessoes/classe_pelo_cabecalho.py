# -*- coding: utf-8 -*-
"""Confere a etiqueta `classe_processo` contra a CLASSE OFICIAL do cabeçalho do teor.

O acórdão/decisão gravado no corpo começa com a classe por extenso:
  "INSTRUÇÃO (11544) N. 0600749-95.2019.6.00.0000 (PJe) - BRASÍLIA..."
  "RECURSO ESPECIAL ELEITORAL (11549) Nº ..."
Isso é fonte OFICIAL e objetiva — melhor que inferir do tema, que engana: as normas
sobre prestação de contas viravam "PC" quando são "Instrução" (caso 0600749-95).

Só reporta/corrige quando a FAMÍLIA diverge (PC × Instrução), nunca por especificidade
(REspe × AgRg-REspe, que descrevem o mesmo processo em fases diferentes) e nunca quando
o número do cabeçalho difere do número da página (teor de outro processo).

Uso: classe_pelo_cabecalho.py [--apply]
"""
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

# classe por extenso no cabeçalho -> option da base. Ordem importa (mais específico antes).
MAPA = [
    (r"AGRAVO REGIMENTAL NO RECURSO ESPECIAL ELEITORAL", "AgRg-REspe"),
    (r"AGRAVO REGIMENTAL NO AGRAVO EM RECURSO ESPECIAL ELEITORAL", "AgRg-AREspe"),
    (r"AGRAVO REGIMENTAL NO RECURSO ORDIN[ÁA]RIO", "AgRg-RO"),
    (r"AGRAVO EM RECURSO ESPECIAL ELEITORAL", "AREspe"),
    (r"RECURSO ESPECIAL ELEITORAL", "REspe"),
    (r"RECURSO ORDIN[ÁA]RIO", "RO"),
    (r"RECURSO CONTRA EXPEDI[ÇC][ÃA]O DE DIPLOMA", "RCED"),
    (r"A[ÇC][ÃA]O DE INVESTIGA[ÇC][ÃA]O JUDICIAL ELEITORAL", "AIJE"),
    (r"PRESTA[ÇC][ÃA]O DE CONTAS", "PC"),
    (r"PROCESSO ADMINISTRATIVO", "PA"),
    (r"INSTRU[ÇC][ÃA]O", "Instrução"),
    (r"CONSULTA", "CTA"),
    (r"MANDADO DE SEGURAN[ÇC]A", "MS"),
    (r"HABEAS CORPUS", "HC"),
    (r"LISTA TR[ÍI]PLICE", "Lista Tríplice"),
    (r"REGISTRO DE CANDIDATURA", "RCand"),
    (r"REPRESENTA[ÇC][ÃA]O", "Representação"),
    (r"A[ÇC][ÃA]O CAUTELAR", "Ação Cautelar"),
    (r"AGRAVO DE INSTRUMENTO", "AI"),
    (r"PETI[ÇC][ÃA]O", "Petição"),
    (r"RECLAMA[ÇC][ÃA]O", "Reclamação"),
    (r"RECURSO EM MANDADO DE SEGURAN[ÇC]A", "RMS"),
    (r"REVIS[ÃA]O DE ELEITORADO", "RvE"),
]
# famílias: divergência DENTRO da família é especificidade, não erro
FAMILIA = {
    "REspe": "recurso_especial", "AREspe": "recurso_especial", "AgRg-REspe": "recurso_especial",
    "AgRg-AREspe": "recurso_especial", "ED-REspe": "recurso_especial",
    "ED-AREspe": "recurso_especial", "ED-AgRg-AREspe": "recurso_especial",
    "RO": "ordinario", "AgRg-RO": "ordinario", "ED-RO": "ordinario",
    "PC": "contas", "AgRg-PC": "contas", "ED-PC": "contas",
    "Rp": "representacao", "Representação": "representacao",
    "PET": "peticao", "Petição": "peticao", "PetCiv": "peticao",
    "Rcl": "reclamacao", "Reclamação": "reclamacao",
    "Lista Tríplice": "lista", "ED-Lista Tríplice": "lista",
    "MS": "mandado", "Ref.-MS": "mandado", "AgRg-MS": "mandado",
}


def familia(c):
    return FAMILIA.get(c, c)


def classe_do_cabecalho(corpo):
    """Extrai a classe SÓ do cabeçalho FORMAL, nunca da ementa.

    Confiável:  "INSTRUÇÃO (11544) N. 0600749-95.2019.6.00.0000 (PJe) - BRASÍLIA"
                "RECURSO ESPECIAL ELEITORAL (11549) Nº 0600941-38.2020..."
    NÃO confiável (ementa): "ELEIÇÕES 2016. REGISTRO DE CANDIDATURA. RECURSO ESPECIAL..."
      — ali a primeira expressão é o ASSUNTO, não a classe; e uma menção a
      "prestação de contas" no meio do texto não torna o processo um PC.
    Exigências: a classe abre o texto (até 80 chars) E é seguida do número do
    processo, opcionalmente com o código entre parênteses.
    """
    cab = corpo[:300].upper().lstrip(" –-—.")
    for padrao, option in MAPA:
        m = re.match(rf"\s*(?:TRIBUNAL SUPERIOR ELEITORAL\s+)?{padrao}\b", cab)
        if not m:
            continue
        resto = cab[m.end():m.end() + 60]
        # precisa vir o número do processo (com ou sem "(11544)" e "N."/"Nº")
        if re.match(r"\s*(?:\(\d{3,6}\))?\s*(?:N[º°.]?|NO\.?)?\s*[\d][\d.\-–]{8,}", resto):
            return option
    return None


client = core.NotionSessoesClient(core.get_notion_api_key())
schema = client.fetch_schema()
pages = client.query_data_source()
print(f"{len(pages)} páginas; APPLY={APPLY}", flush=True)

divergentes, stats = [], Counter()
for i, p in enumerate(pages):
    def t(c):
        return client._extract_property_text(p, schema, c)
    atual = t("classe_processo")
    try:
        ch = fit.get_all_children(client, p["id"])
        idx = fit.marker_index(ch)
        if idx is None:
            stats["sem_teor"] += 1
            continue
        corpo = fit.norm_ws(" ".join(fit._paragrafos_da_secao(ch, idx)))
    except Exception:
        stats["erro_leitura"] += 1
        continue
    oficial = classe_do_cabecalho(corpo)
    if not oficial:
        stats["cabecalho_sem_classe"] += 1
        continue
    # guarda: o teor precisa ser DESTE processo
    dig_pagina = re.sub(r"\D", "", t("numero_processo"))[:20]
    dig_cab = re.sub(r"\D", "", (re.search(r"[\d.\-–]{15,}", corpo[:400]) or re.match("", "")).group(0)
                     if re.search(r"[\d.\-–]{15,}", corpo[:400]) else "")[:20]
    if dig_pagina and dig_cab and dig_pagina != dig_cab:
        stats["teor_de_outro_processo"] += 1
        continue
    if not atual:
        stats["classe_vazia"] += 1
    elif familia(atual) == familia(oficial):
        stats["ok"] += 1
        continue
    else:
        stats["divergente"] += 1
    divergentes.append({"page_id": p["id"], "numero": t("numero_processo"),
                        "data": t("data_sessao")[:10], "atual": atual, "oficial": oficial,
                        "tema": t("tema")[:80], "cabecalho": corpo[:120]})
    if (i + 1) % 400 == 0:
        print(f"  {i+1}/{len(pages)} {dict(stats)}", flush=True)

print("stats:", dict(stats), flush=True)
print(f"\ndivergências: {len(divergentes)}")
print(" pares mais comuns:", Counter((d["atual"], d["oficial"]) for d in divergentes).most_common(10))
for d in divergentes[:8]:
    print(f"   {d['data']} {d['numero'][:24]} | {d['atual']!r} -> {d['oficial']!r} | {d['tema'][:44]}")

log = []
if APPLY:
    for d in divergentes:
        try:
            notion_request_with_retry(client, "PATCH", f"/pages/{d['page_id']}", json={
                "properties": {"classe_processo": {"select": {"name": d["oficial"]}}}})
            log.append({**d, "status": "aplicado"})
        except Exception as exc:
            log.append({**d, "status": "erro", "erro": str(exc)[:150]})

out = ART / f"classe_cabecalho_{'apply' if APPLY else 'dry'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
out.write_text(json.dumps({"stats": dict(stats), "divergentes": divergentes, "aplicados": log},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print("salvo:", out)
