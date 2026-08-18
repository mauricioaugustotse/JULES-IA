#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harmoniza a página de SESSÕES com o acórdão publicado no DJe. Sem IA.

Fecha o ciclo pedido pelo usuário: quando um julgamento que estava "Suspenso por vista"
sai no DJe, não basta corrigir a etiqueta (isso o `suspenso_crosscheck_csv.py` já faz) —
o mérito, a composição e as partes daquele acórdão têm de descer para a página.

PRECEDÊNCIA: **o DJe vence** (decisão do usuário, 30/07/2026). O acórdão publicado é o
documento oficial; a página de sessão nasce de vídeo e transcrição, que erram mais.
Toda sobrescrita fica registrada no changes.json, com o valor anterior, para desfazer.

O QUE ESCREVE (só campos com fonte oficial no CSV):
  · resultado  — MÉRITO extraído do dispositivo (Provido/Desprovido/Não conhecido/...)
  · votacao    — Unânime / Por maioria, do próprio dispositivo
  · composicao — do bloco "Composição:" do acórdão, canonizada
  · partes     — coluna `partes` do CSV
O que NÃO toca: tema, punchline, análise jurídica, youtube_link, precedentes_citados —
tudo que nasce do vídeo e não tem equivalente no DJe.

Uso (no pipeline, recebe --input-dir do staging):
    python harmonizar_sessao_com_dje.py --input-dir <pasta>            # dry-run
    python harmonizar_sessao_com_dje.py --input-dir <pasta> --apply
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, r"C:\Users\mauri\ProjetoConversor")   # vocabulário único de ministros

from audit_notion_sessoes_round2 import notion_request_with_retry
from import_dje_faltantes import extract_dispositivo, resultado_votacao_from_dispositivo
from local_secrets import get_secret
from tse_normalization import (MINISTROS_JURISTAS, MINISTROS_STF, MINISTROS_STJ,
                               normalize_class_text, normalize_ministro_name)
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient
try:
    from _ministros_canonico import canonizar as canonizar_ministro
except Exception:  # noqa: BLE001 — degrada sem quebrar
    def canonizar_ministro(x):  # type: ignore
        return x

ROSTER = MINISTROS_STF | MINISTROS_STJ | MINISTROS_JURISTAS


def _tokens(nome: str) -> list[str]:
    return [t for t in normalize_class_text(re.sub(r"^Min\.\s*", "", nome)).split()
            if t not in {"de", "da", "do", "dos", "das", "e"}]


def _subsequencia(curto: list[str], longo: list[str]) -> bool:
    it = iter(longo)
    return bool(curto) and all(t in it for t in curto)


def canonizar(nome: str) -> str:
    """Canoniza o nome vindo do CSV do DJe, ou devolve '' se não reconhecer.

    O `normalize_ministro_name` (MINISTRO_ALIAS_MAP) tem a palavra final porque é ele
    que define as opções do select na base de sessões: sem essa âncora, o
    fallback-identidade do import acima grava a grafia crua do DJe e recria as
    variantes ('Min. André Ramos Tavares', 'Min. ... Bucchianeri Pinheiro').
    Nome fora do roster é DESCARTADO em vez de gravado — como a composição só é
    escrita com 3 a 7 nomes, um desconhecido faz o campo ser deixado como está,
    que é o modo de falhar seguro (não poluir o select)."""
    canon = normalize_ministro_name(canonizar_ministro(nome)) or ""
    if not canon:
        return ""
    if canon in ROSTER:
        return canon
    # nome longo do DJe ('Min. Gilmar Ferreira Mendes') contendo o canônico curto
    alvo = _tokens(canon)
    for conhecido in ROSTER:
        if _subsequencia(_tokens(conhecido), alvo):
            return conhecido
    LOGGER.warning("ministro fora do roster, composicao ignorada: %r "
                   "(catalogar em tse_normalization se for legitimo)", nome)
    return ""

csv.field_size_limit(500 * 1024 * 1024)
LOGGER = logging.getLogger("harmonizar")
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "notion_harmonizar_dje"

COMP_RE = re.compile(r"Composi[cç][aã]o(?:\s+do\s+julgamento)?\s*:\s*Ministros?\s*\(?a?s?\)?\s*([^.]{10,400})", re.I)


def d20(v) -> str:
    return re.sub(r"\D", "", str(v or ""))[:20]


def iso(v: str) -> str:
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", (v or "").strip())
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else ""


def e_acordao(row: dict) -> bool:
    if (row.get("siglaTribunalJE") or "").strip().upper() not in ("", "TSE"):
        return False
    t = (row.get("descricaoTipoDecisao") or "").strip().lower()
    return bool(t) and "monocr" not in t and "despacho" not in t and "ac" in t and "rd" in t


def extrair(row: dict) -> dict:
    """Campos com fonte oficial no acórdão. Só devolve o que conseguiu extrair."""
    td = row.get("textoDecisao") or ""
    out: Dict[str, object] = {}
    # Usa a extracao desenhada para DISPOSITIVO DE ACORDAO (import_dje_faltantes), nao
    # `normalize_resultado_final` -- esta ultima foi feita para o texto RESUMIDO do
    # video e, diante do dispositivo longo, devolve a propria frase de entrada.
    disp = extract_dispositivo(td)
    res, vot = resultado_votacao_from_dispositivo(disp)
    if res:
        out["resultado"] = res
    if vot:
        out["votacao"] = vot
    mc = COMP_RE.search(td)
    if mc:
        nomes = []
        for p in re.split(r",| e ", re.sub(r"\(.*?\)", "", mc.group(1))):
            nm = " ".join(p.split()).strip(" .;")
            if 4 <= len(nm) <= 45:
                canon = canonizar(nm if nm.startswith("Min.") else f"Min. {nm}")
                if canon:
                    nomes.append(canon)
        nomes = list(dict.fromkeys(nomes))
        if 3 <= len(nomes) <= 7:
            out["composicao"] = nomes
    partes = (row.get("partes") or "").strip()
    if partes:
        out["partes"] = partes
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=str(Path(__file__).resolve().parent
                                               / "artifacts" / "jurisprudencia_csv"))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(message)s")

    # (CNJ-20, data) -> campos oficiais do ACÓRDÃO
    oficial: Dict[tuple, dict] = {}
    for fp in sorted(glob.glob(str(Path(args.input_dir) / "*.csv"))):
        try:
            with open(fp, encoding="utf-8-sig", newline="") as h:
                for row in csv.DictReader(h):
                    if not e_acordao(row):
                        continue
                    cnj, dt = d20(row.get("numeroUnico")), iso(row.get("dataDecisao"))
                    if len(cnj) == 20 and dt:
                        dados = extrair(row)
                        if dados:
                            oficial[(cnj, dt)] = dados
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("erro no CSV %s: %s", Path(fp).name, str(exc)[:110])
    LOGGER.info("acordaos com dado oficial: %s", len(oficial))
    if not oficial:
        LOGGER.info("nada a harmonizar.")
        return 0

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"),
                                 data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f) or ""

    mudancas, ok, falhas = [], 0, 0
    for p in pages:
        cnj = d20(t(p, "numero_processo"))
        dt = (t(p, "data_sessao") or "")[:10]
        dados = oficial.get((cnj, dt))
        if not dados:
            continue
        props, reg = {}, []
        for campo, novo in dados.items():
            atual_txt = t(p, campo)
            atual = [x.strip() for x in atual_txt.split(",") if x.strip()] \
                if campo == "composicao" else atual_txt.strip()
            if (novo == atual) or (not novo):
                continue
            # PRECEDÊNCIA DO DJe: sobrescreve mesmo se a página já tinha valor.
            built = client._build_property_value(schema, campo, novo)
            if not built:
                continue
            props[campo] = built
            reg.append({"campo": campo, "de": atual, "para": novo})
        if not props:
            continue
        mudancas.append({"page_id": p.get("id"), "processo": t(p, "numero_processo"),
                         "data": dt, "mudancas": reg})
        if args.apply:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}",
                                          json={"properties": props})
                ok += 1
                time.sleep(0.15)
            except Exception as exc:  # noqa: BLE001
                falhas += 1
                if falhas <= 3:
                    LOGGER.warning("falha %s: %s", cnj, str(exc)[:110])

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    saida = ARTIFACT_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}_changes.json"
    saida.write_text(json.dumps(mudancas, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    por_campo = Counter(m["campo"] for x in mudancas for m in x["mudancas"])
    LOGGER.info("RESUMO: %s", json.dumps(
        {"modo": "apply" if args.apply else "dry-run", "paginas": len(mudancas),
         "por_campo": dict(por_campo), "aplicadas": ok, "falhas": falhas,
         "relatorio": str(saida)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
