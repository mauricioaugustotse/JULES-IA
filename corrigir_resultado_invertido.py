#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corrige páginas cujo `resultado` ficou INVERTIDO por um bug do normalizador.

O BUG (corrigido em 30/07/2026, tse_normalization.py:1962-1973): a lista de negativas
de `normalize_resultado_final` cobria "negado/negou/nego/nega provimento", mas **não**
"negar" nem "negaram" — justamente a forma do dispositivo colegiado ("ACORDAM os
Ministros em NEGAR provimento ao recurso"). A frase caía no regex genérico
`\\bprovido\\b|\\bprovimento\\b` logo abaixo e o resultado saía como **"Provido"**.

Quem gravou assim: `infer_resultado_from_row_text` (tse_youtube_notion_core.py:1829-1852),
que roda em toda publicação de vídeo com `resultado` vazio, e o normalizador de saída
(`:7130`).

Corrigir o código NÃO conserta o que já está no Notion — daí este script. Num RAG
jurídico é o pior defeito possível: a base afirma o CONTRÁRIO do que o tribunal decidiu.

CRITÉRIO CONSERVADOR — só corrige quando as três condições valem juntas:
  1. `resultado` gravado é "Provido" (ou "Provido em parte");
  2. os textos da página contêm "negar/negaram provimento";
  3. NÃO contêm sinal de provimento real ("dar/deram provimento", "parcial provimento").
Qualquer dúvida vai para o relatório sem ser aplicada.

Uso:
    python corrigir_resultado_invertido.py                 # dry-run
    python corrigir_resultado_invertido.py --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import normalize_resultado_final
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("corrigir_resultado")
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "notion_resultado_invertido"

TEXT_FIELDS = ("punchline", "raciocinio_juridico", "analise_do_conteudo_juridico",
               "fundamentacao_normativa", "tema")

# a forma exata que o bug engolia
NEGAR_RE = re.compile(r"\bneg(?:ar|aram)\s+provimento\b", re.I)
# sinais de provimento REAL — se houver, nao mexer (pode ser parcial, ou multiplos pedidos)
PROVER_RE = re.compile(r"\b(?:dar|deram|dou|da|dado)\s+provimento\b|"
                       r"\bparcial\s+provimento\b|\bprovido\s+em\s+parte\b", re.I)
SUSPEITOS = {"Provido", "Provido em parte"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(message)s")

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"),
                                 data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f) or ""

    corrigir, revisar = [], []
    for p in pages:
        atual = t(p, "resultado").strip()
        if atual not in SUSPEITOS:
            continue
        blob = "  ".join(t(p, f) for f in TEXT_FIELDS)
        if not NEGAR_RE.search(blob):
            continue
        proc = t(p, "numero_processo")
        # o que o normalizador CORRIGIDO diria sobre o mesmo texto
        novo = normalize_resultado_final(blob, t(p, "classe_processo"))
        item = {"page_id": p.get("id"), "processo": proc, "de": atual, "para": novo,
                "trecho": NEGAR_RE.search(blob).group(0)}
        if PROVER_RE.search(blob):
            item["motivo"] = "texto tem negar E dar provimento — ambíguo, revisar à mão"
            revisar.append(item)
        elif novo == "Desprovido":
            corrigir.append(item)
        else:
            item["motivo"] = f"normalizador corrigido devolveu {novo!r}, nao 'Desprovido'"
            revisar.append(item)

    LOGGER.info("paginas na base ......: %s", len(pages))
    LOGGER.info("com resultado suspeito: %s", len(corrigir) + len(revisar))
    LOGGER.info("  a CORRIGIR .........: %s  (Provido -> Desprovido)", len(corrigir))
    LOGGER.info("  a REVISAR a mao ....: %s", len(revisar))

    ok = falhas = 0
    if args.apply:
        for it in corrigir:
            try:
                notion_request_with_retry(
                    client, "PATCH", f"/pages/{it['page_id']}",
                    json={"properties": {"resultado": {"select": {"name": "Desprovido"}}}})
                ok += 1
                time.sleep(0.15)
            except Exception as exc:  # noqa: BLE001
                falhas += 1
                LOGGER.warning("falha %s: %s", it["processo"], str(exc)[:110])

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    saida = ARTIFACT_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}_changes.json"
    saida.write_text(json.dumps({"corrigir": corrigir, "revisar": revisar},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    for it in (corrigir[:5] + revisar[:5]):
        LOGGER.info("  %s | %s -> %s | %r", it["processo"], it["de"],
                    it.get("para"), it["trecho"])
    LOGGER.info("RESUMO: %s", json.dumps(
        {"modo": "apply" if args.apply else "dry-run", "corrigidas": ok,
         "falhas": falhas, "a_revisar": len(revisar), "relatorio": str(saida)},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
