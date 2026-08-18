# -*- coding: utf-8 -*-
"""Desfaz os 8 REVERTS de `composicao` da rodada 20260730_095009.

O que aconteceu (30/07/2026): um teste isolado de `ProjetoConversor\\tse_acervo.py`
escreveu um CSV "delta" na pasta de PRODUCAO por causa de um argumento default
avaliado na definicao da funcao (`destino_dir: Path = PASTA_DJE`), que ignorava a
reatribuicao feita pelo teste. O watcher WatchDJe_Notion, que roda com --apply,
consumiu esse delta de 2 linhas de 1988; como quase nenhuma pagina casava com ele,
a logica de revert de `fill_composicao_from_jurisprudencia.py` (linhas 160-162)
desfez 8 gravacoes boas -- 3 delas zerando a composicao.

Este script le o changes.json daquela rodada e restaura o campo `old` de cada
registro. Idempotente: se a pagina ja esta com o valor bom, nao faz nada.

Uso:
    python _restaurar_composicao_30jul.py            # dry-run
    python _restaurar_composicao_30jul.py --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("restaurar_composicao")
CHANGES = Path(__file__).resolve().parent / "artifacts" / "notion_composicao_csv" / "20260730_095009" / "changes.json"


def digits(v: str) -> str:
    return "".join(c for c in (v or "") if c.isdigit())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--changes", default=str(CHANGES))
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    registros = json.loads(Path(args.changes).read_text(encoding="utf-8"))
    reverts = [r for r in registros if r.get("acao") == "reverter"]
    LOGGER.info("registros: %s | reverts a desfazer: %s", len(registros), len(reverts))
    alvo = {digits(r["cnj"])[:20]: r["old"] for r in reverts}

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"),
                                 data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    ok = ja = falhas = 0
    for p in pages:
        cnj = digits(client._extract_property_text(p, schema, "numero_processo"))[:20]
        if cnj not in alvo:
            continue
        bom = alvo[cnj]
        atual = parse_multi_value_text(client._extract_property_text(p, schema, "composicao"))
        if atual == bom:
            ja += 1
            LOGGER.info("  [ja ok]   %s (%s ministros)", cnj, len(bom))
            continue
        LOGGER.info("  [%s] %s: %s -> %s ministros",
                    "APLICA" if args.apply else "dry-run", cnj, len(atual), len(bom))
        if args.apply:
            try:
                built = (client._build_property_value(schema, "composicao", bom)
                         or client._build_empty_property_value(schema, "composicao"))
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}",
                                          json={"properties": {"composicao": built}})
                ok += 1
                time.sleep(0.15)
            except Exception as exc:  # noqa: BLE001
                falhas += 1
                LOGGER.warning("  falha %s: %s", cnj, str(exc)[:120])

    LOGGER.info("RESUMO: restauradas=%s | ja_corretas=%s | falhas=%s | alvo=%s",
                ok, ja, falhas, len(alvo))
    return 0 if not falhas else 1


if __name__ == "__main__":
    raise SystemExit(main())
