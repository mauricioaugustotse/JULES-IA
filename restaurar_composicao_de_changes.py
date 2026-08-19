# -*- coding: utf-8 -*-
"""Desfaz as reversoes de `composicao` registradas em um changes.json de
fill_composicao_from_jurisprudencia.py, restaurando o campo `old` de cada registro.

POR QUE ISTO EXISTE
-------------------
O ramo "reverter" de fill_composicao_from_jurisprudencia.py ignorava o CSV do lote e reescrevia
paginas a partir do historico de rodadas anteriores. Detonou tres vezes:

  10/07/2026  20260710_190051   469 reversoes  -- nunca reparada
  30/07/2026  20260730_095009     8 reversoes  -- reparada pelo _restaurar_composicao_30jul.py (f93e6ea)
  19/08/2026  20260819_102631  1325 reversoes  -- 337 delas gravaram composicao VAZIA

Em 19/08/2026 o ramo foi desligado por padrao (--permitir-reverter). Este script generaliza o
one-off de 30/07 para reparar qualquer uma dessas rodadas.

DUAS GUARDAS QUE O ONE-OFF NAO TINHA (e que sao obrigatorias em escala)
----------------------------------------------------------------------
1. So restaura se a pagina AINDA carrega o dano (`atual == registro["new"]`). O original
   sobrescrevia cego: com 8 paginas passava; com 1325 atropelaria quem corrigiu algo no meio.
2. Recusa registro cujo `old` esteja vazio -- nao ha o que restaurar, e gravar vazio foi
   justamente o estrago que se quer desfazer.

Casa por (CNJ-20, data da sessao), nao so por CNJ: o mesmo processo tem varias paginas, uma por
sessao, e o changes.json de composicao nao grava page_id nas rodadas antigas.

Idempotente: rodar duas vezes nao faz nada na segunda.

Uso:
    python restaurar_composicao_de_changes.py --changes artifacts/notion_composicao_csv/<rodada>/changes.json
    python restaurar_composicao_de_changes.py --changes ... --apply
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
from tse_normalization import parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("restaurar_composicao")
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "notion_composicao_restore"


def digits(v: str) -> str:
    return "".join(c for c in (v or "") if c.isdigit())


def iso_date(s: str) -> str:
    s = str(s or "").strip()[:10]
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", s)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return s[:10] if re.match(r"(\d{4})-(\d{2})-(\d{2})", s) else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--changes", required=True, help="changes.json da rodada a desfazer.")
    ap.add_argument("--apply", action="store_true", help="Grava (padrao: dry-run).")
    ap.add_argument("--acao", default="reverter",
                    help="Qual acao desfazer (padrao: reverter). 'todas' para ignorar o filtro.")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(message)s")

    registros = json.loads(Path(args.changes).read_text(encoding="utf-8"))
    if args.acao == "todas":
        alvos = list(registros)
    else:
        alvos = [r for r in registros if r.get("acao") == args.acao]

    # Recusa registros sem valor bom para restaurar.
    sem_old = [r for r in alvos if not r.get("old")]
    alvos = [r for r in alvos if r.get("old")]
    LOGGER.info("registros: %d | acao=%s: %d | recusados por 'old' vazio: %d",
                len(registros), args.acao, len(alvos), len(sem_old))

    # (cnj20, data) -> registro. Chave composta porque um CNJ tem uma pagina por sessao.
    alvo: dict[tuple, dict] = {}
    for r in alvos:
        alvo[(digits(r["cnj"])[:20], iso_date(r.get("data")))] = r

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"),
                                 data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()
    LOGGER.info("paginas na base: %d", len(pages))

    feitas, ja_ok, divergentes, falhas, nao_encontradas = [], 0, [], 0, 0
    vistos = set()

    for p in pages:
        cnj = digits(client._extract_property_text(p, schema, "numero_processo"))[:20]
        iso = iso_date(client._extract_property_text(p, schema, "data_sessao"))
        reg = alvo.get((cnj, iso))
        if not reg:
            continue
        vistos.add((cnj, iso))
        bom = reg["old"]
        danificado = reg.get("new")
        atual = parse_multi_value_text(client._extract_property_text(p, schema, "composicao"))

        if atual == bom:
            ja_ok += 1
            continue
        # GUARDA: so mexe se a pagina ainda esta exatamente como a rodada ruim a deixou.
        if atual != danificado:
            divergentes.append({"cnj": cnj, "data": iso, "atual": atual,
                                "esperado_danificado": danificado, "bom": bom})
            LOGGER.warning("  [divergente] %s (%s): atual tem %d ministros, nao bate com o dano "
                           "registrado -- PULADO", cnj, iso, len(atual))
            continue

        LOGGER.info("  [%s] %s (%s): %d -> %d ministros",
                    "APLICA" if args.apply else "dry-run", cnj, iso, len(atual), len(bom))
        if args.apply:
            try:
                built = client._build_property_value(schema, "composicao", bom)
                if not built:
                    LOGGER.warning("  recusado (valor bom nao construiu) %s", cnj)
                    continue
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}",
                                          json={"properties": {"composicao": built}})
                feitas.append({"cnj": cnj, "data": iso, "page_id": p["id"],
                               "old": atual, "new": bom})
                time.sleep(0.15)
            except Exception as exc:  # noqa: BLE001
                falhas += 1
                LOGGER.warning("  falha %s: %s", cnj, str(exc)[:120])
        else:
            feitas.append({"cnj": cnj, "data": iso, "page_id": p["id"],
                           "old": atual, "new": bom})

    nao_encontradas = len(alvo) - len(vistos)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(feitas, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    if divergentes:
        (run_dir / "divergentes.json").write_text(
            json.dumps(divergentes, ensure_ascii=False, indent=1), encoding="utf-8")
    resumo = {
        "mode": "apply" if args.apply else "dry-run",
        "origem": str(args.changes),
        "alvo": len(alvo),
        "restauradas": len(feitas) if args.apply else 0,
        "a_restaurar": 0 if args.apply else len(feitas),
        "ja_corretas": ja_ok,
        "divergentes": len(divergentes),
        "recusadas_old_vazio": len(sem_old),
        "nao_encontradas_na_base": nao_encontradas,
        "falhas": falhas,
    }
    (run_dir / "summary.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(resumo, ensure_ascii=False))
    LOGGER.info("artefatos: %s", run_dir)
    return 0 if not falhas else 1


if __name__ == "__main__":
    raise SystemExit(main())
