"""Backfill/correção via CNJ DataJud das colunas numero_processo, classe_processo e
votacao no Notion. Regras CONSERVADORAS (não sobrescreve dado específico):

- numero_processo: completa para o CNJ de 20 dígitos quando o lookup é inequívoco e
  consistente com o número atual (mesmo prefixo NNNNNNN+DD).
- classe_processo: preenche VAZIO e corrige 'PA' com a classe oficial do CNJ; para os
  demais só corrige se a sigla do CNJ for mais específica/igual (nunca rebaixa
  AgRg-*/ED-* para a base via guard de downgrade).
- votacao: marca 'Suspenso*' quando há pedido de vista E julgamento nos movimentos, em
  classe de recurso jurisdicional, com resultado definitivo e votacao atual
  Unânime/Por maioria (padrão de pedido de vista julgado depois em plenário virtual).

NÃO mexe em origem (o órgão julgador do CNJ é a sede da zona eleitoral, que difere do
município do caso) nem em partes/advogados (indisponíveis na API pública do CNJ por LGPD).

Uso:
    python backfill_notion_cnj.py --limit 80          # dry-run amostra
    python backfill_notion_cnj.py --apply             # base toda
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from audit_notion_sessoes_round2 import notion_request_with_retry
from cnj_datajud import format_cnj_number, lookup_process
from local_secrets import get_secret
from tse_normalization import canonicalize_numero_processo
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NotionSessoesClient,
    classe_is_specificity_downgrade,
)

LOGGER = logging.getLogger("backfill_notion_cnj")
ARTIFACT_ROOT = Path("artifacts") / "notion_cnj_backfill"
APPLY_SLEEP_SECONDS = 0.2
DEFINITIVE_RESULTS_EXCLUDE = {"", "Suspenso por vista", "Suspenso mas julgado depois"}
# Só usamos vista+julgamento p/ marcar Suspenso* em classes de recurso jurisdicional
# (onde 'pedido de vista' afeta a votação), evitando 'Cumprimento de sentença' etc.
SUSPENSO_STAR_CLASSES = {"REspe", "AREspe", "RO", "AgRg-REspe", "AgRg-AREspe", "AgRg-RO", "RMS", "RHC", "RCED", "AR", "AIJE", "Rp", "MS", "RvE"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill via CNJ DataJud (numero/classe/origem/votacao).")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    client = NotionSessoesClient(api_key=key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()
    if args.limit and args.limit < len(pages):
        pages = pages[:: max(1, len(pages) // args.limit)][: args.limit]
    valid_classe = {
        str(o.get("name", "")).strip()
        for o in schema.raw_payload.get("properties", {}).get("classe_processo", {}).get("select", {}).get("options", [])
        if str(o.get("name", "")).strip()
    }
    cnj_session = requests.Session()

    def t(page, f):
        return client._extract_property_text(page, schema, f)

    changes: list[dict[str, Any]] = []
    stats = {"processed": 0, "cnj_hit": 0, "num": 0, "classe": 0, "votacao": 0, "errors": 0}
    for i, p in enumerate(pages, 1):
        stats["processed"] += 1
        try:
            num = t(p, "numero_processo"); trib = t(p, "tribunal"); yr = (t(p, "data_sessao") or "")[:4]
            cur_cls = t(p, "classe_processo").strip(); cur_vot = t(p, "votacao").strip()
            cur_res = t(p, "resultado").strip()
            info = lookup_process(num, tribunal=trib, year=yr, session=cnj_session)
            if not info:
                continue
            stats["cnj_hit"] += 1
            props: dict[str, Any] = {}
            detail: dict[str, Any] = {}
            cur_digits = re.sub(r"\D", "", num)
            cnj_digits = re.sub(r"\D", "", info.numero_completo)
            # numero_processo: completa para o CNJ de 20 dígitos quando consistente com o
            # número atual (mesmo prefixo NNNNNNN+DD), sem encurtar números já completos.
            if len(cnj_digits) == 20 and len(cur_digits) < 20 and len(cur_digits) >= 9 and cnj_digits.startswith(cur_digits[:9]):
                new_num = format_cnj_number(cnj_digits)
                if new_num and new_num != num:
                    props["numero_processo"] = {"rich_text": [{"text": {"content": new_num}}]}
                    detail["numero_processo"] = {"old": num, "new": new_num}; stats["num"] += 1
            # classe_processo (preenche vazio / corrige PA; não rebaixa AgRg-*/ED-*)
            sigla = info.classe_sigla
            if sigla and sigla in valid_classe and ((not cur_cls) or cur_cls == "PA") and sigla != cur_cls:
                if not classe_is_specificity_downgrade(cur_cls, sigla):
                    props["classe_processo"] = {"select": {"name": sigla}}; detail["classe_processo"] = {"old": cur_cls, "new": sigla}; stats["classe"] += 1
            # votacao -> Suspenso* (vista + julgamento + resultado definitivo) só em classes
            # de recurso jurisdicional, evitando falso-positivo (ex.: Cumprimento de sentença).
            if info.has_vista and info.has_julgamento and info.classe_sigla in SUSPENSO_STAR_CLASSES and cur_res not in DEFINITIVE_RESULTS_EXCLUDE and cur_vot in {"Unânime", "Por maioria"}:
                props["votacao"] = {"select": {"name": "Suspenso*"}}; detail["votacao"] = {"old": cur_vot, "new": "Suspenso*"}; stats["votacao"] += 1
            if props:
                rec = {"page_id": p["id"], "numero": canonicalize_numero_processo(num), "detail": detail, "cnj": info.as_dict()}
                changes.append(rec)
                if args.apply:
                    try:
                        notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props}); rec["status"] = "updated"
                    except Exception as exc:
                        rec["status"] = "failed"; rec["error"] = str(exc)
                    time.sleep(APPLY_SLEEP_SECONDS)
        except Exception as exc:  # um registro ruim não derruba a corrida inteira
            stats["errors"] += 1
            LOGGER.warning("Falha no registro %s (%s): %s", i, p.get("id"), exc)
        if i % 50 == 0:
            LOGGER.info("Processados %s/%s | CNJ hits %s | mudancas %s | erros %s", i, len(pages), stats["cnj_hit"], len(changes), stats["errors"])

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats, "pages_changed": len(changes),
               "applied": sum(1 for c in changes if c.get("status") == "updated")}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s | relatorios em %s", json.dumps(summary, ensure_ascii=False), run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
