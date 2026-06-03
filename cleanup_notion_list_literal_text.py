"""Limpa ruído de 'list-literal do Python' em colunas rich_text do Notion.

Alguns registros antigos ficaram com o texto gravado como repr de lista, p.ex.
``['Resolução TSE 23.520/2017', 'Resolução TSE 23.422/2014']`` em vez do texto limpo
``Resolução TSE 23.520/2017, Resolução TSE 23.422/2014``. Este script detecta células
que são um list-literal puro (``[...]`` que faz parse via ast.literal_eval), tira
colchetes/aspas e re-grava os itens juntados por ', '.

Colunas tratadas: resoluções_citadas e precedentes_citados (mesma causa-raiz).

Uso:
    python cleanup_notion_list_literal_text.py            # dry-run (só relata)
    python cleanup_notion_list_literal_text.py --apply     # grava
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("cleanup_notion_list_literal_text")
ARTIFACT_ROOT = Path("artifacts") / "notion_list_literal_cleanup"
TARGET_COLUMNS = ["resoluções_citadas", "precedentes_citados"]
APPLY_SLEEP_SECONDS = 0.2


def clean_list_literal(value: str) -> str:
    """Se ``value`` for um list-literal puro do Python, devolve os itens juntados por
    ', '. Caso contrário, devolve o valor original inalterado."""
    text = (value or "").strip()
    if not (text.startswith("[") and text.endswith("]")):
        return value
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value
    if not isinstance(parsed, (list, tuple)):
        return value
    items = [str(item).strip() for item in parsed if str(item).strip()]
    return ", ".join(items)


def main() -> int:
    parser = argparse.ArgumentParser(description="Limpa list-literal em colunas rich_text do Notion.")
    parser.add_argument("--apply", action="store_true", help="Grava (padrão: dry-run).")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    client = NotionSessoesClient(api_key=key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    changes: list[dict[str, Any]] = []
    for page in pages:
        props: dict[str, Any] = {}
        detail: dict[str, Any] = {}
        for col in TARGET_COLUMNS:
            if col not in schema.properties:
                continue
            current = client._extract_property_text(page, schema, col)
            cleaned = clean_list_literal(current)
            if cleaned != current:
                props[col] = {"rich_text": [{"text": {"content": cleaned}}]} if cleaned else {"rich_text": []}
                detail[col] = {"old": current, "new": cleaned}
        if props:
            rec = {"page_id": str(page.get("id", "")), "detail": detail}
            changes.append(rec)
            if args.apply:
                try:
                    notion_request_with_retry(client, "PATCH", f"/pages/{page['id']}", json={"properties": props})
                    rec["status"] = "updated"
                except Exception as exc:
                    rec["status"] = "failed"
                    rec["error"] = str(exc)
                time.sleep(APPLY_SLEEP_SECONDS)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "pages": len(pages),
        "pages_changed": len(changes),
        "by_column": {col: sum(1 for c in changes if col in c["detail"]) for col in TARGET_COLUMNS},
        "applied": sum(1 for c in changes if c.get("status") == "updated"),
        "failed": sum(1 for c in changes if c.get("status") == "failed"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s | relatorios em %s", json.dumps(summary, ensure_ascii=False), run_dir)
    for c in changes:
        for col, d in c["detail"].items():
            LOGGER.info("  %s: %r -> %r", col, d["old"][:90], d["new"][:90])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
