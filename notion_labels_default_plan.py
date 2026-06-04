"""Gera o plano (read-only via API do Notion) para padronizar as etiquetas das colunas
`partes`, `advogados` e `origem` da base de sessões na cor `default` e identificar as
etiquetas ÓRFÃS (opções do schema sem nenhum uso real nas páginas) para exclusão.

Saída: um CSV por coluna em artifacts/notion_labels_default/<coluna>_plano_manual.csv
com as colunas: coluna, etiqueta, cor_atual, cor_alvo, remover_se_apply.
- remover_se_apply=1  -> etiqueta órfã (não usada por nenhuma página) -> excluir na UI.
- cor_alvo=default    -> cor a aplicar (a UI só recolore as que não estão em default).

A aplicação de cor/exclusão acontece DEPOIS, na UI do Notion via Playwright
(notion_labels_default_playwright.py) — a API oficial não recolore opção existente e
PATCH em options é destrutivo (REPLACE + limite 100). Este passo é só leitura.

Uso:
    python notion_labels_default_plan.py            # todas as colunas
    python notion_labels_default_plan.py --columns partes
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("notion_labels_default_plan")
OUTPUT_DIR = Path("artifacts") / "notion_labels_default"
TARGET_COLUMNS = ["partes", "advogados", "origem"]
CSV_FIELDS = ["coluna", "etiqueta", "cor_atual", "cor_alvo", "remover_se_apply"]


def _schema_options(schema: Any, column: str) -> tuple[str, list[dict[str, Any]]]:
    prop = schema.raw_payload.get("properties", {}).get(column, {})
    ptype = prop.get("type", "")
    options = prop.get(ptype, {}).get("options", []) if ptype in {"select", "multi_select"} else []
    return ptype, [o for o in options if isinstance(o, dict)]


def _used_names(pages: list[dict[str, Any]], column: str, ptype: str) -> set[str]:
    used: set[str] = set()
    for page in pages:
        value = page.get("properties", {}).get(column, {})
        if ptype == "multi_select":
            for item in value.get("multi_select", []) or []:
                name = (item or {}).get("name", "")
                if name:
                    used.add(name)
        elif ptype == "select":
            sel = value.get("select")
            if isinstance(sel, dict) and sel.get("name"):
                used.add(sel["name"])
    return used


def build_rows(schema: Any, pages: list[dict[str, Any]], column: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    ptype, options = _schema_options(schema, column)
    if not options:
        raise RuntimeError(f"Coluna '{column}' não é select/multi_select ou não tem opções (tipo={ptype}).")
    used = _used_names(pages, column, ptype)
    rows: list[dict[str, str]] = []
    orphans = 0
    nondefault = 0
    for option in options:
        name = str(option.get("name", "")).strip()
        if not name:
            continue
        color = str(option.get("color", "") or "default")
        is_orphan = name not in used
        if is_orphan:
            orphans += 1
        if color != "default":
            nondefault += 1
        rows.append(
            {
                "coluna": column,
                "etiqueta": name,
                "cor_atual": color,
                "cor_alvo": "default",
                "remover_se_apply": "1" if is_orphan else "0",
            }
        )
    stats = {
        "tipo": ptype,
        "total": len(rows),
        "usadas": len(used),
        "orfas": orphans,
        "fora_do_default": nondefault,
        # opções que exigem ação na UI: excluir órfã OU recolorir não-default
        "acoes_ui": sum(1 for r in rows if r["remover_se_apply"] == "1" or r["cor_atual"] != "default"),
    }
    return rows, stats


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plano (read-only) para recolorir/excluir etiquetas no Notion.")
    parser.add_argument("--columns", nargs="*", default=TARGET_COLUMNS, help="Colunas-alvo (padrão: partes advogados origem).")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    client = NotionSessoesClient(api_key=key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()
    LOGGER.info("Base carregada: %s páginas.", len(pages))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resumo: dict[str, Any] = {"pages": len(pages), "columns": {}}
    orphan_samples: dict[str, list[str]] = {}
    for column in args.columns:
        rows, stats = build_rows(schema, pages, column)
        csv_path = out_dir / f"{column}_plano_manual.csv"
        write_csv(csv_path, rows)
        resumo["columns"][column] = {**stats, "csv": str(csv_path)}
        orphan_samples[column] = [r["etiqueta"] for r in rows if r["remover_se_apply"] == "1"]
        LOGGER.info(
            "[%s] tipo=%s total=%s usadas=%s ORFAS=%s fora_default=%s acoes_ui=%s -> %s",
            column, stats["tipo"], stats["total"], stats["usadas"], stats["orfas"],
            stats["fora_do_default"], stats["acoes_ui"], csv_path,
        )
    resumo["orfas"] = orphan_samples
    (out_dir / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Resumo salvo em %s", out_dir / "resumo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
