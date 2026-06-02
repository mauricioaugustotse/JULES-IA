"""Audita a cor das etiquetas das colunas indicadas (por padrão ``origem`` e
``partes``) no Notion e lista as que NÃO estão em ``default``.

LIMITAÇÃO DA API DO NOTION: não é possível alterar a cor de uma opção de select já
existente via API — a API responde 400 ("Cannot update color of select ..."). Logo,
a recoloração das etiquetas existentes precisa ser feita MANUALMENTE na interface do
Notion (clicar na opção → escolher "Default"). Este script apenas exporta a lista das
opções coloridas (relatório JSON) para orientar essa correção manual.

Uso:
    python cleanup_notion_label_colors.py
    python cleanup_notion_label_colors.py --properties origem partes votacao
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from local_secrets import get_secret
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NotionSessoesClient,
    audit_label_colors,
)


LOGGER = logging.getLogger("cleanup_notion_label_colors")
ARTIFACT_ROOT = Path("artifacts") / "notion_label_colors"
DEFAULT_PROPERTIES = ["origem", "partes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita a cor das etiquetas (origem/partes) e lista as coloridas.")
    parser.add_argument("--properties", nargs="+", default=DEFAULT_PROPERTIES)
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    api_key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY/NOTION_TOKEN não encontrado.")
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    client = NotionSessoesClient(api_key=api_key, data_source_id=args.data_source_id)
    report = audit_label_colors(client, args.properties)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for prop, info in report.items():
        LOGGER.info(
            "%s: %s opções, %s coloridas (status=%s)",
            prop,
            info.get("total_options"),
            info.get("non_default"),
            info.get("status"),
        )
    LOGGER.info("A recoloração de etiquetas EXISTENTES precisa ser feita manualmente na UI do Notion.")
    LOGGER.info("Lista das opções coloridas exportada em %s/report.json", artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
