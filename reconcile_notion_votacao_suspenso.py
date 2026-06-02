"""Reconcilia a coluna ``votacao`` na base do Notion: rebaixa a etiqueta ``Suspenso``
para ``Suspenso*`` quando o MESMO processo foi, em data igual ou posterior, julgado
de forma definitiva (``Unânime`` ou ``Por maioria``).

Assim, sempre que um processo volta à pauta e é definitivamente votado, os registros
anteriores que ficaram apenas ``Suspenso`` passam a ``Suspenso*`` — mantendo o Notion
coerente ao longo das sessões.

A regra de decisão vive em ``tse_youtube_notion_core.compute_suspenso_star_updates`` e
é compartilhada com a reconciliação automática feita na publicação pela GUI.

Uso:
    python reconcile_notion_votacao_suspenso.py            # dry-run (apenas relatório)
    python reconcile_notion_votacao_suspenso.py --apply    # aplica as trocas
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from local_secrets import get_secret
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NotionSessoesClient,
    reconcile_suspenso_marks,
)


LOGGER = logging.getLogger("reconcile_notion_votacao_suspenso")
ARTIFACT_ROOT = Path("artifacts") / "notion_votacao_suspenso_reconcile"


def write_reports(artifact_dir: Path, changes: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcilia Suspenso -> Suspenso* na coluna votacao do Notion.")
    parser.add_argument("--apply", action="store_true", help="Aplica as trocas (padrão: apenas dry-run).")
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
    schema = client.fetch_schema()
    changes = reconcile_suspenso_marks(client, schema, apply=args.apply)
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "candidates": len(changes),
        "applied": sum(1 for c in changes if c.get("status") == "updated"),
        "failed": sum(1 for c in changes if c.get("status") == "failed"),
        "examples": changes[:25],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_reports(artifact_dir, changes, summary)
    LOGGER.info(
        "Candidatos a Suspenso*: %s (mode=%s, aplicados=%s, falhas=%s)",
        summary["candidates"],
        summary["mode"],
        summary["applied"],
        summary["failed"],
    )
    LOGGER.info("Relatórios gravados em %s", artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
