"""Publica no Notion os julgamentos SEM numero (casos antigos/administrativos) que ja
foram extraidos num lote, reusando os artefatos (sem re-rodar o Gemini). So lanca as
linhas que o assess_row_publishability aprova ('publish') e que nao tem numero — as
demais (anuncios, ruido) seguem barradas pelas guardas. Identidade/upsert pela URL EXATA
do julgamento (find_existing_row sem numero), entao re-rodar ATUALIZA, nao duplica.

Uso:
    python publish_numberless_from_artifacts.py --batch-dir <dir do lote>           # dry-run
    python publish_numberless_from_artifacts.py --batch-dir <dir do lote> --apply
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path

from local_secrets import get_secret
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NotionSessoesClient,
    PublishPreviewRow,
    assess_row_publishability,
    build_video_only_youtube_link,
    canonicalize_numero_processo,
    publish_preview_rows,
    validate_preview_row,
)

LOGGER = logging.getLogger("publish_numberless_from_artifacts")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-dir", required=True, help="Diretorio do lote (batch_gui/<timestamp>).")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    # carrega as linhas (prefere a versao enriquecida com noticias)
    rows: list[PublishPreviewRow] = []
    for vdir in sorted(glob.glob(os.path.join(args.batch_dir, "*"))):
        if not os.path.isdir(vdir):
            continue
        path = os.path.join(vdir, "04b_enriched_preview_rows.json")
        if not os.path.exists(path):
            path = os.path.join(vdir, "04_preview_rows.json")
        if not os.path.exists(path):
            continue
        for rd in json.load(open(path, encoding="utf-8")):
            try:
                rows.append(PublishPreviewRow.model_validate(rd))
            except Exception as exc:
                LOGGER.warning("linha ignorada em %s: %s", os.path.basename(vdir), exc)

    # filtra: SEM numero e que o assess aprova
    candidatos = []
    for row in rows:
        if canonicalize_numero_processo(row.numero_processo):
            continue  # tem numero -> ja foi tratado no fluxo normal
        disp, _ = assess_row_publishability(row)
        if disp == "publish":
            candidatos.append(row)
    LOGGER.info("Linhas SEM numero aprovadas para publicar: %s (de %s totais)", len(candidatos), len(rows))

    key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    client = NotionSessoesClient(
        api_key=key, data_source_id=args.data_source_id,
        normalize_multiselect_colors_post_write=False,  # nao mexe em options (incidente conhecido)
    )
    schema = client.fetch_schema()

    # casa por URL EXATA (idempotente): re-rodar atualiza em vez de duplicar
    for row in candidatos:
        if not row.youtube_link:
            continue
        match = client.find_existing_row(schema, youtube_link=row.youtube_link, numero_processo="")
        if match:
            row.page_id = match.page_id
            row.action = "update"
        else:
            row.action = "create"

    for row in candidatos:
        acao = "ATUALIZA" if row.action == "update" else "CRIA"
        LOGGER.info("  [%s] %s | %s | res=%s vot=%s | %s",
                    acao, row.classe_processo or "-", (row.youtube_link or "")[-30:],
                    row.resultado or "-", row.votacao or "-", row.tema[:50])

    if not args.apply:
        LOGGER.info("DRY-RUN: nada publicado. Use --apply para lancar os %s.", len(candidatos))
        return 0

    results = publish_preview_rows(candidatos, client, schema)
    cr = sum(1 for r in results if r.get("status") == "created")
    up = sum(1 for r in results if r.get("status") == "updated")
    bl = sum(1 for r in results if r.get("status") == "blocked")
    sk = sum(1 for r in results if r.get("status") == "skipped")
    LOGGER.info("APLICADO: %s criadas, %s atualizadas, %s bloqueadas, %s ignoradas", cr, up, bl, sk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
