"""Migra a coluna 'advogados' de multi_select -> rich_text (resolve o limite de tamanho de
schema do Notion: ordenacao alfabetica e cores falham com 2200+ opcoes). Faz BACKUP de todos os
advogados ANTES, altera o tipo da propriedade no data source e REPOPULA cada pagina como texto
' , '-juntado (formato que parse_multi_value_text le de volta). Idempotente.
Espelho de migrate_partes_to_richtext.py.

Uso:
  python migrate_advogados_to_richtext.py            # dry-run: so backup + plano
  python migrate_advogados_to_richtext.py --apply    # backup + muda tipo + repopula
  python migrate_advogados_to_richtext.py --apply --repopulate-only   # so repopula
"""
from __future__ import annotations

import argparse, json, logging, time
from datetime import datetime
from pathlib import Path

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("migrate_advogados_to_richtext")
ARTIFACT_ROOT = Path("artifacts") / "notion_advogados_migration"
COL = "advogados"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--repopulate-only", action="store_true")
    ap.add_argument("--backup-file", default="")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    cur_type = schema.properties[COL].type
    LOGGER.info("tipo atual de '%s': %s", COL, cur_type)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.backup_file:
        backup = json.loads(Path(args.backup_file).read_text(encoding="utf-8"))
        LOGGER.info("backup carregado de %s (%d paginas)", args.backup_file, len(backup))
    else:
        pages = client.query_data_source()
        backup = {}
        for p in pages:
            vals = parse_multi_value_text(client._extract_property_text(p, schema, COL))
            backup[p["id"]] = {"numero": client._extract_property_text(p, schema, "numero_processo"), COL: vals}
        (run_dir / f"backup_{COL}.json").write_text(json.dumps(backup, ensure_ascii=False, indent=1), encoding="utf-8")
        LOGGER.info("BACKUP salvo: %s (%d paginas, %d com %s)", run_dir / f"backup_{COL}.json",
                    len(backup), sum(1 for b in backup.values() if b[COL]), COL)

    if not args.apply:
        LOGGER.info("DRY-RUN: nada alterado. Use --apply para migrar.")
        return 0

    if cur_type != "rich_text" and not args.repopulate_only:
        notion_request_with_retry(client, "PATCH", f"/data_sources/{args.data_source_id}",
                                  json={"properties": {COL: {"rich_text": {}}}})
        schema = client.fetch_schema()
        if schema.properties[COL].type != "rich_text":
            LOGGER.error("FALHA: tipo de %s ainda e %s apos PATCH", COL, schema.properties[COL].type)
            return 1
        LOGGER.info("tipo de '%s' alterado para rich_text", COL)
    else:
        schema = client.fetch_schema()

    n = applied = failed = 0
    for pid, b in backup.items():
        if not b.get(COL):
            continue
        n += 1
        built = client._build_property_value(schema, COL, b[COL])
        if built is None:
            continue
        try:
            notion_request_with_retry(client, "PATCH", f"/pages/{pid}", json={"properties": {COL: built}})
            applied += 1
        except Exception as exc:
            failed += 1
            LOGGER.warning("falha pagina %s: %s", pid, str(exc)[:120])
        time.sleep(0.12)
    LOGGER.info("REPOPULADO: %d/%d paginas (falhas=%d)", applied, n, failed)
    (run_dir / "summary.json").write_text(json.dumps(
        {"cur_type_inicial": cur_type, "repopuladas": applied, "alvo": n, "falhas": failed}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
