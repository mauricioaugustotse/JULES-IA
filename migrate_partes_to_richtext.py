"""Migra a coluna 'partes' de multi_select -> rich_text (resolve o limite de tamanho de
schema do Notion). Faz BACKUP de todas as partes ANTES, altera o tipo da propriedade no
data source e REPOPULA cada pagina como texto ' , '-juntado (formato que parse_multi_value_text
le de volta). Idempotente: rodar de novo so reescreve.

Uso:
  python migrate_partes_to_richtext.py            # dry-run: so backup + plano
  python migrate_partes_to_richtext.py --apply    # backup + muda tipo + repopula
  python migrate_partes_to_richtext.py --apply --repopulate-only   # so repopula (do backup mais recente)
"""
from __future__ import annotations

import argparse, json, logging, time
from datetime import datetime
from pathlib import Path

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("migrate_partes_to_richtext")
ARTIFACT_ROOT = Path("artifacts") / "notion_partes_migration"


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
    cur_type = schema.properties["partes"].type
    LOGGER.info("tipo atual de 'partes': %s", cur_type)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # BACKUP (sempre, a partir do estado atual)
    if args.backup_file:
        backup = json.loads(Path(args.backup_file).read_text(encoding="utf-8"))
        LOGGER.info("backup carregado de %s (%d paginas)", args.backup_file, len(backup))
    else:
        pages = client.query_data_source()
        backup = {}
        for p in pages:
            vals = parse_multi_value_text(client._extract_property_text(p, schema, "partes"))
            backup[p["id"]] = {"numero": client._extract_property_text(p, schema, "numero_processo"), "partes": vals}
        (run_dir / "backup_partes.json").write_text(json.dumps(backup, ensure_ascii=False, indent=1), encoding="utf-8")
        LOGGER.info("BACKUP salvo: %s (%d paginas, %d com partes)", run_dir / "backup_partes.json",
                    len(backup), sum(1 for b in backup.values() if b["partes"]))

    if not args.apply:
        LOGGER.info("DRY-RUN: nada alterado. Use --apply para migrar.")
        return 0

    # ALTERA O TIPO (se ainda nao for rich_text e nao for repopulate-only)
    if cur_type != "rich_text" and not args.repopulate_only:
        notion_request_with_retry(client, "PATCH", f"/data_sources/{args.data_source_id}",
                                  json={"properties": {"partes": {"rich_text": {}}}})
        schema = client.fetch_schema()
        if schema.properties["partes"].type != "rich_text":
            LOGGER.error("FALHA: tipo de partes ainda e %s apos PATCH", schema.properties["partes"].type)
            return 1
        LOGGER.info("tipo de 'partes' alterado para rich_text")
    else:
        schema = client.fetch_schema()

    # REPOPULA (texto ' , '-juntado) a partir do backup
    n = applied = failed = 0
    for pid, b in backup.items():
        if not b.get("partes"):
            continue
        n += 1
        built = client._build_property_value(schema, "partes", b["partes"])
        if built is None:
            continue
        try:
            notion_request_with_retry(client, "PATCH", f"/pages/{pid}", json={"properties": {"partes": built}})
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
