"""Recuperação emergencial da coluna advogados.

PHASE harvest: itera as 627 opções originais (backup com ids) em lotes <=90,
restaurando cada lote no schema e colhendo, por página, os nomes de advogados que
ressurgem. A UNIÃO entre lotes reconstrói a lista completa por página.

PHASE restore: re-grava cada página (PATCH só da propriedade advogados, por NOME),
o que recria as opções de forma aditiva e devolve os valores permanentemente.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

BACKUP = Path("artifacts/_advogados_options_backup.json")
HARVEST = Path("artifacts/_advogados_recovered.json")
BATCH = 90


def client() -> NotionSessoesClient:
    key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    return NotionSessoesClient(api_key=key, data_source_id=DEFAULT_NOTION_DATA_SOURCE_ID)


def harvest() -> None:
    c = client()
    backup = json.loads(BACKUP.read_text(encoding="utf-8"))
    page_adv: dict[str, set[str]] = defaultdict(set)
    batches = [backup[i : i + BATCH] for i in range(0, len(backup), BATCH)]
    for bi, batch in enumerate(batches, start=1):
        options = [{"id": o["id"], "name": o["name"], "color": o.get("color", "default")} for o in batch]
        notion_request_with_retry(
            c, "PATCH", f"/data_sources/{c.data_source_id}",
            json={"properties": {"advogados": {"multi_select": {"options": options}}}},
        )
        pages = c.query_data_source()
        for page in pages:
            vals = page.get("properties", {}).get("advogados", {}).get("multi_select", []) or []
            for v in vals:
                name = v.get("name", "")
                if name:
                    page_adv[page["id"]].add(name)
        print(f"[harvest] lote {bi}/{len(batches)} -> paginas com advogados acumuladas: {len(page_adv)}", flush=True)
    out = {pid: sorted(names) for pid, names in page_adv.items()}
    HARVEST.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    total_vals = sum(len(v) for v in out.values())
    print(f"[harvest] DONE: {len(out)} paginas, {total_vals} valores de advogados salvos em {HARVEST}", flush=True)


def restore() -> None:
    c = client()
    data = json.loads(HARVEST.read_text(encoding="utf-8"))
    items = list(data.items())
    print(f"[restore] paginas a restaurar: {len(items)}", flush=True)
    failed = 0
    for j, (pid, names) in enumerate(items, start=1):
        try:
            notion_request_with_retry(
                c, "PATCH", f"/pages/{pid}",
                json={"properties": {"advogados": {"multi_select": [{"name": n} for n in names]}}},
            )
        except Exception as exc:
            failed += 1
            print(f"[restore] FALHA {pid}: {exc}", flush=True)
        time.sleep(0.15)
        if j % 50 == 0:
            print(f"[restore] {j}/{len(items)} (falhas={failed})", flush=True)
    print(f"[restore] DONE: {len(items)} paginas, falhas={failed}", flush=True)


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "harvest"
    if phase == "harvest":
        harvest()
    elif phase == "restore":
        restore()
    else:
        print("uso: _recover_advogados.py [harvest|restore]")
