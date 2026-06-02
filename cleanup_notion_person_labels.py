"""Padroniza os valores das colunas ``partes``, ``advogados`` e ``composicao`` no
Notion aplicando os normalizadores do projeto (que agora incluem: aliases de
ministros p/ não duplicar a mesma pessoa — ex.: Stella->Estela Aranha; padronização
de tribunais como parte -> 'TRE/XX' / 'TSE'; remoção de ruído descritivo que não é
pessoa). Só reescreve a página quando o valor normalizado difere do atual.

Para a coluna ``composicao`` (que tem <100 opções) também remove as etiquetas que
ficaram sem uso após a padronização. Para ``partes``/``advogados`` (>100 opções) a
API do Notion não permite remover opções com segurança, então apenas os VALORES das
páginas são padronizados (as etiquetas órfãs precisam ser removidas manualmente).

Uso:
    python cleanup_notion_person_labels.py            # dry-run
    python cleanup_notion_person_labels.py --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NotionSessoesClient,
    normalize_advogado_list,
    normalize_composition_list,
    normalize_party_list,
)

LOGGER = logging.getLogger("cleanup_notion_person_labels")
ARTIFACT_ROOT = Path("artifacts") / "notion_person_labels"
APPLY_SLEEP_SECONDS = 0.2
NORMALIZERS = {
    "partes": normalize_party_list,
    "advogados": normalize_advogado_list,
    "composicao": normalize_composition_list,
}


def _vals(page: dict[str, Any], col: str) -> list[str]:
    return [x.get("name", "") for x in (page.get("properties", {}).get(col, {}).get("multi_select", []) or []) if x.get("name", "")]


def build_changes(client: NotionSessoesClient) -> tuple[list[dict[str, Any]], dict[str, int]]:
    schema = client.fetch_schema()
    pages = client.query_data_source()
    changes: list[dict[str, Any]] = []
    counts = {c: 0 for c in NORMALIZERS}
    for page in pages:
        props_changed: dict[str, Any] = {}
        detail: dict[str, Any] = {}
        for col, normalizer in NORMALIZERS.items():
            if col not in schema.properties:
                continue
            current = _vals(page, col)
            if not current:
                continue
            new = normalizer(current)
            if new != current:
                props_changed[col] = {"multi_select": [{"name": n} for n in new]}
                detail[col] = {"old": current, "new": new}
                counts[col] += 1
        if props_changed:
            changes.append(
                {
                    "page_id": str(page.get("id", "")),
                    "url": str(page.get("url", "")),
                    "detail": detail,
                    "props": props_changed,
                }
            )
    return changes, counts


def apply_changes(client: NotionSessoesClient, changes: list[dict[str, Any]]) -> None:
    for i, ch in enumerate(changes, start=1):
        try:
            notion_request_with_retry(client, "PATCH", f"/pages/{ch['page_id']}", json={"properties": ch["props"]})
            ch["status"] = "updated"
        except Exception as exc:
            ch["status"] = "failed"
            ch["error"] = str(exc)
        if APPLY_SLEEP_SECONDS:
            time.sleep(APPLY_SLEEP_SECONDS)
        if i % 25 == 0:
            LOGGER.info("Aplicados: %s/%s", i, len(changes))


def cleanup_composicao_options(client: NotionSessoesClient) -> dict[str, Any]:
    """Remove etiquetas de composicao sem uso (coluna <100 -> seguro via merge)."""
    schema = client.fetch_schema()
    pages = client.query_data_source()
    raw = schema.raw_payload.get("properties", {}).get("composicao", {})
    options = raw.get("multi_select", {}).get("options", []) or []
    used = {name for page in pages for name in _vals(page, "composicao")}
    keep = [o for o in options if str(o.get("name", "")).strip() in used]
    removed = [str(o.get("name", "")).strip() for o in options if str(o.get("name", "")).strip() not in used]
    if removed and len(keep) <= 100:
        payload = [{"name": o.get("name"), "color": str(o.get("color") or "default")} for o in keep]
        notion_request_with_retry(
            client, "PATCH", f"/data_sources/{client.data_source_id}",
            json={"properties": {"composicao": {"multi_select": {"options": payload}}}},
        )
        return {"status": "patched", "removed": removed, "kept": len(keep)}
    return {"status": "noop", "removed": removed, "kept": len(keep)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Padroniza partes/advogados/composicao (nomes/ruído/tribunais).")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--clean-composicao-options", action="store_true", help="Também remove etiquetas de composicao sem uso.")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    api_key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY/NOTION_TOKEN não encontrado.")
    client = NotionSessoesClient(api_key=api_key, data_source_id=args.data_source_id)
    changes, counts = build_changes(client)
    LOGGER.info("Páginas a padronizar: %s | por coluna: %s", len(changes), counts)
    if args.apply and changes:
        apply_changes(client, changes)
    comp_opts = {}
    if args.apply and args.clean_composicao_options:
        comp_opts = cleanup_composicao_options(client)
        LOGGER.info("Limpeza de opções de composicao: %s", comp_opts)
    artifact_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", "pages_changed": len(changes), "by_column": counts,
               "applied": sum(1 for ch in changes if ch.get("status") == "updated"),
               "failed": sum(1 for ch in changes if ch.get("status") == "failed"),
               "composicao_options": comp_opts}
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Resumo: %s | relatórios em %s", json.dumps(summary, ensure_ascii=False), artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
