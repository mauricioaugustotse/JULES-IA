"""Varre a base do Notion e corrige a coluna ``origem`` quando ela ficou na capital
do estado por fallback do TRE, mas o texto do julgamento traz o município real.

Causa tratada: o modelo às vezes devolve a origem como o tribunal ("Tribunal
Regional Eleitoral do Pará"), que é normalizado para a capital ("Belém/PA"), em vez
do município efetivo do processo (ex.: "Santo Antônio do Tauá/PA"). O município
correto costuma estar no texto (análise/raciocínio), e é recuperado por
``infer_origin_from_row_text``.

Segurança (dry-run por padrão):
- só propõe troca quando a origem atual é uma CAPITAL de estado (sinal de fallback
  do TRE) e o município inferido é da MESMA UF;
- só aplica automaticamente quando o município inferido já existe como opção da
  coluna ``origem`` no Notion (evita criar etiquetas novas sem revisão);
- municípios inferidos fora das opções são listados à parte para revisão manual.

Uso:
    python cleanup_notion_origem.py            # dry-run (apenas relatório)
    python cleanup_notion_origem.py --apply    # aplica as trocas seguras
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import UF_CAPITALS, normalize_origem_value
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NotionSessoesClient,
    PublishPreviewRow,
    infer_origin_from_row_text,
)


LOGGER = logging.getLogger("cleanup_notion_origem")
ARTIFACT_ROOT = Path("artifacts") / "notion_origem_cleanup"
APPLY_SLEEP_SECONDS = 0.2
CAPITAL_SET = set(UF_CAPITALS.values())
INFERENCE_TEXT_FIELDS = (
    "tema",
    "punchline",
    "resultado",
    "analise_do_conteudo_juridico",
    "raciocinio_juridico",
    "fundamentacao_normativa",
    "precedentes_citados",
)


@dataclass
class OriginProposal:
    page_id: str
    page_url: str
    numero_processo: str
    old: str
    new: str
    status: str = "would_update"
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_url": self.page_url,
            "numero_processo": self.numero_processo,
            "old": self.old,
            "new": self.new,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class AuditResult:
    proposals: list[OriginProposal] = field(default_factory=list)
    needs_new_option: list[dict[str, Any]] = field(default_factory=list)
    pages_scanned: int = 0


def _page_text(client: NotionSessoesClient, schema: Any, page: dict[str, Any], field_name: str) -> str:
    if field_name not in schema.properties:
        return ""
    try:
        return client._extract_property_text(page, schema, field_name)
    except Exception:
        return ""


def _row_from_page(client: NotionSessoesClient, schema: Any, page: dict[str, Any]) -> PublishPreviewRow:
    values: dict[str, Any] = {
        "numero_processo": _page_text(client, schema, page, "numero_processo"),
        "origem": _page_text(client, schema, page, "origem"),
        "tribunal": _page_text(client, schema, page, "tribunal"),
    }
    for field_name in INFERENCE_TEXT_FIELDS:
        values[field_name] = _page_text(client, schema, page, field_name)
    return PublishPreviewRow(**values)


def _uf_of(value: str) -> str:
    match = re.search(r"/([A-Z]{2})$", value or "")
    return match.group(1) if match else ""


def build_audit(client: NotionSessoesClient) -> tuple[AuditResult, set[str]]:
    schema = client.fetch_schema()
    origem_prop = schema.raw_payload.get("properties", {}).get("origem", {})
    origem_options = {
        str(option.get("name", "")).strip()
        for option in origem_prop.get("select", {}).get("options", []) or []
        if str(option.get("name", "")).strip()
    }
    pages = client.query_data_source()
    result = AuditResult(pages_scanned=len(pages))
    for page in pages:
        row = _row_from_page(client, schema, page)
        current = normalize_origem_value(row.origem)
        inferred = normalize_origem_value(infer_origin_from_row_text(row))
        if not inferred or not re.search(r"/[A-Z]{2}$", inferred) or inferred.upper().startswith("TRE"):
            continue
        if inferred == current:
            continue
        # Só age quando a origem atual é uma capital (sinal de fallback do TRE) e o
        # município inferido é da mesma UF (evita confundir com precedente de outro estado).
        if current not in CAPITAL_SET or _uf_of(inferred) != _uf_of(current):
            continue
        page_id = str(page.get("id", ""))
        page_url = str(page.get("url", ""))
        numero = row.numero_processo
        if inferred not in origem_options:
            result.needs_new_option.append(
                {
                    "page_id": page_id,
                    "page_url": page_url,
                    "numero_processo": numero,
                    "old": current,
                    "inferred": inferred,
                }
            )
            continue
        result.proposals.append(
            OriginProposal(page_id=page_id, page_url=page_url, numero_processo=numero, old=current, new=inferred)
        )
    return result, origem_options


def apply_proposals(client: NotionSessoesClient, proposals: list[OriginProposal], *, max_pages: int = 0) -> None:
    selected = proposals[:max_pages] if max_pages > 0 else proposals
    for index, proposal in enumerate(selected, start=1):
        try:
            notion_request_with_retry(
                client,
                "PATCH",
                f"/pages/{proposal.page_id}",
                json={"properties": {"origem": {"select": {"name": proposal.new}}}},
            )
            proposal.status = "updated"
        except Exception as exc:
            proposal.status = "failed"
            proposal.error = str(exc)
        if APPLY_SLEEP_SECONDS:
            time.sleep(APPLY_SLEEP_SECONDS)
        if index % 25 == 0:
            LOGGER.info("Páginas aplicadas: %s/%s", index, len(selected))


def write_reports(artifact_dir: Path, result: AuditResult, summary: dict[str, Any]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "proposals.json").write_text(
        json.dumps([p.as_dict() for p in result.proposals], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_dir / "needs_new_option.json").write_text(
        json.dumps(result.needs_new_option, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corrige a coluna origem (capital do TRE -> município do texto).")
    parser.add_argument("--apply", action="store_true", help="Aplica as trocas seguras (padrão: apenas dry-run).")
    parser.add_argument(
        "--create-options",
        action="store_true",
        help="Também aplica os municípios inferidos que ainda não são opção da coluna (cria a etiqueta nova).",
    )
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--max-pages", type=int, default=0, help="Limita quantas páginas aplicar (0 = todas).")
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
    result, _options = build_audit(client)
    if args.create_options:
        # Promove os municípios fora das opções a propostas aplicáveis (a etiqueta
        # nova é criada automaticamente pelo Notion ao gravar o select).
        for entry in result.needs_new_option:
            result.proposals.append(
                OriginProposal(
                    page_id=str(entry.get("page_id", "")),
                    page_url=str(entry.get("page_url", "")),
                    numero_processo=str(entry.get("numero_processo", "")),
                    old=str(entry.get("old", "")),
                    new=str(entry.get("inferred", "")),
                )
            )
        result.needs_new_option = []
    LOGGER.info(
        "Páginas lidas: %s; trocas propostas: %s; municípios fora das opções: %s",
        result.pages_scanned,
        len(result.proposals),
        len(result.needs_new_option),
    )
    if args.apply and result.proposals:
        apply_proposals(client, result.proposals, max_pages=args.max_pages)
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "pages_scanned": result.pages_scanned,
        "proposals": len(result.proposals),
        "applied": sum(1 for p in result.proposals if p.status == "updated"),
        "failed": sum(1 for p in result.proposals if p.status == "failed"),
        "needs_new_option": len(result.needs_new_option),
        "examples": [p.as_dict() for p in result.proposals[:25]],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_reports(artifact_dir, result, summary)
    LOGGER.info("Relatórios gravados em %s", artifact_dir)
    LOGGER.info("Resumo: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
