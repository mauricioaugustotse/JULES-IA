# -*- coding: utf-8 -*-
"""Fila de vistoria do fluxo TSE YouTube→Notion.

Consolida itens que NÃO foram publicados automaticamente (skipped/blocked),
divergências de contagem do rito e faltantes apontados pelo DJE, para revisão
humana. A fila é um JSONL global append-only (last-status-wins por id): cada
linha é um item completo ou um patch {"id", "status", ...} aplicado na leitura.

Nada aqui publica no Notion sem chamada explícita de publish_approved_items
(disparada pelo botão "Aprovar e publicar" da GUI ou por script).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from tse_youtube_notion_core import (
    ARTIFACT_ROOT,
    NotionDataSourceSchema,
    NotionSessoesClient,
    PublishPreviewRow,
    assess_row_publishability,
    publish_preview_rows,
    validate_preview_row,
)

VISTORIA_DIR = ARTIFACT_ROOT / "vistoria"
QUEUE_FILE = VISTORIA_DIR / "vistoria_queue.jsonl"
VALID_STATUSES = {"pending", "approved", "rejected", "published"}
APPROVED_WARNING_PREFIX = "Aprovado em vistoria"


def _item_id(source: str, video_id: str, numero: str, start_seconds: int, disposition: str) -> str:
    raw = f"{source}|{video_id}|{numero}|{start_seconds}|{disposition}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def make_vistoria_item(
    *,
    source: str,
    video_id: str,
    youtube_url: str,
    disposition: str,
    reasons: list[str],
    row: Optional[dict[str, Any]] = None,
    artifact_dir: str = "",
    data_sessao: str = "",
    extra: Optional[dict[str, Any]] = None,
    dedupe_key: str = "",
) -> dict[str, Any]:
    numero = ""
    start_seconds = -1
    if row:
        numero = str(row.get("numero_processo", "") or "")
        start_seconds = int(row.get("source_start_seconds", -1) or -1)
        data_sessao = data_sessao or str(row.get("data_sessao", "") or "")
    item: dict[str, Any] = {
        "id": _item_id(source, video_id, numero or dedupe_key, start_seconds, disposition),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source,
        "video_id": video_id,
        "youtube_url": youtube_url,
        "data_sessao": data_sessao,
        "disposition": disposition,
        "reasons": [str(reason) for reason in (reasons or []) if str(reason).strip()],
        "row": row,
        "artifact_dir": str(artifact_dir or ""),
        "status": "pending",
        "published_page_id": "",
    }
    if extra:
        item["extra"] = extra
    return item


def _read_all(queue_file: Path | None = None) -> dict[str, dict[str, Any]]:
    path = queue_file or QUEUE_FILE
    merged: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return merged
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        item_id = str(payload.get("id", "") or "")
        if not item_id:
            continue
        if item_id in merged:
            merged[item_id].update(payload)
        else:
            merged[item_id] = payload
    return merged


def _append_lines(lines: list[dict[str, Any]], queue_file: Path | None = None) -> None:
    path = queue_file or QUEUE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for payload in lines:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_items(items: list[dict[str, Any]], queue_file: Path | None = None) -> int:
    """Anexa itens novos; ids já presentes (qualquer status) são pulados."""
    existing = _read_all(queue_file)
    fresh = [item for item in items if item.get("id") and item["id"] not in existing]
    if fresh:
        _append_lines(fresh, queue_file)
    return len(fresh)


def load_items(status: Optional[str] = "pending", queue_file: Path | None = None) -> list[dict[str, Any]]:
    merged = _read_all(queue_file)
    items = list(merged.values())
    if status:
        items = [item for item in items if item.get("status") == status]
    items.sort(key=lambda item: (item.get("data_sessao", ""), item.get("video_id", ""), item.get("id", "")))
    return items


def update_status(
    item_ids: list[str],
    status: str,
    extra: Optional[dict[str, Any]] = None,
    queue_file: Path | None = None,
) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"status inválido: {status}")
    patches = []
    for item_id in item_ids:
        patch: dict[str, Any] = {"id": item_id, "status": status, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if extra:
            patch.update(extra)
        patches.append(patch)
    _append_lines(patches, queue_file)


def collect_video_vistoria_items(
    rows: list[PublishPreviewRow],
    publish_results: list[dict[str, Any]],
    *,
    video_id: str,
    youtube_url: str,
    artifact_dir: str = "",
    rito_check: Optional[dict[str, Any]] = None,
    published: bool = True,
) -> list[dict[str, Any]]:
    """Itens de vistoria de um vídeo processado.

    publish_preview_rows devolve os primeiros len(rows) resultados na MESMA
    ordem das rows (reconciliações vêm depois) — o pareamento é por posição.
    Quando o lote rodou sem publicar, a disposição é recomputada offline.
    """
    items: list[dict[str, Any]] = []
    if published and publish_results:
        paired = list(zip(rows, publish_results[: len(rows)]))
        for row, result in paired:
            status = str(result.get("status", "") or "")
            if status not in {"skipped", "blocked"}:
                continue
            reasons = list(result.get("errors") or []) + list(result.get("warnings") or [])
            items.append(
                make_vistoria_item(
                    source="batch",
                    video_id=video_id,
                    youtube_url=youtube_url,
                    disposition=status,
                    reasons=reasons,
                    row=row.model_dump(mode="json"),
                    artifact_dir=artifact_dir,
                )
            )
    else:
        for row in rows:
            disposition, reasons = assess_row_publishability(row)
            if disposition not in {"skipped", "blocked"}:
                continue
            items.append(
                make_vistoria_item(
                    source="batch",
                    video_id=video_id,
                    youtube_url=youtube_url,
                    disposition=disposition,
                    reasons=list(reasons) + list(row.errors),
                    row=row.model_dump(mode="json"),
                    artifact_dir=artifact_dir,
                )
            )
    if rito_check and rito_check.get("verdict") not in (None, "ok"):
        data_sessao = rows[0].data_sessao if rows else ""
        items.append(
            make_vistoria_item(
                source="rito",
                video_id=video_id,
                youtube_url=youtube_url,
                disposition="contagem_rito",
                reasons=[
                    "Contagem do rito diverge: "
                    f"{rito_check.get('apregoamentos')} apregoamentos individuais na transcrição "
                    f"× {rito_check.get('rows')} linhas extraídas (delta {rito_check.get('delta')})."
                ],
                row=None,
                artifact_dir=artifact_dir,
                data_sessao=data_sessao,
            )
        )
    return items


def publish_approved_items(
    items: list[dict[str, Any]],
    notion_client: NotionSessoesClient,
    notion_schema: NotionDataSourceSchema,
    *,
    apply: bool = True,
) -> list[dict[str, Any]]:
    """Publica itens aprovados na vistoria (só os que carregam row).

    Erros originais viram warnings prefixados com "Aprovado em vistoria:" —
    marca que também rebaixa o error recomputável de pedido_vista em
    apply_rag_consistency_checks. Itens que permanecerem bloqueados após a
    revalidação NÃO são gravados (aparecem como blocked no retorno).
    """
    publishable: list[tuple[dict[str, Any], PublishPreviewRow]] = []
    results: list[dict[str, Any]] = []
    for item in items:
        row_payload = item.get("row")
        if not row_payload:
            results.append({"id": item.get("id"), "status": "sem_row", "errors": [], "warnings": []})
            continue
        row = PublishPreviewRow.model_validate(row_payload)
        for error in row.errors:
            row.add_warning(f"{APPROVED_WARNING_PREFIX}: {error}")
        row.errors = []
        row = validate_preview_row(row, notion_schema)
        publishable.append((item, row))
    if not publishable:
        return results
    if not apply:
        for item, row in publishable:
            disposition, reasons = assess_row_publishability(row)
            results.append(
                {
                    "id": item.get("id"),
                    "status": f"dry-run:{disposition}",
                    "numero_processo": row.numero_processo,
                    "errors": list(row.errors),
                    "warnings": reasons,
                }
            )
        return results
    publish_results = publish_preview_rows([row for _, row in publishable], notion_client, notion_schema)
    for (item, row), result in zip(publishable, publish_results[: len(publishable)]):
        merged = dict(result)
        merged["id"] = item.get("id")
        results.append(merged)
    return results
