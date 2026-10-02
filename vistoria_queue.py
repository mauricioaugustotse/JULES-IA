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

import re
import unicodedata

from tse_youtube_notion_core import (
    ARTIFACT_ROOT,
    AnalysisResult,
    JudgmentBundleExtraction,
    NotionDataSourceSchema,
    NotionSessoesClient,
    PublishPreviewRow,
    SessionExtraction,
    assess_row_publishability,
    build_preview_rows,
    publish_preview_rows,
    validate_preview_row,
)

VISTORIA_DIR = ARTIFACT_ROOT / "vistoria"
QUEUE_FILE = VISTORIA_DIR / "vistoria_queue.jsonl"
VALID_STATUSES = {"pending", "approved", "rejected", "published", "resolved"}
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
    """Anexa candidatos e absorve alertas do monitor sobre a mesma linha."""
    existing = _read_all(queue_file)
    fresh = []
    patches = []
    for original in items:
        item = dict(original)
        if not item.get("id") or item["id"] in existing:
            continue
        if item.get("source") == "batch" and item.get("row"):
            related = [old for old in existing.values()
                       if old.get("video_id") == item.get("video_id")
                       and _same_candidate(old, item)]
            # A rerun can change an incomplete CNJ without creating a new review.
            previous = next((old for old in related if old.get("source") == "batch"), None)
            if previous:
                if previous.get("status") in {"pending", "approved"}:
                    patch = {**item, "id": previous["id"], "status": previous["status"],
                             "created_at": previous.get("created_at", item.get("created_at"))}
                    patches.append(patch)
                    existing[previous["id"]] = patch
                continue
            for old in related:
                if old.get("source") != "monitor" or old.get("status") != "pending":
                    continue
                issues = _coverage_issues(old)
                extra = {**item.get("extra", {})}
                extra.setdefault("base_reasons", list(item.get("reasons", [])))
                extra["coverage_issues"] = _unique_issues(extra.get("coverage_issues", []) + issues)
                item["extra"] = extra
                item["reasons"] = list(dict.fromkeys(item.get("reasons", []) + old.get("reasons", [])))
                patches.append({"id": old["id"], "status": "resolved",
                                "resolution_kind": "merged", "superseded_by": item["id"],
                                "resolution_note": "Alertas reunidos na revisão do processo."})
        fresh.append(item)
        existing[item["id"]] = item
    if fresh:
        _append_lines(fresh, queue_file)
    if patches:
        _append_lines(patches, queue_file)
    return len(fresh)


def _coverage_issues(item: dict[str, Any]) -> list[dict[str, Any]]:
    extra = item.get("extra") or {}
    return list(extra.get("coverage_issues") or ([extra["coverage_issue"]] if extra.get("coverage_issue") else []))


def _unique_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list({json.dumps(issue, sort_keys=True, ensure_ascii=False): issue for issue in issues}.values())


def _candidate_keys(item: dict[str, Any]) -> set[str]:
    row = item.get("row") or {}
    keys = set()
    for number in [row.get("numero_processo"), item.get("numero_hint"),
                   *[issue.get("numero_processo") for issue in _coverage_issues(item)]]:
        digits = _digits(number)
        if len(digits) >= 9:
            keys.add("cnj:" + digits[:9])
    timestamp = row.get("source_start_seconds", -1)
    if isinstance(timestamp, (int, float)) and timestamp >= 0:
        # Item index separates cited numbers emitted inside the same judgment.
        keys.add(f"position:{int(timestamp)}:{row.get('source_bundle_index', 0)}:{row.get('source_item_index', 0)}")
    return keys


def _same_candidate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    dates = [str(item.get("data_sessao") or (item.get("row") or {}).get("data_sessao") or "")
             for item in (left, right)]
    if all(dates) and dates[0] != dates[1]:
        return False
    return bool(_candidate_keys(left) & _candidate_keys(right))


def sync_monitor_issues(
    issues: list[dict[str, Any]],
    *,
    video_id: str,
    youtube_url: str,
    artifact_dir: str = "",
    data_sessao: str = "",
    rows: Optional[list[Any]] = None,
    queue_file: Path | None = None,
) -> dict[str, Any]:
    """Replace a video's active coverage snapshot, preserving human decisions.

    Call even when ``issues`` is empty. The append-only history records obsolete
    alerts as resolved. Several checks on one process become one review with its
    complete candidate row; alerts without a row remain diagnostic items.
    """
    row_data = [row.model_dump(mode="json") if hasattr(row, "model_dump") else dict(row) for row in (rows or [])]
    data_sessao = data_sessao or next((row.get("data_sessao", "") for row in row_data if row.get("data_sessao")), "")
    existing = _read_all(queue_file)
    scoped = [item for item in existing.values() if item.get("video_id") == video_id]
    groups: dict[str, dict[str, Any]] = {}
    for issue in _unique_issues(issues):
        if issue.get("severity") == "info":
            continue
        index = issue.get("row_index")
        row = row_data[index] if isinstance(index, int) and 0 <= index < len(row_data) else None
        digits = _digits(issue.get("numero_processo"))
        if row is None and len(digits) >= 9:
            matches = [candidate for candidate in row_data if _digits(candidate.get("numero_processo"))[:9] == digits[:9]]
            if len(matches) == 1:
                row = matches[0]
        # Missing a number is a reason to retain the row, not to create an empty page.
        key = ("row:" + str(row_data.index(row)) if row is not None else
               "cnj:" + digits[:9] if len(digits) >= 9 else
               "diagnostic:" + str(issue.get("code", "coverage")))
        group = groups.setdefault(key, {"row": row, "issues": []})
        group["issues"].append(issue)

    active_ids = set()
    changes = []
    counters = {"added": 0, "updated": 0, "resolved": 0}
    for key, group in groups.items():
        row, grouped_issues = group["row"], group["issues"]
        start = row.get("source_start_seconds", -1) if row else next((i.get("start_seconds") for i in grouped_issues if i.get("start_seconds") is not None), -1)
        url = str(row.get("youtube_link") or youtube_url) if row else youtube_url
        if isinstance(start, (int, float)) and start >= 0 and not re.search(r"[?&]t=", url):
            url += ("&" if "?" in url else "?") + f"t={int(start)}s"
        reasons = list(dict.fromkeys(str(issue.get("message") or issue.get("code") or "Verificar cobertura.") for issue in grouped_issues))
        item = make_vistoria_item(source="monitor", video_id=video_id, youtube_url=url,
                                 disposition="cobertura", reasons=reasons, row=row,
                                 artifact_dir=artifact_dir, data_sessao=data_sessao,
                                 extra={"coverage_issues": grouped_issues, "coverage_issue": grouped_issues[0]},
                                 dedupe_key="group:" + key)
        related = [old for old in scoped if _same_candidate(old, item)]
        batch = next((old for old in related if old.get("source") == "batch"), None)
        # A newly detected error on a published page still needs review. Keep
        # the publication record intact and open an independent monitor item.
        if batch and batch.get("status") == "published":
            batch = None
        if batch:
            active_ids.add(batch["id"])
            if batch.get("status") not in {"pending", "approved"}:
                continue
            base_reasons = (batch.get("extra") or {}).get("base_reasons", batch.get("reasons", []))
            item = {**batch, "row": row or batch.get("row"),
                    "data_sessao": data_sessao or batch.get("data_sessao", ""),
                    "reasons": list(dict.fromkeys(base_reasons + reasons)),
                    "extra": {**(batch.get("extra") or {}), **item["extra"], "base_reasons": base_reasons}}
        previous = existing.get(item["id"])
        active_ids.add(item["id"])
        if previous and previous.get("status") in {"published", "rejected", "approved"}:
            continue
        if previous:
            item["created_at"] = previous.get("created_at", item["created_at"])
            # Compare only synchronized fields; old resolution history stays visible.
            if all(previous.get(field) == item.get(field) for field in ("status", "row", "reasons", "extra", "data_sessao", "artifact_dir", "youtube_url")):
                continue
        changes.append(item)
        counters["updated" if previous else "added"] += 1
    for old in scoped:
        if old.get("source") == "monitor" and old.get("status") == "pending" and old["id"] not in active_ids:
            changes.append({"id": old["id"], "status": "resolved", "resolution_kind": "coverage_resolved",
                            "resolution_note": "Alerta resolvido ou reunido com a revisão atual do processo.",
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            counters["resolved"] += 1
        elif old.get("source") == "batch" and old.get("status") == "pending" and old["id"] not in active_ids and _coverage_issues(old):
            extra = dict(old.get("extra") or {})
            extra.pop("coverage_issue", None)
            extra.pop("coverage_issues", None)
            changes.append({"id": old["id"], "extra": extra,
                            "reasons": extra.pop("base_reasons", old.get("reasons", []))})
            counters["updated"] += 1
    if changes:
        _append_lines(changes, queue_file)
    counters["items"] = [item for item in load_items("pending", queue_file) if item.get("video_id") == video_id]
    return counters


def reconcile_published_items(
    rows: list[Any],
    results: list[dict[str, Any]],
    verification: list[dict[str, Any]],
    *,
    video_id: str,
    reconciliation: Optional[dict[str, Any]] = None,
    queue_file: Path | None = None,
) -> int:
    """Close recovered candidates and proven citations after parent readback."""
    existing = [item for item in _read_all(queue_file).values()
                if item.get("video_id") == video_id and item.get("status") in {"pending", "approved"}]
    checks = {(check.get("row_index"), check.get("page_id")): check for check in verification}
    patches = {}
    verified_rows = []
    for index, (row, result) in enumerate(zip(rows, results)):
        page_id = result.get("page_id")
        if result.get("status") not in {"created", "updated"} or not page_id:
            continue
        if checks.get((index, page_id), {}).get("status") != "verified":
            continue
        payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else dict(row)
        verified_rows.append((payload, page_id, checks[(index, page_id)]))
        candidate = {"row": payload, "data_sessao": payload.get("data_sessao", "")}
        for old in existing:
            if not _same_candidate(old, candidate):
                continue
            is_batch = old.get("source") == "batch"
            patches[old["id"]] = {
                "id": old["id"], "status": "published" if is_batch else "resolved",
                "published_page_id": page_id, "verification": checks[(index, page_id)],
                "resolution_kind": "verified_publication",
                "resolution_note": "Processo publicado automaticamente e confirmado por releitura do Notion.",
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **({"row": payload, "data_sessao": payload.get("data_sessao", "")} if is_batch else {}),
            }
    phases = (reconciliation or {}).get("phases", [reconciliation or {}])
    for phase in phases:
        for exclusion in phase.get("exclusions", []):
            if exclusion.get("code") == "institutional_act":
                payload = exclusion.get("row") or {}
                day = payload.get("data_sessao")
                if (phase.get("status") != "complete" or not day or phase.get("session_date") != day
                        or not exclusion.get("evidence")
                        or any(payload.get(k) for k in ("numero_processo", "numero_origem_video", "classe_processo", "partes", "relator"))
                        or not payload.get("source_bundle_index") or not payload.get("source_item_index")
                        or not isinstance(payload.get("source_start_seconds"), (int, float))
                        or payload["source_start_seconds"] < 0):
                    continue
                candidate = {"row": payload, "data_sessao": day}
                for old in existing:
                    if (old.get("status") != "pending" or old.get("source") not in {"batch", "monitor"}
                            or old["id"] in patches or not _same_candidate(old, candidate)):
                        continue
                    patches[old["id"]] = {
                        "id": old["id"], "status": "resolved", "resolution_kind": "institutional_exclusion",
                        "automatic_exclusion": exclusion,
                        "resolution_note": exclusion.get("reason", "Ato institucional sem julgamento individual."),
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                continue
            if exclusion.get("code") != "cited_process_number" or not exclusion.get("evidence"):
                continue
            parent_number = _digits(exclusion.get("parent_numero_processo"))
            cited_number = _digits(exclusion.get("numero_processo"))
            if len(parent_number) < 9 or len(cited_number) < 9 or parent_number[:9] == cited_number[:9]:
                continue
            parents = [(payload, page_id, check) for payload, page_id, check in verified_rows
                       if _digits(payload.get("numero_processo")) == parent_number
                       or (len(parent_number) == 9 and _digits(payload.get("numero_processo"))[:9] == parent_number)]
            if len(parents) != 1:
                continue
            parent, page_id, check = parents[0]
            candidate = {"row": exclusion.get("row") or {"numero_processo": exclusion["numero_processo"]},
                         "data_sessao": parent.get("data_sessao", "")}
            for old in existing:
                if old.get("status") != "pending" or old.get("source") not in {"batch", "monitor"}:
                    continue
                if old["id"] in patches or not _same_candidate(old, candidate):
                    continue
                patches[old["id"]] = {
                    "id": old["id"], "status": "resolved", "resolution_kind": "verified_citation",
                    "parent_page_id": page_id, "parent_numero_processo": parent["numero_processo"],
                    "verification": check, "automatic_exclusion": exclusion,
                    "resolution_note": "Número citado dentro de outro julgamento; processo principal "
                                       + parent["numero_processo"] + " confirmado no Notion (" + page_id + ").",
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
    if patches:
        _append_lines(list(patches.values()), queue_file)
    return len(patches)


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


def _digits(value) -> str:
    return re.sub(r"\D", "", str(value or ""))


def _norm_text(value) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    return re.sub(r"\s+", " ", decomposed.encode("ascii", "ignore").decode().lower()).strip()


def rebuild_row_from_artifacts(item: dict[str, Any]) -> Optional[PublishPreviewRow]:
    """Reconstrói a linha de um item SEM row (backlog antigo) a partir dos
    02_judgment_NN.json do vídeo — zero IA. Escolhe a row do item casando, nesta
    ordem: número (núcleo), timestamp do apregoamento (±90s) e tema."""
    from pathlib import Path as _Path

    video_dir = _Path(item.get("artifact_dir") or "")
    if not video_dir.exists():
        return None
    bundles: list[JudgmentBundleExtraction] = []
    for bundle_path in sorted(video_dir.glob("02_judgment_*.json")):
        try:
            bundles.append(
                JudgmentBundleExtraction.model_validate(
                    json.loads(bundle_path.read_text(encoding="utf-8", errors="ignore"))
                )
            )
        except Exception:
            continue
    if not bundles:
        return None
    data_sessao = item.get("data_sessao", "")
    composicao: list[str] = []
    for bundle in bundles:
        for it in bundle.items:
            if not composicao and it.composicao:
                composicao = list(it.composicao)
    video_id = item.get("video_id") or video_dir.name.split("_", 1)[-1]
    url = f"https://www.youtube.com/watch?v={video_id}"
    analysis = AnalysisResult(
        session=SessionExtraction(data_sessao=data_sessao, composicao=composicao),
        bundles=bundles,
    )
    rows = build_preview_rows(analysis, url, None, None)
    if not rows:
        return None

    numero_hint = _digits(
        (item.get("row") or {}).get("numero_processo")
        or ((item.get("extra") or {}).get("dje") or {}).get("numeroUnico")
        or item.get("numero_hint")
    )
    ts_match = re.search(r"[?&]t=(\d+)", str(item.get("youtube_url") or ""))
    timestamp = int(ts_match.group(1)) if ts_match else None
    tema_hint = _norm_text(item.get("tema_hint"))

    def pick() -> Optional[PublishPreviewRow]:
        if numero_hint and len(numero_hint) >= 9:
            hits = [r for r in rows if _digits(r.numero_processo)[:9] == numero_hint[:9]]
            if len(hits) == 1:
                return hits[0]
            if hits and timestamp is not None:
                timed = [r for r in hits if abs(r.source_start_seconds - timestamp) <= 90]
                if len(timed) == 1:
                    return timed[0]
        if timestamp is not None:
            hits = [r for r in rows if abs(r.source_start_seconds - timestamp) <= 90]
            if len(hits) == 1:
                return hits[0]
            if hits and tema_hint:
                themed = [r for r in hits if tema_hint[:50] in _norm_text(r.tema)]
                if len(themed) == 1:
                    return themed[0]
        if tema_hint:
            hits = [r for r in rows if tema_hint[:50] in _norm_text(r.tema)]
            if len(hits) == 1:
                return hits[0]
        return None

    row = pick()
    if row is None:
        return None
    if not _digits(row.numero_processo) and numero_hint:
        raw = str(
            (item.get("row") or {}).get("numero_processo") or item.get("numero_hint") or ""
        )
        row.numero_processo = raw
    row.add_warning("Linha reconstruída dos artifacts do backlog na aprovação em vistoria.")
    return row


def next_judgment_number_for_dates(
    client: NotionSessoesClient, schema: NotionDataSourceSchema, dates: set[str]
) -> dict[str, int]:
    """Maior N de "Julgamento N" por data (uma varredura da base)."""
    highest: dict[str, int] = {data: 0 for data in dates if data}
    if not highest:
        return {}
    for page in client.query_data_source():
        data = (client._extract_property_text(page, schema, "data_sessao") or "")[:10]
        if data not in highest:
            continue
        match = re.match(r"Julgamento\s+(\d+)", client._extract_property_text(page, schema, "tipo_registro") or "")
        if match:
            highest[data] = max(highest[data], int(match.group(1)))
    return highest


def approval_eligibility(item: dict[str, Any]) -> tuple[bool, str]:
    """Tell the UI whether approval has an identified, usable proposal to write."""
    payload = item.get("row")
    if not payload:
        return False, "Este alerta não contém uma proposta de julgamento. Abra as evidências e corrija ou recupere o processo."
    try:
        row = PublishPreviewRow.model_validate(payload)
    except (TypeError, ValueError):
        return False, "A proposta de julgamento está incompleta ou inválida."
    if any(str(error).startswith("[oficial:") for error in row.errors):
        return False, "Há divergência com o registro oficial. Corrija os campos indicados antes de publicar."
    if any(issue.get("severity") == "error" and str(issue.get("code", "")).startswith("official_")
           for issue in _coverage_issues(item)):
        return False, "Há divergência com o registro oficial. Corrija os campos indicados antes de publicar."
    # Human review can override an extraction heuristic, but cannot turn a
    # diagnostic alert containing only a process number into a judgment page.
    proposal = row.model_copy(deep=True)
    proposal.errors = []
    proposal.action = "create"
    disposition, reasons = assess_row_publishability(proposal)
    if disposition != "publish":
        return False, "; ".join(reasons) or "Faltam dados do julgamento para publicar."
    return True, "Proposta pronta para revisão e publicação."


def publish_approved_items(
    items: list[dict[str, Any]],
    notion_client: NotionSessoesClient,
    notion_schema: NotionDataSourceSchema,
    *,
    apply: bool = True,
) -> list[dict[str, Any]]:
    """Publica itens aprovados na vistoria (só os que carregam row).

    Heurísticas de extração aprovadas viram avisos de auditoria. Conflitos com
    fontes oficiais e propostas sem conteúdo suficiente precisam ser corrigidos.
    A validação é repetida antes da escrita; aprovação não ignora esses bloqueios.
    """
    publishable: list[tuple[dict[str, Any], PublishPreviewRow, bool]] = []
    results: list[dict[str, Any]] = []
    for item in items:
        row_payload = item.get("row")
        row: Optional[PublishPreviewRow] = None
        rebuilt = False
        if row_payload:
            row = PublishPreviewRow.model_validate(row_payload)
        elif item.get("disposition") in {"skipped", "blocked"}:
            # Item do backlog antigo (sem linha salva): reconstrói dos artifacts.
            row = rebuild_row_from_artifacts(item)
            rebuilt = row is not None
        if row is None:
            results.append({"id": item.get("id"), "status": "sem_row",
                            "errors": ["O alerta não contém proposta de julgamento publicável."], "warnings": []})
            continue
        eligible, explanation = approval_eligibility({**item, "row": row.model_dump(mode="json")})
        if not eligible:
            results.append({"id": item.get("id"), "status": "blocked" if apply else "dry-run:blocked",
                            "numero_processo": row.numero_processo, "errors": [explanation], "warnings": []})
            continue
        for error in row.errors:
            row.add_warning(f"{APPROVED_WARNING_PREFIX}: {error}")
        row.errors = []
        row = validate_preview_row(row, notion_schema)
        publishable.append((item, row, rebuilt))
    if not publishable:
        return results
    if not apply:
        for item, row, rebuilt in publishable:
            disposition, reasons = assess_row_publishability(row)
            results.append(
                {
                    "id": item.get("id"),
                    "status": f"dry-run:{disposition}" + (" (reconstruída)" if rebuilt else ""),
                    "numero_processo": row.numero_processo,
                    "errors": list(row.errors),
                    "warnings": reasons,
                }
            )
        return results
    # Linhas reconstruídas recebem o próximo "Julgamento N" livre da data — o N
    # original do vídeo pode colidir com julgamentos já publicados da sessão.
    rebuilt_dates = {row.data_sessao for _, row, rebuilt in publishable if rebuilt and row.data_sessao}
    if rebuilt_dates:
        highest = next_judgment_number_for_dates(notion_client, notion_schema, rebuilt_dates)
        counters: dict[str, int] = {}
        for _, row, rebuilt in publishable:
            if rebuilt and row.data_sessao in highest:
                counters[row.data_sessao] = counters.get(row.data_sessao, 0) + 1
                row.tipo_registro = f"Julgamento {highest[row.data_sessao] + counters[row.data_sessao]}"
    for item, row, _rebuilt in publishable:
        disposition, reasons = assess_row_publishability(row)
        if disposition == "publish":
            try:
                match = notion_client.find_existing_row(
                    notion_schema, row.youtube_link, row.numero_processo, row.data_sessao
                )
                row.action = "update" if match else "create"
                row.page_id = match.page_id if match else ""
                publish_results = publish_preview_rows([row], notion_client, notion_schema)
                merged = dict(publish_results[0]) if publish_results else {"status": "erro", "errors": ["sem resultado"]}
                if merged.get("status") in {"created", "updated"}:
                    from tse_workflow_monitor import verify_notion_rows
                    checks = verify_notion_rows([row], [merged], notion_client, notion_schema)
                    merged["verification"] = checks[0] if checks else {"status": "unverified", "error": "Sem releitura da página."}
            except Exception as exc:
                merged = {"numero_processo": row.numero_processo, "status": "erro",
                          "errors": [str(exc)[:1000]], "warnings": list(row.warnings)}
        else:
            merged = {"numero_processo": row.numero_processo, "status": "blocked",
                      "errors": reasons or list(row.errors), "warnings": list(row.warnings)}
        merged["id"] = item.get("id")
        merged["member_ids"] = list(item.get("member_ids") or [item.get("id")])
        results.append(merged)
    return results
