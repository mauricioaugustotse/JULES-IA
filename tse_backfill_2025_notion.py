from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from html import unescape
from pathlib import Path
from typing import Any

import requests

from tse_normalization import (
    STATE_UF,
    UF_CAPITALS,
    build_video_only_youtube_link,
    build_timestamped_youtube_link,
    canonicalize_numero_processo,
    composicao_regimental_issue,
    dedupe_preserve_order,
    extract_chunk_judgment_process_values,
    extract_full_cnj,
    extract_uf_from_text,
    extract_youtube_video_id,
    identity_overlay_class_key,
    infer_session_date_from_video_title,
    normalize_class_text,
    normalize_classe_processo,
    normalize_eleicao_value,
    normalize_ministro_name,
    normalize_numero_processo_display,
    normalize_origem_value,
    normalize_pedido_vista_value,
    normalize_resultado_final,
    normalize_session_date_to_iso,
    normalize_tre,
    normalize_votacao,
    normalize_youtube_link,
    parse_multi_value_text,
)
from tse_youtube_notion_core import (
    ARTIFACT_ROOT,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_NEWS_GEMINI_MODEL,
    GEMINI_CALL_RETRIES,
    GENERAL_NEWS_LIMIT,
    GeminiNewsEnricher,
    GeminiProcessMetadataEnricher,
    GeminiSessionExtractor,
    GeminiThemePunchlineEnricher,
    JudgmentBundleExtraction,
    JudgmentItemExtraction,
    NotionDataSourceSchema,
    NotionSessoesClient,
    PublishPreviewRow,
    RunArtifacts,
    SessionExtraction,
    build_preview_rows,
    build_theme_repair_context,
    build_runtime_context,
    choose_preferred_composition,
    dedupe_preview_rows,
    assess_row_publishability,
    extract_ministro_roles_from_composition_entries,
    infer_classe_from_row_text,
    infer_full_numero_processo_from_row_text,
    infer_origin_from_row_text,
    infer_pedido_vista_from_row_text,
    infer_punchline_from_row_text,
    infer_relator_from_row_text,
    infer_resultado_from_row_text,
    infer_votacao_from_row_text,
    is_generic_institutional_news_url,
    normalize_advogado_list,
    normalize_composition_list,
    normalize_party_list,
    punchline_looks_generic,
    preview_row_sort_key,
    publish_preview_rows,
    repair_theme_from_text_context,
    should_replace_classe_processo,
    strip_legacy_fundamentacao_text,
    strip_legacy_raciocinio_text,
    tema_looks_generic,
    validate_preview_row,
)


LOGGER = logging.getLogger("tse_backfill_2025")
DEFAULT_PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLljYw1P54c4xZp8GKvAW8ogzKaKOSbMp7"
BACKFILL_ROOT = ARTIFACT_ROOT / "backfill_2025"
SCHEMA_SNAPSHOT_NAME = "_schema_snapshot.json"
EXISTING_PAGES_SNAPSHOT_NAME = "_existing_pages_snapshot.json"
WORKER_STDOUT_NAME = "_worker_stdout.log"
WORKER_STDERR_NAME = "_worker_stderr.log"
WORKER_PIPE_ENCODING = "utf-8"
WORKER_PIPE_ERRORS = "replace"
VIDEO_WORKER_TIMEOUT_SECONDS = int(os.getenv("BACKFILL_VIDEO_TIMEOUT_SECONDS") or "1200")
WORKER_HEARTBEAT_SECONDS = float(os.getenv("BACKFILL_WORKER_HEARTBEAT_SECONDS") or "5")
NO_PROGRESS_TIMEOUT_SECONDS = int(os.getenv("BACKFILL_NO_PROGRESS_TIMEOUT_SECONDS") or "300")
DEFAULT_MAX_WORKERS = int(os.getenv("BACKFILL_MAX_WORKERS") or "3")
DEFAULT_INITIAL_WORKERS = int(os.getenv("BACKFILL_INITIAL_WORKERS") or "3")
SCALE_UP_AFTER_HEALTHY_COMPLETIONS = int(os.getenv("BACKFILL_SCALE_UP_AFTER_DONE") or "2")
SCALE_UP_COOLDOWN_SECONDS = int(os.getenv("BACKFILL_SCALE_UP_COOLDOWN_SECONDS") or "60")
SCALE_DOWN_AFTER_CAPACITY_ERRORS = int(os.getenv("BACKFILL_SCALE_DOWN_AFTER_ERRORS") or "2")
SCALE_DOWN_WINDOW_SECONDS = int(os.getenv("BACKFILL_SCALE_DOWN_WINDOW_SECONDS") or "600")
HEALTHY_WORKER_STALENESS_SECONDS = int(os.getenv("BACKFILL_HEALTHY_WORKER_STALENESS_SECONDS") or "45")
MAX_RECENT_EVENTS = int(os.getenv("BACKFILL_MAX_RECENT_EVENTS") or "40")
MANIFEST_REPLACE_RETRIES = int(os.getenv("BACKFILL_MANIFEST_REPLACE_RETRIES") or "30")
MANIFEST_REPLACE_RETRY_SLEEP_SECONDS = float(os.getenv("BACKFILL_MANIFEST_REPLACE_RETRY_SLEEP_SECONDS") or "0.1")
RUNNER_LOCK_NAME = "runner.lock"
MAX_NEAREST_COMPOSITION_SESSION_DELTA_DAYS = 21
PLAYLIST_FETCH_ATTEMPTS = int(os.getenv("BACKFILL_PLAYLIST_FETCH_ATTEMPTS") or "4")
PLAYLIST_FETCH_RETRY_SLEEP_SECONDS = float(os.getenv("BACKFILL_PLAYLIST_FETCH_RETRY_SLEEP_SECONDS") or "1.5")


@dataclass
class PlaylistVideo:
    position: int
    video_id: str
    title: str
    url: str


@dataclass
class ExistingPageRecord:
    page_id: str
    url: str
    video_id: str
    row: PublishPreviewRow


@dataclass(frozen=True)
class IdentityArtifactTarget:
    video_id: str
    numero_processo: str
    process_key: str
    special_key: str
    session_date: str
    start_seconds: int
    tipo_registro: str
    timestamp_trusted: bool = True


@dataclass(frozen=True)
class IdentityRepairUniverse:
    targets_by_process: dict[str, list[IdentityArtifactTarget]]
    targets_by_special: dict[str, list[IdentityArtifactTarget]]
    target_by_video_process: dict[tuple[str, str], IdentityArtifactTarget]
    target_by_video_special: dict[tuple[str, str], IdentityArtifactTarget]
    existing_page_ids_by_video_process: dict[tuple[str, str], set[str]]


@dataclass
class ActiveWorker:
    video: PlaylistVideo
    process: Any
    artifact_dir: Path
    started_at: str
    started_wall_time: float
    deadline_monotonic: float
    last_seen_artifact_name: str
    last_seen_artifact_mtime: float
    last_progress_wall_time: float


@dataclass
class RepairArtifactContext:
    artifact_dir: Path | None
    session_date: str
    session_composicao: list[str]
    ordering_by_process: dict[str, tuple[int, int, int]]
    ordering_by_special_process: dict[str, tuple[int, int, int]]
    published_process_keys: set[str]
    published_special_process_keys: set[str]
    theme_text_by_process: dict[str, str]
    theme_text_by_special_process: dict[str, str]
    item_by_process: dict[str, JudgmentItemExtraction]
    item_by_special_process: dict[str, JudgmentItemExtraction]
    title_hint_by_process: dict[str, str]
    title_hint_by_special_process: dict[str, str]
    trusted_ordering_by_process: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    trusted_ordering_by_special_process: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    trusted_item_by_process: dict[str, JudgmentItemExtraction] = field(default_factory=dict)
    trusted_item_by_special_process: dict[str, JudgmentItemExtraction] = field(default_factory=dict)
    best_valid_item_composicao: list[str] = field(default_factory=list)
    valid_item_composition_by_date: dict[str, list[str]] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _truncate_output(text: str, limit: int = 4000) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[-limit:]


def resolve_worker_python(project_dir: Path | None = None) -> str:
    base_dir = project_dir or Path.cwd()
    candidates = [
        base_dir / ".venv" / "Scripts" / "python.exe",
        base_dir / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_runner_lock(root_dir: Path) -> Path:
    lock_path = root_dir / RUNNER_LOCK_NAME
    payload = {"pid": os.getpid(), "started_at": _now_iso()}
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            existing_pid = int(existing.get("pid") or 0)
            if existing_pid and process_is_alive(existing_pid):
                raise RuntimeError(
                    f"Já existe um runner do backfill em execução (pid={existing_pid})."
                )
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            return lock_path
    raise RuntimeError("Não foi possível adquirir o lock do runner.")


def release_runner_lock(lock_path: Path | None) -> None:
    if not lock_path:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def _extract_playlist_videos_from_payload(payload: Any) -> list[PlaylistVideo]:
    seen: set[str] = set()
    items: list[PlaylistVideo] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            renderer = node.get("playlistVideoRenderer")
            if renderer:
                video_id = renderer.get("videoId", "")
                title_payload = renderer.get("title") or {}
                runs = title_payload.get("runs") or []
                title = "".join(part.get("text", "") for part in runs) if runs else title_payload.get("simpleText", "")
                if video_id and video_id not in seen:
                    seen.add(video_id)
                    items.append(
                        PlaylistVideo(
                            position=len(items) + 1,
                            video_id=video_id,
                            title=unescape(title or ""),
                            url=f"https://www.youtube.com/watch?v={video_id}",
                        )
                    )
            for value in node.values():
                walk(value)
            return
        if isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return items


def _extract_playlist_payload(html: str) -> Any:
    patterns = (
        r"var ytInitialData = (\{.*?\});",
        r"window\[['\"]ytInitialData['\"]\]\s*=\s*(\{.*?\});",
    )
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return json.loads(match.group(1))
    raise RuntimeError("Não foi possível localizar ytInitialData na playlist do YouTube.")


def load_playlist_videos(playlist_url: str) -> list[PlaylistVideo]:
    last_error: Exception | None = None
    last_html_length = 0
    last_zero_like = False
    for attempt in range(1, PLAYLIST_FETCH_ATTEMPTS + 1):
        try:
            response = requests.get(
                playlist_url,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            last_html_length = len(response.text)
            payload = _extract_playlist_payload(response.text)
            items = _extract_playlist_videos_from_payload(payload)
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "Falha ao carregar a playlist (%s), tentativa %s/%s: %s",
                playlist_url,
                attempt,
                PLAYLIST_FETCH_ATTEMPTS,
                exc,
            )
        else:
            if items:
                if attempt > 1:
                    LOGGER.info(
                        "Playlist %s carregada com sucesso na tentativa %s/%s (%s vídeos).",
                        playlist_url,
                        attempt,
                        PLAYLIST_FETCH_ATTEMPTS,
                        len(items),
                    )
                return items
            last_zero_like = True
            LOGGER.warning(
                "Playlist %s retornou HTML sem itens renderizados na tentativa %s/%s (len=%s).",
                playlist_url,
                attempt,
                PLAYLIST_FETCH_ATTEMPTS,
                last_html_length,
            )
        if attempt < PLAYLIST_FETCH_ATTEMPTS:
            time.sleep(PLAYLIST_FETCH_RETRY_SLEEP_SECONDS * attempt)

    if last_error is not None:
        raise RuntimeError(
            "Falha ao carregar a playlist do YouTube após "
            f"{PLAYLIST_FETCH_ATTEMPTS} tentativas: {last_error}"
        ) from last_error
    if last_zero_like:
        raise RuntimeError(
            "A playlist do YouTube respondeu sem itens renderizados após "
            f"{PLAYLIST_FETCH_ATTEMPTS} tentativas consecutivas (último HTML len={last_html_length})."
        )
    return []


def is_relevant_2025_session(video: PlaylistVideo, year: int) -> bool:
    title = video.title.lower()
    if str(year) not in title:
        return False
    if "sessão" not in title and "sessao" not in title:
        return False
    ignore_markers = (
        "imprensa",
        "teste",
        "ensaio",
        "posse",
        "audiência",
        "audiencia",
    )
    return not any(marker in title for marker in ignore_markers)


def parse_property_text(client: NotionSessoesClient, schema: NotionDataSourceSchema, page: dict[str, Any], property_name: str) -> str:
    return client._extract_property_text(page, schema, property_name)


def notion_page_to_row(
    client: NotionSessoesClient,
    schema: NotionDataSourceSchema,
    page: dict[str, Any],
) -> PublishPreviewRow:
    materia_relation_ids = [
        item.get("id", "")
        for item in (page.get("properties", {}).get("materia_semelhante", {}).get("relation", []) or [])
        if item.get("id", "")
    ]
    row = PublishPreviewRow(
        tema=parse_property_text(client, schema, page, schema.title_property_name),
        classe_processo=parse_property_text(client, schema, page, "classe_processo"),
        tipo_registro=parse_property_text(client, schema, page, "tipo_registro"),
        eleicao=parse_property_text(client, schema, page, "eleicao"),
        origem=parse_property_text(client, schema, page, "origem"),
        tribunal=parse_property_text(client, schema, page, "tribunal"),
        numero_processo=parse_property_text(client, schema, page, "numero_processo"),
        youtube_link=parse_property_text(client, schema, page, "youtube_link"),
        relator=parse_property_text(client, schema, page, "relator"),
        pedido_vista=parse_property_text(client, schema, page, "pedido_vista"),
        resultado=parse_property_text(client, schema, page, "resultado"),
        votacao=parse_property_text(client, schema, page, "votacao"),
        data_sessao=parse_property_text(client, schema, page, "data_sessao"),
        partes=parse_multi_value_text(parse_property_text(client, schema, page, "partes")),
        advogados=parse_multi_value_text(parse_property_text(client, schema, page, "advogados")),
        composicao=parse_multi_value_text(parse_property_text(client, schema, page, "composicao")),
        punchline=parse_property_text(client, schema, page, "punchline"),
        analise_do_conteudo_juridico=parse_property_text(client, schema, page, "analise_do_conteudo_juridico"),
        fundamentacao_normativa=parse_property_text(client, schema, page, "fundamentacao_normativa"),
        precedentes_citados=parse_property_text(client, schema, page, "precedentes_citados"),
        raciocinio_juridico=parse_property_text(client, schema, page, "raciocinio_juridico"),
        resolucoes_citadas=parse_property_text(client, schema, page, "resoluções_citadas"),
        materia_semelhante=materia_relation_ids,
        noticia_TSE=parse_property_text(client, schema, page, "noticia_TSE"),
        noticia_TRE=parse_property_text(client, schema, page, "noticia_TRE"),
        noticias_gerais=[
            parse_property_text(client, schema, page, f"noticia_geral_{index}")
            for index in range(1, GENERAL_NEWS_LIMIT + 1)
            if f"noticia_geral_{index}" in schema.properties
        ],
        page_id=page.get("id", ""),
        action="update",
    )
    row.noticias_gerais = [value for value in row.noticias_gerais if value]
    row.youtube_link = normalize_youtube_link(row.youtube_link)
    row.numero_processo = normalize_numero_processo_display(row.numero_processo)
    return row


def load_existing_pages_for_year(
    client: NotionSessoesClient,
    schema: NotionDataSourceSchema,
    year: int,
    *,
    playlist_url: str = "",
) -> dict[str, list[ExistingPageRecord]]:
    grouped: dict[str, list[ExistingPageRecord]] = {}
    valid_video_ids: set[str] = set()
    if playlist_url:
        valid_video_ids = {
            video.video_id
            for video in load_playlist_videos(playlist_url)
            if is_relevant_2025_session(video, year)
        }
    for page in client.query_data_source():
        date_value = (page.get("properties", {}).get("data_sessao", {}).get("date") or {}).get("start") or ""
        youtube_link = parse_property_text(client, schema, page, "youtube_link")
        video_id = extract_youtube_video_id(youtube_link)
        if not video_id:
            continue
        if valid_video_ids:
            if video_id not in valid_video_ids:
                continue
        elif not date_value.startswith(f"{year}-"):
            continue
        record = ExistingPageRecord(
            page_id=page.get("id", ""),
            url=page.get("url", ""),
            video_id=video_id,
            row=notion_page_to_row(client, schema, page),
        )
        grouped.setdefault(video_id, []).append(record)
    return grouped


def load_existing_pages_for_news(
    client: NotionSessoesClient,
    schema: NotionDataSourceSchema,
    *,
    year: int = 0,
) -> list[ExistingPageRecord]:
    records: list[ExistingPageRecord] = []
    for page in client.query_data_source():
        row = notion_page_to_row(client, schema, page)
        date_value = normalize_session_date_to_iso(row.data_sessao)
        if year and not date_value.startswith(f"{year}-"):
            continue
        video_id = extract_youtube_video_id(row.youtube_link) or ""
        records.append(
            ExistingPageRecord(
                page_id=page.get("id", ""),
                url=page.get("url", ""),
                video_id=video_id,
                row=row,
            )
        )
    return records


def _row_match_score(new_row: PublishPreviewRow, existing: ExistingPageRecord) -> int:
    score = 0
    new_process = canonicalize_numero_processo(new_row.numero_processo)
    existing_process = canonicalize_numero_processo(existing.row.numero_processo)
    if new_process and existing_process and new_process == existing_process:
        score += 10
    if new_row.classe_processo and new_row.classe_processo == existing.row.classe_processo:
        score += 3
    if new_row.relator and new_row.relator == existing.row.relator:
        score += 2
    if new_row.resultado and new_row.resultado == existing.row.resultado:
        score += 1
    return score


def assign_existing_matches(
    rows: list[PublishPreviewRow],
    existing_pages: list[ExistingPageRecord],
) -> tuple[list[PublishPreviewRow], list[ExistingPageRecord]]:
    unused = existing_pages[:]
    matched_page_ids: set[str] = set()

    for row in rows:
        candidates = [
            record for record in unused
            if extract_youtube_video_id(record.row.youtube_link) == extract_youtube_video_id(row.youtube_link)
        ]
        if not candidates:
            continue

        scored = sorted(
            ((record, _row_match_score(row, record)) for record in candidates),
            key=lambda item: item[1],
            reverse=True,
        )
        best_record, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else -1

        should_assign = False
        if best_score >= 10:
            should_assign = True
        elif len(candidates) == 1 and len(rows) == 1:
            should_assign = True
        elif best_score >= 5 and best_score > second_score:
            should_assign = True

        if not should_assign:
            continue

        row.action = "update"
        row.page_id = best_record.page_id
        matched_page_ids.add(best_record.page_id)
        unused = [record for record in unused if record.page_id != best_record.page_id]

    return rows, [record for record in existing_pages if record.page_id not in matched_page_ids]


def should_trash_unmatched_row(
    row: PublishPreviewRow,
    enricher: GeminiProcessMetadataEnricher,
) -> tuple[bool, PublishPreviewRow]:
    candidate = row.model_copy(deep=True)
    candidate.origem = ""
    assessed = enricher.enrich_rows([candidate])[0]
    precedent_error = any("precedente citado" in error.lower() for error in assessed.errors)
    return precedent_error, assessed


def update_manifest(manifest_path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = _now_iso()
    temp_path = manifest_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    last_error: Exception | None = None
    for _ in range(max(MANIFEST_REPLACE_RETRIES, 1)):
        try:
            temp_path.replace(manifest_path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(max(MANIFEST_REPLACE_RETRY_SLEEP_SECONDS, 0.01))
    if last_error is not None:
        raise last_error


def append_manifest_event(
    manifest: dict[str, Any],
    *,
    level: str,
    message: str,
    event_type: str = "",
    video_id: str = "",
) -> None:
    events = list(manifest.get("recent_events") or [])
    events.append(
        {
            "at": _now_iso(),
            "level": level,
            "type": event_type,
            "video_id": video_id,
            "message": message,
        }
    )
    manifest["recent_events"] = events[-MAX_RECENT_EVENTS:]


def repair_manifest_false_errors(manifest: dict[str, Any], root_dir: Path) -> dict[str, Any]:
    videos = manifest.get("videos") or {}
    for video_id, entry in videos.items():
        status = entry.get("status")
        artifact_name = str(entry.get("last_artifact") or "")
        if status != "error" or artifact_name != "07_backfill_summary.json":
            continue
        position = int(entry.get("position") or 0)
        if position <= 0:
            continue
        artifact_dir = root_dir / f"{position:03d}_{video_id}"
        summary_path = artifact_dir / "07_backfill_summary.json"
        if not summary_path.exists():
            continue
        entry["status"] = "done"
        entry["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        entry["last_step"] = "done"
        entry["finished_at"] = entry.get("finished_at") or _now_iso()
        entry.pop("error", None)
        append_manifest_event(
            manifest,
            level="INFO",
            event_type="repair_false_error",
            video_id=video_id,
            message=f"{video_id} reparado para done a partir de 07_backfill_summary.json.",
        )
    return manifest


def normalize_manifest_for_resume(manifest: dict[str, Any]) -> dict[str, Any]:
    videos = manifest.get("videos") or {}
    for entry in videos.values():
        if entry.get("status") == "running":
            entry["status"] = "pending"
            entry["error"] = "Execução anterior interrompida durante status=running; item recolocado em pending."
            entry["last_step"] = "resume_reset_running"
            entry["finished_at"] = _now_iso()
    manifest["completed_at"] = ""
    return manifest


def dump_schema_snapshot(root_dir: Path, schema: NotionDataSourceSchema) -> Path:
    path = root_dir / SCHEMA_SNAPSHOT_NAME
    payload = {
        "data_source_id": schema.data_source_id,
        "raw_payload": schema.raw_payload,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_schema_snapshot(root_dir: Path) -> NotionDataSourceSchema:
    payload = json.loads((root_dir / SCHEMA_SNAPSHOT_NAME).read_text(encoding="utf-8"))
    return NotionDataSourceSchema(
        data_source_id=payload["data_source_id"],
        raw_payload=payload["raw_payload"],
    )


def dump_existing_pages_snapshot(
    root_dir: Path,
    existing_pages_by_video: dict[str, list[ExistingPageRecord]],
) -> Path:
    path = root_dir / EXISTING_PAGES_SNAPSHOT_NAME
    payload = {
        video_id: [
            {
                "page_id": record.page_id,
                "url": record.url,
                "video_id": record.video_id,
                "row": record.row.model_dump(mode="json"),
            }
            for record in records
        ]
        for video_id, records in existing_pages_by_video.items()
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_existing_pages_snapshot(root_dir: Path) -> dict[str, list[ExistingPageRecord]]:
    path = root_dir / EXISTING_PAGES_SNAPSHOT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[ExistingPageRecord]] = {}
    for video_id, records in payload.items():
        grouped[video_id] = [
            ExistingPageRecord(
                page_id=item["page_id"],
                url=item["url"],
                video_id=item["video_id"],
                row=PublishPreviewRow.model_validate(item["row"]),
            )
            for item in records
        ]
    return grouped


def _manifest_video_fallback(root_dir: Path, video_id: str) -> PlaylistVideo | None:
    manifest_path = root_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    entry = (manifest.get("videos") or {}).get(video_id) or {}
    url = str(entry.get("url") or "").strip()
    title = str(entry.get("title") or "").strip()
    position = int(entry.get("position") or 0)
    if not (url and title and position > 0):
        return None
    return PlaylistVideo(position=position, video_id=video_id, title=title, url=url)


def _find_manifest_video_fallback(
    playlist_url: str,
    year: int,
    video_id: str,
    root_dir: Path | None = None,
) -> PlaylistVideo | None:
    checked: set[Path] = set()
    candidate_dirs: list[Path] = []
    if root_dir is not None:
        candidate_dirs.append(root_dir)
    candidate_dirs.extend(iter_backfill_run_dirs(playlist_url, year))
    for candidate_dir in candidate_dirs:
        if candidate_dir in checked:
            continue
        checked.add(candidate_dir)
        fallback = _manifest_video_fallback(candidate_dir, video_id)
        if fallback is not None:
            return fallback
    return None


def find_target_video(playlist_url: str, year: int, video_id: str, root_dir: Path | None = None) -> PlaylistVideo:
    playlist_videos = [
        video for video in load_playlist_videos(playlist_url)
        if is_relevant_2025_session(video, year)
    ]
    for video in playlist_videos:
        if video.video_id == video_id:
            return video
    fallback = _find_manifest_video_fallback(playlist_url, year, video_id, root_dir=root_dir)
    if fallback is not None:
        return fallback
    raise RuntimeError(f"Vídeo {video_id} não localizado na playlist filtrada de {year}.")


def _valid_playlist_video_ids(playlist_url: str, year: int) -> set[str]:
    return {
        video.video_id
        for video in load_playlist_videos(playlist_url)
        if is_relevant_2025_session(video, year)
    }


def _video_ids_from_artifact_run_dir(run_dir: Path) -> set[str]:
    return {
        path.name.split("_", 1)[-1]
        for path in run_dir.glob("*_*")
        if path.is_dir()
    }


def _rerun_dir_matches_playlist(
    run_dir: Path,
    playlist_url: str,
    year: int,
    valid_video_ids: set[str] | None,
) -> bool:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        summary_playlist = str(payload.get("playlist_url") or "").strip()
        summary_year = int(payload.get("year") or 0)
        if summary_playlist:
            return summary_playlist == playlist_url and summary_year in {0, year}

    candidate_video_ids = _video_ids_from_artifact_run_dir(run_dir)
    if valid_video_ids is None:
        return bool(candidate_video_ids)
    return bool(candidate_video_ids & valid_video_ids)


def iter_backfill_run_dirs(playlist_url: str, year: int) -> list[Path]:
    playlist_id = extract_playlist_id(playlist_url)
    run_dirs: list[Path] = []
    current_dir = BACKFILL_ROOT / f"{year}_{playlist_id}"
    archived_root = BACKFILL_ROOT / "_archived_runs"
    archived_dirs: list[Path] = []
    if archived_root.is_dir():
        archived_dirs = sorted(
            (path for path in archived_root.glob(f"{year}_{playlist_id}*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    valid_video_ids: set[str] | None = set()
    if current_dir.is_dir():
        valid_video_ids.update(_video_ids_from_artifact_run_dir(current_dir))
    for archived_dir in archived_dirs:
        valid_video_ids.update(_video_ids_from_artifact_run_dir(archived_dir))
    if not valid_video_ids:
        try:
            valid_video_ids = _valid_playlist_video_ids(playlist_url, year)
        except Exception as exc:
            LOGGER.warning(
                "Não foi possível validar a playlist %s para filtrar reruns locais de %s; "
                "usando apenas metadados e diretórios locais: %s",
                playlist_url,
                year,
                exc,
            )
            valid_video_ids = None

    rerun_dirs = sorted(
        (
            path
            for path in BACKFILL_ROOT.glob(f"_rerun_errors_{year}_*")
            if path.is_dir() and _rerun_dir_matches_playlist(path, playlist_url, year, valid_video_ids)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    run_dirs.extend(rerun_dirs)
    if current_dir.is_dir():
        run_dirs.append(current_dir)
    run_dirs.extend(archived_dirs)
    return run_dirs


def find_artifact_dir_for_video(playlist_url: str, year: int, video_id: str) -> Path | None:
    for run_dir in iter_backfill_run_dirs(playlist_url, year):
        candidates = sorted(
            (path for path in run_dir.glob(f"*_{video_id}") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def _read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_start_refinement_seconds(path: Path) -> int:
    payload = _read_optional_json(path)
    exact = payload.get("exact_start_seconds") or payload.get("start_time")
    if exact is not None:
        try:
            return max(0, int(float(exact)))
        except Exception:
            return 0
    if path.exists():
        try:
            list_payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        if isinstance(list_payload, list):
            for item in list_payload:
                if not isinstance(item, dict):
                    continue
                exact = item.get("exact_start_seconds") or item.get("start_time")
                if exact is None:
                    continue
                try:
                    return max(0, int(float(exact)))
                except Exception:
                    return 0
    return 0


def _artifact_item_score(item: JudgmentItemExtraction) -> tuple[int, int, int]:
    populated = sum(
        1
        for value in [
            item.tema,
            item.classe_processo,
            item.origem,
            item.tre,
            item.relator,
            item.votacao,
            item.resultado_final,
            item.punchline,
            item.analise_do_conteudo_juridico,
            item.fundamentacao_normativa,
            item.raciocinio_juridico,
        ]
        if str(value or "").strip()
    )
    return (populated, len(item.composicao or []), len(item.partes or []))


def _choose_preferred_artifact_item(
    current: JudgmentItemExtraction | None,
    candidate: JudgmentItemExtraction,
) -> JudgmentItemExtraction:
    if current is None:
        return candidate
    return candidate if _artifact_item_score(candidate) > _artifact_item_score(current) else current


def _special_process_lookup_key(numero_processo: str, classe_hint: str = "") -> str:
    process_key = canonicalize_numero_processo(numero_processo)
    overlay_classe = identity_overlay_class_key(classe_hint)
    if process_key and overlay_classe:
        return f"{overlay_classe} {process_key}"

    display = normalize_numero_processo_display(numero_processo)
    if not display:
        display = str(numero_processo or "").strip().upper()
    match = re.search(r"(?i)\b(ADI|ADO)\s*(\d{1,5})\b", display)
    if match:
        return f"{match.group(1).upper()} {match.group(2).lstrip('0') or '0'}"
    digits = re.sub(r"\D", "", display)
    classe = normalize_classe_processo(classe_hint)
    if classe in {"ADI", "ADO"} and 1 <= len(digits) <= 5:
        return f"{classe} {digits.lstrip('0') or '0'}"
    return ""


def _prefer_process_map_item(
    current: JudgmentItemExtraction | None,
    candidate: JudgmentItemExtraction,
) -> JudgmentItemExtraction:
    if current is None:
        return candidate
    current_is_special = bool(_special_process_lookup_key(current.numero_processo, current.classe_processo))
    candidate_is_special = bool(_special_process_lookup_key(candidate.numero_processo, candidate.classe_processo))
    if current_is_special != candidate_is_special:
        return current if not current_is_special else candidate
    return _choose_preferred_artifact_item(current, candidate)


def _normalize_chunk_process_probe(value: str) -> str:
    normalized = normalize_class_text(unescape(str(value or "")))
    if not normalized:
        return ""
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _iter_chunk_judgment_entries(candidate_dir: Path) -> list[tuple[str, str, int]]:
    entries: list[tuple[str, str, int]] = []
    for chunk_path in sorted(candidate_dir.glob("raw_global_response_chunk_*.txt")):
        try:
            payload = json.loads(chunk_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for session_payload in payload:
            if not isinstance(session_payload, dict):
                continue
            judgments = session_payload.get("julgamentos") or []
            if not isinstance(judgments, list):
                continue
            for judgment in judgments:
                if not isinstance(judgment, dict) or judgment.get("should_ignore") is True:
                    continue
                start_seconds = 0
                try:
                    start_seconds = max(0, int(float(judgment.get("timestamp_inicial") or 0)))
                except Exception:
                    start_seconds = 0
                for raw_value in extract_chunk_judgment_process_values(judgment):
                    probe = _normalize_chunk_process_probe(str(raw_value or ""))
                    if probe:
                        entries.append((chunk_path.name, probe, start_seconds))
    return entries


def _chunk_support_for_candidate_item(
    chunk_entries: list[tuple[str, str, int]],
    *,
    item: JudgmentItemExtraction,
    bundle_title_hint: str,
    candidate_start_seconds: int,
) -> tuple[bool, bool]:
    process_key = canonicalize_numero_processo(item.numero_processo)
    short_key = _short_process_lookup_key(process_key) if process_key else ""
    special_key = _special_process_lookup_key(
        item.numero_processo,
        item.classe_processo or bundle_title_hint,
    )
    overlay_key = identity_overlay_class_key(item.classe_processo or bundle_title_hint)
    probes = {
        _normalize_chunk_process_probe(item.numero_processo),
        _normalize_chunk_process_probe(normalize_numero_processo_display(item.numero_processo)),
        _normalize_chunk_process_probe(process_key),
        _normalize_chunk_process_probe(short_key),
        _normalize_chunk_process_probe(special_key),
        _normalize_chunk_process_probe(bundle_title_hint),
    }
    probes.discard("")
    matched_chunks: set[str] = set()
    timestamp_matched = False
    for chunk_name, chunk_probe, chunk_start_seconds in chunk_entries:
        if not any(
            probe == chunk_probe or probe in chunk_probe or chunk_probe in probe
            for probe in probes
        ):
            continue
        matched_chunks.add(chunk_name)
        if candidate_start_seconds > 0 and chunk_start_seconds > 0 and chunk_start_seconds == candidate_start_seconds:
            timestamp_matched = True
    association_trusted = len(matched_chunks) >= 2 or (bool(overlay_key) and len(matched_chunks) >= 1)
    timestamp_trusted = association_trusted and timestamp_matched
    return association_trusted, timestamp_trusted


def load_repair_artifact_context(playlist_url: str, year: int, video_id: str) -> RepairArtifactContext:
    artifact_dir = find_artifact_dir_for_video(playlist_url, year, video_id)
    session_date_candidates: list[str] = []
    session_composicao_candidates: list[list[str]] = []
    best_valid_item_composicao: list[str] = []
    valid_item_composition_by_date: dict[str, list[str]] = {}
    ordering_by_process: dict[str, tuple[int, int, int]] = {}
    ordering_by_special_process: dict[str, tuple[int, int, int]] = {}
    trusted_ordering_by_process: dict[str, tuple[int, int, int]] = {}
    trusted_ordering_by_special_process: dict[str, tuple[int, int, int]] = {}
    trusted_item_by_process: dict[str, JudgmentItemExtraction] = {}
    trusted_item_by_special_process: dict[str, JudgmentItemExtraction] = {}
    published_process_keys: set[str] = set()
    published_special_process_keys: set[str] = set()
    theme_text_by_process: dict[str, str] = {}
    theme_text_by_special_process: dict[str, str] = {}
    item_by_process: dict[str, JudgmentItemExtraction] = {}
    item_by_special_process: dict[str, JudgmentItemExtraction] = {}
    title_hint_by_process: dict[str, str] = {}
    title_hint_by_special_process: dict[str, str] = {}
    authoritative_video_session_date = ""

    for run_dir in iter_backfill_run_dirs(playlist_url, year):
        snapshot_path = run_dir / EXISTING_PAGES_SNAPSHOT_NAME
        if snapshot_path.exists():
            try:
                snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except Exception:
                snapshot_payload = {}
            for item in snapshot_payload.get(video_id, []):
                row_payload = item.get("row") or {}
                composicao = parse_multi_value_text(row_payload.get("composicao", []))
                if composicao:
                    session_composicao_candidates.append(composicao)

        for candidate_dir in sorted(path for path in run_dir.glob(f"*_{video_id}") if path.is_dir()):
            playlist_video_path = candidate_dir / "00_playlist_video.json"
            if not authoritative_video_session_date and playlist_video_path.exists():
                try:
                    playlist_payload = json.loads(playlist_video_path.read_text(encoding="utf-8"))
                except Exception:
                    playlist_payload = {}
                authoritative_video_session_date = infer_session_date_from_video_title(playlist_payload.get("title", ""))
            chunk_entries = _iter_chunk_judgment_entries(candidate_dir)

            metadata_index = 0
            normalized_session_date = ""
            session_path = candidate_dir / "01_session_windows.json"
            if session_path.exists():
                session = SessionExtraction.model_validate(json.loads(session_path.read_text(encoding="utf-8")))
                normalized_session_date = normalize_session_date_to_iso(session.data_sessao)
                if normalized_session_date:
                    session_date_candidates.append(normalized_session_date)
                if session.composicao:
                    session_composicao_candidates.append(list(session.composicao))

            for bundle_path in sorted(candidate_dir.glob("02_judgment_*.json")):
                match = re.search(r"02_judgment_(\d+)\.json$", bundle_path.name)
                bundle_index = int(match.group(1)) if match else 999999
                bundle = JudgmentBundleExtraction.model_validate(json.loads(bundle_path.read_text(encoding="utf-8")))
                bundle_class_hint = normalize_classe_processo(bundle.title_hint)
                raw_text = "\n\n".join(
                    value
                    for value in [
                        _read_optional_text(candidate_dir / f"raw_detail_{bundle_index:02d}.txt"),
                        _read_optional_text(candidate_dir / f"raw_detail_transcript_{bundle_index:02d}.txt"),
                        _read_optional_text(candidate_dir / f"raw_detail_transcript_{bundle_index:02d}.input.txt"),
                    ]
                    if value.strip()
                ).strip()
                for item_index, item in enumerate(bundle.items, start=1):
                    metadata_index += 1
                    process_key = canonicalize_numero_processo(item.numero_processo)
                    special_key = _special_process_lookup_key(
                        item.numero_processo,
                        item.classe_processo or bundle.title_hint,
                    )
                    if not process_key and not special_key:
                        continue
                    candidate_item = item.model_copy(deep=True)
                    item_session_date = normalize_session_date_to_iso(candidate_item.data_sessao)
                    item_composicao = normalize_composition_list(candidate_item.composicao)
                    if item_composicao and not _composition_size_issue(item_composicao):
                        best_valid_item_composicao = choose_preferred_composition(
                            best_valid_item_composicao,
                            item_composicao,
                        )
                        if item_session_date:
                            current_valid = valid_item_composition_by_date.get(item_session_date, [])
                            valid_item_composition_by_date[item_session_date] = choose_preferred_composition(
                                current_valid,
                                item_composicao,
                            )
                    if not candidate_item.classe_processo and bundle_class_hint:
                        candidate_item.classe_processo = bundle_class_hint
                    refined_start_seconds = _read_start_refinement_seconds(
                        candidate_dir / f"raw_start_refinement_{metadata_index:02d}.txt"
                    )
                    candidate_start_seconds = int(refined_start_seconds or bundle.start_seconds or 0)
                    candidate_ordering = (candidate_start_seconds, bundle_index, item_index)
                    metadata_payload = _read_optional_json(candidate_dir / f"04a_process_metadata_{metadata_index:02d}.json")
                    parsed_metadata = metadata_payload.get("parsed") or {}
                    applied_metadata = metadata_payload.get("applied") or {}
                    metadata_session_date = normalize_session_date_to_iso(applied_metadata.get("data_sessao", ""))
                    metadata_proves_association = bool(
                        parsed_metadata.get("is_judged_process") is True
                        and metadata_session_date
                        and (
                            not authoritative_video_session_date
                            or metadata_session_date == authoritative_video_session_date
                        )
                    )
                    chunk_proves_association, chunk_proves_timestamp = _chunk_support_for_candidate_item(
                        chunk_entries,
                        item=candidate_item,
                        bundle_title_hint=bundle.title_hint,
                        candidate_start_seconds=candidate_start_seconds,
                    )
                    association_trusted = True
                    if authoritative_video_session_date:
                        association_trusted = False
                        if item_session_date == authoritative_video_session_date:
                            association_trusted = True
                        elif normalized_session_date == authoritative_video_session_date:
                            association_trusted = True
                        elif metadata_proves_association:
                            association_trusted = True
                        elif chunk_proves_association:
                            association_trusted = True
                    timestamp_trusted = False
                    if association_trusted:
                        if refined_start_seconds > 0:
                            timestamp_trusted = True
                        elif chunk_proves_timestamp:
                            timestamp_trusted = True
                        elif not authoritative_video_session_date and candidate_start_seconds > 0:
                            timestamp_trusted = True
                    if process_key:
                        item_by_process[process_key] = _prefer_process_map_item(
                            item_by_process.get(process_key),
                            candidate_item,
                        )
                        current_ordering = ordering_by_process.get(process_key)
                        if current_ordering is None or candidate_ordering < current_ordering:
                            ordering_by_process[process_key] = candidate_ordering
                        if association_trusted:
                            trusted_item_by_process[process_key] = _prefer_process_map_item(
                                trusted_item_by_process.get(process_key),
                                candidate_item,
                            )
                        if timestamp_trusted:
                            current_trusted_ordering = trusted_ordering_by_process.get(process_key)
                            if current_trusted_ordering is None or candidate_ordering < current_trusted_ordering:
                                trusted_ordering_by_process[process_key] = candidate_ordering
                        if bundle.title_hint and process_key not in title_hint_by_process:
                            title_hint_by_process[process_key] = bundle.title_hint
                        if process_key not in theme_text_by_process:
                            theme_text_by_process[process_key] = "\n".join(
                                value
                                for value in [
                                    bundle.title_hint,
                                    item.punchline,
                                    item.analise_do_conteudo_juridico,
                                    item.raciocinio_juridico,
                                    item.fundamentacao_normativa,
                                    raw_text[:4000] if raw_text else "",
                                ]
                                if str(value or "").strip()
                            ).strip()
                    if special_key:
                        item_by_special_process[special_key] = _choose_preferred_artifact_item(
                            item_by_special_process.get(special_key),
                            candidate_item,
                        )
                        current_special_ordering = ordering_by_special_process.get(special_key)
                        if current_special_ordering is None or candidate_ordering < current_special_ordering:
                            ordering_by_special_process[special_key] = candidate_ordering
                        if association_trusted:
                            trusted_item_by_special_process[special_key] = _choose_preferred_artifact_item(
                                trusted_item_by_special_process.get(special_key),
                                candidate_item,
                            )
                        if timestamp_trusted:
                            current_special_trusted_ordering = trusted_ordering_by_special_process.get(special_key)
                            if current_special_trusted_ordering is None or candidate_ordering < current_special_trusted_ordering:
                                trusted_ordering_by_special_process[special_key] = candidate_ordering
                        if bundle.title_hint and special_key not in title_hint_by_special_process:
                            title_hint_by_special_process[special_key] = bundle.title_hint
                        if special_key not in theme_text_by_special_process:
                            theme_text_by_special_process[special_key] = (
                                theme_text_by_process.get(process_key, "")
                                or "\n".join(
                                    value
                                    for value in [
                                        bundle.title_hint,
                                        item.punchline,
                                        item.analise_do_conteudo_juridico,
                                        item.raciocinio_juridico,
                                        item.fundamentacao_normativa,
                                        raw_text[:4000] if raw_text else "",
                                    ]
                                    if str(value or "").strip()
                                ).strip()
                            )

            summary_path = candidate_dir / "07_backfill_summary.json"
            if summary_path.exists():
                try:
                    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    summary_payload = {}
                for result in summary_payload.get("publish_results", []):
                    process_key = canonicalize_numero_processo(result.get("numero_processo"))
                    if process_key:
                        published_process_keys.add(process_key)
                    special_key = _special_process_lookup_key(
                        result.get("numero_processo", ""),
                        result.get("classe_processo", ""),
                    )
                    if special_key:
                        published_special_process_keys.add(special_key)

    session_composicao: list[str] = []
    for candidate in session_composicao_candidates:
        session_composicao = choose_preferred_composition(session_composicao, candidate)

    session_date = ""
    if session_date_candidates:
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        for index, candidate in enumerate(session_date_candidates):
            counts[candidate] = counts.get(candidate, 0) + 1
            first_seen.setdefault(candidate, index)
        session_date = sorted(
            counts,
            key=lambda item: (-counts[item], first_seen[item]),
        )[0]

    return RepairArtifactContext(
        artifact_dir=artifact_dir,
        session_date=session_date,
        session_composicao=session_composicao,
        best_valid_item_composicao=best_valid_item_composicao,
        valid_item_composition_by_date=valid_item_composition_by_date,
        ordering_by_process=ordering_by_process,
        ordering_by_special_process=ordering_by_special_process,
        trusted_ordering_by_process=trusted_ordering_by_process,
        trusted_ordering_by_special_process=trusted_ordering_by_special_process,
        trusted_item_by_process=trusted_item_by_process,
        trusted_item_by_special_process=trusted_item_by_special_process,
        published_process_keys=published_process_keys,
        published_special_process_keys=published_special_process_keys,
        theme_text_by_process=theme_text_by_process,
        theme_text_by_special_process=theme_text_by_special_process,
        item_by_process=item_by_process,
        item_by_special_process=item_by_special_process,
        title_hint_by_process=title_hint_by_process,
        title_hint_by_special_process=title_hint_by_special_process,
    )


def _identity_record_group_key(row: PublishPreviewRow) -> str:
    special_key = _special_process_lookup_key(
        row.numero_processo,
        row.classe_processo or infer_classe_from_row_text(row),
    )
    if special_key:
        return f"special:{special_key}"
    process_key = canonicalize_numero_processo(row.numero_processo)
    if process_key:
        return f"process:{process_key}"
    return ""


def _authoritative_video_session_date(
    *,
    video_title: str = "",
    year: int,
    artifact_session_date: str = "",
) -> str:
    title_date = infer_session_date_from_video_title(video_title)
    if title_date:
        return title_date
    artifact_date = normalize_session_date_to_iso(artifact_session_date)
    expected_prefix = f"{year}-"
    if artifact_date.startswith(expected_prefix):
        return artifact_date
    return artifact_date or ""


def _build_identity_repair_universe(
    *,
    playlist_url: str,
    year: int,
    video_ids: list[str],
    grouped: dict[str, list[ExistingPageRecord]],
    playlist_title_by_video: dict[str, str] | None = None,
) -> IdentityRepairUniverse:
    targets_by_process: dict[str, list[IdentityArtifactTarget]] = {}
    targets_by_special: dict[str, list[IdentityArtifactTarget]] = {}
    target_by_video_process: dict[tuple[str, str], IdentityArtifactTarget] = {}
    target_by_video_special: dict[tuple[str, str], IdentityArtifactTarget] = {}
    existing_page_ids_by_video_process: dict[tuple[str, str], set[str]] = {}

    for current_video_id, records in grouped.items():
        for record in records:
            process_key = canonicalize_numero_processo(record.row.numero_processo)
            if not process_key:
                continue
            existing_page_ids_by_video_process.setdefault((current_video_id, process_key), set()).add(record.page_id)
            short_key = _short_process_lookup_key(process_key)
            if short_key and short_key != process_key:
                existing_page_ids_by_video_process.setdefault((current_video_id, short_key), set()).add(record.page_id)

    for video_id in video_ids:
        artifact_context = load_repair_artifact_context(playlist_url, year, video_id)
        authoritative_session_date = _authoritative_video_session_date(
            video_title=(playlist_title_by_video or {}).get(video_id, ""),
            year=year,
            artifact_session_date=artifact_context.session_date,
        )
        staged_targets: list[tuple[tuple[int, int, int], str, str, str, str]] = []
        seen_identities: set[tuple[str, str]] = set()

        for process_key, item in artifact_context.trusted_item_by_process.items():
            if not process_key:
                continue
            special_key = _special_process_lookup_key(item.numero_processo, item.classe_processo)
            identity_key = ("process", process_key)
            if identity_key in seen_identities:
                continue
            seen_identities.add(identity_key)
            ordering = artifact_context.ordering_by_process.get(process_key) or artifact_context.ordering_by_special_process.get(
                special_key
            ) or (10**9, 10**9, 10**9)
            trusted_ordering = artifact_context.trusted_ordering_by_process.get(process_key) or artifact_context.trusted_ordering_by_special_process.get(
                special_key
            )
            session_date = authoritative_session_date
            numero_processo = normalize_numero_processo_display(item.numero_processo) or process_key
            staged_targets.append((ordering, numero_processo, process_key, special_key, session_date, trusted_ordering))

        for special_key, item in artifact_context.trusted_item_by_special_process.items():
            if not special_key:
                continue
            process_key = canonicalize_numero_processo(item.numero_processo)
            identity_key = ("special", special_key)
            if identity_key in seen_identities:
                continue
            seen_identities.add(identity_key)
            ordering = artifact_context.ordering_by_special_process.get(special_key) or artifact_context.ordering_by_process.get(
                process_key
            ) or (10**9, 10**9, 10**9)
            trusted_ordering = artifact_context.trusted_ordering_by_special_process.get(special_key) or artifact_context.trusted_ordering_by_process.get(
                process_key
            )
            session_date = authoritative_session_date
            numero_processo = normalize_numero_processo_display(item.numero_processo) or special_key
            staged_targets.append((ordering, numero_processo, process_key, special_key, session_date, trusted_ordering))

        staged_targets.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        for index, (ordering, numero_processo, process_key, special_key, session_date, trusted_ordering) in enumerate(staged_targets, start=1):
            target = IdentityArtifactTarget(
                video_id=video_id,
                numero_processo=numero_processo,
                process_key=process_key,
                special_key=special_key,
                session_date=session_date,
                start_seconds=int(trusted_ordering[0]) if trusted_ordering else 0,
                tipo_registro=f"Julgamento {index}",
                timestamp_trusted=bool(trusted_ordering),
            )
            if process_key:
                targets_by_process.setdefault(process_key, []).append(target)
                target_by_video_process.setdefault((video_id, process_key), target)
                short_key = _short_process_lookup_key(process_key)
                if short_key and short_key != process_key:
                    targets_by_process.setdefault(short_key, []).append(target)
                    target_by_video_process.setdefault((video_id, short_key), target)
            if special_key:
                targets_by_special.setdefault(special_key, []).append(target)
                target_by_video_special.setdefault((video_id, special_key), target)

    return IdentityRepairUniverse(
        targets_by_process=targets_by_process,
        targets_by_special=targets_by_special,
        target_by_video_process=target_by_video_process,
        target_by_video_special=target_by_video_special,
        existing_page_ids_by_video_process=existing_page_ids_by_video_process,
    )


def _dedupe_identity_targets(targets: list[IdentityArtifactTarget]) -> list[IdentityArtifactTarget]:
    deduped: dict[tuple[str, str, str], IdentityArtifactTarget] = {}
    for target in targets:
        key = (target.video_id, target.process_key, target.special_key)
        current = deduped.get(key)
        if current is None or target.start_seconds < current.start_seconds:
            deduped[key] = target
    return list(deduped.values())


def _select_identity_target_for_exact_video(
    row: PublishPreviewRow,
    *,
    current_video_id: str,
    identity_universe: IdentityRepairUniverse,
) -> IdentityArtifactTarget | None:
    process_key = canonicalize_numero_processo(row.numero_processo)
    special_key = _special_process_lookup_key(
        row.numero_processo,
        row.classe_processo or infer_classe_from_row_text(row),
    )
    if special_key:
        exact = identity_universe.target_by_video_special.get((current_video_id, special_key))
        if exact is not None:
            return exact
    if process_key:
        exact = identity_universe.target_by_video_process.get((current_video_id, process_key))
        if exact is not None:
            return exact
    return None


def _select_unique_identity_target(
    row: PublishPreviewRow,
    *,
    current_video_id: str,
    identity_universe: IdentityRepairUniverse,
) -> IdentityArtifactTarget | None:
    exact = _select_identity_target_for_exact_video(
        row,
        current_video_id=current_video_id,
        identity_universe=identity_universe,
    )
    if exact is not None:
        return exact

    process_key = canonicalize_numero_processo(row.numero_processo)
    special_key = _special_process_lookup_key(
        row.numero_processo,
        row.classe_processo or infer_classe_from_row_text(row),
    )
    candidates: list[IdentityArtifactTarget] = []
    if process_key:
        candidates.extend(identity_universe.targets_by_process.get(process_key, []))
    if special_key:
        candidates.extend(identity_universe.targets_by_special.get(special_key, []))
    candidates = _dedupe_identity_targets(candidates)
    if len(candidates) == 1:
        return candidates[0]

    current_session_date = normalize_session_date_to_iso(row.data_sessao)
    if current_session_date:
        same_date = _dedupe_identity_targets(
            [candidate for candidate in candidates if candidate.session_date == current_session_date]
        )
        if len(same_date) == 1:
            return same_date[0]

    return None


def _safe_normalize_origem_for_repair(value: str, tribunal: str = "") -> str:
    normalized = normalize_origem_value(value)
    if normalized:
        if not re.search(r"/[A-Z]{2}$", normalized) and not normalized.startswith("TRE/"):
            tribunal_normalized = str(tribunal or "").strip().upper()
            match = re.match(r"^TRE-([A-Z]{2})$", tribunal_normalized)
            if match:
                return f"{normalized}/{match.group(1)}"
        return normalized
    tribunal_normalized = str(tribunal or "").strip().upper()
    match = re.match(r"^TRE-([A-Z]{2})$", tribunal_normalized)
    if match:
        return UF_CAPITALS.get(match.group(1), "")
    if tribunal_normalized == "TSE":
        return UF_CAPITALS["DF"]
    return normalized or ""


def _origin_from_artifact_item(item: JudgmentItemExtraction | None) -> str:
    if item is None:
        return ""
    normalized = normalize_origem_value(item.origem)
    if normalized:
        return normalized
    tre_value = str(item.tre or "").strip().upper()
    match = re.match(r"^TRE-([A-Z]{2})$", tre_value)
    if match:
        return UF_CAPITALS.get(match.group(1), "")
    return ""


def _numero_process_specificity(value: str) -> int:
    text = normalize_numero_processo_display(value)
    if not text:
        return 0
    if re.fullmatch(r"(?:ADO|ADI)\s+\d+", text, flags=re.IGNORECASE):
        return 3
    if extract_full_cnj(text):
        return 4
    if re.fullmatch(r"\d{3,7}-\d{2}", text):
        return 2
    if re.fullmatch(r"\d+", re.sub(r"\D", "", text)):
        return 1
    return 0


def _short_process_lookup_key(value: str) -> str:
    text = normalize_numero_processo_display(value)
    match = re.search(r"(\d{3,7}-\d{2})", text)
    return match.group(1) if match else ""


def _prefer_specific_numero_processo(current: str, candidate: str) -> str:
    current_display = normalize_numero_processo_display(current)
    candidate_display = normalize_numero_processo_display(candidate)
    if not candidate_display:
        return current_display
    if not current_display:
        return candidate_display
    current_special = _special_process_lookup_key(current_display)
    candidate_special = _special_process_lookup_key(candidate_display)
    if candidate_special and current_special == candidate_special:
        return candidate_display
    current_digits = re.sub(r"\D", "", current_display)
    candidate_digits = re.sub(r"\D", "", candidate_display)
    if candidate_special and current_digits and current_digits == candidate_digits:
        return candidate_display
    current_key = canonicalize_numero_processo(current_display)
    candidate_key = canonicalize_numero_processo(candidate_display)
    if current_key and candidate_key and current_key != candidate_key:
        return current_display
    return (
        candidate_display
        if _numero_process_specificity(candidate_display) > _numero_process_specificity(current_display)
        else current_display
    )


def _origin_specificity(value: str) -> int:
    normalized = normalize_origem_value(value)
    if not normalized:
        return 0
    if re.search(r"/[A-Z]{2}$", normalized) and not normalized.startswith("TRE/"):
        return 3
    if normalized == "TSE":
        return 1
    if normalized.startswith("TRE/"):
        return 1
    return 0


def _prefer_specific_origem(current: str, candidate: str, tribunal: str = "") -> str:
    current_normalized = _safe_normalize_origem_for_repair(current, tribunal)
    candidate_normalized = _safe_normalize_origem_for_repair(candidate, tribunal)
    if not candidate_normalized:
        return current_normalized
    if not current_normalized:
        return candidate_normalized
    current_score = _origin_specificity(current_normalized)
    candidate_score = _origin_specificity(candidate_normalized)
    if candidate_score > current_score:
        return candidate_normalized
    if candidate_score == current_score and len(candidate_normalized) > len(current_normalized):
        return candidate_normalized
    return current_normalized


def _repaired_row_diff(before: PublishPreviewRow, after: PublishPreviewRow) -> dict[str, dict[str, Any]]:
    changed: dict[str, dict[str, Any]] = {}
    for field_name in [
        "tema",
        "classe_processo",
        "tipo_registro",
        "eleicao",
        "data_sessao",
        "origem",
        "numero_processo",
        "youtube_link",
        "relator",
        "resultado",
        "votacao",
        "punchline",
        "partes",
        "advogados",
        "composicao",
        "fundamentacao_normativa",
        "raciocinio_juridico",
    ]:
        before_value = getattr(before, field_name)
        after_value = getattr(after, field_name)
        if before_value != after_value:
            changed[field_name] = {"before": before_value, "after": after_value}
    return changed


FOCUSED_REPAIR_FIELDS: dict[str, set[str]] = {
    "partes-advogados": {"partes", "advogados"},
    "deterministic-core": {"relator", "resultado", "votacao", "eleicao"},
    "schema-core": {"relator", "pedido_vista", "resultado", "votacao", "eleicao", "classe_processo", "tribunal", "origem", "tipo_registro"},
    "identity-core": {"youtube_link", "data_sessao", "numero_processo", "tipo_registro"},
    "composition": {"composicao"},
}


def _row_needs_partes_advogados_repair(row: PublishPreviewRow) -> bool:
    normalized_partes = normalize_party_list(list(row.partes or []))
    normalized_advogados = normalize_advogado_list(list(row.advogados or []))
    return (
        (not row.partes)
        or (not row.advogados)
        or normalized_partes != list(row.partes or [])
        or normalized_advogados != list(row.advogados or [])
    )


def _restrict_repaired_row_to_focus(
    original: PublishPreviewRow,
    repaired: PublishPreviewRow,
    repair_focus: str,
) -> PublishPreviewRow:
    target_fields = FOCUSED_REPAIR_FIELDS.get(repair_focus)
    if not target_fields:
        return repaired
    restricted = repaired.model_copy(deep=True)
    for field_name in PublishPreviewRow.model_fields:
        if field_name in target_fields or field_name in {"page_id", "page_url", "action", "warnings", "errors", "blocked"}:
            continue
        setattr(restricted, field_name, copy.deepcopy(getattr(original, field_name)))
    restricted.clear_properties = [
        value for value in (restricted.clear_properties or [])
        if value in target_fields
    ]
    return restricted


def _extract_youtube_timestamp_seconds(youtube_link: str) -> int | None:
    match = re.search(r"[?&]t=(\d+)", str(youtube_link or ""))
    if not match:
        return None
    return int(match.group(1))


def _composition_size_issue(values: list[str]) -> str:
    return composicao_regimental_issue(normalize_composition_list(values))


def _apply_deterministic_blank_completion_from_artifact(
    original: PublishPreviewRow,
    repaired: PublishPreviewRow,
    artifact_item: JudgmentItemExtraction | None,
) -> None:
    if artifact_item is None:
        return
    artifact_relator, artifact_pedido_vista = extract_ministro_roles_from_composition_entries(
        artifact_item.composicao
    )
    artifact_relator = normalize_ministro_name(artifact_item.relator or artifact_relator)
    artifact_pedido_vista = normalize_pedido_vista_value(
        artifact_item.pedido_vista or artifact_pedido_vista
    )
    if not original.relator and artifact_relator:
        repaired.relator = artifact_relator
    if not original.pedido_vista and artifact_pedido_vista:
        repaired.pedido_vista = artifact_pedido_vista
    if not original.resultado and artifact_item.resultado_final:
        repaired.resultado = artifact_item.resultado_final
    if not original.votacao and artifact_item.votacao:
        repaired.votacao = artifact_item.votacao
    if not original.eleicao and artifact_item.eleicao:
        repaired.eleicao = artifact_item.eleicao


def _apply_schema_core_rewrite_from_artifact(
    repaired: PublishPreviewRow,
    artifact_item: JudgmentItemExtraction | None,
) -> None:
    if artifact_item is not None:
        artifact_relator, artifact_pedido_vista = extract_ministro_roles_from_composition_entries(
            artifact_item.composicao
        )
        canonical_relator = normalize_ministro_name(artifact_item.relator or artifact_relator)
        canonical_pedido_vista = normalize_pedido_vista_value(
            artifact_item.pedido_vista or artifact_pedido_vista
        )
        if canonical_relator:
            repaired.relator = canonical_relator
        if canonical_pedido_vista:
            repaired.pedido_vista = canonical_pedido_vista
        if artifact_item.resultado_final:
            repaired.resultado = normalize_resultado_final(
                artifact_item.resultado_final,
                artifact_item.classe_processo or repaired.classe_processo,
            )
        if artifact_item.votacao:
            repaired.votacao = normalize_votacao(artifact_item.votacao)
        if artifact_item.eleicao:
            repaired.eleicao = normalize_eleicao_value(artifact_item.eleicao)
        artifact_classe = _sanitize_classe_candidate(artifact_item.classe_processo)
        if artifact_classe and should_replace_classe_processo(
            repaired.classe_processo,
            artifact_classe,
            repaired,
        ):
            repaired.classe_processo = artifact_classe
        artifact_tribunal = normalize_tre(artifact_item.tre, artifact_item.uf)
        if artifact_tribunal:
            repaired.tribunal = artifact_tribunal
        repaired.origem = _prefer_specific_origem(
            repaired.origem,
            _origin_from_artifact_item(artifact_item),
            repaired.tribunal or artifact_tribunal,
        )

    if not repaired.relator:
        repaired.relator = infer_relator_from_row_text(repaired)
    if not repaired.pedido_vista:
        repaired.pedido_vista = infer_pedido_vista_from_row_text(repaired)
    if not repaired.resultado:
        repaired.resultado = infer_resultado_from_row_text(repaired)
    if not repaired.votacao:
        repaired.votacao = infer_votacao_from_row_text(repaired)
    if not repaired.tribunal:
        repaired.tribunal = normalize_tre(repaired.tribunal, extract_uf_from_text(repaired.origem))
    if repaired.classe_processo:
        repaired.classe_processo = normalize_classe_processo(repaired.classe_processo)
    if repaired.resultado:
        repaired.resultado = normalize_resultado_final(repaired.resultado, repaired.classe_processo)
    if repaired.votacao:
        repaired.votacao = normalize_votacao(repaired.votacao)
    repaired.eleicao = normalize_eleicao_value(repaired.eleicao)


def _expected_ordering_for_row(
    row: PublishPreviewRow,
    artifact_context: RepairArtifactContext,
) -> tuple[int, int, int] | None:
    process_key = canonicalize_numero_processo(row.numero_processo)
    special_key = _special_process_lookup_key(
        row.numero_processo,
        row.classe_processo or infer_classe_from_row_text(row),
    )
    return artifact_context.ordering_by_special_process.get(special_key) or artifact_context.ordering_by_process.get(
        process_key
    )


def _expected_trusted_ordering_for_row(
    row: PublishPreviewRow,
    artifact_context: RepairArtifactContext,
) -> tuple[int, int, int] | None:
    process_key = canonicalize_numero_processo(row.numero_processo)
    special_key = _special_process_lookup_key(
        row.numero_processo,
        row.classe_processo or infer_classe_from_row_text(row),
    )
    return artifact_context.trusted_ordering_by_special_process.get(special_key) or artifact_context.trusted_ordering_by_process.get(
        process_key
    )


def _duplicate_theme_key(row: PublishPreviewRow) -> str:
    return normalize_class_text(row.tema or "")


def _record_duplicate_strength(
    record: ExistingPageRecord,
    artifact_context: RepairArtifactContext,
) -> tuple[int, int, int, int]:
    row = record.row
    score = 0
    numero_text = normalize_numero_processo_display(row.numero_processo)
    if extract_full_cnj(numero_text):
        score += 5
    elif re.fullmatch(r"\d{3,7}-\d{2}", numero_text):
        score += 3
    if row.resultado:
        score += 4
    if row.votacao:
        score += 3
    if row.classe_processo:
        score += 2
    if row.relator:
        score += 1
    if row.origem:
        score += 1
    if row.punchline:
        score += 1
    if row.tema and not tema_looks_generic(row.tema, row):
        score += 2
    start_seconds = _extract_youtube_timestamp_seconds(row.youtube_link)
    expected = _expected_ordering_for_row(row, artifact_context)
    proximity_score = 0
    if start_seconds is not None and expected:
        delta = abs(start_seconds - int(expected[0]))
        if delta <= 10:
            proximity_score = 3
        elif delta <= 45:
            proximity_score = 2
        elif delta <= 120:
            proximity_score = 1
        score += proximity_score
    return (
        score,
        proximity_score,
        1 if row.votacao else 0,
        1 if row.resultado else 0,
    )


def _rows_form_safe_duplicate_group(
    records: list[ExistingPageRecord],
    artifact_context: RepairArtifactContext,
) -> bool:
    if len(records) < 2:
        return False
    tipo_values = [record.row.tipo_registro for record in records if record.row.tipo_registro]
    if len(tipo_values) != len(set(tipo_values)):
        return True
    youtube_values = [normalize_youtube_link(record.row.youtube_link) for record in records if record.row.youtube_link]
    if len(youtube_values) != len(set(youtube_values)):
        return True
    starts = [
        _extract_youtube_timestamp_seconds(record.row.youtube_link)
        for record in records
        if _extract_youtube_timestamp_seconds(record.row.youtube_link) is not None
    ]
    theme_values = {_duplicate_theme_key(record.row) for record in records if _duplicate_theme_key(record.row)}
    if len(starts) >= 2 and (max(starts) - min(starts)) <= 120 and len(theme_values) <= 1:
        return True
    expected_starts = [
        int(expected[0])
        for expected in (_expected_ordering_for_row(record.row, artifact_context) for record in records)
        if expected is not None
    ]
    if expected_starts and starts and len(theme_values) <= 1:
        anchor = min(expected_starts)
        if max(abs(start - anchor) for start in starts) <= 180:
            return True
    return False


def _split_safe_duplicate_records(
    records: list[ExistingPageRecord],
    artifact_context: RepairArtifactContext,
) -> tuple[list[ExistingPageRecord], list[dict[str, Any]]]:
    grouped: dict[str, list[ExistingPageRecord]] = {}
    for record in records:
        key = canonicalize_numero_processo(record.row.numero_processo)
        grouped.setdefault(key, []).append(record)

    kept: list[ExistingPageRecord] = []
    trashed: list[dict[str, Any]] = []
    for process_key, process_records in grouped.items():
        if len(process_records) < 2 or not _rows_form_safe_duplicate_group(process_records, artifact_context):
            kept.extend(process_records)
            continue
        ordered = sorted(
            process_records,
            key=lambda item: _record_duplicate_strength(item, artifact_context),
            reverse=True,
        )
        keeper = ordered[0]
        kept.append(keeper)
        for duplicate in ordered[1:]:
            trashed.append(
                {
                    "page_id": duplicate.page_id,
                    "numero_processo": duplicate.row.numero_processo,
                    "tipo_registro": duplicate.row.tipo_registro,
                    "youtube_link": duplicate.row.youtube_link,
                    "reason": "safe_duplicate_same_video_same_process",
                    "kept_page_id": keeper.page_id,
                    "kept_youtube_link": keeper.row.youtube_link,
                }
            )
    return kept, trashed


def _split_identity_duplicate_records(
    records: list[ExistingPageRecord],
    artifact_context: RepairArtifactContext,
) -> tuple[list[ExistingPageRecord], list[dict[str, Any]]]:
    grouped: dict[str, list[ExistingPageRecord]] = {}
    for record in records:
        key = _identity_record_group_key(record.row)
        if not key:
            grouped.setdefault(f"page:{record.page_id}", []).append(record)
            continue
        grouped.setdefault(key, []).append(record)

    kept: list[ExistingPageRecord] = []
    trashed: list[dict[str, Any]] = []
    for group_key, group_records in grouped.items():
        if len(group_records) < 2 or not group_key.startswith(("process:", "special:")):
            kept.extend(group_records)
            continue
        ordered = sorted(
            group_records,
            key=lambda item: _record_duplicate_strength(item, artifact_context),
            reverse=True,
        )
        keeper = ordered[0]
        kept.append(keeper)
        for duplicate in ordered[1:]:
            trashed.append(
                {
                    "page_id": duplicate.page_id,
                    "numero_processo": duplicate.row.numero_processo,
                    "tipo_registro": duplicate.row.tipo_registro,
                    "youtube_link": duplicate.row.youtube_link,
                    "reason": "identity_duplicate_same_video_same_process",
                    "kept_page_id": keeper.page_id,
                    "kept_youtube_link": keeper.row.youtube_link,
                }
            )
    return kept, trashed


def _row_has_local_association_proof(
    row: PublishPreviewRow,
    artifact_context: RepairArtifactContext,
) -> bool:
    has_process_maps = bool(
        artifact_context.trusted_item_by_process
        or artifact_context.trusted_ordering_by_process
        or artifact_context.trusted_item_by_special_process
        or artifact_context.trusted_ordering_by_special_process
    )
    if not has_process_maps:
        return False
    process_key = canonicalize_numero_processo(row.numero_processo)
    special_key = _special_process_lookup_key(row.numero_processo, row.classe_processo)
    return bool(
        (
            process_key
            and (
                process_key in artifact_context.trusted_item_by_process
                or process_key in artifact_context.trusted_ordering_by_process
            )
        )
        or (
            special_key
            and (
                special_key in artifact_context.trusted_item_by_special_process
                or special_key in artifact_context.trusted_ordering_by_special_process
            )
        )
    )


def _row_has_soft_local_association_signal(
    row: PublishPreviewRow,
    artifact_context: RepairArtifactContext,
) -> bool:
    has_process_maps = bool(
        artifact_context.item_by_process
        or artifact_context.ordering_by_process
        or artifact_context.item_by_special_process
        or artifact_context.ordering_by_special_process
        or artifact_context.published_process_keys
        or artifact_context.published_special_process_keys
    )
    if not has_process_maps:
        return False
    process_key = canonicalize_numero_processo(row.numero_processo)
    special_key = _special_process_lookup_key(row.numero_processo, row.classe_processo)
    return bool(
        (
            process_key
            and (
                process_key in artifact_context.item_by_process
                or process_key in artifact_context.ordering_by_process
                or process_key in artifact_context.published_process_keys
            )
        )
        or (
            special_key
            and (
                special_key in artifact_context.item_by_special_process
                or special_key in artifact_context.ordering_by_special_process
                or special_key in artifact_context.published_special_process_keys
            )
        )
    )


NOTION_REPAIR_UPDATE_RETRIES = 4
NOTION_REPAIR_UPDATE_RETRY_SLEEP_SECONDS = 2.0


def _extract_notion_status_code(exc: Exception) -> int | None:
    match = re.search(r"Notion API error (\d{3})", str(exc))
    if match:
        return int(match.group(1))
    return None


def _is_retryable_notion_update_error(exc: Exception) -> bool:
    return _extract_notion_status_code(exc) in {429, 500, 502, 503, 504}


def update_notion_row_with_retry(
    notion_client: NotionSessoesClient,
    notion_schema: NotionDataSourceSchema,
    page_id: str,
    row: PublishPreviewRow,
    *,
    retries: int = NOTION_REPAIR_UPDATE_RETRIES,
    sleep_seconds: float = NOTION_REPAIR_UPDATE_RETRY_SLEEP_SECONDS,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return notion_client.update_row(notion_schema, page_id, row)
        except Exception as exc:
            last_error = exc
            if attempt >= retries or not _is_retryable_notion_update_error(exc):
                raise
            LOGGER.warning(
                "Falha transitória ao atualizar página %s no Notion (tentativa %s/%s): %s",
                page_id,
                attempt,
                retries,
                exc,
            )
            time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"Falha ao atualizar página {page_id} no Notion: {last_error}") from last_error


def build_news_only_properties_payload(
    schema: NotionDataSourceSchema,
    row: PublishPreviewRow,
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    if "noticia_TSE" in schema.properties:
        properties["noticia_TSE"] = {"url": row.noticia_TSE or None}
    if "noticia_TRE" in schema.properties:
        properties["noticia_TRE"] = {"url": row.noticia_TRE or None}
    for index in range(1, GENERAL_NEWS_LIMIT + 1):
        property_name = f"noticia_geral_{index}"
        if property_name not in schema.properties:
            continue
        url = row.noticias_gerais[index - 1] if index <= len(row.noticias_gerais) else ""
        properties[property_name] = {"url": url or None}
    return properties


def update_notion_news_fields_with_retry(
    notion_client: NotionSessoesClient,
    notion_schema: NotionDataSourceSchema,
    page_id: str,
    row: PublishPreviewRow,
    *,
    retries: int = NOTION_REPAIR_UPDATE_RETRIES,
    sleep_seconds: float = NOTION_REPAIR_UPDATE_RETRY_SLEEP_SECONDS,
) -> dict[str, Any]:
    payload = {"properties": build_news_only_properties_payload(notion_schema, row)}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return notion_client._request("PATCH", f"/pages/{page_id}", json=payload)
        except Exception as exc:
            last_error = exc
            if attempt >= retries or not _is_retryable_notion_update_error(exc):
                raise
            LOGGER.warning(
                "Falha transitória ao atualizar notícias da página %s no Notion (tentativa %s/%s): %s",
                page_id,
                attempt,
                retries,
                exc,
            )
            time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"Falha ao atualizar notícias da página {page_id}: {last_error}") from last_error


SCHEMA_RESIDUE_PROPERTIES = [
    "probe_expand_default_large",
    "partes_default_tmp",
    "advogados_default_tmp",
    "partes__default_tmp",
    "partes__legacy_color",
    "advogados__default_tmp",
    "advogados__legacy_color",
]
FORCE_REMOVE_SCHEMA_RESIDUE_PROPERTIES = {
    "partes_default_tmp",
    "advogados_default_tmp",
    "partes__default_tmp",
    "partes__legacy_color",
    "advogados__default_tmp",
    "advogados__legacy_color",
}


def _property_payload_has_value(prop: dict[str, Any] | None) -> bool:
    if not prop:
        return False
    prop_type = prop.get("type")
    if prop_type == "title":
        return bool(prop.get("title"))
    if prop_type == "rich_text":
        return bool(prop.get("rich_text"))
    if prop_type == "multi_select":
        return bool(prop.get("multi_select"))
    if prop_type == "select":
        return bool((prop.get("select") or {}).get("name"))
    if prop_type == "status":
        return bool((prop.get("status") or {}).get("name"))
    if prop_type == "url":
        return bool(prop.get("url"))
    if prop_type == "date":
        return bool(prop.get("date"))
    if prop_type == "checkbox":
        return bool(prop.get("checkbox"))
    if prop_type == "number":
        return prop.get("number") is not None
    return False


def cleanup_notion_schema_residue(
    notion_client: NotionSessoesClient,
    *,
    normalize_colors: bool = True,
    apply_changes: bool = True,
) -> dict[str, Any]:
    if not hasattr(notion_client, "_request") or not hasattr(notion_client, "query_data_source"):
        return {"removed_properties": [], "blocked_properties": [], "color_summary": {"updated": False, "properties": []}}
    payload = notion_client._request("GET", f"/data_sources/{notion_client.data_source_id}")
    schema = NotionDataSourceSchema(notion_client.data_source_id, payload)
    pages = notion_client.query_data_source()
    removed: list[str] = []
    blocked: list[dict[str, Any]] = []
    for property_name in SCHEMA_RESIDUE_PROPERTIES:
        if property_name not in schema.properties:
            continue
        used_by = [
            page.get("id", "")
            for page in pages
            if _property_payload_has_value((page.get("properties") or {}).get(property_name))
        ]
        if property_name in FORCE_REMOVE_SCHEMA_RESIDUE_PROPERTIES:
            if apply_changes:
                notion_client._request(
                    "PATCH",
                    f"/data_sources/{notion_client.data_source_id}",
                    json={"properties": {property_name: None}},
                )
            removed.append(property_name)
            continue
        if used_by:
            blocked.append({"property": property_name, "used_count": len(used_by), "used_by_pages": used_by[:10]})
            continue
        if apply_changes:
            notion_client._request(
                "PATCH",
                f"/data_sources/{notion_client.data_source_id}",
                json={"properties": {property_name: None}},
            )
        removed.append(property_name)
    color_summary: dict[str, Any] = {"updated": False, "properties": []}
    if normalize_colors:
        refreshed_payload = notion_client._request("GET", f"/data_sources/{notion_client.data_source_id}")
        refreshed_schema = NotionDataSourceSchema(notion_client.data_source_id, refreshed_payload)
        property_summaries: list[dict[str, Any]] = []
        for property_name in ["partes", "advogados"]:
            prop = refreshed_schema.raw_payload.get("properties", {}).get(property_name, {})
            if prop.get("type") != "multi_select":
                continue
            options = prop.get("multi_select", {}).get("options", []) or []
            nondefault_options = sum(1 for option in options if (option.get("color") or "") != "default")
            used_option_names: set[str] = set()
            for page in pages:
                for item in page.get("properties", {}).get(property_name, {}).get("multi_select", []):
                    name = item.get("name", "").strip()
                    if name:
                        used_option_names.add(name)
            property_summaries.append(
                {
                    "property": property_name,
                    "nondefault_options": nondefault_options,
                    "used_options": len(used_option_names),
                    "retro_normalization_supported": False,
                }
            )
        color_summary = {
            "updated": False,
            "apply_changes_requested": apply_changes,
            "retro_normalization_supported": False,
            "properties": property_summaries,
        }
    return {"removed_properties": removed, "blocked_properties": blocked, "color_summary": color_summary}


def _raw_origem_looks_invalid(value: str) -> bool:
    raw = normalize_class_text(value)
    if not raw:
        return True
    if raw in STATE_UF:
        return True
    if raw == "tse" or re.fullmatch(r"(?:tre|tse)\s+[a-z]{2}", raw):
        return True
    invalid_markers = [
        "tribunal de justica",
        "decisoes do tre",
        "decisões do tre",
        "jurisprudencia do tre",
        "jurisprudência do tre",
        "municipal de",
        "titular do tre",
        "suplente do tre",
        "eleitoral de ",
        "tse/",
        "tse-",
    ]
    return any(marker in raw for marker in invalid_markers)


def _row_has_wrong_youtube_video(row: PublishPreviewRow, expected_video_id: str) -> bool:
    actual_video_id = extract_youtube_video_id(row.youtube_link or "")
    return bool(actual_video_id and expected_video_id and actual_video_id != expected_video_id)


def _votacao_is_inconsistent(row: PublishPreviewRow) -> bool:
    if row.resultado == "Suspenso mas julgado depois":
        return row.votacao != "Suspenso*"
    if row.votacao == "Suspenso*":
        return row.resultado != "Suspenso mas julgado depois"
    if row.resultado == "Suspenso por vista" and row.votacao not in {"", "Suspenso"}:
        return True
    if row.votacao == "Suspenso" and row.resultado and row.resultado != "Suspenso por vista":
        return True
    return False


SUSPENSO_JULGADO_DEPOIS_RESULTADO = "Suspenso mas julgado depois"
SUSPENSO_JULGADO_DEPOIS_VOTACAO = "Suspenso*"


def _suspended_resolution_process_key(row: PublishPreviewRow) -> str:
    return (
        canonicalize_numero_processo(row.numero_processo)
        or _short_process_lookup_key(row.numero_processo)
        or _special_process_lookup_key(row.numero_processo, row.classe_processo)
    )


def _is_suspended_for_later_resolution(row: PublishPreviewRow) -> bool:
    return row.resultado == "Suspenso por vista" or row.votacao == "Suspenso"


def _is_definitively_resolved_record(row: PublishPreviewRow) -> bool:
    resultado = normalize_resultado_final(row.resultado, row.classe_processo)
    votacao = normalize_votacao(row.votacao)
    if not resultado:
        return False
    if resultado in {"Suspenso por vista", SUSPENSO_JULGADO_DEPOIS_RESULTADO}:
        return False
    if votacao in {"Suspenso", SUSPENSO_JULGADO_DEPOIS_VOTACAO}:
        return False
    return True


def build_suspended_later_resolution_targets(
    grouped: dict[str, list[ExistingPageRecord]],
) -> dict[str, dict[str, str]]:
    entries_by_process: dict[str, list[dict[str, Any]]] = {}
    for video_id, records in grouped.items():
        for position, record in enumerate(records):
            process_key = _suspended_resolution_process_key(record.row)
            session_date = normalize_session_date_to_iso(record.row.data_sessao)
            if not process_key or not session_date:
                continue
            entries_by_process.setdefault(process_key, []).append(
                {
                    "video_id": video_id,
                    "position": position,
                    "record": record,
                    "session_date": session_date,
                }
            )

    targets: dict[str, dict[str, str]] = {}
    for process_key, entries in entries_by_process.items():
        definitive_entries = sorted(
            [
                entry
                for entry in entries
                if _is_definitively_resolved_record(entry["record"].row)
            ],
            key=lambda entry: (entry["session_date"], entry["video_id"], entry["position"]),
        )
        if not definitive_entries:
            continue
        for entry in entries:
            record = entry["record"]
            if not _is_suspended_for_later_resolution(record.row):
                continue
            later_entry = next(
                (
                    definitive
                    for definitive in definitive_entries
                    if definitive["session_date"] > entry["session_date"]
                ),
                None,
            )
            if later_entry is None:
                continue
            later_record = later_entry["record"]
            targets[record.page_id] = {
                "resultado": SUSPENSO_JULGADO_DEPOIS_RESULTADO,
                "votacao": SUSPENSO_JULGADO_DEPOIS_VOTACAO,
                "process_key": process_key,
                "later_page_id": later_record.page_id,
                "later_video_id": later_entry["video_id"],
                "later_data_sessao": later_entry["session_date"],
            }
    return targets


def _apply_suspended_later_resolution_marker(
    row: PublishPreviewRow,
    page_id: str,
    targets: dict[str, dict[str, str]] | None,
) -> bool:
    if not targets or page_id not in targets:
        return False
    row.resultado = SUSPENSO_JULGADO_DEPOIS_RESULTADO
    row.votacao = SUSPENSO_JULGADO_DEPOIS_VOTACAO
    return True


def _sanitize_classe_candidate(candidate: str) -> str:
    normalized = normalize_classe_processo(candidate)
    if not normalized:
        return ""
    if normalized in {"ADI", "ADO"}:
        return ""
    if re.search(r"\d", normalized):
        return ""
    if len(normalized) > 30:
        return ""
    return normalized


def _classe_mismatch_candidate(
    row: PublishPreviewRow,
    artifact_context: RepairArtifactContext | None = None,
) -> str:
    process_key = canonicalize_numero_processo(row.numero_processo)
    special_key = _special_process_lookup_key(row.numero_processo, row.classe_processo)
    if artifact_context is not None:
        artifact_item = artifact_context.item_by_process.get(process_key) or artifact_context.item_by_special_process.get(
            special_key
        )
        if artifact_item and artifact_item.classe_processo:
            sanitized = _sanitize_classe_candidate(artifact_item.classe_processo)
            if sanitized:
                return sanitized
        title_hint = artifact_context.title_hint_by_process.get(process_key) or artifact_context.title_hint_by_special_process.get(
            special_key
        )
        if title_hint:
            sanitized = _sanitize_classe_candidate(title_hint)
            if sanitized:
                return sanitized
    return _sanitize_classe_candidate(infer_classe_from_row_text(row))


def _choose_authoritative_repair_session_date(
    current_date: str,
    *,
    year: int,
    session_date_hint: str = "",
    artifact_session_date: str = "",
) -> str:
    current = normalize_session_date_to_iso(current_date)
    hint = normalize_session_date_to_iso(session_date_hint)
    artifact = normalize_session_date_to_iso(artifact_session_date)
    expected_prefix = f"{year}-"
    if hint:
        return hint
    if current.startswith(expected_prefix):
        return current
    if artifact.startswith(expected_prefix):
        return artifact
    return current or artifact or hint


def _target_youtube_link(target: IdentityArtifactTarget) -> str:
    base = f"https://www.youtube.com/watch?v={target.video_id}"
    if target.timestamp_trusted and target.start_seconds > 0:
        return build_timestamped_youtube_link(base, target.start_seconds)
    return build_video_only_youtube_link(base)


def repair_existing_video_rows(
    *,
    video_id: str,
    records: list[ExistingPageRecord],
    notion_client: NotionSessoesClient,
    notion_schema: NotionDataSourceSchema,
    playlist_url: str,
    year: int,
    gemini_api_key: str,
    model: str,
    use_theme_api: bool = True,
    session_date_hint: str = "",
    repair_focus: str = "all",
    apply_updates: bool = True,
    best_composition_by_session_date: dict[str, list[str]] | None = None,
    identity_universe: IdentityRepairUniverse | None = None,
    suspended_later_resolution_targets: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    artifact_context = load_repair_artifact_context(playlist_url, year, video_id)
    artifact_store = RunArtifacts(artifact_context.artifact_dir) if artifact_context.artifact_dir else None
    original_rows_count = len(records)
    if repair_focus == "identity-core":
        records, duplicate_pages = _split_identity_duplicate_records(records, artifact_context)
    else:
        records, duplicate_pages = _split_safe_duplicate_records(records, artifact_context)
    duplicate_trash_failures: list[dict[str, Any]] = []
    trashed_unproven_pages: list[dict[str, Any]] = []
    failed_unproven_trash: list[dict[str, Any]] = []
    if apply_updates:
        for duplicate in duplicate_pages:
            try:
                notion_client._request("PATCH", f"/pages/{duplicate['page_id']}", json={"in_trash": True})
            except Exception as exc:
                duplicate_trash_failures.append(
                    {
                        **duplicate,
                        "error": str(exc),
                    }
                )
            else:
                LOGGER.info(
                    "Página duplicada segura enviada para a lixeira: %s (%s) em %s",
                    duplicate["page_id"],
                    duplicate["numero_processo"],
                    video_id,
                )
    repaired_rows: list[PublishPreviewRow] = []
    per_page: list[dict[str, Any]] = []
    review_pages: list[dict[str, Any]] = []
    group_best_composition = artifact_context.session_composicao
    for record in records:
        group_best_composition = choose_preferred_composition(group_best_composition, record.row.composicao)

    for record in records:
        if repair_focus == "identity-core":
            original = record.row.model_copy(deep=True)
            repaired = record.row.model_copy(deep=True)
            target = None
            if identity_universe is not None:
                target = _select_unique_identity_target(
                    repaired,
                    current_video_id=video_id,
                    identity_universe=identity_universe,
                )
            if target is None:
                review_reason = "identity_unproven"
                review_pages.append(
                    {
                        "page_id": record.page_id,
                        "url": record.url,
                        "numero_processo": record.row.numero_processo,
                        "reason": review_reason,
                    }
                )
                per_page.append(
                    {
                        "page_id": record.page_id,
                        "url": record.url,
                        "numero_processo": record.row.numero_processo,
                    }
                )
                if apply_updates:
                    try:
                        notion_client._request("PATCH", f"/pages/{record.page_id}", json={"in_trash": True})
                    except Exception as exc:
                        failed_unproven_trash.append(
                            {
                                "page_id": record.page_id,
                                "url": record.url,
                                "numero_processo": record.row.numero_processo,
                                "reason": review_reason,
                                "error": str(exc),
                            }
                        )
                    else:
                        trashed_unproven_pages.append(
                            {
                                "page_id": record.page_id,
                                "url": record.url,
                                "numero_processo": record.row.numero_processo,
                                "reason": review_reason,
                            }
                        )
                continue

            destination_duplicate = False
            if (
                identity_universe is not None
                and target.process_key
                and target.video_id != video_id
            ):
                destination_pages = identity_universe.existing_page_ids_by_video_process.get(
                    (target.video_id, target.process_key),
                    set(),
                )
                destination_duplicate = bool(destination_pages - {record.page_id})
            if destination_duplicate:
                review_reason = "identity_destination_duplicate"
                review_pages.append(
                    {
                        "page_id": record.page_id,
                        "url": record.url,
                        "numero_processo": record.row.numero_processo,
                        "reason": review_reason,
                        "target_video_id": target.video_id,
                    }
                )
                per_page.append(
                    {
                        "page_id": record.page_id,
                        "url": record.url,
                        "numero_processo": record.row.numero_processo,
                    }
                )
                if apply_updates:
                    try:
                        notion_client._request("PATCH", f"/pages/{record.page_id}", json={"in_trash": True})
                    except Exception as exc:
                        failed_unproven_trash.append(
                            {
                                "page_id": record.page_id,
                                "url": record.url,
                                "numero_processo": record.row.numero_processo,
                                "reason": review_reason,
                                "error": str(exc),
                            }
                        )
                    else:
                        trashed_unproven_pages.append(
                            {
                                "page_id": record.page_id,
                                "url": record.url,
                                "numero_processo": record.row.numero_processo,
                                "reason": review_reason,
                            }
                        )
                continue

            repaired.numero_processo = _prefer_specific_numero_processo(
                repaired.numero_processo,
                target.numero_processo,
            )
            if target.session_date:
                repaired.data_sessao = target.session_date
            repaired.youtube_link = _target_youtube_link(target)
            if target.timestamp_trusted or target.video_id != video_id or not repaired.tipo_registro:
                repaired.tipo_registro = target.tipo_registro
            repaired = validate_preview_row(repaired, notion_schema)
            repaired = _restrict_repaired_row_to_focus(original, repaired, repair_focus)
            repaired_rows.append(repaired)
            per_page.append(
                {
                    "page_id": record.page_id,
                    "url": record.url,
                    "numero_processo": repaired.numero_processo,
                }
            )
            continue

        proof_failed = not _row_has_soft_local_association_signal(record.row, artifact_context)
        wrong_video = _row_has_wrong_youtube_video(record.row, video_id)
        if proof_failed or wrong_video:
            per_page.append(
                {
                    "page_id": record.page_id,
                    "url": record.url,
                    "numero_processo": record.row.numero_processo,
                }
            )
            review_reason = "youtube_video_mismatch" if wrong_video else "association_unproven"
            review_pages.append(
                {
                    "page_id": record.page_id,
                    "url": record.url,
                    "numero_processo": record.row.numero_processo,
                    "reason": review_reason,
                }
            )
            if apply_updates:
                try:
                    notion_client._request("PATCH", f"/pages/{record.page_id}", json={"in_trash": True})
                except Exception as exc:
                    failed_unproven_trash.append(
                        {
                            "page_id": record.page_id,
                            "url": record.url,
                            "numero_processo": record.row.numero_processo,
                            "reason": review_reason,
                            "error": str(exc),
                        }
                    )
                else:
                    trashed_unproven_pages.append(
                        {
                            "page_id": record.page_id,
                            "url": record.url,
                            "numero_processo": record.row.numero_processo,
                            "reason": review_reason,
                        }
                    )
            continue
        original = record.row.model_copy(deep=True)
        repaired = record.row.model_copy(deep=True)
        original_composition_issue = _composition_size_issue(original.composicao)
        process_key = canonicalize_numero_processo(repaired.numero_processo)
        special_key = _special_process_lookup_key(
            repaired.numero_processo,
            repaired.classe_processo or infer_classe_from_row_text(repaired),
        )
        artifact_item = artifact_context.item_by_process.get(process_key) or artifact_context.item_by_special_process.get(
            special_key
        )
        _apply_deterministic_blank_completion_from_artifact(original, repaired, artifact_item)
        if repair_focus == "schema-core":
            _apply_schema_core_rewrite_from_artifact(repaired, artifact_item)
        effective_session_date = _choose_authoritative_repair_session_date(
            repaired.data_sessao,
            year=year,
            session_date_hint=session_date_hint,
            artifact_session_date=artifact_context.session_date,
        )
        if effective_session_date:
            repaired.data_sessao = effective_session_date
        current_session_date = normalize_session_date_to_iso(repaired.data_sessao or original.data_sessao)
        best_session_composition = []
        if best_composition_by_session_date:
            best_session_composition = best_composition_by_session_date.get(current_session_date) or []
        repaired.origem = _safe_normalize_origem_for_repair(repaired.origem, repaired.tribunal)
        repaired.fundamentacao_normativa = strip_legacy_fundamentacao_text(repaired.fundamentacao_normativa)
        repaired.raciocinio_juridico = strip_legacy_raciocinio_text(repaired.raciocinio_juridico)
        repaired.composicao = choose_preferred_composition(repaired.composicao, group_best_composition)
        best_same_video_composition = artifact_context.best_valid_item_composicao
        best_same_video_same_date_composition = []
        if current_session_date:
            best_same_video_same_date_composition = (
                artifact_context.valid_item_composition_by_date.get(current_session_date) or []
            )
        if artifact_item is not None:
            artifact_classe = _sanitize_classe_candidate(artifact_item.classe_processo)
            if artifact_classe and should_replace_classe_processo(
                repaired.classe_processo,
                artifact_classe,
                repaired,
            ):
                repaired.classe_processo = artifact_classe
            artifact_relator, artifact_pedido_vista = extract_ministro_roles_from_composition_entries(
                artifact_item.composicao
            )
            canonical_artifact_relator = normalize_ministro_name(
                artifact_item.relator or artifact_relator
            )
            canonical_artifact_pedido_vista = normalize_pedido_vista_value(
                artifact_item.pedido_vista or artifact_pedido_vista
            )
            if repair_focus == "schema-core" and canonical_artifact_relator:
                repaired.relator = canonical_artifact_relator
            elif repair_focus != "deterministic-core" and canonical_artifact_relator and not repaired.relator:
                repaired.relator = canonical_artifact_relator
            if repair_focus == "schema-core" and canonical_artifact_pedido_vista:
                repaired.pedido_vista = canonical_artifact_pedido_vista
            if repair_focus != "deterministic-core" and artifact_item.resultado_final and not repaired.resultado:
                repaired.resultado = artifact_item.resultado_final
            if repair_focus != "deterministic-core" and artifact_item.votacao and (not repaired.votacao or _votacao_is_inconsistent(repaired)):
                repaired.votacao = artifact_item.votacao
            if repair_focus != "deterministic-core" and artifact_item.eleicao and not repaired.eleicao:
                repaired.eleicao = artifact_item.eleicao
            if artifact_item.numero_processo:
                repaired.numero_processo = _prefer_specific_numero_processo(
                    repaired.numero_processo,
                    artifact_item.numero_processo,
                )
            if artifact_item.punchline and punchline_looks_generic(repaired.punchline, repaired):
                repaired.punchline = artifact_item.punchline
            if artifact_item.advogados and not repaired.advogados:
                repaired.advogados = list(artifact_item.advogados)
            if artifact_item.partes and not repaired.partes:
                repaired.partes = list(artifact_item.partes)
            repaired.origem = _prefer_specific_origem(
                repaired.origem,
                _origin_from_artifact_item(artifact_item),
                repaired.tribunal,
            )
            repaired.composicao = choose_preferred_composition(repaired.composicao, artifact_item.composicao)
        composition_probe = normalize_composition_list(
            list(original.composicao)
            + list(repaired.composicao)
            + list(group_best_composition)
            + list(best_same_video_same_date_composition)
            + list(best_same_video_composition)
            + (list(artifact_item.composicao) if artifact_item is not None else [])
            + list(artifact_context.session_composicao)
        )
        nearest_session_composition = []
        if best_composition_by_session_date:
            nearest_session_composition = _choose_nearest_valid_composition_by_session_date(
                target_date=current_session_date,
                target_year=year,
                probe_values=composition_probe,
                best_composition_by_session_date=best_composition_by_session_date,
            )
        if original_composition_issue and _composition_size_issue(best_session_composition) == "":
            repaired.composicao = normalize_composition_list(best_session_composition)
        elif original_composition_issue and _composition_size_issue(best_same_video_same_date_composition) == "":
            repaired.composicao = normalize_composition_list(best_same_video_same_date_composition)
        elif original_composition_issue and _composition_size_issue(nearest_session_composition) == "":
            repaired.composicao = normalize_composition_list(nearest_session_composition)
        elif original_composition_issue and _composition_size_issue(best_same_video_composition) == "":
            repaired.composicao = normalize_composition_list(best_same_video_composition)
        elif original_composition_issue and _composition_size_issue(group_best_composition) == "":
            repaired.composicao = normalize_composition_list(group_best_composition)
        elif original_composition_issue and artifact_item is not None and _composition_size_issue(artifact_item.composicao) == "":
            repaired.composicao = normalize_composition_list(artifact_item.composicao)
        inference_row = repaired
        if artifact_item is not None:
            inference_row = repaired.model_copy(
                update={
                    "tema": artifact_item.tema or repaired.tema,
                    "punchline": artifact_item.punchline or repaired.punchline,
                    "analise_do_conteudo_juridico": artifact_item.analise_do_conteudo_juridico or repaired.analise_do_conteudo_juridico,
                    "raciocinio_juridico": artifact_item.raciocinio_juridico or repaired.raciocinio_juridico,
                    "fundamentacao_normativa": artifact_item.fundamentacao_normativa or repaired.fundamentacao_normativa,
                    "precedentes_citados": artifact_item.precedentes_citados or repaired.precedentes_citados,
                    "resultado": artifact_item.resultado_final or repaired.resultado,
                    "votacao": artifact_item.votacao or repaired.votacao,
                    "classe_processo": artifact_item.classe_processo or repaired.classe_processo,
                }
            )
        ordering = artifact_context.ordering_by_process.get(process_key) or artifact_context.ordering_by_special_process.get(
            special_key
        )
        trusted_ordering = artifact_context.trusted_ordering_by_process.get(process_key) or artifact_context.trusted_ordering_by_special_process.get(
            special_key
        )
        if ordering:
            repaired.source_start_seconds, repaired.source_bundle_index, repaired.source_item_index = ordering
        if trusted_ordering:
            repaired.youtube_link = build_timestamped_youtube_link(repaired.youtube_link, trusted_ordering[0])
        else:
            repaired.youtube_link = build_video_only_youtube_link(repaired.youtube_link)

        inferred_origin = infer_origin_from_row_text(repaired)
        repaired.origem = _prefer_specific_origem(repaired.origem, inferred_origin, repaired.tribunal)
        inferred_full_cnj = infer_full_numero_processo_from_row_text(repaired)
        if inferred_full_cnj:
            repaired.numero_processo = _prefer_specific_numero_processo(repaired.numero_processo, inferred_full_cnj)
        if repair_focus != "deterministic-core" and not repaired.relator:
            repaired.relator = infer_relator_from_row_text(repaired)
        if repair_focus in {"all", "schema-core"} and not repaired.pedido_vista:
            repaired.pedido_vista = infer_pedido_vista_from_row_text(repaired)
        if repair_focus != "deterministic-core" and not repaired.votacao:
            repaired.votacao = infer_votacao_from_row_text(repaired)
        if repair_focus != "deterministic-core" and not repaired.resultado:
            repaired.resultado = infer_resultado_from_row_text(repaired)
        if repair_focus == "schema-core":
            repaired.resultado = normalize_resultado_final(repaired.resultado, repaired.classe_processo)
            repaired.votacao = normalize_votacao(repaired.votacao)
            repaired.eleicao = normalize_eleicao_value(repaired.eleicao)
            repaired.relator = normalize_ministro_name(repaired.relator)
            repaired.pedido_vista = normalize_pedido_vista_value(repaired.pedido_vista)
            if not repaired.tribunal:
                repaired.tribunal = normalize_tre(repaired.tribunal, extract_uf_from_text(repaired.origem))
        inferred_classe = infer_classe_from_row_text(inference_row)
        if inferred_classe and should_replace_classe_processo(repaired.classe_processo, inferred_classe, repaired):
            repaired.classe_processo = inferred_classe
        if not repaired.classe_processo:
            title_hint = artifact_context.title_hint_by_process.get(process_key) or artifact_context.title_hint_by_special_process.get(
                special_key
            )
            if title_hint:
                title_hint_classe = _sanitize_classe_candidate(title_hint)
                if should_replace_classe_processo(repaired.classe_processo, title_hint_classe, repaired):
                    repaired.classe_processo = title_hint_classe
        repaired.origem = _safe_normalize_origem_for_repair(repaired.origem, repaired.tribunal)
        if punchline_looks_generic(repaired.punchline, repaired):
            repaired.punchline = infer_punchline_from_row_text(inference_row)
        repaired = validate_preview_row(repaired, notion_schema)
        repaired.origem = _prefer_specific_origem(
            repaired.origem,
            original.origem,
            original.tribunal or repaired.tribunal,
        )
        if repair_focus == "composition":
            if not original_composition_issue:
                # Original ja estava regimentalmente OK: preserve-o exatamente, sem
                # troca-lo por uma bancada de maior score porem com ruido.
                repaired.composicao = list(original.composicao)
            elif _composition_size_issue(repaired.composicao):
                # Original tinha problema E o reparo nao alcancou composicao
                # aproveitavel: nao-regressao por qualidade em vez de reverter
                # cegamente ao original (que costuma estar vazio). Assim 6 nomes
                # corretos ou 7 com 1 desconhecido nao sao descartados.
                repaired.composicao = choose_preferred_composition(repaired.composicao, original.composicao)

        needs_theme_repair = tema_looks_generic(repaired.tema, repaired) or not repaired.tema
        if needs_theme_repair:
            repaired.tema = ""
            if artifact_item is not None and artifact_item.tema:
                repaired.tema = artifact_item.tema
            repaired = validate_preview_row(repaired, notion_schema)
            if use_theme_api and repair_focus not in {"schema-core", "deterministic-core", "identity-core", "partes-advogados", "composition"}:
                context_text = build_theme_repair_context(
                    repaired,
                    artifact_context.theme_text_by_process.get(process_key, "")
                    or artifact_context.theme_text_by_special_process.get(special_key, ""),
                )
                if context_text:
                    try:
                        result = repair_theme_from_text_context(
                            api_key=gemini_api_key,
                            model=model,
                            row=repaired,
                            context_text=context_text,
                            artifact_store=artifact_store,
                            logger=LOGGER,
                            artifact_name=f"08_theme_repair_{process_key or record.page_id}.txt",
                        )
                    except Exception as exc:
                        repaired.add_warning(f"Reparo textual de tema indisponível: {exc}")
                    else:
                        repaired.tema = result.tema
                        repaired = validate_preview_row(repaired, notion_schema)
                        repaired.origem = _prefer_specific_origem(
                            repaired.origem,
                            original.origem,
                            original.tribunal or repaired.tribunal,
                        )
        if punchline_looks_generic(repaired.punchline, repaired):
            repaired.punchline = infer_punchline_from_row_text(
                inference_row.model_copy(update={"tema": repaired.tema, "punchline": repaired.punchline})
            )
            repaired = validate_preview_row(repaired, notion_schema)
            repaired.origem = _prefer_specific_origem(
                repaired.origem,
                original.origem,
                original.tribunal or repaired.tribunal,
            )
        repaired_rows.append(repaired)
        per_page.append(
            {
                "page_id": record.page_id,
                "url": record.url,
                "numero_processo": repaired.numero_processo,
            }
        )

    if repair_focus != "identity-core":
        repaired_rows = sorted(repaired_rows, key=preview_row_sort_key)
    changed_pages: list[dict[str, Any]] = []
    failed_pages: list[dict[str, Any]] = []
    for index, repaired in enumerate(repaired_rows, start=1):
        original = next(record.row for record in records if record.page_id == repaired.page_id)
        if repair_focus != "identity-core":
            repaired.tipo_registro = f"Julgamento {index}"
            repaired = validate_preview_row(repaired, notion_schema)
            if _apply_suspended_later_resolution_marker(
                repaired,
                repaired.page_id,
                suspended_later_resolution_targets,
            ):
                repaired = validate_preview_row(repaired, notion_schema)
            repaired.origem = _prefer_specific_origem(
                repaired.origem,
                original.origem,
                original.tribunal or repaired.tribunal,
            )
        repaired = _restrict_repaired_row_to_focus(original, repaired, repair_focus)
        diff = _repaired_row_diff(original, repaired)
        if not diff:
            continue
        if apply_updates:
            try:
                update_notion_row_with_retry(notion_client, notion_schema, repaired.page_id, repaired)
            except Exception as exc:
                failed_pages.append(
                    {
                        "page_id": repaired.page_id,
                        "numero_processo": repaired.numero_processo,
                        "error": str(exc),
                        "changed_fields": sorted(diff),
                    }
                )
                LOGGER.error(
                    "Falha ao atualizar página %s (%s) durante reparo de %s: %s",
                    repaired.page_id,
                    repaired.numero_processo,
                    video_id,
                    exc,
                )
                continue
        changed_pages.append(
            {
                "page_id": repaired.page_id,
                "numero_processo": repaired.numero_processo,
                "changed_fields": sorted(diff),
                "diff": diff,
            }
        )

    return {
        "video_id": video_id,
        "artifact_dir": str(artifact_context.artifact_dir) if artifact_context.artifact_dir else "",
        "rows": original_rows_count,
        "rows_after_dedup": len(records),
        "repair_focus": repair_focus,
        "apply_updates": apply_updates,
        "trashed_duplicates": len(duplicate_pages) - len(duplicate_trash_failures),
        "trashed_duplicate_pages": duplicate_pages,
        "failed_duplicate_trash": duplicate_trash_failures,
        "trashed_unproven_pages": len(trashed_unproven_pages),
        "trashed_unproven": trashed_unproven_pages,
        "failed_unproven_trash": failed_unproven_trash,
        "updated_pages": len(changed_pages),
        "updated": changed_pages,
        "review_pages": review_pages,
        "failed_pages": len(failed_pages),
        "failed": failed_pages,
    }


def load_existing_pages_for_year_with_retry(
    notion_client: NotionSessoesClient,
    notion_schema: NotionDataSourceSchema,
    year: int,
    *,
    playlist_url: str = "",
    retries: int = 3,
    delay_seconds: float = 2.0,
) -> dict[str, list[ExistingPageRecord]]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return load_existing_pages_for_year(notion_client, notion_schema, year, playlist_url=playlist_url)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(delay_seconds * attempt)
    raise RuntimeError(f"Falha ao carregar páginas existentes de {year}: {last_error}") from last_error


def _numero_processo_needs_repair(row: PublishPreviewRow) -> bool:
    text = normalize_numero_processo_display(row.numero_processo)
    classe = normalize_classe_processo(row.classe_processo)
    if not text:
        return classe != "PA"
    if re.fullmatch(r"\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}", text):
        return False
    if re.fullmatch(r"\d{3,7}-\d{2}", text):
        return False
    if re.fullmatch(r"(?:ADO|ADI)\s+\d+", text, flags=re.IGNORECASE):
        return False
    return True


def _expected_tipo_registro_by_page(
    records: list[ExistingPageRecord],
    ordering_by_process: dict[str, tuple[int, int, int]] | None = None,
) -> dict[str, str]:
    rows: list[PublishPreviewRow] = []
    for record in records:
        row = record.row.model_copy(deep=True)
        if ordering_by_process is not None:
            process_key = canonicalize_numero_processo(row.numero_processo)
            ordering = ordering_by_process.get(process_key)
            if not ordering:
                continue
            row.source_start_seconds, row.source_bundle_index, row.source_item_index = ordering
            row.youtube_link = build_timestamped_youtube_link(row.youtube_link, ordering[0])
        rows.append(row)
    ordered = sorted(rows, key=preview_row_sort_key)
    return {row.page_id: f"Julgamento {index}" for index, row in enumerate(ordered, start=1)}


def _video_has_incomplete_composition(records: list[ExistingPageRecord]) -> bool:
    return any(bool(_composition_size_issue(record.row.composicao)) for record in records)


def _build_best_composition_by_session_date(
    grouped: dict[str, list[ExistingPageRecord]],
) -> dict[str, list[str]]:
    best_by_date: dict[str, list[str]] = {}
    for records in grouped.values():
        for record in records:
            session_date = normalize_session_date_to_iso(record.row.data_sessao)
            if not session_date:
                continue
            composition = normalize_composition_list(record.row.composicao)
            if _composition_size_issue(composition):
                continue
            current = best_by_date.get(session_date, [])
            best_by_date[session_date] = choose_preferred_composition(current, composition)
    return best_by_date


def _choose_nearest_valid_composition_by_session_date(
    *,
    target_date: str,
    target_year: int,
    probe_values: list[str],
    best_composition_by_session_date: dict[str, list[str]],
) -> list[str]:
    normalized_target = normalize_session_date_to_iso(target_date)
    if not normalized_target or not normalized_target.startswith(f"{target_year}-"):
        return []
    try:
        target_day = date.fromisoformat(normalized_target)
    except ValueError:
        return []
    probe_names = set(normalize_composition_list(probe_values))
    best_choice: list[str] = []
    best_score: tuple[int, int, int] | None = None
    for session_date, composition in best_composition_by_session_date.items():
        normalized_date = normalize_session_date_to_iso(session_date)
        normalized_composition = normalize_composition_list(composition)
        if not normalized_date or normalized_date == normalized_target:
            continue
        if not normalized_date.startswith(f"{target_year}-"):
            continue
        if _composition_size_issue(normalized_composition):
            continue
        try:
            candidate_day = date.fromisoformat(normalized_date)
        except ValueError:
            continue
        delta_days = abs((candidate_day - target_day).days)
        if delta_days > MAX_NEAREST_COMPOSITION_SESSION_DELTA_DAYS:
            continue
        overlap = len(probe_names.intersection(normalized_composition))
        score = (overlap, -delta_days, 1 if len(normalized_composition) == 7 else 0)
        if best_score is None or score > best_score:
            best_score = score
            best_choice = normalized_composition
    if best_score is None:
        return []
    if best_score[0] <= 0 and -best_score[1] > 3:
        return []
    return best_choice


def audit_existing_year(
    grouped: dict[str, list[ExistingPageRecord]],
    *,
    playlist_url: str = "",
    year: int = 0,
) -> dict[str, Any]:
    playlist_title_by_video: dict[str, str] = {}
    if playlist_url and year:
        playlist_title_by_video = {
            video.video_id: video.title
            for video in load_playlist_videos(playlist_url)
            if is_relevant_2025_session(video, year)
        }
    suspended_later_resolution_targets = build_suspended_later_resolution_targets(grouped)
    stats = {
        "videos": len(grouped),
        "pages": 0,
        "tema_empty": 0,
        "tema_generic": 0,
        "punchline_empty": 0,
        "tipo_blank": 0,
        "tipo_duplicate": 0,
        "tipo_out_of_order": 0,
        "association_unproven": 0,
        "youtube_timestamp_unvalidated": 0,
        "numero_needs_repair": 0,
        "data_sessao_mismatch": 0,
        "identity_duplicate_process": 0,
        "identity_needs_repair": 0,
        "origem_empty": 0,
        "origem_state_only": 0,
        "origem_tre_extenso": 0,
        "origem_invalid_label": 0,
        "origem_downgraded_tre": 0,
        "resultado_empty": 0,
        "votacao_empty": 0,
        "votacao_inconsistent": 0,
        "suspenso_julgado_depois_missing": 0,
        "relator_empty": 0,
        "classe_empty": 0,
        "classe_invalid_stf": 0,
        "classe_mismatch": 0,
        "youtube_video_mismatch": 0,
        "composicao_incomplete": 0,
        "composicao_lt6": 0,
        "composicao_gt7": 0,
        "composicao_regimental": 0,
        "possible_false_positive": 0,
    }
    offenders: dict[str, list[dict[str, str]]] = {
        key: []
        for key in stats
        if key not in {"videos", "pages"}
    }
    for video_id, records in grouped.items():
        stats["pages"] += len(records)
        ordering_by_process: dict[str, tuple[int, int, int]] = {}
        artifact_context: RepairArtifactContext | None = None
        if playlist_url and year:
            artifact_context = load_repair_artifact_context(playlist_url, year, video_id)
            ordering_by_process = artifact_context.trusted_ordering_by_process
        expected_tipo = _expected_tipo_registro_by_page(records, ordering_by_process)
        tipo_counts: dict[str, int] = {}
        for record in records:
            row = record.row
            identity_reasons: list[str] = []
            if not row.tema:
                stats["tema_empty"] += 1
                offenders["tema_empty"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
            elif tema_looks_generic(row.tema, row):
                stats["tema_generic"] += 1
                offenders["tema_generic"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
            if not row.punchline:
                stats["punchline_empty"] += 1
                offenders["punchline_empty"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
            if not row.tipo_registro:
                stats["tipo_blank"] += 1
                offenders["tipo_blank"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
            else:
                tipo_counts[row.tipo_registro] = tipo_counts.get(row.tipo_registro, 0) + 1
                if row.tipo_registro != expected_tipo.get(record.page_id, row.tipo_registro):
                    stats["tipo_out_of_order"] += 1
                    offenders["tipo_out_of_order"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
            if artifact_context is not None and not _row_has_local_association_proof(row, artifact_context):
                stats["association_unproven"] += 1
                offenders["association_unproven"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
                identity_reasons.append("association_unproven")
            if _row_has_wrong_youtube_video(row, video_id):
                stats["youtube_video_mismatch"] += 1
                offenders["youtube_video_mismatch"].append(
                    {
                        "video_id": video_id,
                        "numero_processo": row.numero_processo,
                        "page_id": record.page_id,
                        "youtube_link": row.youtube_link,
                    }
                )
                identity_reasons.append("youtube_video_mismatch")
            process_key = canonicalize_numero_processo(row.numero_processo)
            special_key = _special_process_lookup_key(
                row.numero_processo,
                row.classe_processo or infer_classe_from_row_text(row),
            )
            extracted_ts = re.search(r"[?&]t=(\d+)", row.youtube_link or "")
            expected_ordering: tuple[int, int, int] | None = None
            authoritative_session_date = ""
            if artifact_context is not None:
                expected_ordering = artifact_context.trusted_ordering_by_process.get(process_key) or artifact_context.trusted_ordering_by_special_process.get(
                    special_key
                )
                authoritative_session_date = _authoritative_video_session_date(
                    video_title=playlist_title_by_video.get(video_id, ""),
                    year=year,
                    artifact_session_date=artifact_context.session_date,
                )
            if extracted_ts:
                if expected_ordering and int(extracted_ts.group(1)) != int(expected_ordering[0]):
                    stats["youtube_timestamp_unvalidated"] += 1
                    offenders["youtube_timestamp_unvalidated"].append(
                        {
                            "video_id": video_id,
                            "numero_processo": row.numero_processo,
                            "page_id": record.page_id,
                            "youtube_link": row.youtube_link,
                            "expected_start_seconds": expected_ordering[0],
                        }
                    )
                    identity_reasons.append("youtube_timestamp_unvalidated")
                elif expected_ordering is None:
                    stats["youtube_timestamp_unvalidated"] += 1
                    offenders["youtube_timestamp_unvalidated"].append(
                        {
                            "video_id": video_id,
                            "numero_processo": row.numero_processo,
                            "page_id": record.page_id,
                            "youtube_link": row.youtube_link,
                            "expected_start_seconds": None,
                        }
                    )
                    identity_reasons.append("youtube_timestamp_unvalidated")
            row_session_date = normalize_session_date_to_iso(row.data_sessao)
            if authoritative_session_date and row_session_date and authoritative_session_date != row_session_date:
                stats["data_sessao_mismatch"] += 1
                offenders["data_sessao_mismatch"].append(
                    {
                        "video_id": video_id,
                        "numero_processo": row.numero_processo,
                        "page_id": record.page_id,
                        "data_sessao": row.data_sessao,
                        "expected_data_sessao": authoritative_session_date,
                    }
                )
                identity_reasons.append("data_sessao_mismatch")
            if _numero_processo_needs_repair(row):
                stats["numero_needs_repair"] += 1
                offenders["numero_needs_repair"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
                identity_reasons.append("numero_needs_repair")
            origem = normalize_origem_value(row.origem)
            if not origem:
                stats["origem_empty"] += 1
                offenders["origem_empty"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
            elif normalize_class_text(row.origem) in STATE_UF:
                stats["origem_state_only"] += 1
                offenders["origem_state_only"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id, "origem": origem})
            elif "Tribunal Superior Eleitoral" in (row.origem or "") or "Tribunal Regional Eleitoral" in (row.origem or ""):
                stats["origem_tre_extenso"] += 1
                offenders["origem_tre_extenso"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id, "origem": row.origem})
            if row.origem and _raw_origem_looks_invalid(row.origem):
                stats["origem_invalid_label"] += 1
                offenders["origem_invalid_label"].append(
                    {"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id, "origem": row.origem}
                )
            inferred_origin = infer_origin_from_row_text(row)
            if origem.startswith("TRE/") and inferred_origin and _origin_specificity(inferred_origin) > _origin_specificity(origem):
                stats["origem_downgraded_tre"] += 1
                offenders["origem_downgraded_tre"].append(
                    {
                        "video_id": video_id,
                        "numero_processo": row.numero_processo,
                        "page_id": record.page_id,
                        "origem": row.origem,
                        "preferred_origin": inferred_origin,
                    }
                )
            if not row.resultado:
                stats["resultado_empty"] += 1
                offenders["resultado_empty"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
            if not row.votacao:
                stats["votacao_empty"] += 1
                offenders["votacao_empty"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
            elif _votacao_is_inconsistent(row):
                stats["votacao_inconsistent"] += 1
                offenders["votacao_inconsistent"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
            if record.page_id in suspended_later_resolution_targets:
                stats["suspenso_julgado_depois_missing"] += 1
                target = suspended_later_resolution_targets[record.page_id]
                offenders["suspenso_julgado_depois_missing"].append(
                    {
                        "video_id": video_id,
                        "numero_processo": row.numero_processo,
                        "page_id": record.page_id,
                        "later_video_id": target.get("later_video_id", ""),
                        "later_data_sessao": target.get("later_data_sessao", ""),
                    }
                )
            if not row.relator:
                stats["relator_empty"] += 1
                offenders["relator_empty"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
            classe_norm = normalize_classe_processo(row.classe_processo)
            if classe_norm in {"ADI", "ADO"}:
                stats["classe_invalid_stf"] += 1
                offenders["classe_invalid_stf"].append(
                    {
                        "video_id": video_id,
                        "numero_processo": row.numero_processo,
                        "page_id": record.page_id,
                        "classe_processo": row.classe_processo,
                    }
                )
            elif not row.classe_processo:
                stats["classe_empty"] += 1
                offenders["classe_empty"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
            else:
                classe_candidate = _classe_mismatch_candidate(row, artifact_context)
                if classe_candidate and should_replace_classe_processo(row.classe_processo, classe_candidate, row):
                    stats["classe_mismatch"] += 1
                    offenders["classe_mismatch"].append(
                        {
                            "video_id": video_id,
                            "numero_processo": row.numero_processo,
                            "page_id": record.page_id,
                            "classe_processo": row.classe_processo,
                            "expected_classe": normalize_classe_processo(classe_candidate),
                        }
                    )
            composition_issue = _composition_size_issue(row.composicao)
            if composition_issue:
                stats["composicao_incomplete"] += 1
                offenders["composicao_incomplete"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
                if composition_issue == "lt6":
                    stats["composicao_lt6"] += 1
                    offenders["composicao_lt6"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
                elif composition_issue == "gt7":
                    stats["composicao_gt7"] += 1
                    offenders["composicao_gt7"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
                else:
                    stats["composicao_regimental"] += 1
                    offenders["composicao_regimental"].append(
                        {
                            "video_id": video_id,
                            "numero_processo": row.numero_processo,
                            "page_id": record.page_id,
                            "issue": composition_issue,
                        }
                    )
            publishability, _ = assess_row_publishability(row)
            if publishability == "skipped":
                stats["possible_false_positive"] += 1
                offenders["possible_false_positive"].append({"video_id": video_id, "numero_processo": row.numero_processo, "page_id": record.page_id})
                identity_reasons.append("possible_false_positive")
            if identity_reasons:
                stats["identity_needs_repair"] += 1
                offenders["identity_needs_repair"].append(
                    {
                        "video_id": video_id,
                        "numero_processo": row.numero_processo,
                        "page_id": record.page_id,
                        "reasons": sorted(set(identity_reasons)),
                    }
                )
        for tipo_value, count in tipo_counts.items():
            if count > 1:
                stats["tipo_duplicate"] += count
                for record in records:
                    if record.row.tipo_registro == tipo_value:
                        offenders["tipo_duplicate"].append({"video_id": video_id, "numero_processo": record.row.numero_processo, "page_id": record.page_id, "tipo_registro": tipo_value})
        identity_groups: dict[str, list[ExistingPageRecord]] = {}
        for record in records:
            key = _identity_record_group_key(record.row)
            if key:
                identity_groups.setdefault(key, []).append(record)
        for duplicate_records in identity_groups.values():
            if len(duplicate_records) < 2:
                continue
            stats["identity_duplicate_process"] += len(duplicate_records)
            for record in duplicate_records:
                offenders["identity_duplicate_process"].append(
                    {
                        "video_id": video_id,
                        "numero_processo": record.row.numero_processo,
                        "page_id": record.page_id,
                    }
                )
    return {"stats": stats, "offenders": offenders}


def run_audit_existing_year(args: argparse.Namespace) -> None:
    runtime = build_runtime_context()
    notion_client = NotionSessoesClient(
        api_key=runtime["notion_api_key"],
        data_source_id=runtime["notion_data_source_id"],
        logger=LOGGER,
        normalize_multiselect_colors_post_write=False,
    )
    notion_schema = notion_client.fetch_schema()
    schema_cleanup = cleanup_notion_schema_residue(notion_client, normalize_colors=False, apply_changes=False)
    grouped = load_existing_pages_for_year_with_retry(
        notion_client,
        notion_schema,
        args.year,
        playlist_url=args.playlist_url,
    )
    summary = audit_existing_year(grouped, playlist_url=args.playlist_url, year=args.year)
    summary["schema_cleanup"] = schema_cleanup
    audit_root = BACKFILL_ROOT / f"_audit_{args.year}_{time.strftime('%Y%m%d_%H%M%S')}"
    audit_root.mkdir(parents=True, exist_ok=True)
    (audit_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Auditoria de %s concluída. Resumo: %s", args.year, audit_root / "summary.json")


def run_repair_existing_year(args: argparse.Namespace) -> None:
    runtime = build_runtime_context()
    notion_client = NotionSessoesClient(
        api_key=runtime["notion_api_key"],
        data_source_id=runtime["notion_data_source_id"],
        logger=LOGGER,
        normalize_multiselect_colors_post_write=False,
    )
    schema_cleanup = cleanup_notion_schema_residue(notion_client, normalize_colors=True)
    notion_schema = notion_client.fetch_schema()
    playlist_title_by_video = {
        video.video_id: video.title
        for video in load_playlist_videos(args.playlist_url)
        if is_relevant_2025_session(video, args.year)
    }
    grouped = load_existing_pages_for_year_with_retry(
        notion_client,
        notion_schema,
        args.year,
        playlist_url=args.playlist_url,
    )
    best_composition_by_session_date = _build_best_composition_by_session_date(grouped)
    suspended_later_resolution_targets = build_suspended_later_resolution_targets(grouped)
    suspended_later_video_ids = {
        video_id
        for video_id, records in grouped.items()
        if any(record.page_id in suspended_later_resolution_targets for record in records)
    }
    identity_universe: IdentityRepairUniverse | None = None
    video_ids = sorted(grouped)
    repair_focus = getattr(args, "repair_focus", "all")
    review_only = bool(getattr(args, "review_only", False))
    if repair_focus != "all":
        if repair_focus == "partes-advogados":
            video_ids = [
                video_id
                for video_id in video_ids
                if any(_row_needs_partes_advogados_repair(record.row) for record in grouped[video_id])
            ]
        elif repair_focus == "schema-core":
            video_ids = list(video_ids)
        elif repair_focus == "deterministic-core":
            video_ids = [
                video_id
                for video_id in video_ids
                if any(
                    (not record.row.relator)
                    or (not record.row.resultado)
                    or (not record.row.votacao)
                    or (not record.row.eleicao)
                    for record in grouped[video_id]
                )
                or video_id in suspended_later_video_ids
            ]
        elif repair_focus == "identity-core":
            audit = audit_existing_year(grouped, playlist_url=args.playlist_url, year=args.year)
            offender_video_ids = {
                item["video_id"]
                for key in {
                    "association_unproven",
                    "youtube_timestamp_unvalidated",
                    "youtube_video_mismatch",
                    "numero_needs_repair",
                    "data_sessao_mismatch",
                    "identity_duplicate_process",
                    "possible_false_positive",
                    "identity_needs_repair",
                }
                for item in audit["offenders"].get(key, [])
            }
            video_ids = [video_id for video_id in video_ids if video_id in offender_video_ids]
        else:
            audit = audit_existing_year(grouped, playlist_url=args.playlist_url, year=args.year)
            focus_keys = {
                "association": {"association_unproven", "youtube_timestamp_unvalidated", "youtube_video_mismatch", "possible_false_positive"},
                "tipo": {"tipo_blank", "tipo_duplicate", "tipo_out_of_order"},
                "punchline": {"punchline_empty"},
                "origem": {"origem_empty", "origem_state_only", "origem_tre_extenso", "origem_invalid_label", "origem_downgraded_tre"},
                "classe": {"classe_empty", "classe_invalid_stf", "classe_mismatch"},
                "votacao": {"votacao_empty", "votacao_inconsistent", "suspenso_julgado_depois_missing"},
                "links": {"association_unproven", "youtube_timestamp_unvalidated", "youtube_video_mismatch"},
                "numero": {"numero_needs_repair"},
                "core-fields": {"tema_empty", "tema_generic", "resultado_empty", "votacao_empty", "relator_empty", "classe_empty"},
                "composition": {"composicao_incomplete", "composicao_lt6", "composicao_gt7", "composicao_regimental"},
            }
            offender_video_ids = {
                item["video_id"]
                for key in focus_keys.get(repair_focus, set())
                for item in audit["offenders"].get(key, [])
            }
            video_ids = [video_id for video_id in video_ids if video_id in offender_video_ids]
    requested_video_ids = dedupe_preserve_order(getattr(args, "video_ids", []) or [])
    if requested_video_ids:
        video_ids = [video_id for video_id in video_ids if video_id in set(requested_video_ids)]
    if args.only_composicao_incompleta:
        video_ids = [video_id for video_id in video_ids if _video_has_incomplete_composition(grouped[video_id])]
    if args.limit > 0:
        video_ids = video_ids[:args.limit]
    if repair_focus == "identity-core":
        identity_universe = _build_identity_repair_universe(
            playlist_url=args.playlist_url,
            year=args.year,
            video_ids=sorted(playlist_title_by_video),
            grouped=grouped,
            playlist_title_by_video=playlist_title_by_video,
        )

    repair_kind = "repair_composicao_incompleta" if args.only_composicao_incompleta else f"repair_{args.year}"
    repair_root = BACKFILL_ROOT / f"_{repair_kind}_{time.strftime('%Y%m%d_%H%M%S')}"
    repair_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for video_id in video_ids:
        LOGGER.info("Reparando registros existentes de %s (%s páginas).", video_id, len(grouped[video_id]))
        summary = repair_existing_video_rows(
            video_id=video_id,
            records=grouped[video_id],
            notion_client=notion_client,
            notion_schema=notion_schema,
            playlist_url=args.playlist_url,
            year=args.year,
            gemini_api_key=runtime["gemini_api_key"],
            model=DEFAULT_GEMINI_MODEL,
            use_theme_api=not args.no_theme_api,
            session_date_hint=infer_session_date_from_video_title(playlist_title_by_video.get(video_id, "")),
            repair_focus=repair_focus,
            apply_updates=not review_only,
            best_composition_by_session_date=best_composition_by_session_date,
            identity_universe=identity_universe,
            suspended_later_resolution_targets=suspended_later_resolution_targets,
        )
        summaries.append(summary)
        (repair_root / f"{video_id}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (repair_root / "summary.json").write_text(
        json.dumps(
            {
                "playlist_url": args.playlist_url,
                "year": args.year,
                "schema_cleanup": schema_cleanup,
                "videos": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Reparo retroativo concluído. Resumo: %s", repair_root / "summary.json")


def run_repair_existing_2025(args: argparse.Namespace) -> None:
    run_repair_existing_year(args)


def _news_fields_snapshot(row: PublishPreviewRow) -> dict[str, Any]:
    return {
        "noticia_TSE": row.noticia_TSE,
        "noticia_TRE": row.noticia_TRE,
        "noticias_gerais": list(row.noticias_gerais or []),
    }


def is_news_quota_exhausted_error(error: str) -> bool:
    text = (error or "").lower()
    return (
        "resource_exhausted" in text
        or "exceeded your current quota" in text
        or "rate-limit" in text
        or "rate limits" in text
        or "gemini rest error 429" in text
    )


def load_completed_news_page_ids(resume_from: str | Path | None) -> set[str]:
    if not resume_from:
        return set()
    resume_root = Path(resume_from)
    if not resume_root.exists():
        raise FileNotFoundError(f"Artefato de notícias para retomada não encontrado: {resume_root}")
    page_ids: set[str] = set()
    for summary_path in resume_root.glob("*/news_existing_summary.json"):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        page_id = str(payload.get("page_id") or "").strip()
        after = payload.get("after") or {}
        if any(
            is_generic_institutional_news_url(str(after.get(field) or ""))
            for field in ("noticia_TSE", "noticia_TRE")
        ):
            continue
        if page_id:
            page_ids.add(page_id)
    return page_ids


def run_enrich_existing_news(args: argparse.Namespace) -> None:
    runtime = build_runtime_context()
    notion_client = NotionSessoesClient(
        api_key=runtime["notion_api_key"],
        data_source_id=runtime["notion_data_source_id"],
        logger=LOGGER,
        normalize_multiselect_colors_post_write=False,
    )
    notion_schema = notion_client.fetch_schema()
    target_year = 0 if bool(getattr(args, "all_years", False)) else int(args.year)
    records = load_existing_pages_for_news(
        notion_client,
        notion_schema,
        year=target_year,
    )
    resume_from = str(getattr(args, "news_resume_from", "") or "").strip()
    completed_resume_page_ids = load_completed_news_page_ids(resume_from)
    rows_loaded = len(records)
    if completed_resume_page_ids:
        records = [record for record in records if record.page_id not in completed_resume_page_ids]
    if args.limit > 0:
        records = records[:args.limit]
    label = "all" if target_year == 0 else str(target_year)
    news_root = BACKFILL_ROOT / f"_news_existing_{label}_{time.strftime('%Y%m%d_%H%M%S')}"
    news_root.mkdir(parents=True, exist_ok=True)
    review_only = bool(getattr(args, "review_only", False))
    single_search = bool(getattr(args, "single_news_search", False))
    news_workers = max(1, int(getattr(args, "news_workers", 1) or 1))
    summary: dict[str, Any] = {
        "scope": label,
        "rows_loaded": rows_loaded,
        "rows": len(records),
        "resume_from": resume_from,
        "skipped_already_completed": len(completed_resume_page_ids),
        "review_only": review_only,
        "single_news_search": single_search,
        "news_workers": news_workers,
        "updated_pages": 0,
        "unchanged_pages": 0,
        "failed_pages": 0,
        "aborted_reason": "",
        "grounding_calls_budgeted": 0,
        "updated": [],
        "failed": [],
    }
    summary_path = news_root / "summary.json"

    def process_record(index: int, record: ExistingPageRecord) -> dict[str, Any]:
        row_artifact_dir = news_root / f"{index:04d}_{record.page_id.replace('-', '')[:12]}"
        row_artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_store = RunArtifacts(row_artifact_dir)
        before = _news_fields_snapshot(record.row)
        news_enricher = GeminiNewsEnricher(
            api_key=runtime["gemini_api_key"],
            model=DEFAULT_NEWS_GEMINI_MODEL,
            artifact_store=artifact_store,
            logger=LOGGER,
            allow_institutional_repair=not single_search,
            max_grounding_attempts=1 if single_search else GEMINI_CALL_RETRIES,
        )
        try:
            enriched = news_enricher.enrich_rows([record.row])[0]
            enriched.page_id = record.page_id
            enriched.action = "update"
            enriched = validate_preview_row(enriched, notion_schema)
            after = _news_fields_snapshot(enriched)
            artifact_payload = {
                "page_id": record.page_id,
                "url": record.url,
                "numero_processo": record.row.numero_processo,
                "before": before,
                "after": after,
                "warnings": enriched.warnings,
            }
            artifact_store.write_json("news_existing_summary.json", artifact_payload)
            if before == after:
                return {"status": "unchanged", "budgeted_grounding_calls": 1 if single_search else 0}
            if not review_only:
                update_notion_news_fields_with_retry(
                    notion_client,
                    notion_schema,
                    record.page_id,
                    enriched,
                )
            return {
                "status": "updated",
                "payload": artifact_payload,
                "budgeted_grounding_calls": 1 if single_search else 0,
            }
        except Exception as exc:
            error = str(exc)
            failure = {
                "page_id": record.page_id,
                "url": record.url,
                "numero_processo": record.row.numero_processo,
                "error": error,
            }
            artifact_store.write_json("news_existing_failure.json", failure)
            LOGGER.error("Falha ao enriquecer notícias da página %s: %s", record.page_id, exc)
            return {
                "status": "failed",
                "payload": failure,
                "fatal": is_news_quota_exhausted_error(error),
                "budgeted_grounding_calls": 1 if single_search else 0,
            }

    def apply_result(result: dict[str, Any]) -> bool:
        status = result.get("status")
        summary["grounding_calls_budgeted"] += int(result.get("budgeted_grounding_calls") or 0)
        if status == "updated":
            summary["updated_pages"] += 1
            summary["updated"].append(result.get("payload") or {})
        elif status == "failed":
            summary["failed_pages"] += 1
            summary["failed"].append(result.get("payload") or {})
        else:
            summary["unchanged_pages"] += 1
        fatal = bool(result.get("fatal"))
        if fatal and not summary.get("aborted_reason"):
            summary["aborted_reason"] = "quota_exhausted"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return fatal

    if news_workers == 1:
        for index, record in enumerate(records, start=1):
            if apply_result(process_record(index, record)):
                LOGGER.error("Quota da Gemini API esgotada; interrompendo enriquecimento de notícias.")
                break
    else:
        with ThreadPoolExecutor(max_workers=news_workers) as executor:
            record_iter = iter(enumerate(records, start=1))
            futures: dict[Any, int] = {}
            aborting = False

            def submit_next() -> bool:
                try:
                    index, record = next(record_iter)
                except StopIteration:
                    return False
                futures[executor.submit(process_record, index, record)] = index
                return True

            for _ in range(news_workers):
                if not submit_next():
                    break
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future, None)
                    if future.cancelled():
                        continue
                    if apply_result(future.result()):
                        aborting = True
                if aborting:
                    for future in list(futures):
                        future.cancel()
                    continue
                while len(futures) < news_workers and submit_next():
                    pass
            if aborting:
                LOGGER.error("Quota da Gemini API esgotada; interrompendo enriquecimento de notícias.")
    LOGGER.info("Enriquecimento retroativo de notícias concluído. Resumo: %s", news_root / "summary.json")


def _load_existing_rerun_summary(
    rerun_root: Path,
    *,
    video_id: str,
    title: str,
) -> dict[str, Any] | None:
    summary_file = rerun_root / f"{video_id}.json"
    if summary_file.exists():
        try:
            payload = json.loads(summary_file.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict) and payload.get("status") in {"done", "error"}:
            return payload

    artifact_dirs = sorted(rerun_root.glob(f"*_{video_id}"))
    for artifact_dir in artifact_dirs:
        if not artifact_dir.is_dir():
            continue
        backfill_summary_path = artifact_dir / "07_backfill_summary.json"
        if not backfill_summary_path.exists():
            continue
        try:
            payload = json.loads(backfill_summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        return {
            "video_id": video_id,
            "title": title,
            "status": "done",
            **payload,
        }
    return None


def run_rerun_error_videos(args: argparse.Namespace) -> None:
    resume = bool(getattr(args, "resume", False))
    root_dir = BACKFILL_ROOT / f"{args.year}_{extract_playlist_id(args.playlist_url)}"
    manifest_path = root_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Manifest inexistente para {args.year}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    error_items = [
        (video_id, entry)
        for video_id, entry in (manifest.get("videos") or {}).items()
        if entry.get("status") == "error"
    ]
    requested_video_ids = dedupe_preserve_order(getattr(args, "video_ids", []) or [])
    if requested_video_ids:
        requested_set = set(requested_video_ids)
        error_items = [
            (video_id, entry)
            for video_id, entry in error_items
            if video_id in requested_set
        ]
    if args.limit > 0:
        error_items = error_items[:args.limit]
    rerun_root = Path(args.root_dir) if getattr(args, "root_dir", "") else BACKFILL_ROOT / f"_rerun_errors_{args.year}_{time.strftime('%Y%m%d_%H%M%S')}"
    rerun_root.mkdir(parents=True, exist_ok=True)
    schema_snapshot_path = rerun_root / SCHEMA_SNAPSHOT_NAME
    existing_snapshot_path = rerun_root / EXISTING_PAGES_SNAPSHOT_NAME
    notion_client: NotionSessoesClient | None = None
    notion_schema: NotionDataSourceSchema | None = None
    existing_pages_by_video: dict[str, list[ExistingPageRecord]] | None = None
    if not resume or not schema_snapshot_path.exists() or not existing_snapshot_path.exists():
        runtime = build_runtime_context()
        notion_client = NotionSessoesClient(
            api_key=runtime["notion_api_key"],
            data_source_id=runtime["notion_data_source_id"],
            logger=LOGGER,
            normalize_multiselect_colors_post_write=False,
        )
        notion_schema = notion_client.fetch_schema()
        existing_pages_by_video = load_existing_pages_for_year_with_retry(
            notion_client,
            notion_schema,
            args.year,
            playlist_url=args.playlist_url,
        )
    if not resume or not schema_snapshot_path.exists():
        assert notion_schema is not None
        dump_schema_snapshot(rerun_root, notion_schema)
    if not resume or not existing_snapshot_path.exists():
        assert existing_pages_by_video is not None
        dump_existing_pages_snapshot(rerun_root, existing_pages_by_video)

    summaries: list[dict[str, Any]] = []
    base_video_timeout_seconds = VIDEO_WORKER_TIMEOUT_SECONDS
    base_no_progress_timeout_seconds = NO_PROGRESS_TIMEOUT_SECONDS
    max_attempts = 3
    for video_id, entry in error_items:
        video = find_target_video(args.playlist_url, args.year, video_id, root_dir=root_dir)
        if resume:
            resumed_summary = _load_existing_rerun_summary(
                rerun_root,
                video_id=video_id,
                title=entry.get("title", ""),
            )
            if resumed_summary and resumed_summary.get("status") == "done":
                manifest["videos"][video_id]["status"] = "done"
                manifest["videos"][video_id]["summary"] = resumed_summary
                manifest["videos"][video_id]["finished_at"] = _now_iso()
                manifest["videos"][video_id].pop("error", None)
                update_manifest_eta(manifest, 1)
                update_manifest(manifest_path, manifest)
                summaries.append(resumed_summary)
                continue
        LOGGER.info("Rerodando vídeo com erro: %s - %s", video.video_id, video.title)
        attempt = 1
        current_video_timeout_seconds = base_video_timeout_seconds
        current_no_progress_timeout_seconds = base_no_progress_timeout_seconds
        while True:
            try:
                summary = _run_video_worker_with_limits(
                    video=video,
                    args=args,
                    root_dir=rerun_root,
                    progress_heartbeat=lambda *_args, **_kwargs: None,
                    video_timeout_seconds=current_video_timeout_seconds,
                    no_progress_timeout_seconds=current_no_progress_timeout_seconds,
                )
            except Exception as exc:
                error_text = str(exc)
                if attempt < max_attempts and is_capacity_related_error(error_text):
                    LOGGER.warning(
                        "Rerun do vídeo %s falhou na tentativa %s/%s. Relançando com limites %ss/%ss. Erro: %s",
                        video.video_id,
                        attempt,
                        max_attempts,
                        current_video_timeout_seconds * 2,
                        current_no_progress_timeout_seconds * 2,
                        error_text,
                    )
                    attempt += 1
                    current_video_timeout_seconds *= 2
                    current_no_progress_timeout_seconds *= 2
                    continue
                summary = {
                    "video_id": video_id,
                    "title": entry.get("title", ""),
                    "status": "error",
                    "error": error_text,
                    "attempts": attempt,
                    "video_timeout_seconds": current_video_timeout_seconds,
                    "no_progress_timeout_seconds": current_no_progress_timeout_seconds,
                }
            else:
                summary["attempts"] = attempt
                summary["video_timeout_seconds"] = current_video_timeout_seconds
                summary["no_progress_timeout_seconds"] = current_no_progress_timeout_seconds
                break
            break
        summary_status = str(summary.get("status") or "done").lower()
        if summary_status != "error":
            summary["status"] = "done"
            manifest["videos"][video_id]["status"] = "done"
            manifest["videos"][video_id]["summary"] = summary
            manifest["videos"][video_id]["finished_at"] = _now_iso()
            manifest["videos"][video_id].pop("error", None)
            update_manifest_eta(manifest, 1)
            update_manifest(manifest_path, manifest)
        if summary_status == "error":
            summary["status"] = "error"
            manifest["videos"][video_id]["error"] = summary["error"]
            update_manifest_eta(manifest, 1)
            update_manifest(manifest_path, manifest)
        summaries.append(summary)
        (rerun_root / f"{video_id}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (rerun_root / "summary.json").write_text(
        json.dumps(
            {
                "playlist_url": args.playlist_url,
                "year": args.year,
                "videos": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Rerun de erros concluído. Resumo: %s", rerun_root / "summary.json")


def run_worker_mode(args: argparse.Namespace) -> None:
    if not args.root_dir or not args.worker_video_id:
        raise RuntimeError("Modo worker requer --root-dir e --worker-video-id.")
    root_dir = Path(args.root_dir)
    runtime = build_runtime_context()
    notion_schema = load_schema_snapshot(root_dir)
    existing_pages_by_video = load_existing_pages_snapshot(root_dir)
    notion_client = NotionSessoesClient(
        api_key=runtime["notion_api_key"],
        data_source_id=notion_schema.data_source_id,
        logger=LOGGER,
        normalize_multiselect_colors_post_write=True,
    )
    target_video = find_target_video(args.playlist_url, args.year, args.worker_video_id, root_dir=root_dir)
    summary = process_video(
        video=target_video,
        notion_client=notion_client,
        notion_schema=notion_schema,
        gemini_api_key=runtime["gemini_api_key"],
        model=DEFAULT_GEMINI_MODEL,
        existing_pages_by_video=existing_pages_by_video,
        root_dir=root_dir,
        auto_trash_unmatched_precedents=not args.no_trash_unmatched_precedents,
        skip_news=args.skip_news,
    )
    print(json.dumps(summary, ensure_ascii=False))


def latest_progress_artifact(artifact_dir: Path) -> tuple[str, float] | tuple[None, None]:
    candidates = [
        path for path in artifact_dir.glob("*")
        if path.is_file() and path.name not in {WORKER_STDOUT_NAME, WORKER_STDERR_NAME}
    ]
    if not candidates:
        return None, None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest.name, latest.stat().st_mtime


def build_worker_command(video: PlaylistVideo, args: argparse.Namespace, root_dir: Path) -> tuple[list[str], Path]:
    project_dir = Path.cwd()
    script_name = Path(__file__).name
    try:
        root_dir_arg = str(root_dir.relative_to(project_dir))
    except ValueError:
        root_dir_arg = os.path.relpath(root_dir, project_dir)
    command = [
        resolve_worker_python(project_dir),
        script_name,
        "--playlist-url",
        args.playlist_url,
        "--year",
        str(args.year),
        f"--worker-video-id={video.video_id}",
        f"--root-dir={root_dir_arg}",
    ]
    if not args.skip_news:
        command.append("--with-news")
    if args.no_trash_unmatched_precedents:
        command.append("--no-trash-unmatched-precedents")
    return command, project_dir


def build_worker_popen_kwargs(command: list[str], project_dir: Path) -> dict[str, Any]:
    return {
        "args": command,
        "cwd": str(project_dir),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": WORKER_PIPE_ENCODING,
        "errors": WORKER_PIPE_ERRORS,
    }


def start_video_worker(video: PlaylistVideo, args: argparse.Namespace, root_dir: Path) -> ActiveWorker:
    artifact_dir = root_dir / f"{video.position:03d}_{video.video_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command, project_dir = build_worker_command(video, args, root_dir)
    process = subprocess.Popen(**build_worker_popen_kwargs(command, project_dir))
    artifact_name, artifact_mtime = latest_progress_artifact(artifact_dir)
    return ActiveWorker(
        video=video,
        process=process,
        artifact_dir=artifact_dir,
        started_at=_now_iso(),
        started_wall_time=time.time(),
        deadline_monotonic=time.monotonic() + VIDEO_WORKER_TIMEOUT_SECONDS,
        last_seen_artifact_name=artifact_name or "",
        last_seen_artifact_mtime=float(artifact_mtime or 0.0),
        last_progress_wall_time=time.time(),
    )


def finalize_worker_logs(handle: ActiveWorker, stdout_text: str, stderr_text: str) -> None:
    (handle.artifact_dir / WORKER_STDOUT_NAME).write_text(stdout_text or "", encoding="utf-8")
    (handle.artifact_dir / WORKER_STDERR_NAME).write_text(stderr_text or "", encoding="utf-8")


def finalize_worker_from_summary(handle: ActiveWorker) -> dict[str, Any]:
    if handle.process.poll() is None:
        handle.process.kill()
    stdout_text, stderr_text = handle.process.communicate(timeout=5)
    finalize_worker_logs(handle, stdout_text, stderr_text)
    summary_path = handle.artifact_dir / "07_backfill_summary.json"
    return {
        "status": "done",
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "progress_changed": False,
    }


def poll_active_worker(handle: ActiveWorker) -> dict[str, Any]:
    progress_name, progress_mtime = latest_progress_artifact(handle.artifact_dir)
    progress_changed = False
    if progress_mtime is not None and progress_mtime > handle.last_seen_artifact_mtime:
        handle.last_seen_artifact_mtime = progress_mtime
        handle.last_seen_artifact_name = progress_name or ""
        handle.last_progress_wall_time = time.time()
        progress_changed = True

    return_code = handle.process.poll()
    if return_code is None:
        summary_path = handle.artifact_dir / "07_backfill_summary.json"
        if summary_path.exists() and handle.last_seen_artifact_name == "07_backfill_summary.json":
            return finalize_worker_from_summary(handle)
        if time.monotonic() >= handle.deadline_monotonic:
            handle.process.kill()
            stdout_text, stderr_text = handle.process.communicate(timeout=5)
            finalize_worker_logs(handle, stdout_text, stderr_text)
            return {
                "status": "error",
                "error": f"Timeout de {VIDEO_WORKER_TIMEOUT_SECONDS}s ao processar o vídeo {handle.video.video_id}.",
                "progress_changed": progress_changed,
            }
        if time.time() - handle.last_progress_wall_time >= NO_PROGRESS_TIMEOUT_SECONDS:
            handle.process.kill()
            stdout_text, stderr_text = handle.process.communicate(timeout=5)
            finalize_worker_logs(handle, stdout_text, stderr_text)
            return {
                "status": "error",
                "error": (
                    f"Sem progresso real de artefatos por {NO_PROGRESS_TIMEOUT_SECONDS}s no vídeo "
                    f"{handle.video.video_id}. Último artefato: {handle.last_seen_artifact_name or 'nenhum'}."
                ),
                "progress_changed": progress_changed,
            }
        return {"status": "running", "progress_changed": progress_changed}

    stdout_text, stderr_text = handle.process.communicate(timeout=5)
    finalize_worker_logs(handle, stdout_text, stderr_text)
    if return_code != 0:
        return {
            "status": "error",
            "error": (
                f"Worker do vídeo {handle.video.video_id} terminou com código {return_code}. "
                f"{_truncate_output(stderr_text or stdout_text)}"
            ),
            "progress_changed": progress_changed,
        }
    summary_path = handle.artifact_dir / "07_backfill_summary.json"
    if not summary_path.exists():
        return {
            "status": "error",
            "error": f"Worker do vídeo {handle.video.video_id} terminou sem gerar {summary_path.name}.",
            "progress_changed": progress_changed,
        }
    return {
        "status": "done",
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "progress_changed": progress_changed,
    }


def update_manifest_eta(manifest: dict[str, Any], max_workers: int) -> None:
    durations: list[float] = []
    statuses: dict[str, int] = {}
    for item in (manifest.get("videos") or {}).values():
        status = item.get("status", "pending")
        statuses[status] = statuses.get(status, 0) + 1
        started_at = item.get("started_at")
        finished_at = item.get("finished_at")
        if status == "done" and started_at and finished_at:
            try:
                start_dt = datetime.fromisoformat(started_at)
                finish_dt = datetime.fromisoformat(finished_at)
            except ValueError:
                pass
            else:
                durations.append((finish_dt - start_dt).total_seconds())
                continue
        if status == "done":
            summary = item.get("summary") or {}
            artifact_dir = summary.get("artifact_dir")
            if artifact_dir:
                artifact_path = Path(artifact_dir)
                start_file = artifact_path / "00_playlist_video.json"
                end_file = artifact_path / "07_backfill_summary.json"
                if start_file.exists() and end_file.exists():
                    durations.append(end_file.stat().st_mtime - start_file.stat().st_mtime)
    avg_seconds = (sum(durations) / len(durations)) if durations else None
    remaining_units = statuses.get("pending", 0) + statuses.get("running", 0)
    if avg_seconds and remaining_units:
        eta_seconds = int((avg_seconds * remaining_units) / max(max_workers, 1))
        manifest["eta_seconds"] = eta_seconds
        manifest["eta_at"] = datetime.fromtimestamp(time.time() + eta_seconds).isoformat(timespec="seconds")
        manifest["avg_video_seconds"] = round(avg_seconds, 1)
    else:
        manifest["eta_seconds"] = None
        manifest["eta_at"] = ""
        manifest["avg_video_seconds"] = None


def prune_recent_timestamps(timestamps: list[float], *, now: float, window_seconds: int) -> list[float]:
    return [value for value in timestamps if now - value <= max(window_seconds, 0)]


def is_capacity_related_error(error: str) -> bool:
    text = (error or "").lower()
    return (
        "sem progresso real de artefatos" in text
        or "timeout de" in text
        or "429 client error" in text
        or "too many requests" in text
    )


def _run_video_worker_with_limits(
    *,
    video: PlaylistVideo,
    args: argparse.Namespace,
    root_dir: Path,
    progress_heartbeat: callable,
    video_timeout_seconds: int,
    no_progress_timeout_seconds: int,
) -> dict[str, Any]:
    global VIDEO_WORKER_TIMEOUT_SECONDS, NO_PROGRESS_TIMEOUT_SECONDS
    previous_video_timeout_seconds = VIDEO_WORKER_TIMEOUT_SECONDS
    previous_no_progress_timeout_seconds = NO_PROGRESS_TIMEOUT_SECONDS
    VIDEO_WORKER_TIMEOUT_SECONDS = max(1, int(video_timeout_seconds))
    NO_PROGRESS_TIMEOUT_SECONDS = max(1, int(no_progress_timeout_seconds))
    try:
        return run_video_worker(
            video=video,
            args=args,
            root_dir=root_dir,
            progress_heartbeat=progress_heartbeat,
        )
    finally:
        VIDEO_WORKER_TIMEOUT_SECONDS = previous_video_timeout_seconds
        NO_PROGRESS_TIMEOUT_SECONDS = previous_no_progress_timeout_seconds


def active_workers_are_healthy(active_workers: dict[str, ActiveWorker], *, now: float) -> bool:
    if not active_workers:
        return True
    return all((now - worker.last_progress_wall_time) <= HEALTHY_WORKER_STALENESS_SECONDS for worker in active_workers.values())


def compute_next_worker_target(
    *,
    current_target: int,
    max_target: int,
    min_target: int,
    pending_videos: int,
    active_workers: dict[str, ActiveWorker],
    healthy_completions_since_scale: int,
    recent_capacity_errors: int,
    seconds_since_last_scale_up: float,
    seconds_since_last_scale_down: float,
    now: float,
) -> tuple[int, str | None]:
    floor = max(1, min(min_target, max_target))
    ceiling = max(floor, max_target)

    if (
        current_target > floor
        and recent_capacity_errors >= SCALE_DOWN_AFTER_CAPACITY_ERRORS
        and seconds_since_last_scale_down >= 0
    ):
        return current_target - 1, "scale_down_capacity_errors"

    if current_target >= ceiling:
        return current_target, None
    if pending_videos <= current_target:
        return current_target, None
    if healthy_completions_since_scale < SCALE_UP_AFTER_HEALTHY_COMPLETIONS:
        return current_target, None
    if recent_capacity_errors:
        return current_target, None
    if seconds_since_last_scale_up < SCALE_UP_COOLDOWN_SECONDS:
        return current_target, None
    if not active_workers_are_healthy(active_workers, now=now):
        return current_target, None
    return current_target + 1, "scale_up_healthy"


def update_manifest_scaler_state(
    manifest: dict[str, Any],
    *,
    current_target_workers: int,
    max_workers: int,
    initial_workers: int,
    auto_scale: bool,
    healthy_completions_since_scale: int,
    recent_capacity_error_times: list[float],
    last_scale_reason: str,
    last_scaled_at: str,
) -> None:
    manifest["current_target_workers"] = current_target_workers
    manifest["max_target_workers"] = max_workers
    manifest["initial_workers"] = initial_workers
    manifest["auto_scale_enabled"] = auto_scale
    manifest["healthy_completions_since_scale"] = healthy_completions_since_scale
    manifest["recent_capacity_errors"] = len(recent_capacity_error_times)
    manifest["last_scale_reason"] = last_scale_reason
    manifest["last_scaled_at"] = last_scaled_at


def run_video_worker(
    *,
    video: PlaylistVideo,
    args: argparse.Namespace,
    root_dir: Path,
    progress_heartbeat: callable,
) -> dict[str, Any]:
    artifact_dir = root_dir / f"{video.position:03d}_{video.video_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    project_dir = Path.cwd()
    script_name = Path(__file__).name
    try:
        root_dir_arg = str(root_dir.relative_to(project_dir))
    except ValueError:
        root_dir_arg = os.path.relpath(root_dir, project_dir)
    command = [
        resolve_worker_python(project_dir),
        script_name,
        "--playlist-url",
        args.playlist_url,
        "--year",
        str(args.year),
        f"--worker-video-id={video.video_id}",
        f"--root-dir={root_dir_arg}",
    ]
    if not args.skip_news:
        command.append("--with-news")
    if args.no_trash_unmatched_precedents:
        command.append("--no-trash-unmatched-precedents")

    process = subprocess.Popen(**build_worker_popen_kwargs(command, project_dir))
    deadline = time.monotonic() + VIDEO_WORKER_TIMEOUT_SECONDS
    stdout_text = ""
    stderr_text = ""
    last_progress_name, last_progress_mtime = latest_progress_artifact(artifact_dir)
    if last_progress_mtime is None:
        last_progress_mtime = time.time()
    try:
        while True:
            return_code = process.poll()
            progress_name, progress_mtime = latest_progress_artifact(artifact_dir)
            if progress_mtime is not None and progress_mtime > last_progress_mtime:
                last_progress_mtime = progress_mtime
                last_progress_name = progress_name
                progress_heartbeat(process.pid, progress_name, progress_mtime)
            if return_code is not None:
                stdout_text, stderr_text = process.communicate(timeout=5)
                break
            summary_path = artifact_dir / "07_backfill_summary.json"
            if summary_path.exists() and last_progress_name == "07_backfill_summary.json":
                process.kill()
                stdout_text, stderr_text = process.communicate(timeout=5)
                return_code = 0
                break
            if time.monotonic() >= deadline:
                process.kill()
                stdout_text, stderr_text = process.communicate(timeout=5)
                raise TimeoutError(
                    f"Timeout de {VIDEO_WORKER_TIMEOUT_SECONDS}s ao processar o vídeo {video.video_id}."
                )
            if time.time() - float(last_progress_mtime) >= NO_PROGRESS_TIMEOUT_SECONDS:
                process.kill()
                stdout_text, stderr_text = process.communicate(timeout=5)
                raise TimeoutError(
                    "Sem progresso real de artefatos por "
                    f"{NO_PROGRESS_TIMEOUT_SECONDS}s no vídeo {video.video_id}. "
                    f"Último artefato: {last_progress_name or 'nenhum'}."
                )
            time.sleep(WORKER_HEARTBEAT_SECONDS)
    finally:
        (artifact_dir / WORKER_STDOUT_NAME).write_text(stdout_text or "", encoding="utf-8")
        (artifact_dir / WORKER_STDERR_NAME).write_text(stderr_text or "", encoding="utf-8")

    if return_code != 0:
        details = _truncate_output(stderr_text or stdout_text)
        raise RuntimeError(
            f"Worker do vídeo {video.video_id} terminou com código {return_code}. {details}"
        )

    summary_path = artifact_dir / "07_backfill_summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"Worker do vídeo {video.video_id} terminou sem gerar {summary_path.name}."
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def process_video(
    *,
    video: PlaylistVideo,
    notion_client: NotionSessoesClient,
    notion_schema: NotionDataSourceSchema,
    gemini_api_key: str,
    model: str,
    existing_pages_by_video: dict[str, list[ExistingPageRecord]],
    root_dir: Path,
    auto_trash_unmatched_precedents: bool,
    skip_news: bool,
) -> dict[str, Any]:
    artifact_dir = root_dir / f"{video.position:03d}_{video.video_id}"
    artifact_store = RunArtifacts(artifact_dir)
    artifact_store.write_json(
        "00_playlist_video.json",
        asdict(video),
    )

    extractor = GeminiSessionExtractor(
        api_key=gemini_api_key,
        model=model,
        artifact_store=artifact_store,
        logger=LOGGER,
    )
    metadata_enricher = GeminiProcessMetadataEnricher(
        api_key=gemini_api_key,
        model=model,
        artifact_store=artifact_store,
        logger=LOGGER,
    )
    theme_punchline_enricher = GeminiThemePunchlineEnricher(
        api_key=gemini_api_key,
        model=model,
        artifact_store=artifact_store,
        logger=LOGGER,
    )
    news_enricher = GeminiNewsEnricher(
        api_key=gemini_api_key,
        model=DEFAULT_NEWS_GEMINI_MODEL,
        artifact_store=artifact_store,
        logger=LOGGER,
    )

    analysis = extractor.analyze_session(video.url)
    session_date_hint = infer_session_date_from_video_title(video.title)
    if session_date_hint:
        analysis.session.data_sessao = session_date_hint
    rows = build_preview_rows(
        analysis,
        youtube_url=video.url,
        notion_schema=notion_schema,
        notion_client=None,
    )
    rows = enrich_preview_rows_with_process_metadata(
        rows,
        api_key=gemini_api_key,
        model=model,
        artifact_store=artifact_store,
        logger=LOGGER,
        enricher=metadata_enricher,
        notion_schema=notion_schema,
    )
    rows = dedupe_preview_rows(rows, video.url)
    rows = [validate_preview_row(row, notion_schema) for row in rows]
    if rows:
        rows = theme_punchline_enricher.enrich_rows(rows)
        rows = [validate_preview_row(row, notion_schema) for row in rows]
        rows = dedupe_preview_rows(rows, video.url)
        rows = [validate_preview_row(row, notion_schema) for row in rows]
    if not skip_news and rows:
        rows = news_enricher.enrich_rows(rows)
        rows = [validate_preview_row(row, notion_schema) for row in rows]
        rows = dedupe_preview_rows(rows, video.url)
        rows = [validate_preview_row(row, notion_schema) for row in rows]

    existing_pages = existing_pages_by_video.get(video.video_id, [])
    rows, unmatched_existing = assign_existing_matches(rows, existing_pages)

    publish_results = publish_preview_rows(rows, notion_client, notion_schema)
    published_page_ids = {result.get("page_id", "") for result in publish_results if result.get("page_id")}

    trashed_pages: list[dict[str, str]] = []
    review_pages: list[dict[str, str]] = []
    if unmatched_existing:
        for record in unmatched_existing:
            if record.page_id in published_page_ids:
                continue
            if auto_trash_unmatched_precedents:
                should_trash, assessed = should_trash_unmatched_row(record.row, metadata_enricher)
                if should_trash:
                    notion_client._request("PATCH", f"/pages/{record.page_id}", json={"in_trash": True})
                    trashed_pages.append(
                        {
                            "page_id": record.page_id,
                            "url": record.url,
                            "numero_processo": record.row.numero_processo,
                        }
                    )
                    continue
                review_pages.append(
                    {
                        "page_id": record.page_id,
                        "url": record.url,
                        "numero_processo": record.row.numero_processo,
                        "warnings": assessed.warnings,
                        "errors": assessed.errors,
                    }
                )
            else:
                review_pages.append(
                    {
                        "page_id": record.page_id,
                        "url": record.url,
                        "numero_processo": record.row.numero_processo,
                    }
                )

    summary = {
        "video_id": video.video_id,
        "title": video.title,
        "url": video.url,
        "artifact_dir": str(artifact_dir),
        "rows_extracted": len(rows),
        "created": sum(1 for item in publish_results if item.get("status") == "created"),
        "updated": sum(1 for item in publish_results if item.get("status") == "updated"),
        "blocked": sum(1 for item in publish_results if item.get("status") == "blocked"),
        "skipped": sum(1 for item in publish_results if item.get("status") == "skipped"),
        "publish_results": publish_results,
        "unmatched_existing_review": review_pages,
        "unmatched_existing_trashed": trashed_pages,
    }
    artifact_store.write_json("07_backfill_summary.json", summary)
    return summary


def enrich_preview_rows_with_process_metadata(
    rows: list[PublishPreviewRow],
    *,
    api_key: str,
    model: str,
    artifact_store: RunArtifacts,
    logger: logging.Logger,
    enricher: GeminiProcessMetadataEnricher,
    notion_schema: NotionDataSourceSchema,
) -> list[PublishPreviewRow]:
    del api_key, model, artifact_store, logger
    enriched_rows = enricher.enrich_rows(rows)
    return [validate_preview_row(row, notion_schema) for row in enriched_rows]


def build_manifest(
    *,
    playlist_url: str,
    year: int,
    videos: list[PlaylistVideo],
) -> dict[str, Any]:
    return {
        "playlist_url": playlist_url,
        "year": year,
        "started_at": _now_iso(),
        "completed_at": "",
        "recent_events": [],
        "videos": {
            video.video_id: {
                "position": video.position,
                "title": video.title,
                "url": video.url,
                "status": "pending",
                "attempts": 0,
            }
            for video in videos
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill 2025 do Notion a partir da playlist do TSE.")
    parser.add_argument("--playlist-url", default=DEFAULT_PLAYLIST_URL)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--limit", type=int, default=0, help="Processa apenas os primeiros N vídeos relevantes.")
    parser.add_argument("--skip-news", action="store_true", help="Mantido por compatibilidade; o padrão já é pular notícias.")
    parser.add_argument("--with-news", action="store_true", help="Executa enriquecimento de notícias durante o backfill principal.")
    parser.add_argument(
        "--repair-existing-2025",
        action="store_true",
        help="Reprocessa deterministicamente os registros já publicados de 2025 antes da segunda passada de notícias.",
    )
    parser.add_argument(
        "--repair-existing-year",
        action="store_true",
        help="Reprocessa deterministicamente os registros já publicados do ano informado por --year.",
    )
    parser.add_argument(
        "--video-id",
        dest="video_ids",
        action="append",
        default=[],
        help="Limita o reparo retroativo aos video_ids informados. Pode ser repetido.",
    )
    parser.add_argument(
        "--only-composicao-incompleta",
        action="store_true",
        help="No modo de reparo retroativo, limita a execução aos vídeos que ainda têm composição incompleta.",
    )
    parser.add_argument(
        "--audit-existing-year",
        action="store_true",
        help="Audita os registros já publicados do ano informado por --year e grava um resumo dos resíduos.",
    )
    parser.add_argument(
        "--rerun-error-videos",
        action="store_true",
        help="Reroda focalmente os vídeos marcados como error no manifest do ano informado.",
    )
    parser.add_argument(
        "--enrich-existing-news",
        action="store_true",
        help="Enriquece apenas as colunas de notícias das páginas já existentes no Notion.",
    )
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Com --enrich-existing-news, processa todas as páginas da base em vez de filtrar por --year.",
    )
    parser.add_argument(
        "--single-news-search",
        action="store_true",
        help="Com --enrich-existing-news, limita o pipeline a uma chamada grounded por linha e desativa reparos por nova busca.",
    )
    parser.add_argument(
        "--news-workers",
        type=int,
        default=1,
        help="Número de workers paralelos para --enrich-existing-news.",
    )
    parser.add_argument(
        "--news-resume-from",
        default="",
        help="Com --enrich-existing-news, pula páginas já concluídas em um artefato anterior de notícias.",
    )
    parser.add_argument(
        "--no-theme-api",
        action="store_true",
        help="No modo de reparo, evita até mesmo o reparo textual barato de tema via Gemini.",
    )
    parser.add_argument(
        "--repair-focus",
        default="all",
        choices=["all", "association", "origem", "classe", "votacao", "links", "tipo", "punchline", "numero", "core-fields", "composition", "partes-advogados", "schema-core", "deterministic-core", "identity-core"],
        help="Recorte lógico do reparo retroativo. Atualmente controla a classificação e o sumário, mantendo o reparo completo por padrão.",
    )
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="No modo de reparo, calcula as correções e grava o resumo sem atualizar o Notion.",
    )
    parser.add_argument("--no-trash-unmatched-precedents", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--initial-workers", type=int, default=DEFAULT_INITIAL_WORKERS)
    parser.add_argument("--auto-scale", action="store_true", help="Aumenta progressivamente o número de workers até --max-workers conforme a saúde do lote.")
    parser.add_argument("--worker-video-id")
    parser.add_argument("--root-dir")
    args = parser.parse_args()
    args.skip_news = not bool(args.with_news)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.worker_video_id:
        run_worker_mode(args)
        return

    if args.audit_existing_year:
        run_audit_existing_year(args)
        return

    if args.repair_existing_2025 or args.repair_existing_year:
        run_repair_existing_year(args)
        return

    if args.rerun_error_videos:
        run_rerun_error_videos(args)
        return

    if args.enrich_existing_news:
        run_enrich_existing_news(args)
        return

    raw_playlist_videos = load_playlist_videos(args.playlist_url)
    playlist_videos = [video for video in raw_playlist_videos if is_relevant_2025_session(video, args.year)]
    if not playlist_videos and raw_playlist_videos:
        session_like_videos = [
            video for video in raw_playlist_videos if ("sessão" in video.title.lower() or "sessao" in video.title.lower())
        ]
        sample_titles = "; ".join(video.title for video in raw_playlist_videos[:5])
        raise SystemExit(
            "Nenhum vídeo relevante foi identificado após o filtro anual, apesar de a playlist conter "
            f"{len(raw_playlist_videos)} vídeos brutos e {len(session_like_videos)} títulos de sessão. "
            f"Exemplos: {sample_titles}"
        )
    if args.limit > 0:
        playlist_videos = playlist_videos[:args.limit]

    args.max_workers = max(1, args.max_workers)
    args.initial_workers = max(1, min(args.initial_workers, args.max_workers))
    if not args.auto_scale:
        args.initial_workers = args.max_workers

    root_dir = BACKFILL_ROOT / f"{args.year}_{extract_playlist_id(args.playlist_url)}"
    root_dir.mkdir(parents=True, exist_ok=True)
    runner_lock_path: Path | None = None
    try:
        runner_lock_path = acquire_runner_lock(root_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    manifest_path = root_dir / "manifest.json"
    try:
        runtime = build_runtime_context()
        notion_client = NotionSessoesClient(
            api_key=runtime["notion_api_key"],
            data_source_id=runtime["notion_data_source_id"],
            logger=LOGGER,
            normalize_multiselect_colors_post_write=False,
        )
        notion_schema = notion_client.fetch_schema()

        manifest = build_manifest(playlist_url=args.playlist_url, year=args.year, videos=playlist_videos)
        if args.resume and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = normalize_manifest_for_resume(manifest)
        manifest = repair_manifest_false_errors(manifest, root_dir)
        current_target_workers = args.initial_workers
        healthy_completions_since_scale = 0
        recent_capacity_error_times: list[float] = []
        last_scale_reason = "bootstrap"
        last_scaled_at_wall = time.time()
        append_manifest_event(
            manifest,
            level="INFO",
            event_type="runner_start",
            message=(
                f"Runner iniciado em resume={args.resume} auto_scale={args.auto_scale} "
                f"initial={args.initial_workers} max={args.max_workers} "
                f"with_news={args.with_news}."
            ),
        )
        update_manifest_scaler_state(
            manifest,
            current_target_workers=current_target_workers,
            max_workers=args.max_workers,
            initial_workers=args.initial_workers,
            auto_scale=args.auto_scale,
            healthy_completions_since_scale=healthy_completions_since_scale,
            recent_capacity_error_times=recent_capacity_error_times,
            last_scale_reason=last_scale_reason,
            last_scaled_at=_now_iso(),
        )
        update_manifest(manifest_path, manifest)

        existing_pages_by_video = load_existing_pages_for_year(
            notion_client,
            notion_schema,
            args.year,
            playlist_url=args.playlist_url,
        )
        dump_schema_snapshot(root_dir, notion_schema)
        dump_existing_pages_snapshot(root_dir, existing_pages_by_video)
        LOGGER.info("Vídeos relevantes da playlist: %s", len(playlist_videos))
        LOGGER.info("Vídeos já representados na base %s: %s", args.year, len(existing_pages_by_video))

        todo_videos = []
        for video in playlist_videos:
            current = manifest["videos"][video.video_id]
            if args.resume and current.get("status") == "done":
                LOGGER.info("Pulando %s (%s) por já constar como concluído no manifest.", video.video_id, video.title)
                continue
            todo_videos.append(video)

        active_workers: dict[str, ActiveWorker] = {}
        while todo_videos or active_workers:
            while todo_videos and len(active_workers) < max(1, current_target_workers):
                video = todo_videos.pop(0)
                current = manifest["videos"][video.video_id]
                LOGGER.info("Processando %s - %s", video.video_id, video.title)
                handle = start_video_worker(video, args, root_dir)
                current["status"] = "running"
                current["attempts"] = int(current.get("attempts") or 0) + 1
                current["started_at"] = handle.started_at
                current["heartbeat_at"] = ""
                current["finished_at"] = ""
                current["last_step"] = "worker_start"
                current["last_artifact"] = handle.last_seen_artifact_name or ""
                current["last_artifact_at"] = ""
                current["worker_pid"] = handle.process.pid
                current.pop("summary", None)
                current.pop("error", None)
                active_workers[video.video_id] = handle
                append_manifest_event(
                    manifest,
                    level="INFO",
                    event_type="worker_start",
                    video_id=video.video_id,
                    message=f"{video.video_id} iniciou tentativa {current['attempts']} com pid={handle.process.pid}.",
                )
                update_manifest_scaler_state(
                    manifest,
                    current_target_workers=current_target_workers,
                    max_workers=args.max_workers,
                    initial_workers=args.initial_workers,
                    auto_scale=args.auto_scale,
                    healthy_completions_since_scale=healthy_completions_since_scale,
                    recent_capacity_error_times=recent_capacity_error_times,
                    last_scale_reason=last_scale_reason,
                    last_scaled_at=datetime.fromtimestamp(last_scaled_at_wall).isoformat(timespec="seconds"),
                )
                update_manifest_eta(manifest, current_target_workers)
                update_manifest(manifest_path, manifest)

            completed_ids: list[str] = []
            manifest_changed = False
            now = time.time()
            recent_capacity_error_times = prune_recent_timestamps(
                recent_capacity_error_times,
                now=now,
                window_seconds=SCALE_DOWN_WINDOW_SECONDS,
            )
            for video_id, handle in list(active_workers.items()):
                current = manifest["videos"][video_id]
                polled = poll_active_worker(handle)
                if polled.get("progress_changed"):
                    current["last_step"] = "worker_running"
                    current["last_artifact"] = handle.last_seen_artifact_name or ""
                    current["last_artifact_at"] = (
                        datetime.fromtimestamp(handle.last_seen_artifact_mtime).isoformat(timespec="seconds")
                        if handle.last_seen_artifact_mtime
                        else ""
                    )
                    current["heartbeat_at"] = current["last_artifact_at"]
                    manifest_changed = True
                if polled["status"] == "running":
                    continue
                completed_ids.append(video_id)
                current["finished_at"] = _now_iso()
                if polled["status"] == "done":
                    current["status"] = "done"
                    current["summary"] = polled["summary"]
                    current["last_step"] = "done"
                    healthy_completions_since_scale += 1
                    summary = polled["summary"] or {}
                    append_manifest_event(
                        manifest,
                        level="INFO",
                        event_type="worker_done",
                        video_id=video_id,
                        message=(
                            f"{video_id} concluído: rows={summary.get('rows_extracted', 0)} "
                            f"created={summary.get('created', 0)} updated={summary.get('updated', 0)} "
                            f"blocked={summary.get('blocked', 0)} skipped={summary.get('skipped', 0)}."
                        ),
                    )
                else:
                    current["status"] = "error"
                    current["error"] = polled["error"]
                    current["last_step"] = "error"
                    healthy_completions_since_scale = 0
                    if is_capacity_related_error(polled["error"]):
                        recent_capacity_error_times.append(now)
                    LOGGER.error("Falha ao processar %s - %s", video_id, polled["error"])
                    append_manifest_event(
                        manifest,
                        level="ERROR",
                        event_type="worker_error",
                        video_id=video_id,
                        message=f"{video_id} falhou: {polled['error']}",
                    )
                manifest_changed = True

            for video_id in completed_ids:
                active_workers.pop(video_id, None)

            if args.auto_scale:
                next_target_workers, scale_reason = compute_next_worker_target(
                    current_target=current_target_workers,
                    max_target=args.max_workers,
                    min_target=args.initial_workers,
                    pending_videos=len(todo_videos) + len(active_workers),
                    active_workers=active_workers,
                    healthy_completions_since_scale=healthy_completions_since_scale,
                    recent_capacity_errors=len(recent_capacity_error_times),
                    seconds_since_last_scale_up=(now - last_scaled_at_wall) if last_scale_reason.startswith("scale_up") else SCALE_UP_COOLDOWN_SECONDS,
                    seconds_since_last_scale_down=(now - last_scaled_at_wall) if last_scale_reason.startswith("scale_down") else 0,
                    now=now,
                )
                if next_target_workers != current_target_workers:
                    previous_target = current_target_workers
                    current_target_workers = next_target_workers
                    last_scale_reason = scale_reason or "scale"
                    last_scaled_at_wall = now
                    if scale_reason == "scale_up_healthy":
                        healthy_completions_since_scale = 0
                        LOGGER.info(
                            "Autoescala saudável: alvo de workers %s -> %s.",
                            previous_target,
                            current_target_workers,
                        )
                        append_manifest_event(
                            manifest,
                            level="INFO",
                            event_type="scale_up",
                            message=f"Autoescala saudável: alvo {previous_target} -> {current_target_workers}.",
                        )
                    elif scale_reason == "scale_down_capacity_errors":
                        recent_capacity_error_times = []
                        LOGGER.warning(
                            "Autoescala defensiva: alvo de workers %s -> %s após erros de capacidade.",
                            previous_target,
                            current_target_workers,
                        )
                        append_manifest_event(
                            manifest,
                            level="WARN",
                            event_type="scale_down",
                            message=f"Autoescala defensiva: alvo {previous_target} -> {current_target_workers}.",
                        )
                    manifest_changed = True

            if manifest_changed:
                update_manifest_scaler_state(
                    manifest,
                    current_target_workers=current_target_workers,
                    max_workers=args.max_workers,
                    initial_workers=args.initial_workers,
                    auto_scale=args.auto_scale,
                    healthy_completions_since_scale=healthy_completions_since_scale,
                    recent_capacity_error_times=recent_capacity_error_times,
                    last_scale_reason=last_scale_reason,
                    last_scaled_at=datetime.fromtimestamp(last_scaled_at_wall).isoformat(timespec="seconds"),
                )
                update_manifest_eta(manifest, current_target_workers)
                update_manifest(manifest_path, manifest)

            if active_workers:
                time.sleep(WORKER_HEARTBEAT_SECONDS)

        manifest["completed_at"] = _now_iso()
        update_manifest_scaler_state(
            manifest,
            current_target_workers=current_target_workers,
            max_workers=args.max_workers,
            initial_workers=args.initial_workers,
            auto_scale=args.auto_scale,
            healthy_completions_since_scale=healthy_completions_since_scale,
            recent_capacity_error_times=recent_capacity_error_times,
            last_scale_reason=last_scale_reason,
            last_scaled_at=datetime.fromtimestamp(last_scaled_at_wall).isoformat(timespec="seconds"),
        )
        update_manifest_eta(manifest, current_target_workers)
        update_manifest(manifest_path, manifest)
        LOGGER.info("Backfill concluído. Manifest: %s", manifest_path)
    finally:
        release_runner_lock(runner_lock_path)


def extract_playlist_id(url: str) -> str:
    match = re.search(r"[?&]list=([A-Za-z0-9_-]+)", url)
    return match.group(1) if match else "playlist"


if __name__ == "__main__":
    main()
