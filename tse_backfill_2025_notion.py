from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any

import requests

from tse_normalization import (
    build_timestamped_youtube_link,
    canonicalize_numero_processo,
    extract_youtube_video_id,
    normalize_numero_processo_display,
    normalize_origem_value,
    normalize_youtube_link,
    parse_multi_value_text,
)
from tse_youtube_notion_core import (
    ARTIFACT_ROOT,
    DEFAULT_GEMINI_MODEL,
    GENERAL_NEWS_LIMIT,
    GeminiNewsEnricher,
    GeminiProcessMetadataEnricher,
    GeminiSessionExtractor,
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
    infer_classe_from_row_text,
    infer_origin_from_row_text,
    infer_relator_from_row_text,
    infer_votacao_from_row_text,
    preview_row_sort_key,
    publish_preview_rows,
    repair_theme_from_text_context,
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
    session_composicao: list[str]
    ordering_by_process: dict[str, tuple[int, int, int]]
    theme_text_by_process: dict[str, str]
    item_by_process: dict[str, JudgmentItemExtraction]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _truncate_output(text: str, limit: int = 4000) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[-limit:]


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


def load_playlist_videos(playlist_url: str) -> list[PlaylistVideo]:
    response = requests.get(
        playlist_url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    match = re.search(r"var ytInitialData = (\{.*?\});", response.text)
    if not match:
        raise RuntimeError("Não foi possível localizar ytInitialData na playlist do YouTube.")
    payload = json.loads(match.group(1))

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
) -> dict[str, list[ExistingPageRecord]]:
    grouped: dict[str, list[ExistingPageRecord]] = {}
    for page in client.query_data_source():
        date_value = (page.get("properties", {}).get("data_sessao", {}).get("date") or {}).get("start") or ""
        if not date_value.startswith(f"{year}-"):
            continue
        youtube_link = parse_property_text(client, schema, page, "youtube_link")
        video_id = extract_youtube_video_id(youtube_link)
        if not video_id:
            continue
        record = ExistingPageRecord(
            page_id=page.get("id", ""),
            url=page.get("url", ""),
            video_id=video_id,
            row=notion_page_to_row(client, schema, page),
        )
        grouped.setdefault(video_id, []).append(record)
    return grouped


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


def find_target_video(playlist_url: str, year: int, video_id: str) -> PlaylistVideo:
    playlist_videos = [
        video for video in load_playlist_videos(playlist_url)
        if is_relevant_2025_session(video, year)
    ]
    for video in playlist_videos:
        if video.video_id == video_id:
            return video
    raise RuntimeError(f"Vídeo {video_id} não localizado na playlist filtrada de {year}.")


def iter_backfill_run_dirs(playlist_url: str, year: int) -> list[Path]:
    playlist_id = extract_playlist_id(playlist_url)
    run_dirs: list[Path] = []
    current_dir = BACKFILL_ROOT / f"{year}_{playlist_id}"
    if current_dir.is_dir():
        run_dirs.append(current_dir)
    archived_root = BACKFILL_ROOT / "_archived_runs"
    if archived_root.is_dir():
        archived_dirs = sorted(
            (path for path in archived_root.glob(f"{year}_{playlist_id}*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
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


def load_repair_artifact_context(playlist_url: str, year: int, video_id: str) -> RepairArtifactContext:
    artifact_dir = find_artifact_dir_for_video(playlist_url, year, video_id)
    session_composicao_candidates: list[list[str]] = []
    ordering_by_process: dict[str, tuple[int, int, int]] = {}
    theme_text_by_process: dict[str, str] = {}
    item_by_process: dict[str, JudgmentItemExtraction] = {}

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
            session_path = candidate_dir / "01_session_windows.json"
            if session_path.exists():
                session = SessionExtraction.model_validate(json.loads(session_path.read_text(encoding="utf-8")))
                if session.composicao:
                    session_composicao_candidates.append(list(session.composicao))

            for bundle_path in sorted(candidate_dir.glob("02_judgment_*.json")):
                match = re.search(r"02_judgment_(\d+)\.json$", bundle_path.name)
                bundle_index = int(match.group(1)) if match else 999999
                bundle = JudgmentBundleExtraction.model_validate(json.loads(bundle_path.read_text(encoding="utf-8")))
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
                    process_key = canonicalize_numero_processo(item.numero_processo)
                    if not process_key:
                        continue
                    item_by_process[process_key] = _choose_preferred_artifact_item(
                        item_by_process.get(process_key),
                        item.model_copy(deep=True),
                    )
                    current_ordering = ordering_by_process.get(process_key)
                    candidate_ordering = (int(bundle.start_seconds or 0), bundle_index, item_index)
                    if current_ordering is None or candidate_ordering < current_ordering:
                        ordering_by_process[process_key] = candidate_ordering
                    if process_key not in theme_text_by_process:
                        theme_text_by_process[process_key] = "\n".join(
                            value
                            for value in [
                                item.punchline,
                                item.analise_do_conteudo_juridico,
                                item.raciocinio_juridico,
                                item.fundamentacao_normativa,
                                raw_text[:4000] if raw_text else "",
                            ]
                            if str(value or "").strip()
                        ).strip()

    session_composicao: list[str] = []
    for candidate in session_composicao_candidates:
        session_composicao = choose_preferred_composition(session_composicao, candidate)

    return RepairArtifactContext(
        artifact_dir=artifact_dir,
        session_composicao=session_composicao,
        ordering_by_process=ordering_by_process,
        theme_text_by_process=theme_text_by_process,
        item_by_process=item_by_process,
    )


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
        return f"TRE/{match.group(1)}"
    return normalized or str(value or "").strip()


def _origin_from_artifact_item(item: JudgmentItemExtraction | None) -> str:
    if item is None:
        return ""
    normalized = normalize_origem_value(item.origem)
    if normalized:
        return normalized
    tre_value = str(item.tre or "").strip().upper()
    match = re.match(r"^TRE-([A-Z]{2})$", tre_value)
    if match:
        return f"TRE/{match.group(1)}"
    return ""


def _repaired_row_diff(before: PublishPreviewRow, after: PublishPreviewRow) -> dict[str, dict[str, Any]]:
    changed: dict[str, dict[str, Any]] = {}
    for field_name in [
        "tema",
        "classe_processo",
        "tipo_registro",
        "origem",
        "numero_processo",
        "relator",
        "votacao",
        "partes",
        "composicao",
        "fundamentacao_normativa",
        "raciocinio_juridico",
    ]:
        before_value = getattr(before, field_name)
        after_value = getattr(after, field_name)
        if before_value != after_value:
            changed[field_name] = {"before": before_value, "after": after_value}
    return changed


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
) -> dict[str, Any]:
    artifact_context = load_repair_artifact_context(playlist_url, year, video_id)
    artifact_store = RunArtifacts(artifact_context.artifact_dir) if artifact_context.artifact_dir else None
    repaired_rows: list[PublishPreviewRow] = []
    per_page: list[dict[str, Any]] = []
    group_best_composition = artifact_context.session_composicao
    for record in records:
        group_best_composition = choose_preferred_composition(group_best_composition, record.row.composicao)

    for record in records:
        original = record.row.model_copy(deep=True)
        repaired = record.row.model_copy(deep=True)
        process_key = canonicalize_numero_processo(repaired.numero_processo)
        artifact_item = artifact_context.item_by_process.get(process_key)
        repaired.origem = _safe_normalize_origem_for_repair(repaired.origem, repaired.tribunal)
        repaired.fundamentacao_normativa = strip_legacy_fundamentacao_text(repaired.fundamentacao_normativa)
        repaired.raciocinio_juridico = strip_legacy_raciocinio_text(repaired.raciocinio_juridico)
        repaired.composicao = choose_preferred_composition(repaired.composicao, group_best_composition)
        if artifact_item is not None:
            if artifact_item.classe_processo and not repaired.classe_processo:
                repaired.classe_processo = artifact_item.classe_processo
            if artifact_item.relator and not repaired.relator:
                repaired.relator = artifact_item.relator
            if artifact_item.votacao and not repaired.votacao:
                repaired.votacao = artifact_item.votacao
            if artifact_item.numero_processo:
                repaired.numero_processo = normalize_numero_processo_display(
                    artifact_item.numero_processo or repaired.numero_processo
                )
            if not repaired.origem:
                repaired.origem = _origin_from_artifact_item(artifact_item)
            repaired.composicao = choose_preferred_composition(repaired.composicao, artifact_item.composicao)
        ordering = artifact_context.ordering_by_process.get(process_key)
        if ordering:
            repaired.source_start_seconds, repaired.source_bundle_index, repaired.source_item_index = ordering
            repaired.youtube_link = build_timestamped_youtube_link(repaired.youtube_link, ordering[0])

        inferred_origin = infer_origin_from_row_text(repaired)
        if inferred_origin and (
            not repaired.origem or not re.search(r"/[A-Z]{2}$", repaired.origem)
        ):
            repaired.origem = inferred_origin
        if not repaired.relator:
            repaired.relator = infer_relator_from_row_text(repaired)
        if not repaired.votacao:
            repaired.votacao = infer_votacao_from_row_text(repaired)
        inferred_classe = infer_classe_from_row_text(repaired)
        if inferred_classe and (
            not repaired.classe_processo
            or (repaired.classe_processo == "PA" and inferred_classe != "PA")
        ):
            repaired.classe_processo = inferred_classe
        if not repaired.origem or not re.search(r"/[A-Z]{2}$", repaired.origem):
            repaired.origem = _safe_normalize_origem_for_repair(repaired.origem, repaired.tribunal)
        repaired = validate_preview_row(repaired, notion_schema)
        if original.origem and not repaired.origem:
            repaired.origem = original.origem

        needs_theme_repair = tema_looks_generic(repaired.tema, repaired) or not repaired.tema
        if needs_theme_repair:
            repaired.tema = ""
            if artifact_item is not None and artifact_item.tema:
                repaired.tema = artifact_item.tema
            repaired = validate_preview_row(repaired, notion_schema)
            if use_theme_api:
                context_text = build_theme_repair_context(
                    repaired,
                    artifact_context.theme_text_by_process.get(process_key, ""),
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
                        if original.origem and not repaired.origem:
                            repaired.origem = original.origem

        repaired_rows.append(repaired)
        per_page.append(
            {
                "page_id": record.page_id,
                "url": record.url,
                "numero_processo": repaired.numero_processo,
            }
        )

    repaired_rows = sorted(repaired_rows, key=preview_row_sort_key)
    changed_pages: list[dict[str, Any]] = []
    for index, repaired in enumerate(repaired_rows, start=1):
        repaired.tipo_registro = f"Julgamento {index}"
        repaired = validate_preview_row(repaired, notion_schema)
        original = next(record.row for record in records if record.page_id == repaired.page_id)
        if original.origem and not repaired.origem:
            repaired.origem = original.origem
        diff = _repaired_row_diff(original, repaired)
        if not diff:
            continue
        notion_client.update_row(notion_schema, repaired.page_id, repaired)
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
        "rows": len(records),
        "updated_pages": len(changed_pages),
        "updated": changed_pages,
    }


def run_repair_existing_2025(args: argparse.Namespace) -> None:
    runtime = build_runtime_context()
    notion_client = NotionSessoesClient(
        api_key=runtime["notion_api_key"],
        data_source_id=runtime["notion_data_source_id"],
        logger=LOGGER,
    )
    notion_schema = notion_client.fetch_schema()
    grouped = load_existing_pages_for_year(notion_client, notion_schema, args.year)
    video_ids = sorted(grouped)
    if args.limit > 0:
        video_ids = video_ids[:args.limit]

    repair_root = BACKFILL_ROOT / f"_repair_{args.year}_{time.strftime('%Y%m%d_%H%M%S')}"
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
        )
        summaries.append(summary)
        (repair_root / f"{video_id}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (repair_root / "summary.json").write_text(
        json.dumps({"videos": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Reparo retroativo concluído. Resumo: %s", repair_root / "summary.json")


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
    )
    target_video = find_target_video(args.playlist_url, args.year, args.worker_video_id)
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
        ground_origem_with_search=args.ground_origem_with_search,
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
        sys.executable,
        script_name,
        "--playlist-url",
        args.playlist_url,
        "--year",
        str(args.year),
        "--worker-video-id",
        video.video_id,
        "--root-dir",
        root_dir_arg,
    ]
    if not args.skip_news:
        command.append("--with-news")
    if args.ground_origem_with_search:
        command.append("--ground-origem-with-search")
    if args.no_trash_unmatched_precedents:
        command.append("--no-trash-unmatched-precedents")
    return command, project_dir


def start_video_worker(video: PlaylistVideo, args: argparse.Namespace, root_dir: Path) -> ActiveWorker:
    artifact_dir = root_dir / f"{video.position:03d}_{video.video_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command, project_dir = build_worker_command(video, args, root_dir)
    process = subprocess.Popen(
        command,
        cwd=str(project_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
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
    return "sem progresso real de artefatos" in text or "timeout de" in text


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
        sys.executable,
        script_name,
        "--playlist-url",
        args.playlist_url,
        "--year",
        str(args.year),
        "--worker-video-id",
        video.video_id,
        "--root-dir",
        root_dir_arg,
    ]
    if not args.skip_news:
        command.append("--with-news")
    if args.ground_origem_with_search:
        command.append("--ground-origem-with-search")
    if args.no_trash_unmatched_precedents:
        command.append("--no-trash-unmatched-precedents")

    process = subprocess.Popen(
        command,
        cwd=str(project_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
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
    ground_origem_with_search: bool,
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
        ground_origem_with_search=ground_origem_with_search,
    )
    news_enricher = GeminiNewsEnricher(
        api_key=gemini_api_key,
        model=model,
        artifact_store=artifact_store,
        logger=LOGGER,
    )

    analysis = extractor.analyze_session(video.url)
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
        "--ground-origem-with-search",
        action="store_true",
        help="Permite grounding também para reparar origem isoladamente. Desligado por padrão para reduzir custo.",
    )
    parser.add_argument(
        "--repair-existing-2025",
        action="store_true",
        help="Reprocessa deterministicamente os registros já publicados de 2025 antes da segunda passada de notícias.",
    )
    parser.add_argument(
        "--no-theme-api",
        action="store_true",
        help="No modo de reparo, evita até mesmo o reparo textual barato de tema via Gemini.",
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

    if args.repair_existing_2025:
        run_repair_existing_2025(args)
        return

    playlist_videos = [video for video in load_playlist_videos(args.playlist_url) if is_relevant_2025_session(video, args.year)]
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
                f"with_news={args.with_news} ground_origem_with_search={args.ground_origem_with_search}."
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

        existing_pages_by_video = load_existing_pages_for_year(notion_client, notion_schema, args.year)
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
