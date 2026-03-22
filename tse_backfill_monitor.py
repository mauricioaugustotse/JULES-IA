from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(
    "artifacts/tse_youtube_notion/backfill_2025/2025_PLljYw1P54c4xZp8GKvAW8ogzKaKOSbMp7/manifest.json"
)
DEFAULT_RUNNER_LOG = Path("artifacts/tse_youtube_notion/backfill_2025/runner_stderr.log")


@dataclass
class WorkerView:
    video_id: str
    title: str
    artifact: str
    stage: str
    heartbeat_at: str
    worker_pid: str
    attempts: int


@dataclass
class ErrorView:
    video_id: str
    title: str
    stage: str
    kind: str
    detail: str


@dataclass
class EventView:
    at: str
    level: str
    event_type: str
    video_id: str
    message: str


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def human_duration(seconds: int | float | None) -> str:
    if not seconds:
        return "-"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def human_age(iso_value: str) -> str:
    dt = parse_iso(iso_value)
    if dt is None:
        return "-"
    return human_duration((datetime.now() - dt).total_seconds())


def human_age_from_epoch(epoch_seconds: float | None) -> str:
    if not epoch_seconds:
        return "-"
    return human_duration(time.time() - epoch_seconds)


def truncate(text: str, limit: int = 110) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def stage_from_artifact(filename: str) -> str:
    name = str(filename or "").strip()
    if not name:
        return "sem etapa"
    if name == "00_playlist_video.json":
        return "bootstrap do vídeo"
    if name == "00_scan_windows.json":
        return "duracao e janelas"
    if name == "01_session_windows.json":
        return "sessao consolidada"
    if name == "07_backfill_summary.json":
        return "finalizado, aguardando encerramento"
    matchers = [
        (r"^raw_global_response_chunk_(\d+)\.(json|txt)$", "varredura global chunk {}"),
        (r"^raw_detail_(\d+)\.txt$", "extracao detalhada bloco {}"),
        (r"^raw_start_refinement_(\d+)\.txt$", "refino de inicio bloco {}"),
        (r"^raw_transition_refinement_(\d+)\.txt$", "transicao administrativa bloco {}"),
        (r"^02_judgment_(\d+)\.json$", "julgamento consolidado bloco {}"),
        (r"^04a_process_metadata_(\d+)\.(json|txt)$", "metadados processuais item {}"),
        (r"^06_news_enrichment_(\d+)\.(json|txt)$", "enriquecimento de noticias item {}"),
        (r"^06_news_repair_(?:tse|tre)_(\d+)\.txt$", "reparo de link institucional item {}"),
    ]
    for pattern, label in matchers:
        match = re.match(pattern, name)
        if match:
            return label.format(int(match.group(1)))
    return name


def classify_error_kind(error: str, last_artifact: str) -> str:
    text = (error or "").lower()
    artifact = (last_artifact or "").lower()
    if "sem progresso real de artefatos" in text and artifact == "07_backfill_summary.json":
        return "timeout apos resumo final"
    if "sem progresso real de artefatos" in text:
        return "timeout sem progresso"
    if "timeout de" in text:
        return "timeout total do worker"
    if "terminou com código" in text:
        return "worker encerrou com erro"
    return "erro de processamento"


def build_worker_views(manifest: dict[str, Any]) -> list[WorkerView]:
    rows: list[WorkerView] = []
    for video_id, data in (manifest.get("videos") or {}).items():
        if data.get("status") != "running":
            continue
        rows.append(
            WorkerView(
                video_id=video_id,
                title=str(data.get("title", "") or ""),
                artifact=str(data.get("last_artifact", "") or ""),
                stage=stage_from_artifact(str(data.get("last_artifact", "") or "")),
                heartbeat_at=str(data.get("heartbeat_at", "") or ""),
                worker_pid=str(data.get("worker_pid", "") or ""),
                attempts=int(data.get("attempts") or 0),
            )
        )
    rows.sort(key=lambda item: item.heartbeat_at, reverse=True)
    return rows


def build_error_views(manifest: dict[str, Any], limit: int = 8) -> list[ErrorView]:
    rows: list[ErrorView] = []
    for video_id, data in (manifest.get("videos") or {}).items():
        if data.get("status") != "error":
            continue
        artifact = str(data.get("last_artifact", "") or "")
        error = str(data.get("error", "") or "")
        rows.append(
            ErrorView(
                video_id=video_id,
                title=str(data.get("title", "") or ""),
                stage=stage_from_artifact(artifact),
                kind=classify_error_kind(error, artifact),
                detail=truncate(error, limit=140),
            )
        )
    rows.sort(key=lambda item: (item.kind, item.video_id))
    return rows[:limit]


def summarize_error_kinds(manifest: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for video_id, data in (manifest.get("videos") or {}).items():
        if data.get("status") != "error":
            continue
        kind = classify_error_kind(str(data.get("error", "") or ""), str(data.get("last_artifact", "") or ""))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def build_event_views(manifest: dict[str, Any], limit: int = 8) -> list[EventView]:
    events = []
    for item in (manifest.get("recent_events") or [])[-limit:]:
        events.append(
            EventView(
                at=str(item.get("at") or ""),
                level=str(item.get("level") or "INFO"),
                event_type=str(item.get("type") or ""),
                video_id=str(item.get("video_id") or ""),
                message=str(item.get("message") or ""),
            )
        )
    return list(reversed(events))


def build_log_event_views(log_path: Path, limit: int = 8) -> list[str]:
    if not log_path.exists():
        return []
    raw = log_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    interesting = [
        line for line in lines
        if " INFO " in line or " WARNING " in line or " ERROR " in line
    ]
    return interesting[-limit:]


def recent_done_rows(manifest: dict[str, Any], limit: int = 5) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for video_id, data in (manifest.get("videos") or {}).items():
        if data.get("status") != "done":
            continue
        finished_at = str(data.get("finished_at", "") or "")
        rows.append((finished_at, video_id, str(data.get("title", "") or "")))
    rows.sort(reverse=True)
    return rows[:limit]


def compute_status_counts(manifest: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for data in (manifest.get("videos") or {}).values():
        status = data.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def progress_line(counts: dict[str, int]) -> str:
    total = sum(counts.values()) or 1
    done = counts.get("done", 0)
    running = counts.get("running", 0)
    error = counts.get("error", 0)
    pending = counts.get("pending", 0)
    pct = (done / total) * 100.0
    coverage = ((done + running + error) / total) * 100.0
    return (
        f"publicados: {done}/{total} ({pct:.1f}%)  "
        f"ativos={running}  fila={pending}  falhas_finais={error}  "
        f"cobertura_da_rodada={coverage:.1f}%"
    )


def health_line(manifest: dict[str, Any], workers: list[WorkerView], counts: dict[str, int]) -> str:
    if any(parse_iso(worker.heartbeat_at) is not None and (datetime.now() - parse_iso(worker.heartbeat_at)).total_seconds() > 90 for worker in workers):
        return "saude: atencao  |  ha worker com heartbeat antigo"
    if counts.get("error", 0):
        return "saude: atencao  |  ha falhas finais nesta rodada"
    if workers:
        return "saude: ok  |  workers emitindo artefatos recentes"
    return "saude: ocioso"


def render_snapshot(manifest: dict[str, Any], manifest_path: Path, runner_log_path: Path = DEFAULT_RUNNER_LOG) -> str:
    counts = compute_status_counts(manifest)
    workers = build_worker_views(manifest)
    errors = build_error_views(manifest)
    error_kinds = summarize_error_kinds(manifest)
    events = build_event_views(manifest)
    log_events = build_log_event_views(runner_log_path)
    recent_done = recent_done_rows(manifest)
    manifest_mtime = manifest_path.stat().st_mtime if manifest_path.exists() else None
    runner_log_mtime = runner_log_path.stat().st_mtime if runner_log_path.exists() else None
    width = min(max(shutil.get_terminal_size((120, 40)).columns, 100), 160)
    line = "=" * width

    parts: list[str] = []
    parts.append("[prod] backfill_2025")
    parts.append(line)
    parts.append(f"manifest: {manifest_path}")
    parts.append(
        f"refresh_at: {datetime.now().isoformat(timespec='seconds')}  "
        f"manifest_age={human_age_from_epoch(manifest_mtime)}  "
        f"runner_log_age={human_age_from_epoch(runner_log_mtime)}"
    )
    parts.append(f"updated_at: {manifest.get('updated_at') or '-'}")
    parts.append(progress_line(counts))
    parts.append(health_line(manifest, workers, counts))
    parts.append(
        "eta: "
        f"{human_duration(manifest.get('eta_seconds'))}  "
        f"fim_estimado={manifest.get('eta_at') or '-'}  "
        f"media_video={human_duration(manifest.get('avg_video_seconds'))}"
    )
    if "current_target_workers" in manifest or "max_target_workers" in manifest:
        parts.append(
            "workers: "
            f"alvo={manifest.get('current_target_workers') or '-'}  "
            f"piso={manifest.get('initial_workers') or '-'}  "
            f"teto={manifest.get('max_target_workers') or '-'}  "
            f"auto_scale={'on' if manifest.get('auto_scale_enabled') else 'off'}  "
            f"saudaveis_desde_escala={manifest.get('healthy_completions_since_scale', '-')}  "
            f"erros_capacidade_recent={manifest.get('recent_capacity_errors', '-')}"
        )
        if manifest.get("last_scale_reason") or manifest.get("last_scaled_at"):
            parts.append(
                "ultima_escala: "
                f"motivo={manifest.get('last_scale_reason') or '-'}  "
                f"quando={manifest.get('last_scaled_at') or '-'}"
            )

    if workers:
        parts.append("")
        parts.append("running:")
        for idx, worker in enumerate(workers, start=1):
            parts.append(
                f"  [{idx}] {worker.video_id}  pid={worker.worker_pid or '-'}  tentativa={worker.attempts}"
            )
            parts.append(f"      {truncate(worker.title, 100)}")
            parts.append(
                f"      etapa: {worker.stage}  |  ultimo_artefato: {worker.artifact or '-'}"
            )
            parts.append(
                f"      heartbeat: {worker.heartbeat_at or '-'}  ({human_age(worker.heartbeat_at)} atras)"
            )
    else:
        parts.append("")
        parts.append("running: none")

    parts.append("")
    parts.append("falhas finais:")
    if error_kinds:
        grouped = ", ".join(f"{key}={value}" for key, value in sorted(error_kinds.items()))
        parts.append(f"  resumo: {grouped}")
        for item in errors[:3]:
            parts.append(f"  - {item.video_id} | {item.kind}")
            parts.append(f"    {truncate(item.title, 100)}")
            parts.append(f"    etapa: {item.stage}")
            parts.append(f"    detalhe: {item.detail}")
    else:
        parts.append("  none")

    parts.append("")
    parts.append("eventos recentes:")
    if events:
        for item in events:
            prefix = f"[{item.level}]"
            target = f" {item.video_id}" if item.video_id else ""
            parts.append(f"  - {item.at or '-'} {prefix}{target}")
            parts.append(f"    {truncate(item.message, 140)}")
    else:
        parts.append("  none")

    parts.append("")
    parts.append("log recente do runner:")
    if log_events:
        for line in log_events:
            parts.append(f"  - {truncate(line, 160)}")
    else:
        parts.append("  none")

    parts.append("")
    parts.append("recent done:")
    if recent_done:
        for finished_at, video_id, title in recent_done:
            parts.append(f"  - {finished_at or '-'} | {video_id} | {truncate(title, 90)}")
    else:
        parts.append("  none")

    return "\n".join(parts)


def resolve_manifest_path(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    return DEFAULT_MANIFEST


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor do backfill 2025 do Notion.")
    parser.add_argument("--manifest", help="Caminho para o manifest.json.")
    parser.add_argument("--runner-log", help="Caminho para o runner_stderr.log.")
    parser.add_argument("--watch", action="store_true", help="Atualiza continuamente no terminal.")
    parser.add_argument("--interval", type=float, default=5.0, help="Intervalo de atualização em segundos no modo watch.")
    args = parser.parse_args()

    manifest_path = resolve_manifest_path(args.manifest)
    runner_log_path = Path(args.runner_log) if args.runner_log else DEFAULT_RUNNER_LOG
    if not manifest_path.exists():
        raise SystemExit(f"Manifest não encontrado: {manifest_path}")

    if not args.watch:
        print(render_snapshot(load_manifest(manifest_path), manifest_path, runner_log_path))
        return

    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(render_snapshot(load_manifest(manifest_path), manifest_path, runner_log_path))
            time.sleep(max(args.interval, 1.0))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
