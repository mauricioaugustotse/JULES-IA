from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tse_backfill_2025_notion import (
    BACKFILL_ROOT,
    ExistingPageRecord,
    _authoritative_video_session_date,
    _read_optional_json,
    _short_process_lookup_key,
    extract_playlist_id,
    find_artifact_dir_for_video,
    is_relevant_2025_session,
    load_existing_pages_for_year_with_retry,
    load_playlist_videos,
)
from tse_normalization import (
    canonicalize_numero_processo,
    extract_full_cnj,
    normalize_class_text,
    normalize_numero_processo_display,
    normalize_session_date_to_iso,
)
from tse_youtube_notion_core import (
    JudgmentBundleExtraction,
    NotionSessoesClient,
    SessionExtraction,
    build_runtime_context,
    tema_looks_generic,
)


RUNS_ROOT = Path("artifacts/tse_youtube_notion/semantic_bleed_audit")
SHORT_NUMERO_RE = re.compile(r"^\d{3,7}-\d{2}$")
PROCESS_IN_TITLE_RE = re.compile(r"(?i)\b(?:ADI|ADO|AIJE|AREspe|AgR-?REspe|REspe|RO|RCED|PC|PA|Consulta|CTA|Lista)\b|\d{3,7}-\d{2}")
ADMIN_THEME_PATTERNS = (
    re.compile(r"(?i)\bresolu[cç][aã]o\b"),
    re.compile(r"(?i)\baprova[cç][aã]o\b"),
    re.compile(r"(?i)\bminuta\b"),
    re.compile(r"(?i)\blista tr[ií]plice\b"),
)
ADMIN_ANALYSIS_PATTERNS = (
    re.compile(r"(?i)\baprovou a resolu[cç][aã]o\b"),
    re.compile(r"(?i)\btexto proposto\b"),
    re.compile(r"(?i)\bcolegiado\b"),
    re.compile(r"(?i)\baprova[cç][aã]o un[aâ]nime\b"),
)
JUDICIAL_CASE_PATTERNS = (
    re.compile(r"(?i)\brecurso especial\b"),
    re.compile(r"(?i)\bagr-?respe\b"),
    re.compile(r"(?i)\badiamento\b"),
    re.compile(r"(?i)\bsustenta[cç][aã]o oral\b"),
    re.compile(r"(?i)\bretirado de pauta\b"),
    re.compile(r"(?i)\bproblemas? t[eé]cnic"),
)


@dataclass(frozen=True)
class ArtifactCandidate:
    bundle_index: int
    item_index: int
    title_hint: str
    start_seconds: int
    end_seconds: int | None
    mentioned_process_numbers: list[str]
    item_numero_processo: str
    resolved_numero_processo: str
    process_key: str
    short_key: str
    special_key: str
    data_sessao: str
    tema: str
    relator: str
    origem: str
    partes: list[str]
    advogados: list[str]
    analise_do_conteudo_juridico: str
    raciocinio_juridico: str


def _load_playlist_url(year: int) -> str:
    manifests = sorted(BACKFILL_ROOT.glob(f"{year}_PL*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Manifesto do ano {year} não encontrado em {BACKFILL_ROOT}")
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    playlist_url = str(payload.get("playlist_url") or payload.get("playlist") or "").strip()
    if not playlist_url:
        raise ValueError(f"Manifesto do ano {year} não contém playlist_url")
    return playlist_url


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _excerpt(text: str, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _numero_specificity(value: str) -> int:
    text = normalize_numero_processo_display(value)
    if not text:
        return 0
    if extract_full_cnj(text):
        return 4
    if SHORT_NUMERO_RE.fullmatch(text):
        return 2
    if re.fullmatch(r"\d+", re.sub(r"\D", "", text)):
        return 1
    return 0


def _is_short_numero(value: str) -> bool:
    return bool(SHORT_NUMERO_RE.fullmatch(normalize_numero_processo_display(value)))


def _title_hint_looks_weak(title_hint: str) -> bool:
    text = str(title_hint or "").strip()
    if not text:
        return True
    if PROCESS_IN_TITLE_RE.search(text):
        return False
    normalized = normalize_class_text(text)
    if not normalized:
        return True
    if re.fullmatch(r"[\d\W_]+", text):
        return True
    if len(normalized) <= 6 and normalized.isdigit():
        return True
    return False


def _looks_administrative_theme(text: str) -> bool:
    value = str(text or "").strip()
    return any(pattern.search(value) for pattern in ADMIN_THEME_PATTERNS)


def _looks_administrative_analysis(text: str) -> bool:
    value = str(text or "").strip()
    return any(pattern.search(value) for pattern in ADMIN_ANALYSIS_PATTERNS)


def _looks_judicial_case_text(text: str) -> bool:
    value = str(text or "").strip()
    return any(pattern.search(value) for pattern in JUDICIAL_CASE_PATTERNS)


def _count_structured_fields(candidate: ArtifactCandidate) -> int:
    score = 0
    if extract_full_cnj(candidate.resolved_numero_processo):
        score += 2
    if candidate.origem:
        score += 1
    if candidate.relator:
        score += 1
    if candidate.partes:
        score += 1
    if candidate.advogados:
        score += 1
    if candidate.analise_do_conteudo_juridico:
        score += 1
    return score


def _resolve_metadata_numero(metadata_payload: dict[str, Any], fallback: str) -> str:
    applied = metadata_payload.get("applied") or {}
    value = str(applied.get("numero_processo") or fallback or "").strip()
    return normalize_numero_processo_display(value)


def _resolve_metadata_origem(metadata_payload: dict[str, Any], fallback: str) -> str:
    applied = metadata_payload.get("applied") or {}
    return str(applied.get("origem") or fallback or "").strip()


def _metadata_matches_item_number(metadata_numero: str, item_numero: str) -> bool:
    metadata_display = normalize_numero_processo_display(metadata_numero)
    item_display = normalize_numero_processo_display(item_numero)
    if not metadata_display or not item_display:
        return True
    metadata_key = canonicalize_numero_processo(metadata_display)
    item_key = canonicalize_numero_processo(item_display)
    if metadata_key and item_key and metadata_key == item_key:
        return True
    metadata_short = _short_process_lookup_key(metadata_display)
    item_short = _short_process_lookup_key(item_display)
    return bool(metadata_short and item_short and metadata_short == item_short)


def _load_video_candidates(
    *,
    playlist_url: str,
    year: int,
    video_id: str,
    video_title: str,
) -> tuple[Path | None, str, list[ArtifactCandidate]]:
    artifact_dir = find_artifact_dir_for_video(playlist_url, year, video_id)
    if artifact_dir is None:
        return None, "", []

    session_path = artifact_dir / "01_session_windows.json"
    session = SessionExtraction.model_validate(json.loads(session_path.read_text(encoding="utf-8"))) if session_path.exists() else SessionExtraction()
    authoritative_session_date = _authoritative_video_session_date(
        video_title=video_title,
        year=year,
        artifact_session_date=session.data_sessao,
    )
    windows_by_index = {index: window for index, window in enumerate(session.judgments, start=1)}

    candidates: list[ArtifactCandidate] = []
    metadata_index = 0
    for bundle_path in sorted(artifact_dir.glob("02_judgment_*.json")):
        match = re.search(r"02_judgment_(\d+)\.json$", bundle_path.name)
        if not match:
            continue
        bundle_index = int(match.group(1))
        bundle = JudgmentBundleExtraction.model_validate(json.loads(bundle_path.read_text(encoding="utf-8")))
        window = windows_by_index.get(bundle_index)
        mentioned = list((window.mentioned_process_numbers if window else []) or [])
        title_hint = window.title_hint if window else bundle.title_hint
        start_seconds = int(bundle.start_seconds or (window.start_seconds if window else 0) or 0)
        end_seconds = bundle.end_seconds if bundle.end_seconds is not None else (window.end_seconds if window else None)
        for item_index, item in enumerate(bundle.items, start=1):
            metadata_index += 1
            metadata_payload = _read_optional_json(artifact_dir / f"04a_process_metadata_{metadata_index:02d}.json")
            metadata_numero = _resolve_metadata_numero(metadata_payload, item.numero_processo)
            metadata_matches_item = _metadata_matches_item_number(metadata_numero, item.numero_processo)
            resolved_numero = metadata_numero if metadata_matches_item else normalize_numero_processo_display(item.numero_processo)
            resolved_origem = _resolve_metadata_origem(metadata_payload, item.origem) if metadata_matches_item else str(item.origem or "")
            process_key = canonicalize_numero_processo(resolved_numero)
            short_key = _short_process_lookup_key(resolved_numero or item.numero_processo)
            if not process_key and not short_key:
                continue
            candidates.append(
                ArtifactCandidate(
                    bundle_index=bundle_index,
                    item_index=item_index,
                    title_hint=title_hint,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    mentioned_process_numbers=mentioned,
                    item_numero_processo=normalize_numero_processo_display(item.numero_processo),
                    resolved_numero_processo=resolved_numero,
                    process_key=process_key,
                    short_key=short_key,
                    special_key="",
                    data_sessao=str(item.data_sessao or ""),
                    tema=str(item.tema or ""),
                    relator=str(item.relator or ""),
                    origem=resolved_origem,
                    partes=[str(value).strip() for value in item.partes if str(value).strip()],
                    advogados=[str(value).strip() for value in item.advogados if str(value).strip()],
                    analise_do_conteudo_juridico=str(item.analise_do_conteudo_juridico or ""),
                    raciocinio_juridico=str(item.raciocinio_juridico or ""),
                )
            )
    return artifact_dir, authoritative_session_date, candidates


def _pick_primary_candidate(row: ExistingPageRecord, candidates: list[ArtifactCandidate]) -> ArtifactCandidate | None:
    display_numero = normalize_numero_processo_display(row.row.numero_processo)
    process_key = canonicalize_numero_processo(display_numero)
    short_key = _short_process_lookup_key(display_numero)

    exact = [candidate for candidate in candidates if process_key and candidate.process_key == process_key]
    if exact:
        exact.sort(key=lambda candidate: (candidate.resolved_numero_processo != display_numero, candidate.bundle_index, candidate.item_index))
        return exact[0]

    by_display = [candidate for candidate in candidates if normalize_numero_processo_display(candidate.resolved_numero_processo) == display_numero]
    if by_display:
        by_display.sort(key=lambda candidate: (candidate.bundle_index, candidate.item_index))
        return by_display[0]

    by_short = [candidate for candidate in candidates if short_key and candidate.short_key == short_key]
    if by_short:
        by_short.sort(key=lambda candidate: (_numero_specificity(candidate.resolved_numero_processo), -candidate.bundle_index, -candidate.item_index))
        return by_short[0]
    return None


def _find_richer_neighbors(
    row: ExistingPageRecord,
    candidates: list[ArtifactCandidate],
    primary: ArtifactCandidate | None,
) -> list[ArtifactCandidate]:
    short_key = _short_process_lookup_key(row.row.numero_processo)
    if not short_key:
        return []
    row_specificity = _numero_specificity(row.row.numero_processo)
    neighbors = [
        candidate
        for candidate in candidates
        if candidate.short_key == short_key and normalize_numero_processo_display(candidate.resolved_numero_processo) != normalize_numero_processo_display(row.row.numero_processo)
    ]
    neighbors = [
        candidate
        for candidate in neighbors
        if _numero_specificity(candidate.resolved_numero_processo) > row_specificity or _count_structured_fields(candidate) > (_count_structured_fields(primary) if primary else 0)
    ]
    neighbors.sort(
        key=lambda candidate: (
            -(abs((candidate.bundle_index - (primary.bundle_index if primary else candidate.bundle_index))) <= 1),
            -_numero_specificity(candidate.resolved_numero_processo),
            -_count_structured_fields(candidate),
            candidate.bundle_index,
            candidate.item_index,
        )
    )
    return neighbors


def _analyze_record(
    *,
    year: int,
    video_id: str,
    video_title: str,
    record: ExistingPageRecord,
    authoritative_session_date: str,
    candidates: list[ArtifactCandidate],
) -> dict[str, Any] | None:
    row = record.row
    primary = _pick_primary_candidate(record, candidates)
    richer_neighbors = _find_richer_neighbors(record, candidates, primary)
    reasons: list[str] = []
    risk_score = 0

    if primary is None:
        return None

    if _is_short_numero(row.numero_processo) and richer_neighbors:
        reasons.append("short_numero_conflicts_with_richer_neighbor")
        risk_score += 2
    if not primary.mentioned_process_numbers:
        reasons.append("matched_window_has_no_mentioned_process_numbers")
        risk_score += 2
    if _title_hint_looks_weak(primary.title_hint):
        reasons.append("matched_window_title_hint_is_weak")
        risk_score += 1
    primary_item_date = normalize_session_date_to_iso(primary.data_sessao)
    if authoritative_session_date and primary_item_date and primary_item_date != authoritative_session_date:
        reasons.append("primary_item_date_conflicts_with_video_session_date")
        risk_score += 1

    row_has_admin_semantics = (
        tema_looks_generic(row.tema, row)
        or _looks_administrative_theme(row.tema)
        or _looks_administrative_analysis(row.analise_do_conteudo_juridico)
        or _looks_administrative_analysis(row.raciocinio_juridico)
    )
    neighbor_has_judicial_semantics = any(
        _looks_judicial_case_text(candidate.tema)
        or _looks_judicial_case_text(candidate.analise_do_conteudo_juridico)
        or candidate.origem
        or candidate.partes
        or candidate.advogados
        for candidate in richer_neighbors
    )
    if row_has_admin_semantics and richer_neighbors and neighbor_has_judicial_semantics:
        reasons.append("administrative_semantics_conflict_with_richer_neighbor_case")
        risk_score += 2

    if richer_neighbors and not row.origem and any(candidate.origem for candidate in richer_neighbors):
        reasons.append("row_missing_origem_while_richer_neighbor_has_origin")
        risk_score += 1
    if richer_neighbors and not row.partes and any(candidate.partes for candidate in richer_neighbors):
        reasons.append("row_missing_partes_while_richer_neighbor_has_partes")
        risk_score += 1
    if richer_neighbors and not row.advogados and any(candidate.advogados for candidate in richer_neighbors):
        reasons.append("row_missing_advogados_while_richer_neighbor_has_advogados")
        risk_score += 1

    adjacent_neighbor = next(
        (
            candidate
            for candidate in richer_neighbors
            if abs(candidate.bundle_index - primary.bundle_index) <= 1
        ),
        None,
    )
    if adjacent_neighbor is not None:
        reasons.append("adjacent_neighbor_with_same_short_numero_is_richer")
        risk_score += 1

    if risk_score < 3:
        return None

    risk_level = "high" if risk_score >= 5 else "medium"
    top_neighbors = richer_neighbors[:3]
    return {
        "year": year,
        "video_id": video_id,
        "video_title": video_title,
        "page_id": record.page_id,
        "page_url": record.url,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "reasons": reasons,
        "row": {
            "numero_processo": row.numero_processo,
            "tipo_registro": row.tipo_registro,
            "youtube_link": row.youtube_link,
            "data_sessao": row.data_sessao,
            "relator": row.relator,
            "tema": row.tema,
            "origem": row.origem,
            "partes": row.partes,
            "advogados": row.advogados,
            "analise_excerpt": _excerpt(row.analise_do_conteudo_juridico),
            "raciocinio_excerpt": _excerpt(row.raciocinio_juridico),
        },
        "primary_artifact": {
            "bundle_index": primary.bundle_index,
            "item_index": primary.item_index,
            "title_hint": primary.title_hint,
            "start_seconds": primary.start_seconds,
            "mentioned_process_numbers": primary.mentioned_process_numbers,
            "numero_processo": primary.resolved_numero_processo,
            "data_sessao": primary.data_sessao,
            "relator": primary.relator,
            "tema": primary.tema,
            "origem": primary.origem,
            "partes": primary.partes,
            "advogados": primary.advogados,
            "analise_excerpt": _excerpt(primary.analise_do_conteudo_juridico),
        },
        "neighbor_candidates": [
            {
                "bundle_index": candidate.bundle_index,
                "item_index": candidate.item_index,
                "title_hint": candidate.title_hint,
                "start_seconds": candidate.start_seconds,
                "mentioned_process_numbers": candidate.mentioned_process_numbers,
                "numero_processo": candidate.resolved_numero_processo,
                "data_sessao": candidate.data_sessao,
                "relator": candidate.relator,
                "tema": candidate.tema,
                "origem": candidate.origem,
                "partes": candidate.partes,
                "advogados": candidate.advogados,
                "analise_excerpt": _excerpt(candidate.analise_do_conteudo_juridico),
            }
            for candidate in top_neighbors
        ],
        "recommended_action": (
            "revisar a página contra o bundle vizinho mais específico e reprocessar o vídeo focalmente"
            if richer_neighbors
            else "revisar o bundle primário do vídeo"
        ),
    }


def audit_semantic_bleed_year(
    *,
    year: int,
    playlist_url: str,
) -> dict[str, Any]:
    runtime = build_runtime_context()
    notion_client = NotionSessoesClient(
        api_key=runtime["notion_api_key"],
        data_source_id=runtime["notion_data_source_id"],
        normalize_multiselect_colors_post_write=False,
    )
    notion_schema = notion_client.fetch_schema()
    grouped = load_existing_pages_for_year_with_retry(
        notion_client,
        notion_schema,
        year,
        playlist_url=playlist_url,
    )
    playlist_title_by_video = {
        video.video_id: video.title
        for video in load_playlist_videos(playlist_url)
        if is_relevant_2025_session(video, year)
    }

    stats = Counter(
        {
            "videos": len(grouped),
            "pages": 0,
            "pages_with_short_numero": 0,
            "pages_flagged": 0,
            "pages_flagged_high": 0,
            "artifact_missing": 0,
        }
    )
    reason_counts: Counter[str] = Counter()
    flagged: list[dict[str, Any]] = []

    for video_id, records in grouped.items():
        video_title = playlist_title_by_video.get(video_id, "")
        artifact_dir, authoritative_session_date, candidates = _load_video_candidates(
            playlist_url=playlist_url,
            year=year,
            video_id=video_id,
            video_title=video_title,
        )
        if artifact_dir is None:
            stats["artifact_missing"] += len(records)
            continue
        for record in records:
            stats["pages"] += 1
            if _is_short_numero(record.row.numero_processo):
                stats["pages_with_short_numero"] += 1
            offender = _analyze_record(
                year=year,
                video_id=video_id,
                video_title=video_title,
                record=record,
                authoritative_session_date=authoritative_session_date,
                candidates=candidates,
            )
            if offender is None:
                continue
            flagged.append(offender)
            stats["pages_flagged"] += 1
            if offender["risk_level"] == "high":
                stats["pages_flagged_high"] += 1
            reason_counts.update(offender["reasons"])

    flagged.sort(
        key=lambda item: (
            {"high": 0, "medium": 1}.get(item["risk_level"], 9),
            -int(item["risk_score"]),
            int(item["year"]),
            str(item["video_id"]),
            str(item["row"]["numero_processo"]),
        )
    )
    return {
        "year": year,
        "playlist_url": playlist_url,
        "stats": dict(stats),
        "reason_counts": dict(reason_counts.most_common()),
        "flagged_pages": flagged,
        "top_flagged": flagged[:50],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Auditoria dirigida para detectar contaminação semântica entre bundles adjacentes."
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2025, 2024, 2023, 2022, 2021])
    args = parser.parse_args(argv)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = RUNS_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_root": str(run_root),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "years": [],
    }
    summary_path = run_root / "summary.json"
    _write_json(summary_path, summary)

    aggregate_reason_counts: Counter[str] = Counter()
    aggregate_stats: Counter[str] = Counter()

    for year in args.years:
        playlist_url = _load_playlist_url(year)
        year_payload = audit_semantic_bleed_year(year=year, playlist_url=playlist_url)
        year_path = run_root / f"{year}.json"
        _write_json(year_path, year_payload)

        aggregate_reason_counts.update(year_payload.get("reason_counts") or {})
        aggregate_stats.update(year_payload.get("stats") or {})
        summary["years"].append(
            {
                "year": year,
                "playlist_url": playlist_url,
                "status": "done",
                "report": str(year_path),
                "stats": year_payload.get("stats") or {},
                "top_reasons": dict(list((year_payload.get("reason_counts") or {}).items())[:10]),
            }
        )
        _write_json(summary_path, summary)

    summary["status"] = "done"
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["aggregate_stats"] = dict(aggregate_stats)
    summary["aggregate_reason_counts"] = dict(aggregate_reason_counts.most_common())
    _write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
