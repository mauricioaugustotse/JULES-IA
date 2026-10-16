"""Positive video evidence required for a new session judgment page.

Official documents may complete identity and outcome. They cannot supply this
evidence, which is captured from the original own-item video analysis.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel
from tse_normalization import canonicalize_numero_processo


class VideoDebateEvidence(BaseModel):
    source: str = "video_analysis"
    video_id: str = ""
    session_date: str = ""
    start_seconds: int = -1
    end_seconds: int | None = None
    numero_processo: str = ""
    title_hint: str = ""
    analysis_excerpt: str = ""
    source_bundle_index: int = -1
    source_item_index: int = -1


def _fold(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()


def _process_key(value: str) -> str:
    value = canonicalize_numero_processo(value) if value.strip() else ""
    match = re.match(r"^(\d{1,7})[-–](\d{2})(?:\.|$)", value.strip())
    if match:
        return match[1].zfill(7) + match[2]
    return re.sub(r"\W", "", value).lower()


def video_debate_issues(row) -> list[str]:
    """Fail closed on absent, mismatched, collective or non-debate evidence."""
    proof = row.video_debate_evidence
    if proof is None:
        return ["Nova inclusão exige análise do vídeo e trecho identificável do debate individual."]
    issues = []
    if proof.source != "video_analysis":
        issues.append("Certidão, SJUR e DJe não comprovam debate no vídeo.")
    parsed = urlparse(row.youtube_link)
    host = (parsed.hostname or "").lower()
    query = parse_qs(parsed.query)
    video_id = (query.get("v") or [""])[0] if host in {"youtube.com", "www.youtube.com", "m.youtube.com"} else ""
    if host == "youtu.be":
        video_id = parsed.path.strip("/")
    if not re.fullmatch(r"[\w-]{11}", proof.video_id) or video_id != proof.video_id:
        issues.append("Vídeo da inclusão diverge da evidência original do debate.")
    if (proof.start_seconds < 0 or proof.end_seconds is None
            or proof.end_seconds <= proof.start_seconds):
        issues.append("Trecho do debate sem início e fim válidos; não inventar timestamps.")
    timestamp = (query.get("t") or query.get("start") or parse_qs(parsed.fragment).get("t") or [""])[0]
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", timestamp)
    seconds = None
    if timestamp.isdigit():
        seconds = int(timestamp)
    elif match and any(match.groups()):
        seconds = int(match[1] or 0) * 3600 + int(match[2] or 0) * 60 + int(match[3] or 0)
    if (seconds is None or proof.end_seconds is None
            or not proof.start_seconds <= seconds < proof.end_seconds):
        issues.append("Link do YouTube deve apontar para o trecho comprovado do debate.")
    try:
        session_date = date.fromisoformat(proof.session_date).isoformat()
    except (ValueError, TypeError):
        session_date = ""
    if not session_date or session_date != row.data_sessao:
        issues.append("Data da inclusão diverge da sessão analisada no vídeo.")
    if _process_key(proof.numero_processo) != _process_key(row.numero_processo):
        issues.append("Processo da inclusão diverge do item próprio analisado no vídeo.")
    excerpt = _fold(proof.title_hint + "\n" + proof.analysis_excerpt)
    if len(proof.analysis_excerpt.strip()) < 80:
        issues.append("Análise original do vídeo insuficiente para comprovar debate individual.")
    if re.search(r"julgament\w*\s+(?:em\s+(?:lista|bloco)|coletiv)|"
                 r"processos?\s+(?:julgad\w*\s+)?em\s+lista|"
                 r"(?:^|\n)\s*lista\s*(?:n[.o]*\s*)?\d+(?:\s*[-–:]|\s*$)", excerpt):
        issues.append("Julgamento coletivo em lista não autoriza nova inclusão.")
    if re.search(r"apenas\s+(?:a\s+)?leitura\s+(?:da\s+)?(?:ata|pauta)|"
                 r"(?:apenas|somente)\s+(?:anunci|aprego)|"
                 r"convoca\w*\s+uma\s+sessao\s+extraordinaria", excerpt):
        issues.append("Leitura de ata ou anúncio de pauta não comprova debate individual.")
    return issues
