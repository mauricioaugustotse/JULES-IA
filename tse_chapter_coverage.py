"""Inventário independente dos processos nos capítulos oficiais do vídeo.

Os capítulos dão evidência de que um processo precisa ser conferido, mas não
substituem a extração do julgamento. Nenhuma função deste módulo publica dados
nem elimina linhas extraídas que não estejam na descrição.
"""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from tse_youtube_notion_core import (
    GeminiSessionExtractor,
    extract_youtube_video_id,
    fetch_youtube_description,
    parse_youtube_chapter_entries,
)


INVENTORY_FILENAME = "00_chapter_inventory.json"
_TIMESTAMP = re.compile(r"^\s*(\d{1,3}):(\d{2})(?::(\d{2}))?(?:\s|$)")


def _chapter_boundaries(description: str) -> list[int]:
    """Inclui abertura/encerramento/lista para limitar o último caso individual."""
    boundaries: set[int] = set()
    for line in description.splitlines():
        match = _TIMESTAMP.match(line)
        if not match:
            continue
        a, b = int(match[1]), int(match[2])
        seconds = a * 3600 + b * 60 + int(match[3]) if match[3] else a * 60 + b
        boundaries.add(seconds)
    return sorted(boundaries)


def ensure_chapter_inventory(artifact_store: Any, url: str) -> dict[str, Any]:
    """Captura uma vez o inventário válido; ausência/falha permite nova tentativa.

    A descrição fica no mesmo JSON para que a comparação possa ser reproduzida
    offline, mesmo se o canal editar os capítulos posteriormente.
    """
    video_id = extract_youtube_video_id(url) or ""
    try:
        cached = artifact_store.read_json(INVENTORY_FILENAME)
    except (OSError, ValueError, TypeError):
        cached = None
    if (
        isinstance(cached, dict)
        and cached.get("status") == "available"
        and cached.get("video_id") == video_id
        and isinstance(cached.get("chapters"), list)
        and cached["chapters"]
    ):
        return cached

    payload: dict[str, Any] = {
        "status": "unavailable",
        "video_id": video_id,
        "source_url": f"https://www.youtube.com/watch?v={video_id}" if video_id else url,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "chapters": [],
        "description": "",
    }
    if not video_id:
        payload["error"] = "Não foi possível identificar o vídeo para conferir os capítulos."
    else:
        try:
            description = fetch_youtube_description(video_id)
            payload["description"] = description
            entries = parse_youtube_chapter_entries(description)
            boundaries = _chapter_boundaries(description)
            for numero, entry in entries.items():
                start = int(entry["seconds"])
                payload["chapters"].append({
                    "numero_processo": numero,
                    "start_seconds": start,
                    "end_seconds": next((value for value in boundaries if value > start), None),
                    "classe": entry.get("classe", ""),
                    "classe_raw": entry.get("classe_raw", ""),
                })
            payload["chapters"].sort(key=lambda item: (item["start_seconds"], item["numero_processo"]))
            if payload["chapters"]:
                payload["status"] = "available"
            else:
                payload["error"] = (
                    "A descrição do vídeo não contém capítulos processuais utilizáveis."
                    if description else "A descrição do vídeo não pôde ser obtida."
                )
        except Exception as exc:
            # Registra só o tipo: mensagens de clientes HTTP podem conter detalhes
            # de autenticação/configuração que não pertencem ao relatório público.
            payload["error"] = f"Falha ao conferir capítulos do vídeo ({type(exc).__name__})."
    artifact_store.write_json(INVENTORY_FILENAME, payload)
    return payload


def compare_chapters(inventory: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compara identidades, sem aceitar excesso de linhas como prova de cobertura."""
    if inventory.get("status") != "available":
        return [{
            "kind": "chapter_inventory_unavailable",
            "severity": "warning",
            "source": "youtube_chapters",
            "message": inventory.get("error") or "Cobertura pelos capítulos do vídeo indisponível.",
        }]

    process_key = GeminiSessionExtractor._chave_de_processo
    row_keys = {process_key(str(row.get("numero_processo") or "")) for row in rows}
    row_keys.discard("")
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chapter in inventory.get("chapters", []):
        numero = str(chapter.get("numero_processo") or "")
        key = process_key(numero)
        if not key or key in seen:
            continue
        seen.add(key)
        if key in row_keys:
            continue
        issues.append({
            "kind": "chapter_missing_from_rows",
            "severity": "error",
            "source": "youtube_chapters",
            "numero_processo": numero,
            "start_seconds": chapter.get("start_seconds"),
            "message": f"Processo {numero} consta dos capítulos do vídeo e não aparece nas linhas extraídas.",
        })
    return issues
