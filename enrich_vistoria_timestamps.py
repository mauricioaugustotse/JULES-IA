# -*- coding: utf-8 -*-
"""Enriquece EM LOTE os itens pendentes da fila de vistoria com o timestamp do
apregoamento (t=) e com dicas de número/tema, lendo os artifacts de cada vídeo.

Estratégia por item sem t= no link:
  1. Pelo NÚMERO: menor start_seconds dos 02_judgment_NN.json cujo conteúdo
     contém o núcleo do processo (>= 9 dígitos).
  2. Sem número: localiza no 07_backfill_summary.json o publish_result que
     originou o item (errors/warnings contidos nos reasons) e casa o TEMA dele
     com o tema/title_hint dos bundles; só usa se o match for único.
A descoberta é gravada na própria fila (patch por id): a GUI passa a mostrar a
coluna ⏱ preenchida e ordenar esses itens primeiro.

Uso: python enrich_vistoria_timestamps.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path

import vistoria_queue

LOGGER = logging.getLogger("enrich_vistoria_ts")


def digits(value: str) -> str:
    return re.sub(r"\D", "", str(value or ""))


def norm(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    return re.sub(r"\s+", " ", decomposed.encode("ascii", "ignore").decode().lower()).strip()


def has_timestamp(item: dict) -> bool:
    row = item.get("row") or {}
    link = str(row.get("youtube_link") or item.get("youtube_url") or "")
    return bool(re.search(r"[?&]t=\d+", link))


class VideoBundles:
    """Cache dos bundles de um vídeo: (start_seconds, digits_do_json, tema_norm)."""

    def __init__(self, video_dir: Path) -> None:
        self.entries: list[tuple[int, str, str]] = []
        for bundle_path in sorted(video_dir.glob("02_judgment_*.json")):
            try:
                payload = json.loads(bundle_path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            start = int(payload.get("start_seconds", 0) or 0)
            blob = json.dumps(payload, ensure_ascii=False)
            temas = [payload.get("title_hint", "")]
            for it in payload.get("items") or []:
                temas.append(it.get("tema", ""))
            self.entries.append((start, digits(blob), norm(" | ".join(t for t in temas if t))))

    def by_numero(self, numero_digits: str) -> int | None:
        core = numero_digits[:9]
        hits = [start for start, blob, _ in self.entries if core and core in blob]
        return min(hits) if hits else None

    def by_tema(self, tema: str) -> int | None:
        tema_n = norm(tema)
        if len(tema_n) < 15:
            return None
        hits = [start for start, _, temas in self.entries if tema_n[:60] in temas]
        if len(hits) == 1:
            return hits[0]
        return None


def backlog_result_for_item(item: dict, video_dir: Path) -> dict | None:
    summary_path = video_dir / "07_backfill_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    reason_text = " ".join(item.get("reasons") or [])
    best: dict | None = None
    best_score = 0
    tied = False
    for result in summary.get("publish_results") or []:
        if result.get("status") != item.get("disposition"):
            continue
        texts = [t for t in (result.get("errors") or []) + (result.get("warnings") or []) if t]
        # Os reasons do item foram truncados na auditoria: casa por prefixo de cada texto.
        score = sum(1 for t in texts if t[:120] in reason_text)
        if score > best_score:
            best, best_score, tied = result, score, False
        elif score == best_score and score > 0 and result is not best:
            tied = True
    return best if best_score > 0 and not tied else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    items = [i for i in vistoria_queue.load_items("pending") if not has_timestamp(i)]
    if args.limit:
        items = items[: args.limit]
    LOGGER.info("Itens pendentes sem timestamp: %s", len(items))

    cache: dict[str, VideoBundles] = {}
    patched = por_numero = por_tema = sem_artifacts = sem_match = 0
    patches: list[dict] = []
    for index, item in enumerate(items, start=1):
        if index % 50 == 0:
            LOGGER.info("  ... %s/%s (%s enriquecidos)", index, len(items), patched)
        video_dir = Path(item.get("artifact_dir") or "")
        if not video_dir.exists():
            sem_artifacts += 1
            continue
        key = str(video_dir)
        if key not in cache:
            cache[key] = VideoBundles(video_dir)
        bundles = cache[key]

        row = item.get("row") or {}
        dje = (item.get("extra") or {}).get("dje") or {}
        numero = str(row.get("numero_processo") or dje.get("numeroUnico") or item.get("numero_hint") or "")
        tema = str(row.get("tema") or item.get("tema_hint") or "")
        result = None
        if len(digits(numero)) < 9 or not tema:
            result = backlog_result_for_item(item, video_dir)
            if result:
                numero = numero if len(digits(numero)) >= 9 else str(result.get("numero_processo") or "")
                tema = tema or str(result.get("tema") or "")

        timestamp = None
        if len(digits(numero)) >= 9:
            timestamp = bundles.by_numero(digits(numero))
            if timestamp is not None:
                por_numero += 1
        if timestamp is None and tema:
            timestamp = bundles.by_tema(tema)
            if timestamp is not None:
                por_tema += 1

        patch: dict = {}
        if timestamp is not None:
            video_id = item.get("video_id") or video_dir.name.split("_", 1)[-1]
            base = str(item.get("youtube_url") or f"https://www.youtube.com/watch?v={video_id}")
            base = re.sub(r"[?&]t=\d+", "", base)
            separator = "&" if "?" in base else "?"
            patch["youtube_url"] = f"{base}{separator}t={timestamp}"
        if numero and not row.get("numero_processo") and not dje.get("numeroUnico"):
            patch["numero_hint"] = numero
        if tema and not row.get("tema"):
            patch["tema_hint"] = tema[:160]
        if patch:
            patch["id"] = item["id"]
            patch["status"] = item.get("status", "pending")
            patches.append(patch)
            if timestamp is not None:
                patched += 1
        else:
            sem_match += 1

    if patches:
        vistoria_queue._append_lines(patches)
    LOGGER.info(
        "RESUMO: %s itens | com t= gravado: %s (numero=%s, tema=%s) | dicas numero/tema: %s | sem artifacts: %s | sem match: %s",
        len(items), patched, por_numero, por_tema, len(patches) - patched, sem_artifacts, sem_match,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
