"""TAREFA 4: adiciona &t=<seg> aos links do YouTube no Notion que NAO tem timestamp, usando o
start_seconds da JANELA do julgamento no backlog (02_judgment). Indexa o backlog (video_id ->
[(start_seconds, numero_digits, tema_fold)]) e casa cada pagina com link sem 't=' por
video_id + tema (exato) ou numero. So atua quando o match e UNICO (tempo claramente demarcado).

Uso:
  python build_youtube_timestamp.py --backlog-root "H:\\Meu Drive\\TSE_YOUTUBE_NOTION_BACKLOG\\backfill_2025"            # dry-run
  python build_youtube_timestamp.py --backlog-root "..." --apply
"""
from __future__ import annotations

import argparse, glob, json, logging, os, re, time, unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import extract_youtube_video_id
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("build_youtube_timestamp")
ARTIFACT_ROOT = Path("artifacts") / "notion_youtube_timestamp"


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    return re.sub(r"\s+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def _vid(name: str) -> str:
    m = re.match(r"\d+_(.+)$", name)
    return m.group(1) if m else name


def build_index(root: str) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = defaultdict(list)
    pls = [d for d in os.listdir(root) if re.match(r"\d{4}_PL", d) and os.path.isdir(os.path.join(root, d))]
    for pl in pls:
        pld = os.path.join(root, pl)
        for vname in os.listdir(pld):
            vd = os.path.join(pld, vname)
            if not os.path.isdir(vd):
                continue
            vid = _vid(vname)
            for bp in glob.glob(os.path.join(vd, "02_judgment_*.json")):
                try:
                    data = json.load(open(bp, encoding="utf-8"))
                except Exception:
                    continue
                start = data.get("start_seconds")
                if not isinstance(start, int) or start <= 0:
                    continue
                for it in data.get("items", []):
                    idx[vid].append({"start": start, "num": re.sub(r"\D", "", it.get("numero_processo", "") or ""),
                                     "tema": _fold(it.get("tema", ""))})
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backlog-root", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    LOGGER.info("Indexando backlog...")
    idx = build_index(args.backlog_root)
    LOGGER.info("Index: %s videos, %s items", len(idx), sum(len(v) for v in idx.values()))

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    changes: list[dict[str, Any]] = []
    stats = {"sem_timestamp": 0, "match_tema": 0, "match_numero": 0, "ambiguo": 0, "sem_match": 0, "applied": 0, "failed": 0}
    for p in pages:
        link = (t(p, "youtube_link") or "").strip()
        if not link or "t=" in link:
            continue
        stats["sem_timestamp"] += 1
        vid = extract_youtube_video_id(link)
        items = idx.get(vid, [])
        if not items:
            stats["sem_match"] += 1
            continue
        tema = _fold(t(p, "tema"))
        num = re.sub(r"\D", "", t(p, "numero_processo") or "")
        cand = [it for it in items if tema and it["tema"] == tema]
        via = "tema"
        if not cand and num:
            cand = [it for it in items if it["num"] and it["num"] == num]
            via = "numero"
        starts = sorted({it["start"] for it in cand})
        if len(starts) != 1:
            stats["ambiguo" if cand else "sem_match"] += 1
            continue
        stats["match_tema" if via == "tema" else "match_numero"] += 1
        new_link = link + ("&" if "?" in link else "?") + "t=" + str(starts[0])
        rec = {"page_id": p["id"], "numero": t(p, "numero_processo"), "old": link, "new": new_link, "via": via}
        if args.apply:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}",
                                          json={"properties": {"youtube_link": client._build_property_value(schema, "youtube_link", new_link)}})
                stats["applied"] += 1
            except Exception as exc:
                stats["failed"] += 1; rec["erro"] = str(exc)
            time.sleep(0.15)
        changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s | %s", json.dumps(summary, ensure_ascii=False), run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
