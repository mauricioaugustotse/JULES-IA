"""Renumera tipo_registro ('Julgamento N') nas sessoes onde ha DUPLICATA (tipicamente
sessoes com mais de um video, onde a numeracao reinicia por video). Por sessao, ordena os
julgamentos por (video_id, timestamp do youtube_link) e atribui 'Julgamento 1..N'. So toca
sessoes com duplicata; escrita page-value segura.

Uso:
  python fix_tipo_registro_renumber.py            # dry-run
  python fix_tipo_registro_renumber.py --apply
"""
from __future__ import annotations

import argparse, json, logging, re, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import extract_youtube_video_id
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("fix_tipo_registro_renumber")
ARTIFACT_ROOT = Path("artifacts") / "notion_tipo_registro_renumber"


def vid_t(link: str) -> tuple[str, int]:
    vid = extract_youtube_video_id(link or "")
    m = re.search(r"[?&]t=(\d+)", link or "")
    return (vid, int(m.group(1)) if m else 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    client = NotionSessoesClient(api_key=key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    by_sess: dict[str, list[Any]] = defaultdict(list)
    for p in pages:
        ds = (t(p, "data_sessao") or "")[:10]
        tr = (t(p, "tipo_registro") or "").strip()
        if ds and tr:
            by_sess[ds].append(p)

    changes: list[dict[str, Any]] = []
    stats = {"sessoes_dup": 0, "paginas_renumeradas": 0, "applied": 0, "failed": 0}
    for ds, ps in by_sess.items():
        trs = [(t(p, "tipo_registro") or "").strip() for p in ps]
        if len(set(trs)) == len(trs):
            continue  # sem duplicata
        stats["sessoes_dup"] += 1
        ordered = sorted(ps, key=lambda p: vid_t(t(p, "youtube_link")))
        for i, p in enumerate(ordered, 1):
            new = f"Julgamento {i}"
            old = (t(p, "tipo_registro") or "").strip()
            if new == old:
                continue
            stats["paginas_renumeradas"] += 1
            rec = {"page_id": p["id"], "data": ds, "numero": t(p, "numero_processo"), "old": old, "new": new}
            if args.apply:
                built = client._build_property_value(schema, "tipo_registro", new)
                try:
                    notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": {"tipo_registro": built}})
                    rec["status"] = "updated"; stats["applied"] += 1
                except Exception as exc:
                    rec["status"] = "failed"; rec["error"] = str(exc); stats["failed"] += 1
                time.sleep(0.2)
            changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:15]:
        LOGGER.info("  [%s] %s -> %s (%s)", c["data"], c["old"], c["new"], c["numero"])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
