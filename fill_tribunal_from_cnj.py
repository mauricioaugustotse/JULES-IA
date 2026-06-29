"""Preenche a coluna `tribunal` VAZIA derivando do proprio CNJ-20 da pagina (segmento do
tribunal): TR=='00' -> 'TSE' (processo originario/nacional do TSE); senao -> 'TRE-<UF>', onde a
UF e aprendida do mapeamento TR->UF observado nas paginas que JA tem CNJ-20 + origem.
So PREENCHE vazios (nao sobrescreve os ja preenchidos). Conservador: pula quando nao ha CNJ-20
ou o codigo de tribunal nao mapeia para uma UF conhecida. dry-run por padrao.

Uso:
  python fill_tribunal_from_cnj.py            # dry-run
  python fill_tribunal_from_cnj.py --apply
"""
from __future__ import annotations

import argparse, collections, json, logging, re, time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("fill_tribunal_from_cnj")
ARTIFACT_ROOT = Path("artifacts") / "notion_tribunal_cnj"


def digits(s):
    return re.sub(r"\D", "", str(s or ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    # TR code (CNJ[14:16]) -> UF, aprendido das paginas que JA tem CNJ-20 + tribunal TRE-XX
    tr_counter = collections.Counter()
    for p in pages:
        u = digits(t(p, "numero_processo"))
        if len(u) >= 20 and u[14:16] != "00":
            m = re.search(r"\bTRE-([A-Z]{2})\b", t(p, "tribunal"))
            if not m:
                m = re.search(r"/([A-Z]{2})$", t(p, "origem"))
            if m:
                tr_counter[(u[14:16], m.group(1))] += 1
    TR2UF: dict[str, str] = {}
    for (tr, uf), _n in tr_counter.most_common():
        TR2UF.setdefault(tr, uf)
    LOGGER.info("mapa TR->UF aprendido: %s codigos", len(TR2UF))

    changes: list[dict[str, Any]] = []
    stats = collections.Counter()
    for p in pages:
        if t(p, "tribunal").strip():
            continue  # so preenche vazios
        stats["vazios"] += 1
        u = digits(t(p, "numero_processo"))
        if len(u) < 20:
            stats["sem_cnj"] += 1
            continue
        tr = u[14:16]
        if tr == "00":
            novo = "TSE"
        else:
            uf = TR2UF.get(tr)
            if not uf:
                stats["tr_nao_mapeado"] += 1
                continue
            novo = f"TRE-{uf}"
        stats["preenchidos"] += 1
        stats[novo if novo == "TSE" else "TRE-*"] += 1
        rec = {"page_id": p["id"], "numero": t(p, "numero_processo"), "new": novo}
        if args.apply:
            try:
                built = client._build_property_value(schema, "tribunal", novo)
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": {"tribunal": built}})
                rec["status"] = "updated"; stats["applied"] += 1
            except Exception as exc:
                rec["status"] = "failed"; rec["error"] = str(exc); stats["failed"] += 1
            time.sleep(0.12)
        changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **dict(stats)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:15]:
        LOGGER.info("  %s -> %s", c["numero"], c["new"])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
