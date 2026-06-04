"""Re-checa via CNJ DataJud os processos marcados como SUSPENSOS por vista
(votacao="Suspenso" + resultado="Suspenso por vista") e, quando o DataJud mostra um
movimento DECISIVO (julgamento de merito / provimento / negacao de seguimento / baixa
definitiva / transito em julgado) com data POSTERIOR a sessao da suspensao, conclui que
o processo foi julgado depois e troca as etiquetas:

    votacao   "Suspenso"          -> "Suspenso*"
    resultado "Suspenso por vista" -> "Suspenso mas julgado depois"

Conservador: so atua em processos com CNJ de 20 digitos e com movimentos no DataJud; se
nao houver movimento decisivo posterior (ou nao houver movimentos), NAO mexe.

Uso:
    python recheck_suspenso_via_datajud.py            # dry-run (relatorio)
    python recheck_suspenso_via_datajud.py --apply    # aplica as trocas
    python recheck_suspenso_via_datajud.py --limit 30 # amostra
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from audit_notion_sessoes_round2 import notion_request_with_retry
from cnj_datajud import API_BASE, API_KEY, _fold, _indices_for
from local_secrets import get_secret
from tse_normalization import canonicalize_numero_processo
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("recheck_suspenso")
ARTIFACT_ROOT = Path("artifacts") / "notion_suspenso_recheck"
APPLY_SLEEP_SECONDS = 0.2

# Movimentos que indicam JULGAMENTO/CONCLUSAO definitiva (substrings, texto foldado).
# Calibrado pelo vocabulario real dos movimentos pos-sessao desses processos.
DECISIVE_TERMS = (
    "transito em julgado", "baixa definitiva", "definitivo", "merito",
    "provimento", "nao-provimento", "nao provimento", "negacao de seguimento",
    "negativa de seguimento", "negacao de provimento", "acolhimento de embargos",
    "nao-acolhimento de embargos", "improcedencia", "procedencia", "homologacao",
    "extincao", "perda do objeto", "nao conhecimento", "nao-conhecimento",
    "julgamento", "acordao",
)
# Substrings que NAO contam como decisivas mesmo contendo um termo acima.
NON_DECISIVE = ("mero expediente", "ato ordinatorio", "outras decisoes", "conclusao para")


def is_decisive(nome: str) -> bool:
    n = _fold(nome)
    if any(bad in n for bad in NON_DECISIVE):
        return False
    return any(term in n for term in DECISIVE_TERMS)


def _digits(s: Any) -> str:
    return re.sub(r"\D", "", str(s or ""))


def datajud_movimentos(num: str, trib: str, session: requests.Session) -> list[tuple[str, str]]:
    """Retorna [(data_iso, nome_foldado)] do processo (uniao entre os indices)."""
    d20 = _digits(num)[:20]
    if len(d20) < 20:
        return []
    movs: set[tuple[str, str]] = set()
    for alias in _indices_for(trib):
        url = f"{API_BASE}/{alias}/_search"
        try:
            r = session.post(url, headers={"Authorization": API_KEY, "Content-Type": "application/json"},
                             json={"query": {"match": {"numeroProcesso": d20}}, "size": 10}, timeout=40)
            hits = r.json().get("hits", {}).get("hits", []) if r.status_code < 400 else []
        except Exception:
            hits = []
        time.sleep(0.12)
        for h in hits:
            s = h.get("_source", {})
            if _digits(s.get("numeroProcesso", ""))[:20] != d20:
                continue
            for m in (s.get("movimentos", []) or []):
                dh = str(m.get("dataHora", ""))[:10]
                if dh:
                    movs.add((dh, _fold(m.get("nome", ""))))
    return sorted(movs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-checa Suspenso por vista via CNJ DataJud (julgado depois?).")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    client = NotionSessoesClient(api_key=key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    targets = [p for p in pages if t(p, "votacao").strip() == "Suspenso" and t(p, "resultado").strip() == "Suspenso por vista"]
    short = [p for p in targets if len(_digits(t(p, "numero_processo"))) < 20]
    full = [p for p in targets if len(_digits(t(p, "numero_processo"))) >= 20]
    if args.limit and args.limit < len(full):
        full = full[: args.limit]
    LOGGER.info("Alvos: %s (CNJ-20: %s | curtos pulados: %s)", len(targets), len(full), len(short))

    sess = requests.Session()
    changes: list[dict[str, Any]] = []
    stats = {"checados": 0, "flip": 0, "sem_movimentos": 0, "pendente": 0, "sem_data_sessao": 0, "applied": 0, "failed": 0}
    for i, p in enumerate(full, 1):
        num = t(p, "numero_processo"); trib = t(p, "tribunal"); ds = (t(p, "data_sessao") or "")[:10]
        stats["checados"] += 1
        if not re.match(r"\d{4}-\d{2}-\d{2}", ds):
            stats["sem_data_sessao"] += 1
            continue
        movs = datajud_movimentos(num, trib, sess)
        if not movs:
            stats["sem_movimentos"] += 1
            continue
        decisive_after = [(dh, nm) for dh, nm in movs if dh > ds and is_decisive(nm)]
        if not decisive_after:
            stats["pendente"] += 1
            continue
        stats["flip"] += 1
        trigger = decisive_after[0]
        rec = {
            "page_id": p["id"], "numero": canonicalize_numero_processo(num), "tribunal": trib,
            "data_sessao": ds, "gatilho": {"data": trigger[0], "movimento": trigger[1]},
            "decisivos_apos": [{"data": d, "movimento": n} for d, n in decisive_after[:6]],
            "detail": {"votacao": {"old": "Suspenso", "new": "Suspenso*"},
                       "resultado": {"old": "Suspenso por vista", "new": "Suspenso mas julgado depois"}},
        }
        if args.apply:
            props = {"votacao": {"select": {"name": "Suspenso*"}},
                     "resultado": {"select": {"name": "Suspenso mas julgado depois"}}}
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                rec["status"] = "updated"; stats["applied"] += 1
            except Exception as exc:
                rec["status"] = "failed"; rec["error"] = str(exc); stats["failed"] += 1
            time.sleep(APPLY_SLEEP_SECONDS)
        changes.append(rec)
        if i % 20 == 0:
            LOGGER.info("...%s/%s | flips %s", i, len(full), stats["flip"])

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats, "curtos_pulados": len(short)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:20]:
        LOGGER.info("  FLIP %s [%s] sessao=%s -> julgado %s (%s)", c["numero"], c["tribunal"], c["data_sessao"],
                    c["gatilho"]["data"], c["gatilho"]["movimento"])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
