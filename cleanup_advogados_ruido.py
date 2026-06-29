"""Limpa RUIDO de borda na coluna `advogados` (rich_text) da base de sessoes:
  - travessao/hifen/bullet residual nas bordas (ex.: "Dr. Fulano –" -> "Dr. Fulano"), sobra do OAB
    removido (o strip antigo so tirava '-', nao en-dash '–'/em-dash '—');
  - parentese aberto sem fechar no fim (ex.: "Dra. Fulana (pelo" -> "Dra. Fulana");
  - pontuacao de borda; descarta entradas que sobrarem vazias / so com prefixo (Dr./Dra.).
NAO re-normaliza nomes validos nem mexe em travessao/hifen no MEIO do nome. dry-run por padrao.

Uso:
  python cleanup_advogados_ruido.py            # dry-run
  python cleanup_advogados_ruido.py --apply
"""
from __future__ import annotations

import argparse, json, logging, re, time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import dedupe_preserve_order, parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("cleanup_advogados_ruido")
ARTIFACT_ROOT = Path("artifacts") / "notion_advogados_ruido"
BORDAS = " \t.,;:/\\-–—•·"
# "e outro"/"e outros"/"e outra(s)" [+ qualquer coisa] ate o fim -> marcador, nao nome
_E_OUTROS_RE = re.compile(r"(?i)\s*\be\s+outr[oa]s?\b.*$")
_RUIDO_CORPO = {"e", "e outro", "e outros", "e outra", "e outras", "outro", "outros", "outra", "outras"}


def limpa(nome: str) -> str:
    n = str(nome or "")
    n = re.sub(r"\s*\([^)]*$", "", n)      # parentese aberto sem fechar no fim
    n = _E_OUTROS_RE.sub("", n)             # marcador "e outro(s)" colado ao nome/prefixo
    n = re.sub(r"\s{2,}", " ", n)
    n = n.strip(BORDAS)                      # bordas: espaco, pontuacao, travessoes, bullets
    return n


def valido(n: str) -> bool:
    if not n:
        return False
    corpo = re.sub(r"(?i)^dra?\.\s*", "", n).strip(BORDAS)
    if len(corpo) < 3:
        return False  # vazio / so prefixo "Dr."/"Dra."
    if corpo.lower() in _RUIDO_CORPO:
        return False  # corpo e so marcador de ruido (ex.: "Dr. e outro")
    return True


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

    changes: list[dict[str, Any]] = []
    stats = {"paginas": 0, "paginas_mudadas": 0, "entradas_limpas": 0, "entradas_removidas": 0, "applied": 0, "failed": 0}
    for p in pages:
        stats["paginas"] += 1
        cur = parse_multi_value_text(t(p, "advogados"))
        if not cur:
            continue
        novo: list[str] = []
        limpas = removidas = 0
        for n in cur:
            c = limpa(n)
            if not valido(c):
                removidas += 1
                continue
            if c != n:
                limpas += 1
            novo.append(c)
        novo = dedupe_preserve_order(novo)
        if novo == cur:
            continue
        stats["paginas_mudadas"] += 1
        stats["entradas_limpas"] += limpas
        stats["entradas_removidas"] += removidas
        rec = {"page_id": p["id"], "numero": t(p, "numero_processo"), "old": cur, "new": novo}
        if args.apply:
            try:
                built = client._build_property_value(schema, "advogados", novo) or client._build_empty_property_value(schema, "advogados")
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": {"advogados": built}})
                rec["status"] = "updated"; stats["applied"] += 1
            except Exception as exc:
                rec["status"] = "failed"; rec["error"] = str(exc); stats["failed"] += 1
            time.sleep(0.12)
        changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:12]:
        difs = [f"{o!r}->{n!r}" for o, n in zip(c["old"], c["new"]) if o != n] or [f"{len(c['old'])}->{len(c['new'])} itens"]
        LOGGER.info("  [%s] %s", c["numero"], "; ".join(difs[:3]))
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
