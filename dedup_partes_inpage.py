"""Remove DUPLICATAS da mesma pessoa/entidade DENTRO da mesma pagina na coluna partes
(rich_text). Pega variacoes de GRAFIA (Arthur/Artur, Souza/Sousa) e de FORMA de instituicao
(Corregedoria ... (CRE-SP) vs Corregedoria ... do Tre/Sp) — que o fold de acento/caixa nao
captura. Fuzzy match (difflib ratio >= 0.85) OU mesma cabeca institucional + 3 tokens iniciais
iguais. Mantem a forma mais completa/acentuada/oficial. So atua DENTRO da pagina (contexto do
mesmo caso reduz o risco).

Uso:
  python dedup_partes_inpage.py            # dry-run
  python dedup_partes_inpage.py --apply
"""
from __future__ import annotations

import argparse, difflib, json, logging, re, time, unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import dedupe_preserve_order, parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("dedup_partes_inpage")
ARTIFACT_ROOT = Path("artifacts") / "notion_partes_dedup_inpage"
CONN = {"de", "da", "do", "dos", "das", "e", "em", "no", "na", "a", "o"}
# instituicoes SINGULARES por regiao (sufixo = forma, nao entidade distinta). NAO inclui
# federacao/diretorio/coligacao/comissao (sufixo de local distingue entidades reais).
INST = {"corregedoria", "tribunal", "procuradoria", "cartorio", "juizo", "ministerio",
        "promotoria", "delegacia", "ouvidoria", "defensoria"}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    return re.sub(r"\s+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def diacritics(s):
    return sum(1 for c in unicodedata.normalize("NFKD", str(s)) if unicodedata.combining(c))


def tokens(v):
    base = re.sub(r"(?i)^\s*dr[a]?\.?\s+", "", str(v or ""))  # tira Dr./Dra. (advogados)
    base = re.sub(r"\([^)]*\)", " ", base)
    return [t for t in fold(base).split() if len(t) > 1 and t not in CONN]


def same_entity(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return False
    # VARIANTE DE GRAFIA: mesma contagem de tokens, na mesma ordem, cada par quase-igual
    # (ratio>=0.8). Rejeita 1 token TOTALMENTE diferente (Carvalho vs Dantas, Avancar vs Renascer)
    # e troca de GENERO no fim (Fernanda/Fernando, Roberta/Roberto = pessoa diferente).
    if len(ta) == len(tb) and ta != tb:
        gender = any(x[:-1] == y[:-1] and {x[-1:], y[-1:]} == {"a", "o"} for x, y in zip(ta, tb) if x != y)
        if not gender and all(x == y or difflib.SequenceMatcher(None, x, y).ratio() >= 0.8 for x, y in zip(ta, tb)):
            return True
    # INSTITUICAO SINGULAR: mesma cabeca institucional + 3 tokens iniciais iguais (forma/sigla varia)
    if len(ta) >= 3 and len(tb) >= 3 and ta[:3] == tb[:3] and ta[0] in INST:
        return True
    return False


def canonical(group):
    return max(group, key=lambda v: (diacritics(v), 1 if re.search(r"\([A-Za-z]{2,}\)", v) else 0, len(v)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--column", default="partes", help="partes ou advogados")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    col = args.column
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    changes: list[dict[str, Any]] = []
    stats = {"paginas": 0, "afetadas": 0, "fusoes": 0, "applied": 0, "failed": 0}
    for p in pages:
        stats["paginas"] += 1
        cur = parse_multi_value_text(client._extract_property_text(p, schema, col))
        if len(cur) < 2:
            continue
        clusters: list[list[str]] = []
        for v in cur:
            for cl in clusters:
                if same_entity(v, cl[0]):
                    cl.append(v)
                    break
            else:
                clusters.append([v])
        new = dedupe_preserve_order([canonical(cl) for cl in clusters])
        if new == cur:
            continue
        merges = [cl for cl in clusters if len(cl) > 1]
        stats["afetadas"] += 1
        stats["fusoes"] += sum(len(cl) - 1 for cl in merges)
        rec = {"page_id": p["id"], "numero": client._extract_property_text(p, schema, "numero_processo"),
               "fusoes": [{"mantido": canonical(cl), "removidos": [x for x in cl if x != canonical(cl)]} for cl in merges]}
        if args.apply:
            built = client._build_property_value(schema, col, new) or client._build_empty_property_value(schema, col)
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": {col: built}})
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
    for c in changes[:30]:
        for f in c["fusoes"]:
            LOGGER.info("  [%s] mantem %r <- %r", c["numero"], f["mantido"], f["removidos"])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
