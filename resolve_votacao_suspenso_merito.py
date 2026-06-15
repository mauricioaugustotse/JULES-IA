"""Resolve a etiqueta ``votacao = 'Suspenso*'`` para a forma REAL da votacao
(``Unânime`` ou ``Por maioria``) nos registros que foram EFETIVAMENTE JULGADOS na
propria sessao — isto e, cujo ``resultado`` e de merito (Provido, Desprovido,
Improcedente, Procedente, Rejeitados, ...), e NAO ``Suspenso por vista`` nem
``Suspenso mas julgado depois``. Nesses casos ``Suspenso*`` esta incorreto: o
processo nao ficou suspenso, foi decidido.

Fonte da forma de votacao: a descricao do julgamento ja presente no proprio
registro (campos gerados a partir do video da sessao). Regra:
  - menciona "por maioria" / "voto divergente" / "vencido" / "divergencia" / "dissidencia"
    -> ``Por maioria``;
  - sem qualquer mencao de divergencia -> ``Unânime`` (caso default do TSE; quando ha
    divergencia, ela e sempre registrada no resumo).

NAO toca os registros com ``resultado`` suspenso (esses ``Suspenso*`` estao corretos).
Idempotente: re-consulta o estado vivo e so altera paginas AINDA em ``Suspenso*``.
``votacao`` e um SELECT e ambas as opcoes ('Unânime', 'Por maioria') ja existem — o
PATCH e apenas page-value, nao mexe nas opcoes/cores (sem o footgun de multi_select).

Uso:
  python resolve_votacao_suspenso_merito.py            # dry-run + relatorio
  python resolve_votacao_suspenso_merito.py --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("resolve_votacao_suspenso_merito")
ARTIFACT_ROOT = Path("artifacts") / "notion_votacao_suspenso_merito"

SUSPENSO_STAR = "Suspenso*"
# resultados que indicam que NAO houve julgamento de merito naquele registro
RESULTADO_NAO_MERITO = {"", "Suspenso por vista", "Suspenso mas julgado depois"}
# campos do registro que descrevem o julgamento (geram o sinal de votacao)
TEXT_FIELDS = ("punchline", "raciocinio_juridico", "analise_do_conteudo_juridico",
               "fundamentacao_normativa", "tema")
# sinal FORTE de votacao por maioria / divergencia (TSE registra sempre que ha)
MAIORIA_RE = re.compile(
    r"\bpor maioria\b|\bmaioria de votos\b|\bvotos? divergentes?\b|\bvoto divergente\b|"
    r"\bvencid\w*\b|\bdivergiu\b|\bdivergenc\w*\b|\bdissid\w*\b",
    re.I,
)


def deaccent(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    return "".join(c for c in s if not unicodedata.combining(c))


def classify(textos: str) -> tuple[str, str]:
    """Retorna (votacao_nova, frase_evidencia). 'Por maioria' se houver sinal de
    divergencia; caso contrario 'Unânime'."""
    flat = deaccent(textos)
    m = MAIORIA_RE.search(flat)
    if m:
        # frase em torno do match, p/ auditoria
        ini = flat.rfind(".", 0, m.start()) + 1
        fim = flat.find(".", m.end())
        fim = fim + 1 if fim != -1 else m.end() + 80
        return "Por maioria", flat[ini:fim].strip()[:240]
    return "Unânime", ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resolve votacao Suspenso* -> Unânime/Por maioria nos julgados de merito.")
    ap.add_argument("--apply", action="store_true", help="Aplica as trocas (padrao: dry-run).")
    ap.add_argument("--plan", default="", help="JSON {page_id: 'Unânime'|'Por maioria'} curado; "
                    "sobrepoe a classificacao por regex (fallback) quando presente.")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--artifact-dir", default="")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    api_key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY/NOTION_TOKEN nao encontrado.")
    client = NotionSessoesClient(api_key=api_key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    opts = schema.properties["votacao"].options
    for needed in ("Unânime", "Por maioria"):
        if needed not in opts:
            raise RuntimeError(f"Opcao de votacao ausente no schema: {needed!r} (opcoes: {opts})")
    plan: dict[str, str] = {}
    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        LOGGER.info("Plano curado carregado: %s entradas (%s).", len(plan), args.plan)

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    pages = client.query_data_source()
    changes: list[dict[str, Any]] = []
    applied = failed = skipped = 0
    for p in pages:
        if t(p, "votacao").strip() != SUSPENSO_STAR:
            continue
        resultado = t(p, "resultado").strip()
        if resultado in RESULTADO_NAO_MERITO:
            continue  # 'Suspenso*' esta correto nesses (suspenso de fato)
        blob = "  ".join(t(p, f) for f in TEXT_FIELDS)
        nova_regex, evidencia = classify(blob)
        nova = plan.get(p["id"], nova_regex)
        source = "plan" if p["id"] in plan else "regex"
        rec = {
            "page_id": p["id"],
            "url": p.get("url", ""),
            "numero_processo": t(p, "numero_processo"),
            "data_sessao": t(p, "data_sessao"),
            "resultado": resultado,
            "old": SUSPENSO_STAR,
            "new": nova,
            "regex": nova_regex,
            "source": source,
            "evidencia": evidencia,
            "status": "planned",
        }
        if args.apply:
            try:
                built = client._build_property_value(schema, "votacao", nova)
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": {"votacao": built}})
                rec["status"] = "updated"
                applied += 1
                time.sleep(0.12)
            except Exception as exc:
                rec["status"] = "failed"
                rec["error"] = str(exc)[:200]
                failed += 1
                LOGGER.warning("falha %s: %s", rec["numero_processo"], str(exc)[:120])
        changes.append(rec)

    n_maioria = sum(1 for c in changes if c["new"] == "Por maioria")
    n_unanime = sum(1 for c in changes if c["new"] == "Unânime")
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "alvos": len(changes),
        "por_maioria": n_maioria,
        "unanime": n_unanime,
        "applied": applied,
        "failed": failed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    art = Path(args.artifact_dir) if args.artifact_dir else ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    art.mkdir(parents=True, exist_ok=True)
    (art / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    (art / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    LOGGER.info("Relatorio: %s", art)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
