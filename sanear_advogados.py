"""Saneamento DEFINITIVO da coluna advogados (multi_select):
 1) html.unescape (decodifica &#39; etc.) + remove fragmentos truncados (terminam em apostrofo);
 2) ESTIRPA junk de papel bled do SADP (Assistente do/Orgao Coator:/Litisconsorte/Recorrido:/
    Coligacao/Partido...) — corta o nome no 1o rotulo de papel;
 3) FUNDE mesma-pessoa: (a) fuzzy (mesmo 1o nome + difflib ratio >= 0.90, ex.: Arraes/Arrais,
    Sousa/Souza, acento) e (b) subset (parcial e prefixo de UM unico mais completo). Canonico =
    mais comum/acentuado/longo. Ambiguos (parcial com varios completos distintos) -> NAO funde.
Reescreve cada pagina (page-value) + dedup. Conservador: na ambiguidade, mantem.

Uso:
  python sanear_advogados.py            # dry-run
  python sanear_advogados.py --apply
"""
from __future__ import annotations

import argparse, collections, difflib, html, json, logging, re, time, unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import dedupe_preserve_order, parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("sanear_advogados")
ARTIFACT_ROOT = Path("artifacts") / "notion_advogados_sanear"
_CONN = {"de", "da", "do", "dos", "das", "e"}
# corta o nome no 1o rotulo de papel/parte que vazou do detalhe do SADP
ROLE_CUT = re.compile(
    r"(?i)\s+(?:assistente\w*|[oó]rg[aã]o\s+coator\w*|litisconsorte\w*|recorr\w+|reclam\w+|interessad\w+|"
    r"impetr\w+|embarg\w+|agrav\w+|requer\w+|denunci\w+|coliga\w*|partido\w*|federa\w*|terceir\w+|"
    r"represent\w+|paciente\w*|exeq\w+|execut\w+|coator\w*|passivo\w*|ativo\w*|\br[eé]u?\s*:|\bautor\s*:)\b.*$")


def fold(s: str) -> str:
    s = re.sub(r"^Dra?\.\s*", "", str(s or ""))
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"\s+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def sig(v: str) -> list[str]:
    return [tk for tk in re.sub(r"[^a-z ]", " ", fold(v)).split() if len(tk) > 1 and tk not in _CONN]


def diac(s: str) -> int:
    return sum(1 for c in unicodedata.normalize("NFKD", str(s)) if unicodedata.combining(c))


def clean_adv(v: str) -> str:
    v = html.unescape(str(v or "")).strip()
    v = re.sub(r"(?i)\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ]+\s*:\s.*$", "", v)  # corta em "Rotulo: ..." (papel bled)
    v = ROLE_CUT.sub("", v).strip(" ,:;-")
    return re.sub(r"\s+", " ", v)


def is_fragmento(v: str) -> bool:
    body = re.sub(r"^Dra?\.\s*", "", v).strip()
    return body.endswith("'") or body.endswith("’") or len(sig(v)) < 1


def _is_subseq(ta: list[str], tb: list[str]) -> bool:
    """ta e subsequencia ORDENADA de tb (tb pode ter tokens extras no meio): catches
    'Sidney Neves' [sidney,neves] dentro de 'Sidney Sa das Neves' [sidney,sa,neves]."""
    it = iter(tb)
    return all(tok in it for tok in ta)


def build_canonical(distinct: dict[str, int]) -> dict[str, str]:
    vals = list(distinct)
    tokmap = {v: sig(v) for v in vals}
    parent: dict[str, str] = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x]); x = parent[x]
        return parent.get(x, x)

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 1) FUZZY full por 1o nome (mesma estrutura, transposicoes/acentos: Andreive/Andréive)
    by_first: dict[str, list[str]] = collections.defaultdict(list)
    for v in vals:
        if tokmap[v]:
            by_first[tokmap[v][0]].append(v)
    for grp in by_first.values():
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                if difflib.SequenceMatcher(None, fold(grp[i]), fold(grp[j])).ratio() >= 0.90:
                    union(grp[i], grp[j])
    # 2) VARIANTE DE GRAFIA por token: mesmo nro de tokens, exatamente 1 posicao difere e o
    #    token-variante e similar. 'chave-curinga' = tokens menos a posicao i. Limiar mais
    #    folgado quando ha >=2 tokens batendo (Socrates Jose Nicklevisk/Niclevicz) e mais
    #    rigido quando so 1 bate (Sidnei/Sidney/Sydney Neves SIM; Carlos/Carla Neves NAO).
    by_wild: dict[tuple, list[tuple[str, int]]] = collections.defaultdict(list)
    for v in vals:
        tv = tokmap[v]
        if len(tv) < 2:
            continue
        for i in range(len(tv)):
            by_wild[(len(tv), tuple(tv[:i]), tuple(tv[i + 1:]))].append((v, i))
    for key, members in by_wild.items():
        thr = 0.70 if (key[0] - 1) >= 2 else 0.80
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                (va, ia), (vb, _ib) = members[x], members[y]
                if va == vb:
                    continue
                if difflib.SequenceMatcher(None, tokmap[va][ia], tokmap[vb][ia]).ratio() >= thr:
                    union(va, vb)
    # canonico por cluster: mais PALAVRAS > mais ACENTO > mais LONGO > mais comum
    canon: dict[str, str] = {}
    clusters: dict[str, list[str]] = collections.defaultdict(list)
    for v in vals:
        clusters[find(v)].append(v)
    for grp in clusters.values():
        if len(grp) > 1:
            best = max(grp, key=lambda v: (len(v.split()), diac(v), len(v), distinct[v]))
            for v in grp:
                if v != best:
                    canon[v] = best
    # 3) SUBSEQUENCIA: parcial e subsequencia ordenada de UM unico mais completo (nome do meio
    #    omitido: 'Sidney Neves' -> 'Sidney Sa das Neves')
    reduced = sorted(set(canon.get(v, v) for v in vals))
    rtok = {v: sig(v) for v in reduced}
    for v in reduced:
        tv = rtok[v]
        if len(tv) < 2:
            continue
        fullers = [w for w in reduced if w != v and len(rtok[w]) > len(tv) and _is_subseq(tv, rtok[w])]
        if len(fullers) == 1:
            canon[v] = fullers[0]
    # resolve cadeias: Sydney->Sidney->'Sidney Sa das Neves' vira Sydney->'Sidney Sa das Neves'
    def _resolve(v: str) -> str:
        seen = set()
        while v in canon and v not in seen:
            seen.add(v); v = canon[v]
        return v
    return {v: _resolve(v) for v in canon}


def main() -> int:
    ap = argparse.ArgumentParser()
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

    # distinct dos valores JA limpos (unescape+role-strip), p/ casar o canonical
    distinct = collections.Counter()
    for p in pages:
        for a in parse_multi_value_text(t(p, "advogados")):
            c = clean_adv(a)
            if c and not is_fragmento(c):
                distinct[c] += 1
    canon = build_canonical(distinct)

    changes: list[dict[str, Any]] = []
    stats = {"distintos_limpos": len(distinct), "merges": len(canon), "paginas_alteradas": 0,
             "fragmentos_removidos": 0, "applied": 0, "falhas": 0}
    for p in pages:
        cur = parse_multi_value_text(t(p, "advogados"))
        if not cur:
            continue
        new: list[str] = []
        for a in cur:
            c = clean_adv(a)
            if not c or is_fragmento(c):
                stats["fragmentos_removidos"] += 1
                continue
            new.append(canon.get(c, c))
        new = dedupe_preserve_order(new)
        if new == cur:
            continue
        stats["paginas_alteradas"] += 1
        rec = {"page_id": p["id"], "old": cur, "new": new}
        if args.apply:
            props = {"advogados": client._build_property_value(schema, "advogados", new) or client._build_empty_property_value(schema, "advogados")}
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                stats["applied"] += 1
            except Exception as exc:
                stats["falhas"] += 1; rec["erro"] = str(exc)
            time.sleep(0.15)
        changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "canonical.json").write_text(json.dumps(canon, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s | %s", json.dumps(summary, ensure_ascii=False), run_dir)
    for v, cc in list(canon.items())[:25]:
        LOGGER.info("  merge: %r -> %r", v, cc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
