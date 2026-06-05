"""Limpa etiquetas ruins de partes (SEM gerar duplicidade):
  1) FRAGMENTO de localidade: valor que comeca com conector minusculo (de/da/do/no/na/em)
     -> remove (ex.: 'de Nova Era/MG'). Preserva 'De Volta Ao Trabalho' (D maiusculo).
  2) Sufixo 'e Outro(s)/Outra(s)/e mais N' -> estripa (ex.: 'Jair ... e Outro' -> 'Jair ...').
  3) Gemeos de ACENTO/CAIXA -> consolida para a forma ACENTUADA e PROPRIA (preserva siglas,
     que nao tem gemeo). Tudo deduplicado por pagina.

Uso:
  python clean_partes_labels.py            # dry-run
  python clean_partes_labels.py --apply
"""
from __future__ import annotations

import argparse, json, logging, re, time, unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import dedupe_preserve_order, parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("clean_partes_labels")
ARTIFACT_ROOT = Path("artifacts") / "notion_partes_clean"
CONN = {"de", "da", "do", "dos", "das", "no", "na", "em"}
EOUTRO_RE = re.compile(r"(?i)\s*,?\s*\be\s+(outr[oa]s?|mais\s+\d+)\b\s*$")
INITIALS_RE = re.compile(r"^(?:[A-Za-zÀ-ÿ]\.?\s+)+[A-Za-zÀ-ÿ]\.?$")  # ex.: 'A. C', 'L. D. D. P. A'
# valores que sao SO papel/ruido (nunca nome real de parte)
JUNK_WORDS = {"interessados", "interessado", "assessoria", "coligacao", "coligacoes",
              "requerente", "requerentes", "recorrente", "recorrentes", "embargante",
              "embargantes", "agravante", "agravantes", "outros", "outro", "outras",
              "advogado", "advogados", "advogada", "eleitor", "eleitores", "candidato",
              "candidatos", "partido", "diretorio", "federacao"}
# frases-ruido (designacoes processuais que viraram "parte")
JUNK_PHRASES = {"destinatario para ciencia publica", "para ciencia publica",
                "ciencia publica", "interessado nao identificado", "nao identificado",
                "a quem possa interessar", "terceiro interessado", "terceiros interessados"}


def _fold0(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    return re.sub(r"\s+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def is_fragment(v: str) -> bool:
    toks = str(v or "").split()
    return bool(toks) and toks[0][:1].islower() and toks[0].lower() in CONN


def is_junk(v: str) -> bool:
    v = str(v or "").strip()
    if _fold0(v) in JUNK_WORDS or _fold0(v) in JUNK_PHRASES:
        return True
    core = re.sub(r"[^A-Za-zÀ-ÿ]", "", v)
    if len(core) <= 6 and INITIALS_RE.match(v):  # iniciais soltas
        return True
    return False


def strip_eoutro(v: str) -> str:
    prev = None
    while prev != v:
        prev = v
        v = EOUTRO_RE.sub("", v).strip().rstrip(",;").strip()
    return v


_ROLE = (r"recorrente|recorrid[oa]s?|reclamante|reclamad[oa]s?|agravante|agravad[oa]s?|"
         r"embargante|embargad[oa]s?|requerente|requerid[oa]s?|assistente(?:\s+simples)?|"
         r"consultante|consulente|petrante|impugnante|litisconsorte\w*|paciente|investigad[oa]|"
         r"interessad[oa]s?|autoridade\s+coatora|(?:primeir[oa]|segund[oa]|terceir[oa])\s+\w+")
ALIAS_PREFIX = re.compile(r"(?i)^(?:registrad[oa]\s+civilmente\s+como|conhecid[oa]\s+como|"
                          r"tamb[eé]m\s+conhecid[oa]\s+como|vulgo|nome\s+social)\s*:?\s+")
# prefixo de papel, inclusive COMPOSTO: "Recorrido(a):", "Recorrido/Reclamado:", "Agravados/Embargados:"
PREFIX_ROLE = re.compile(rf"(?i)^(?:{_ROLE})(?:\s*\([a-zà-ÿ]+\)|\s*/\s*[a-zà-ÿ]+)*\s*:\s*")
TRAIL_ROLE = re.compile(rf"(?i)\s*\((?:{_ROLE})[^)]*\)\s*$")


def strip_roles(v: str) -> str:
    v = ALIAS_PREFIX.sub("", str(v or "")).strip()
    v = PREFIX_ROLE.sub("", v).strip()
    v = TRAIL_ROLE.sub("", v).strip()
    return v


def clean_value(v: str) -> str:
    prev = None
    v = str(v or "")
    while prev != v:
        prev = v
        v = strip_roles(strip_eoutro(v))
    return v


ENTITY = (r"(?:Coliga|Partido|Federa[cç]|Diret[oó]rio|Comiss[aã]o\s+Provis|Movimento\s+Democr|"
          r"Sindicato|Associa[cç]|Funda[cç][aã]o|Ju[ií]zo|Tribunal|C[aâ]mara\s+Munic)")
_SEP = ""


def split_parties(v: str) -> list[str]:
    """Divide multi-partes em entidades unicas SO em fronteiras seguras: ' x ', '(papel) e
    <Entidade>' e ' e <Entidade>'. NAO divide ' e ' generico (pode ser nome de coligacao)."""
    v = str(v or "")
    sep = "\x01"
    v = re.sub(r"\s+x\s+", sep, v)  # ' x ' minusculo (litigio); evita 'Pio X Fernandes' (inicial)
    # parentese de PAPEL seguido de ' e ' = fronteira clara (qualquer coisa depois)
    v = re.sub(rf"(?i)(\((?:{_ROLE})[^)]*\))\s+e\s+", r"\1" + sep, v)
    v = re.sub(rf"(?i)\)\s+e\s+(?={ENTITY})", ")" + sep, v)
    v = re.sub(rf"(?i)\s+e\s+(?={ENTITY})", sep, v)
    parts = [p.strip() for p in v.split(sep) if p.strip()]
    out = []
    for p in parts:
        out.extend(_split_person_e(p))
    return out


def has_ambiguous_e(v: str) -> bool:
    """Sobra ' e ' (fora de entidade) apos split -> possivel multi-parte nao resolvido."""
    return bool(re.search(r"(?i)\s+e\s+", v)) and not re.search(rf"(?i)\s+e\s+{ENTITY}", v)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "").lower())
    return re.sub(r"\s+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


COMPANY_MARK = re.compile(r"(?i)\b(ltda|eireli|epp|s\.?\s*/?\s*a\.?\b|\bme\b|ind[uú]stria|com[eé]rcio|comunica|coopera|eventos|projetos|consultoria|entretenimento|educa[cç]|r[aá]dio|\btv\b|gr[aá]fica|editora|associa|sindicato|funda[cç])")
NAME_HEAD = re.compile(r"(?i)^(coliga|partido|federa|diret|frente|movimento|comiss|tribunal|ju[ií]zo|corregedoria|procuradoria|minist[eé]rio|c[aâ]mara|prefeitura)")


def _sig_tokens(s):
    base = re.sub(r"\([^)]*\)", " ", str(s or ""))
    return [t for t in fold(base).split() if len(t) > 1 and t not in CONN]


def is_person_name(s):
    s = str(s or "").strip()
    if not s or s[:1].islower() or COMPANY_MARK.search(s) or NAME_HEAD.match(s):
        return False
    return 2 <= len(_sig_tokens(s)) <= 4


BOUNDARY = re.compile(r"(?i)(\)|[-–]\s*(nacional|estadual|municipal|distrital|regional))\s*$")


def _split_person_e(p):
    """Divide na ULTIMA ' e ' quando a direita e NOME DE PESSOA e a esquerda TERMINA em
    fronteira CLARA de entidade (sigla '(...)' ou '- Nacional/Estadual/...'). Isso evita
    quebrar nomes de coligacao/partido ('Coligacao Coragem e Atitude', 'Partido Socialismo
    e Liberdade') e sobrenomes ('Nogueira e Silva'). Recursa a esquerda."""
    for m in reversed(list(re.finditer(r"\s+e\s+", p))):
        right = p[m.end():].strip()
        left = p[:m.start()].strip()
        if is_person_name(right) and BOUNDARY.search(left) and len(_sig_tokens(left)) >= 2:
            return _split_person_e(left) + [right]
    return [p]


def diacritics(s: str) -> int:
    return sum(1 for c in unicodedata.normalize("NFKD", str(s)) if unicodedata.combining(c))


def has_lower(s: str) -> int:
    return 1 if any(c.islower() for c in s) else 0


def pick_canonical(variants: list[str]) -> str:
    # prefere: mais acentos, depois forma propria (tem minuscula > CAIXA ALTA), depois mais longa
    return max(variants, key=lambda v: (diacritics(v), has_lower(v), len(v)))


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

    distinct: set[str] = set()
    for p in pages:
        distinct.update(parse_multi_value_text(client._extract_property_text(p, schema, "partes")))

    # canonical por fold (sobre valores nao-fragmento, ja com e-outro estripado)
    def parts_of(v: str) -> list[str]:
        if is_fragment(v) or is_junk(v):
            return []
        out = []
        for s in split_parties(v):
            c = clean_value(s)
            if c and not is_fragment(c) and not is_junk(c):
                out.append(c)
        return out

    byfold: dict[str, set[str]] = defaultdict(set)
    ambiguous: set[str] = set()
    for v in distinct:
        for part in parts_of(v):
            byfold[fold(part)].add(part)
            if has_ambiguous_e(part):
                ambiguous.add(part)
    canon = {k: pick_canonical(sorted(cs)) for k, cs in byfold.items()}

    def expand(v: str) -> list[str]:
        return [canon.get(fold(part), part) for part in parts_of(v)]

    changes: list[dict[str, Any]] = []
    stats = {"removidos": 0, "divididos": 0, "limpos_ou_merge": 0,
             "paginas_alteradas": 0, "applied": 0, "failed": 0}
    for p in pages:
        cur = parse_multi_value_text(client._extract_property_text(p, schema, "partes"))
        if not cur:
            continue
        new = dedupe_preserve_order([c for v in cur for c in expand(v)])
        if new == cur:
            continue
        for v in cur:
            ps = parts_of(v)
            if not ps:
                stats["removidos"] += 1
            elif len(ps) > 1:
                stats["divididos"] += 1
            elif ps[0] != v:
                stats["limpos_ou_merge"] += 1
        stats["paginas_alteradas"] += 1
        rec = {"page_id": p["id"], "numero": client._extract_property_text(p, schema, "numero_processo"),
               "old": cur, "new": new}
        if args.apply:
            built = client._build_property_value(schema, "partes", new) or client._build_empty_property_value(schema, "partes")
            props = {"partes": built}
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                rec["status"] = "updated"; stats["applied"] += 1
            except Exception as exc:
                rec["status"] = "failed"; rec["error"] = str(exc); stats["failed"] += 1
            time.sleep(0.2)
        changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "ambiguos_e_revisao.json").write_text(json.dumps(sorted(ambiguous), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", "ambiguos_e_revisao": len(ambiguous), **stats}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:25]:
        LOGGER.info("  [%s] %s -> %s", c["numero"], c["old"], c["new"])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
