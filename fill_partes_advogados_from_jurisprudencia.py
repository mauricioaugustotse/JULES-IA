"""Preenche/substitui partes e advogados na base de sessões a partir dos CSVs de
jurisprudência do TSE (https://jurisprudencia.tse.jus.br) — a via que, diferente do CNJ
DataJud, traz PARTES e ADVOGADOS (estes no cabeçalho do `textoDecisao`).

Fonte: CSV(s) exportados do portal. Para cada processo:
- partes: da coluna `partes` (lista de nomes), proper-case + `normalize_party_entry`.
- advogados: parseados do `textoDecisao` (blocos "ADVOGADO(S):"), OAB removido,
  proper-case + `normalize_advogado_name` (acrescenta Dr./Dra. por gênero inferido).

Casamento com a base: por número CNJ de 20 dígitos (`numeroUnico` ↔ `numero_processo`).
Só processa páginas com CNJ COMPLETO (>=20 díg.). Escopo: MESCLA — preserva os nomes já
vindos da sessão (vídeo), completa com o nome cheio quando o oficial for mais completo,
e acrescenta os que só aparecem na publicação oficial do TSE (sem apagar os da sessão).

Escreve como page-values multi_select (seguro — NÃO mexe nas options/schema).

Uso:
    python fill_partes_advogados_from_jurisprudencia.py            # dry-run + relatório
    python fill_partes_advogados_from_jurisprudencia.py --apply     # grava
    python fill_partes_advogados_from_jurisprudencia.py --input-dir <pasta com CSVs>
"""

from __future__ import annotations

import argparse
import csv
import glob
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
from tse_normalization import (
    clean_partes_list,
    dedupe_preserve_order,
    normalize_advogado_name,
    normalize_party_entry,
    parse_multi_value_text,
)
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("fill_partes_advogados_jurisprudencia")
ARTIFACT_ROOT = Path("artifacts") / "jurisprudencia_partes_advogados"
DEFAULT_INPUT_DIRS = [
    str(Path("artifacts") / "jurisprudencia_csv"),
    r"C:\Users\mauri\ProjetoConversor\DJE",
]
APPLY_SLEEP_SECONDS = 0.2

_LOWER_CONNECTORS = {"de", "da", "do", "das", "dos", "e", "di", "del", "la", "das"}
_OAB_RE = re.compile(r"\(\s*OAB[^)]*\)|\bOAB[\s/:.\-].*?(?=(?:,|;|\bE\b|$))", re.IGNORECASE)
_E_OUTROS_RE = re.compile(r"\b[Ee]\s+outr[oa]s?\b")
# blocos de advogados no cabeçalho do textoDecisao, até o próximo rótulo em maiúsculas
_ADV_BLOCK_RE = re.compile(
    r"ADVOGAD[OA]S?\s*:\s*(.+?)(?=\s+(?:AGRAVANTE|AGRAVAD|RECORR|REQUER|REQUERID|IMPETR|"
    r"EMBARG|APEL|ADVOGAD|RELATOR|RELATORA|DECIS[ÃA]O|EMENTA|INTERESSAD|PACIENTE|"
    r"ASSISTENTE|AUTOR|R[ÉE]U|REPRESENT|LITISCONS|TERCEIRO|MINIST[ÉE]RIO\s+P[ÚU]BLICO)\b|$)",
    re.IGNORECASE | re.DOTALL,
)


def _digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def proper_case(name: str) -> str:
    """Caixa própria: baixa tudo e capitaliza a 1ª letra de cada palavra; conectivos
    (de/da/do/dos/e) minúsculos; algarismos romanos preservados. Acentos preservados."""
    name = re.sub(r"\s+", " ", str(name or "").strip())
    if not name:
        return ""
    out: list[str] = []
    for i, w in enumerate(name.split(" ")):
        wl = w.lower()
        core = re.sub(r"[^0-9a-zà-ÿ]", "", wl)
        if core in _LOWER_CONNECTORS and i > 0:
            out.append(wl)
            continue
        if re.fullmatch(r"[ivxlcdm]{1,}", core) and len(core) >= 2:
            out.append(w.upper())  # algarismo romano (II, III, IV)
            continue
        out.append(re.sub(r"(^|[-'/(.])([a-zà-ÿ0-9])", lambda m: m.group(1) + m.group(2).upper(), wl))
    return " ".join(out)


def _strip_accents(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(value or "")) if not unicodedata.combining(c))


def _person_tokens(name: str) -> list[str]:
    """Tokens significativos de um nome para COMPARAÇÃO (sem acento, minúsculos, sem
    Dr./Dra. nem conectivos). Siglas em parênteses ALL-CAPS (ex.: (ALMAGIS)) viram token;
    papéis processuais entre parênteses (ex.: (Recorrente)) são descartados."""
    raw = str(name or "")
    base = re.sub(r"(?i)^\s*dr[a]?\.?\s+", "", raw)
    acronyms: list[str] = []
    for inner in re.findall(r"\(([^)]*)\)", base):
        cand = re.sub(r"[^0-9A-Za-zÀ-ÿ]", "", inner)
        if cand and re.fullmatch(r"[0-9A-ZÀ-Ý]{2,}", cand):  # só ALL-CAPS = sigla
            acronyms.append(_strip_accents(cand).lower())
    base = re.sub(r"\([^)]*\)", " ", base)
    base = re.sub(r"[^0-9A-Za-z ]", " ", _strip_accents(base)).lower()
    toks = [tok for tok in base.split() if tok and tok not in _LOWER_CONNECTORS and len(tok) > 1]
    return toks + acronyms


def _same_person(a: str, b: str) -> bool:
    """Heurística (insensível a acento/caixa): mesmo nome se um conjunto de tokens
    significativos contém o outro (nome parcial vs completo) OU se compartilham primeiro
    e último token."""
    ta, tb = _person_tokens(a), _person_tokens(b)
    if not ta or not tb:
        return False
    sa, sb = set(ta), set(tb)
    if sa <= sb or sb <= sa:
        return True
    return ta[0] == tb[0] and ta[-1] == tb[-1]


def _diacritic_count(value: str) -> int:
    return sum(1 for c in unicodedata.normalize("NFKD", str(value or "")) if unicodedata.combining(c))


def _name_score(value: str) -> tuple[int, int, int]:
    """Qualidade de uma forma do nome: + tokens (mais completo), + acentos (forma
    acentuada), + comprimento. Usado para escolher entre variantes da mesma pessoa."""
    return (len(_person_tokens(value)), _diacritic_count(value), len(str(value or "")))


def merge_names(current: list[str], official: list[str]) -> list[str]:
    """Mescla preservando os nomes da sessão (current) e enriquecendo com os oficiais
    (official): mesmo nome -> usa a MELHOR forma (mais tokens; empate -> acentuada);
    nome só-da-sessão -> mantém; nome só-oficial -> acrescenta. Comparação insensível a
    acento/caixa para não duplicar 'Sílvia'/'Silvia'."""
    result = list(current)
    for off in official:
        matched = None
        for idx, cur in enumerate(result):
            if _same_person(cur, off):
                matched = idx
                break
        if matched is None:
            result.append(off)
        elif _name_score(off) > _name_score(result[matched]):
            result[matched] = off  # oficial é a forma melhor (mais completa/acentuada)
    return dedupe_preserve_order(result)


def parse_advogados_from_texto(texto: str) -> list[str]:
    """Extrai advogados do cabeçalho do textoDecisao: blocos 'ADVOGADO(S):', remove OAB
    e 'e outros', proper-case e aplica normalize_advogado_name (Dr./Dra.)."""
    out: list[str] = []
    for match in _ADV_BLOCK_RE.finditer(texto or ""):
        block = match.group(1)
        block = _OAB_RE.sub("", block)
        block = _E_OUTROS_RE.sub("", block)
        for piece in re.split(r"[;,]|\bE\b(?=\s+[A-ZÀ-Ý])", block):
            raw = piece.strip(" .,;- ")
            if not raw or len(raw.split()) < 2 or "OAB" in raw.upper():
                continue
            normalized = normalize_advogado_name(proper_case(raw))
            if normalized:
                out.append(normalized)
    return dedupe_preserve_order(out)


# --- Advogados a partir do CABEÇALHO (textoEmenta) -------------------------------
# Os CSVs de jurisprudência trazem o roster de advogados no cabeçalho da `textoEmenta`
# (AGRAVANTE/ADVOGADOS/AGRAVADO/...), NÃO na `textoDecisao`. Para PRECISÃO: pegamos só os
# blocos rotulados ADVOGADO(S)/representante e, dentro deles, nomes ancorados na OAB.
# Padrões portados de SJUR_csv_to_csv_NOTIONfriendly_v2.py (ProjetoConversor).
_OAB_NUMBER_PATTERN = r"[\d./]+(?:[-–—][A-Z])?(?:/[A-Z]{2})?"
_OAB_TAIL_PATTERN = rf"(?:[-–—]\s*)?OAB(?:/[A-Z]{{2}})?(?:\s*[-–—]\s*|\s*:?\s*){_OAB_NUMBER_PATTERN}"
_PERSON_NAME_PATTERN = (
    r"[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`´^~.-]*"
    r"(?:\s+(?:[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`´^~.-]*|da|de|do|das|dos|e|d|del|la|van|von|di)){1,12}"
)
# reg EXIGE 'OAB' (descarta ruído da ementa que não tem OAB)
_ATTORNEY_REG_PATTERN = rf"(?:\(\s*OAB[^)]*\)|{_OAB_TAIL_PATTERN})"
_ATTORNEY_WITH_REG_RE = re.compile(
    rf"(?P<name>{_PERSON_NAME_PATTERN})\s*(?P<reg>{_ATTORNEY_REG_PATTERN})",
    re.IGNORECASE | re.UNICODE,
)
_PARTY_ROLE_PATTERN = (
    r"(?:recorrentes?|recorrid[oa]s?|agravantes?|agravad[oa]s?|impetrantes?|impetrad[oa]s?|"
    r"requerentes?|requerid[oa]s?|exequentes?|executad[oa]s?|embargantes?|embargad[oa]s?|"
    r"apelantes?|apelad[oa]s?|autor(?:es)?|r[eé]us?|interessad[oa]s?|representantes?|"
    r"representad[oa]s?|reclamantes?|reclamad[oa]s?|pacientes?|impugnantes?|impugnad[oa]s?|"
    r"noticiantes?|noticiad[oa]s?|investigantes?|investigad[oa]s?|org[aã]o\s+coator|"
    r"autoridade\s+coator[ao]|relator(?:a)?)"
)
_ATTORNEY_LABEL_PATTERN = rf"(?:advogad(?:o|a|os|as)|representantes?\s+(?:do\s*\(a\)|do|da|dos|das)\s+{_PARTY_ROLE_PATTERN})"
_HEADER_LABEL_RE = re.compile(rf"(?i)\b(?P<label>{_ATTORNEY_LABEL_PATTERN}|{_PARTY_ROLE_PATTERN})\s*:")


def _attorney_label_blocks(text: str):
    """Itera os blocos do cabeçalho rotulados como ADVOGADO(S)/representante, delimitados
    pelo próximo rótulo (parte ou advogado)."""
    labels = list(_HEADER_LABEL_RE.finditer(text or ""))
    for i, m in enumerate(labels):
        lab = re.sub(r"\s+", " ", m.group("label")).strip().lower()
        if lab.startswith("advogad") or lab.startswith("representante"):
            start = m.end()
            end = labels[i + 1].start() if i + 1 < len(labels) else len(text)
            yield text[start:end]


def parse_advogados_from_header(*texts: str) -> list[str]:
    """Extrai advogados dos blocos ADVOGADO(S): do cabeçalho (ex.: textoEmenta), ancorando
    na OAB; normaliza ao padrão do Notion (Dr./Dra., sem OAB)."""
    out: list[str] = []
    for texto in texts:
        for block in _attorney_label_blocks(texto):
            for m in _ATTORNEY_WITH_REG_RE.finditer(block):
                nome = (m.group("name") or "").strip(" .,;-")
                if not nome or len(nome.split()) < 2:
                    continue
                normalized = normalize_advogado_name(proper_case(nome))
                if normalized:
                    out.append(normalized)
    return dedupe_preserve_order(out)


def parse_partes_from_column(partes_raw: str) -> list[str]:
    out: list[str] = []
    for piece in str(partes_raw or "").split(","):
        raw = piece.strip()
        if not raw:
            continue
        normalized = normalize_party_entry(proper_case(raw))
        if normalized:
            out.append(normalized)
    return dedupe_preserve_order(out)


def load_jurisprudencia(input_dirs: list[str]) -> dict[str, dict[str, list[str]]]:
    """numero CNJ (20 díg.) -> {'partes': [...], 'advogados': [...]}."""
    data: dict[str, dict[str, list[str]]] = {}
    files: list[str] = []
    for d in input_dirs:
        files.extend(glob.glob(str(Path(d) / "*.csv")))
    LOGGER.info("CSVs de jurisprudência encontrados: %s", len(files))
    for path in files:
        try:
            with open(path, encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "numeroUnico" not in reader.fieldnames:
                    continue
                for row in reader:
                    num = _digits(row.get("numeroUnico")) or _digits(row.get("numeroProcesso"))
                    if len(num) < 20:
                        continue
                    num = num[:20]
                    partes = parse_partes_from_column(row.get("partes", ""))
                    advogados = dedupe_preserve_order(
                        parse_advogados_from_texto(row.get("textoDecisao", ""))
                        + parse_advogados_from_header(row.get("textoEmenta", ""), row.get("textoDecisao", ""))
                    )
                    if not partes and not advogados:
                        continue
                    # se já existe (CSV duplicado), prefere o que tem mais dados
                    prev = data.get(num)
                    if prev and (len(prev["partes"]) + len(prev["advogados"])) >= (len(partes) + len(advogados)):
                        continue
                    data[num] = {"partes": partes, "advogados": advogados}
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Falha lendo %s: %s", Path(path).name, exc)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Preenche partes/advogados via jurisprudência TSE.")
    parser.add_argument("--apply", action="store_true", help="Grava (padrão: dry-run).")
    parser.add_argument("--input-dir", action="append", default=None, help="Pasta(s) com CSVs de jurisprudência.")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    input_dirs = args.input_dir or DEFAULT_INPUT_DIRS
    juris = load_jurisprudencia(input_dirs)
    LOGGER.info("Processos com partes/advogados na jurisprudência: %s", len(juris))

    key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    client = NotionSessoesClient(api_key=key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(page: dict, field: str) -> str:
        return client._extract_property_text(page, schema, field)

    changes: list[dict[str, Any]] = []
    stats = {"paginas": 0, "cnj_completo": 0, "match": 0, "muda_partes": 0, "muda_advogados": 0}
    for page in pages:
        stats["paginas"] += 1
        num = _digits(t(page, "numero_processo"))
        if len(num) < 20:
            continue
        stats["cnj_completo"] += 1
        info = juris.get(num[:20])
        if not info:
            continue
        stats["match"] += 1
        cur_partes = parse_multi_value_text(t(page, "partes"))
        cur_adv = parse_multi_value_text(t(page, "advogados"))
        # MESCLA: preserva os nomes da sessão e enriquece/completa com os oficiais do TSE
        merged_partes = merge_names(cur_partes, info["partes"])
        merged_adv = merge_names(cur_adv, info["advogados"])
        props: dict[str, Any] = {}
        detail: dict[str, Any] = {}
        # going-forward redondo: limpa o merge (estirpa papeis + remove lixo + dedup fuzzy)
        # via clean_partes_list ANTES de gravar (sem re-dividir -> preserva nomes de empresa).
        merged_partes = clean_partes_list(merged_partes)
        if merged_partes and merged_partes != cur_partes:
            props["partes"] = client._build_property_value(schema, "partes", merged_partes)  # schema-driven (rich_text)
            detail["partes"] = {"old": ", ".join(cur_partes), "new": ", ".join(merged_partes)}
            stats["muda_partes"] += 1
        if merged_adv and merged_adv != cur_adv:
            props["advogados"] = {"multi_select": [{"name": n} for n in merged_adv]}
            detail["advogados"] = {"old": ", ".join(cur_adv), "new": ", ".join(merged_adv)}
            stats["muda_advogados"] += 1
        if props:
            rec = {"page_id": page["id"], "numero": num, "detail": detail}
            changes.append(rec)
            if args.apply:
                try:
                    notion_request_with_retry(client, "PATCH", f"/pages/{page['id']}", json={"properties": props})
                    rec["status"] = "updated"
                except Exception as exc:
                    rec["status"] = "failed"; rec["error"] = str(exc)
                time.sleep(APPLY_SLEEP_SECONDS)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        **stats,
        "paginas_com_mudanca": len(changes),
        "applied": sum(1 for c in changes if c.get("status") == "updated"),
        "failed": sum(1 for c in changes if c.get("status") == "failed"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:8]:
        for col, d in c["detail"].items():
            LOGGER.info("  [%s] %s: %r -> %r", c["numero"][:20], col, d["old"][:70], d["new"][:90])
    LOGGER.info("Relatórios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
