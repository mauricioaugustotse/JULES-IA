from __future__ import annotations

import ast
import csv
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, urlencode, urlparse


CNJ_REGEX = r"\b\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b"
SHORT_PROCESSO_REGEX = r"\b\d{3,7}-\d{2}\b"
LABELED_PROCESSO_REGEX = r"(?i)\bn(?:[º°]|\.)\s*(\d{3,7})\b"

CANON_CSV_FILENAME = "padrões para canonizar.csv"
CANON_DATA: Optional[dict[str, Any]] = None

CLASSE_PROCESSO_MAP = [
    (r"\bembargos de declaracao\b.*\b(agravo regimental|agravo interno|agrg)\b.*\b(agravo em recurso especial eleitoral|arespe)\b", "ED-AgRg-AREspe"),
    (r"\bed\s+agrg\s+arespe\b", "ED-AgRg-AREspe"),
    (r"\bembargos de declaracao\b.*\b(agravo em recurso especial eleitoral|arespe)\b", "ED-AREspe"),
    (r"\bembargos de declaracao\b.*\b(recurso especial eleitoral|respe)\b", "ED-REspe"),
    (r"\bembargos de declaracao\b.*\b(recurso ordinario|ro)\b", "ED-RO"),
    (r"\bembargos de declaracao\b.*\b(prestacao de contas|pc)\b", "ED-PC"),
    (r"\bembargos de declaracao\b.*\b(lista triplice|lt)\b", "ED-Lista Tríplice"),
    (r"\bed\s+lt\b", "ED-Lista Tríplice"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(recurso especial eleitoral|respe)\b", "AgRg-REspe"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(agravo em recurso especial eleitoral|arespe)\b", "AgRg-AREspe"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(recurso ordinario|ro)\b", "AgRg-RO"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(mandado de seguranca|ms)\b", "AgRg-MS"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(prestacao de contas|pc)\b", "AgRg-PC"),
    (r"\breferend\w*\b.*\b(tutela cautelar antecedente|tutcautant)\b", "Ref-TutCautAnt"),
    (r"\bref\s*tutcautant\b", "Ref-TutCautAnt"),
    (r"\breferend\w*\b.*\b(mandado de seguranca|ms)\b", "Ref.-MS"),
    (r"\bref\s*ms\b", "Ref.-MS"),
    (r"\btutela cautelar antecedente\b|\btutcautant\b", "TutCautAnt"),
    (r"\blista triplice\b|\blt\b", "Lista Tríplice"),
    (r"\bprocesso administrativo\b|\bpa\b", "PA"),
    (r"\bprestacao de contas\b|\bpc\b", "PC"),
    (r"\bconsulta\b|\bcta\b", "CTA"),
    (r"\bquestao de ordem\b|\bqo\b", "QO"),
    (r"\bpeticao civel\b|\bpetciv\b", "PetCiv"),
    (r"\brecurso especial eleitoral\b|\brespe\b", "REspe"),
    (r"\bagravo em recurso especial eleitoral\b|\barespe\b", "AREspe"),
    (r"\brecurso ordinario\b|\bro\b", "RO"),
    (r"\brecurso em habeas corpus\b|\brhc\b", "RHC"),
    (r"\brecurso em mandado de seguranca\b|\brms\b", "RMS"),
    (r"\bmandado de seguranca\b|\bms\b", "MS"),
    (r"\bregistro\s+de\s+partido\s+pol[ií]tico\b", "RPP"),
    (r"\balter[aã]c[aã]o\s+(?:do|de|no)\s+registro\s+de\s+partido(?:\s+pol[ií]tico)?\b", "RPP"),
    (r"\balter[aã]c[aã]o\s+de\s+esta?tuto(?:\s+partid[aá]rio)?\b", "RPP"),
    (r"\brevis[aã]o\s+do\s+eleitorado\b", "RvE"),
    (r"\brpp\b", "RPP"),
    (r"\brve\b", "RvE"),
]

STATE_UF = {
    "acre": "AC",
    "alagoas": "AL",
    "amapá": "AP",
    "amapa": "AP",
    "amazonas": "AM",
    "bahia": "BA",
    "ceará": "CE",
    "ceara": "CE",
    "distrito federal": "DF",
    "espírito santo": "ES",
    "espirito santo": "ES",
    "goiás": "GO",
    "goias": "GO",
    "maranhão": "MA",
    "maranhao": "MA",
    "mato grosso": "MT",
    "mato grosso do sul": "MS",
    "minas gerais": "MG",
    "pará": "PA",
    "para": "PA",
    "paraíba": "PB",
    "paraiba": "PB",
    "paraná": "PR",
    "parana": "PR",
    "pernambuco": "PE",
    "piauí": "PI",
    "piaui": "PI",
    "rio de janeiro": "RJ",
    "rio grande do norte": "RN",
    "rio grande do sul": "RS",
    "rondônia": "RO",
    "rondonia": "RO",
    "roraima": "RR",
    "santa catarina": "SC",
    "são paulo": "SP",
    "sao paulo": "SP",
    "sergipe": "SE",
    "tocantins": "TO",
}

MINISTRO_ALIAS_MAP = {
    "antonio carlos ferreira": "Min. Antônio Carlos Ferreira",
    "maria isabel gallotti": "Min. Isabel Gallotti",
    "isabel gallotti": "Min. Isabel Gallotti",
    "andre ramos tavares": "Min. Ramos Tavares",
    "ramos tavares": "Min. Ramos Tavares",
    "carmen lucia": "Min. Cármen Lúcia",
    "vera lucia": "Min. Vera Lúcia Santana Araújo",
    "vera lucia santana araujo": "Min. Vera Lúcia Santana Araújo",
    "edilene lobo": "Min. Edilene Lôbo",
    "ricardo villas boas cueva": "Min. Ricardo Villas Bôas Cueva",
    "villas boas cueva": "Min. Ricardo Villas Bôas Cueva",
    "vilas boas cueva": "Min. Ricardo Villas Bôas Cueva",
}

EMPTY_ADVOGADOS_REGEX = re.compile(
    r"(?i)\b("
    r"n[ãa]o\s+(?:citad\w*|mencionad\w*|informad\w*|consta\w*|houve|h[áa]|aplic[aá]vel)"
    r"|sem\s+sustenta[cç][aã]o"
    r"|sem\s+advogad\w*"
    r"|n/?a"
    r")\b"
)
MPE_REFERENCE_REGEX = re.compile(r"(?i)\bminist[ée]rio\s+p[úu]blico\s+eleitoral\b")
MINISTERIO_PUBLICO_REGEX = re.compile(r"(?i)\bminist[ée]rio\s+p[úu]blico\b")
MPE_ABBREV_REGEX = re.compile(r"(?i)\bmp\s*eleitoral\b|\bm\.?\s*p\.?\s*e\.?\b|\bmpe\b")
MPE_REPRESENTATIVE_REGEX = re.compile(
    r"(?i)\b(vice-?procurador(?:-geral)?\s+eleitoral|"
    r"procurador(?:-geral)?(?:\s+regional)?\s+eleitoral|"
    r"procuradoria\s+geral\s+eleitoral|pge)\b"
)
PARTY_ATTORNEY_MARKER_REGEX = re.compile(
    r"(?i)\b("
    r"dr\.?|dra\.?|doutor(?:a)?|adv\.?|advogad[oa]s?|patrono(?:s)?|patrona(?:s)?|"
    r"defensor(?:a|es|as)?|procurador(?:a|es|as)?"
    r")\b"
)
PARTY_OAB_REGEX = re.compile(
    r"(?i)\bOAB(?:/[A-Z]{2})?\s*[-:]?\s*[\d./-]+\b"
)
PARTY_PLACEHOLDER_REGEX = re.compile(
    r"(?i)^\s*("
    r"n[ãa]o\s+especificad\w*|"
    r"n[ãa]o\s+informad\w*|"
    r"n[ãa]o\s+identificad\w*|"
    r"desconhecid\w*|"
    r"ignorado\w*|"
    r"sem\s+identifica[cç][aã]o|"
    r"sem\s+informa[cç][aã]o"
    r")\s*$"
)
PARTY_PROCESSUAL_ROLE_MAP = {
    "agravante": "Agravante",
    "agravantes": "Agravante",
    "agravado": "Agravado",
    "agravada": "Agravada",
    "agravados": "Agravado",
    "agravadas": "Agravada",
    "apelante": "Apelante",
    "apelantes": "Apelante",
    "apelado": "Apelado",
    "apelada": "Apelada",
    "apelados": "Apelado",
    "apeladas": "Apelada",
    "autor": "Autor",
    "autora": "Autora",
    "autores": "Autor",
    "autoras": "Autora",
    "candidato": "Candidato",
    "candidata": "Candidata",
    "candidatos": "Candidato",
    "candidatas": "Candidata",
    "embargante": "Embargante",
    "embargantes": "Embargante",
    "embargado": "Embargado",
    "embargada": "Embargada",
    "embargados": "Embargado",
    "embargadas": "Embargada",
    "impugnante": "Impugnante",
    "impugnantes": "Impugnante",
    "impugnado": "Impugnado",
    "impugnada": "Impugnada",
    "impugnados": "Impugnado",
    "impugnadas": "Impugnada",
    "impetrante": "Impetrante",
    "impetrantes": "Impetrante",
    "impetrado": "Impetrado",
    "impetrada": "Impetrada",
    "impetrados": "Impetrado",
    "impetradas": "Impetrada",
    "interessado": "Interessado",
    "interessada": "Interessada",
    "interessados": "Interessado",
    "interessadas": "Interessada",
    "investigado": "Investigado",
    "investigada": "Investigada",
    "investigados": "Investigado",
    "investigadas": "Investigada",
    "investigante": "Investigante",
    "investigantes": "Investigante",
    "recorrente": "Recorrente",
    "recorrentes": "Recorrente",
    "recorrido": "Recorrido",
    "recorrida": "Recorrida",
    "recorridos": "Recorrido",
    "recorridas": "Recorrida",
    "representante": "Representante",
    "representantes": "Representante",
    "representado": "Representado",
    "representada": "Representada",
    "representados": "Representado",
    "representadas": "Representada",
    "requerente": "Requerente",
    "requerentes": "Requerente",
    "requerido": "Requerido",
    "requerida": "Requerida",
    "requeridos": "Requerido",
    "requeridas": "Requerida",
    "consulente": "Consulente",
    "consulentes": "Consulente",
    "reu": "Réu",
    "réu": "Réu",
    "reus": "Réu",
    "réus": "Réu",
    "re": "Ré",
    "ré": "Ré",
    "res": "Ré",
    "rés": "Ré",
}
PARTY_PROCESSUAL_ROLE_REGEX = re.compile(
    r"(?i)^\s*(?P<label>"
    + "|".join(re.escape(label) for label in PARTY_PROCESSUAL_ROLE_MAP)
    + r")\s*[:\-–]\s*(?P<name>.+?)\s*$"
)
PROCURADOR_GENERIC_REGEX = re.compile(r"(?i)\b(vice-?procurador(?:-geral)?|procurador(?:-geral)?|procuradoria)\b")

FEMALE_NAME_HINTS = {
    "ana", "maria", "mariana", "marina", "carla", "claudia", "clara", "camila",
    "carolina", "caroline", "beatriz", "bianca", "renata", "fernanda", "patricia",
    "luciana", "lucia", "marcia", "sandra", "silvana", "viviane", "oneida",
    "andreia", "adriana", "tatiana", "vanessa", "aline", "leticia", "raquel",
    "cristina", "cristiane", "gisele", "giovana", "gabriela", "isabel", "marta",
    "carmen", "estela", "juliana", "julia", "amanda", "bruna", "daniela",
}


def normalize_text(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00ad": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def clean_label_value(value: str) -> str:
    value = value.replace("**", "").strip()
    value = re.sub(r"^\*+\s*", "", value).strip()
    value = re.sub(r"\s*\*+$", "", value).strip()
    return value.rstrip(".").strip()


def normalize_token(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def normalize_class_text(value: str) -> str:
    if not value:
        return ""
    value = normalize_text(value)
    value = normalize_token(value)
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def get_canonization_data() -> dict[str, Any]:
    global CANON_DATA
    if CANON_DATA is not None:
        return CANON_DATA

    data: dict[str, Any] = {
        "classes": set(),
        "results": set(),
        "class_results": {},
        "class_norm_map": {},
        "result_norm_map": {},
    }
    path = Path(__file__).resolve().parent / CANON_CSV_FILENAME
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            class_results: dict[str, set[str]] = {}
            for row in reader:
                cls = (row.get("classe_processo") or "").strip()
                res = (row.get("resultado") or "").strip()
                if cls:
                    data["classes"].add(cls)
                    class_results.setdefault(cls, set())
                    if res:
                        class_results[cls].add(res)
                if res:
                    data["results"].add(res)
            data["class_results"] = class_results

    data["class_norm_map"] = {
        normalize_class_text(cls): cls for cls in data["classes"]
    }
    data["result_norm_map"] = {
        normalize_class_text(res): res for res in data["results"]
    }
    CANON_DATA = data
    return data


def is_mpe_noise_entry(text: str) -> bool:
    if not text:
        return False
    return bool(
        MPE_REFERENCE_REGEX.search(text)
        or MPE_ABBREV_REGEX.search(text)
        or MPE_REPRESENTATIVE_REGEX.search(text)
        or MINISTERIO_PUBLICO_REGEX.search(text)
        or PROCURADOR_GENERIC_REGEX.search(text)
    )


def normalize_mpe_reference(value: str) -> str:
    if not value:
        return ""
    value = re.sub(MPE_REFERENCE_REGEX, "MPE", value)
    value = re.sub(MPE_ABBREV_REGEX, "MPE", value)
    return value


def remove_mpe_from_partes(value: str) -> str:
    if not value:
        return ""
    normalized = normalize_mpe_reference(value)
    parts = [part.strip() for part in re.split(r"\s*,\s*|\s*;\s*", normalized) if part.strip()]
    cleaned: list[str] = []
    for part in parts:
        if part == "MPE" or is_mpe_noise_entry(part):
            continue
        cleaned.append(part)
    return ", ".join(dedupe_preserve_order(cleaned))


def looks_like_advogado_party_entry(value: str) -> bool:
    normalized = normalize_mpe_reference(value or "")
    if not normalized:
        return False
    if PARTY_OAB_REGEX.search(normalized):
        return True
    if PARTY_ATTORNEY_MARKER_REGEX.search(normalized):
        return True
    return bool(re.search(r"(?i)\badvogad[oa]\b", normalized))


def _cleanup_party_name(value: str) -> str:
    cleaned = PARTY_OAB_REGEX.sub("", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;,:-")
    return cleaned.strip()


def _normalize_party_role_label(value: str) -> str:
    normalized = normalize_token(value or "")
    return PARTY_PROCESSUAL_ROLE_MAP.get(normalized, value.strip())


def normalize_party_entry(value: str) -> str:
    if not value:
        return ""
    cleaned = normalize_mpe_reference(value)
    cleaned = re.sub(r"(?i)^\s*part[ea]\s*[:\-]\s*", "", cleaned).strip()
    if not cleaned or cleaned == "MPE" or is_mpe_noise_entry(cleaned) or PARTY_PLACEHOLDER_REGEX.match(cleaned):
        return ""
    match = PARTY_PROCESSUAL_ROLE_REGEX.match(cleaned)
    if match:
        role_label = _normalize_party_role_label(match.group("label"))
        name = _cleanup_party_name(match.group("name"))
        if not name or is_mpe_noise_entry(name) or looks_like_advogado_party_entry(name) or PARTY_PLACEHOLDER_REGEX.match(name):
            return ""
        return f"{name} ({role_label})"
    suffix_match = re.match(r"^(?P<name>.+?)\s*\((?P<label>[^()]+)\)\s*$", cleaned)
    if suffix_match and normalize_token(suffix_match.group("label")) in PARTY_PROCESSUAL_ROLE_MAP:
        role_label = _normalize_party_role_label(suffix_match.group("label"))
        name = _cleanup_party_name(suffix_match.group("name"))
        if not name or is_mpe_noise_entry(name) or looks_like_advogado_party_entry(name) or PARTY_PLACEHOLDER_REGEX.match(name):
            return ""
        return f"{name} ({role_label})"
    cleaned = _cleanup_party_name(cleaned)
    if not cleaned or looks_like_advogado_party_entry(cleaned) or is_mpe_noise_entry(cleaned) or PARTY_PLACEHOLDER_REGEX.match(cleaned):
        return ""
    return cleaned


def _flatten_structured_party_payload(value: Any, label_hint: str = "") -> list[str]:
    if isinstance(value, dict):
        flattened: list[str] = []
        for key, nested in value.items():
            key_text = str(key or "").strip().strip("'\"")
            next_label = "" if normalize_token(key_text) in {"parte", "partes"} else key_text
            flattened.extend(_flatten_structured_party_payload(nested, next_label or label_hint))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_structured_party_payload(item, label_hint))
        return flattened
    text = str(value or "").strip().strip("'\"")
    if not text:
        return []
    if label_hint and not PARTY_PROCESSUAL_ROLE_REGEX.match(text):
        return [f"{label_hint}: {text}"]
    return [text]


def _looks_like_structured_party_fragment(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text.startswith(("{", "[")):
        return True
    return bool(re.match(r"""^\s*['"][^'"]+['"]\s*:\s*""", text))


def _parse_structured_partes_payload(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    if not text.startswith(("{", "[")):
        label_match = re.match(r"""^\s*['"]?(?P<label>[^'":{}\[\],]+)['"]?\s*:\s*(?P<raw>.+?)\s*$""", text)
        if not label_match:
            return []
        label = label_match.group("label").strip()
        label_norm = normalize_token(label)
        if label_norm not in PARTY_PROCESSUAL_ROLE_MAP:
            return []
        raw_value = label_match.group("raw").strip().rstrip("}").strip()
        if not raw_value:
            return []
        parsed_value: Any
        try:
            parsed_value = ast.literal_eval(raw_value)
        except Exception:
            cleaned = raw_value.strip().strip("'\"")
            if not cleaned:
                return []
            values = [item.strip().strip("'\"") for item in cleaned.split(",") if item.strip().strip("'\"")]
            parsed_value = values if len(values) > 1 else cleaned
        else:
            if isinstance(parsed_value, str) and "," in parsed_value and label_norm.endswith("s"):
                values = [item.strip().strip("'\"") for item in parsed_value.split(",") if item.strip().strip("'\"")]
                if len(values) > 1:
                    parsed_value = values
        return _flatten_structured_party_payload({label: parsed_value})
    try:
        payload = ast.literal_eval(text)
    except Exception:
        return []
    return _flatten_structured_party_payload(payload)


def normalize_partes_list(value: str | list[str]) -> str:
    if isinstance(value, list):
        raw_parts = [str(item or "").strip() for item in value if str(item or "").strip()]
        structured_entries = []
        if raw_parts and any(_looks_like_structured_party_fragment(part) for part in raw_parts):
            structured_entries = _parse_structured_partes_payload(", ".join(raw_parts))
        if structured_entries:
            raw_parts = structured_entries
    else:
        structured_entries = _parse_structured_partes_payload(str(value or ""))
        if structured_entries:
            raw_parts = structured_entries
        else:
            raw_parts = [part.strip() for part in re.split(r"\s*,\s*|\s*;\s*", str(value or "")) if part.strip()]
    normalized: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        entry = normalize_party_entry(part)
        if not entry or entry in seen:
            continue
        seen.add(entry)
        normalized.append(entry)
    return ", ".join(normalized)


def is_empty_advogados_value(value: str) -> bool:
    if not value:
        return True
    return bool(EMPTY_ADVOGADOS_REGEX.search(value))


def infer_advogado_prefix(name: str, label_hint: str = "") -> str:
    hint = label_hint.lower()
    if "advogada" in hint:
        return "Dra."
    if "advogado" in hint:
        return "Dr."
    first = name.split()[0] if name.split() else ""
    if normalize_token(first) in FEMALE_NAME_HINTS:
        return "Dra."
    return "Dr."


def normalize_advogado_name(name: str, label_hint: str = "") -> str:
    name = name.strip()
    if not name or is_mpe_noise_entry(name) or is_empty_advogados_value(name):
        return ""
    suffix = ""
    if "(" in name:
        base, extra = name.split("(", 1)
        name = base.strip()
        suffix = " (" + extra.strip()
    name = name.rstrip(".;:,").strip()
    if not name or is_empty_advogados_value(name):
        return ""
    prefix = ""
    if re.match(r"(?i)^dra\.?\s+", name):
        prefix = "Dra."
        name = re.sub(r"(?i)^dra\.?\s+", "", name).strip()
    elif re.match(r"(?i)^dr\.?\s+", name):
        prefix = "Dr."
        name = re.sub(r"(?i)^dr\.?\s+", "", name).strip()
    elif re.match(r"(?i)^doutora\s+", name):
        prefix = "Dra."
        name = re.sub(r"(?i)^doutora\s+", "", name).strip()
    elif re.match(r"(?i)^doutor\s+", name):
        prefix = "Dr."
        name = re.sub(r"(?i)^doutor\s+", "", name).strip()
    elif re.match(r"(?i)^(sra|srta|senhora)\.?\s+", name):
        prefix = "Dra."
        name = re.sub(r"(?i)^(sra|srta|senhora)\.?\s+", "", name).strip()
    elif re.match(r"(?i)^(sr|senhor)\.?\s+", name):
        prefix = "Dr."
        name = re.sub(r"(?i)^(sr|senhor)\.?\s+", "", name).strip()
    if not name:
        return ""
    if not prefix:
        prefix = infer_advogado_prefix(name, label_hint)
    return f"{prefix} {name}{suffix}".strip()


def split_advogados_entries(text: str) -> list[str]:
    parts: list[str] = []
    buffer = ""
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        if depth == 0:
            if text[index:index + 3].lower() == " e ":
                if buffer.strip():
                    parts.append(buffer.strip())
                buffer = ""
                index += 3
                continue
            if char in ",;":
                if buffer.strip():
                    parts.append(buffer.strip())
                buffer = ""
                index += 1
                continue
        buffer += char
        index += 1
    if buffer.strip():
        parts.append(buffer.strip())
    return parts


def normalize_advogados_list(value: str, label_hint: str = "") -> str:
    if not value:
        return ""
    if is_empty_advogados_value(value) and not re.search(r"(?i)[,;]|\bdr\.|\bdra\.|\be\b", value):
        return ""
    text = normalize_text(value)
    normalized: list[str] = []
    for part in split_advogados_entries(text):
        part = re.sub(r"(?i)^(advogad[oa]s?|defensor[oa]s?)\s*:?\s*", "", part).strip()
        if not part or is_mpe_noise_entry(part) or is_empty_advogados_value(part):
            continue
        normalized_name = normalize_advogado_name(part, label_hint)
        if normalized_name:
            normalized.append(normalized_name)
    return ", ".join(dedupe_preserve_order(normalized))


def extract_full_cnj(text: str) -> str:
    match = re.search(CNJ_REGEX, normalize_text(text))
    return match.group(0) if match else ""


def extract_short_processo(text: str) -> str:
    match = re.search(SHORT_PROCESSO_REGEX, normalize_text(text))
    return match.group(0) if match else ""


def extract_labeled_short_processo(text: str) -> str:
    match = re.search(LABELED_PROCESSO_REGEX, normalize_text(text))
    return match.group(1) if match else ""


def format_short_process_number_from_digits(value: str) -> str:
    digits = re.sub(r"\D", "", normalize_text(value))
    if len(digits) in {8, 9}:
        return f"{digits[:-2]}-{digits[-2:]}"
    return ""


def normalize_numero_processo_display(value: str) -> str:
    if not value:
        return ""
    full_cnj = extract_full_cnj(value)
    if full_cnj:
        return full_cnj
    short = extract_short_processo(value)
    if short:
        return short
    digits_short = format_short_process_number_from_digits(value)
    if digits_short:
        return digits_short
    labeled = extract_labeled_short_processo(value)
    return labeled if labeled else value.strip()


def normalize_processo_num(value: str) -> str:
    if not value:
        return ""
    full_cnj = extract_full_cnj(value)
    if full_cnj:
        short = extract_short_processo(full_cnj)
        return short if short else full_cnj
    short = extract_short_processo(value)
    if short:
        return short
    digits_short = format_short_process_number_from_digits(value)
    if digits_short:
        return digits_short
    labeled = extract_labeled_short_processo(value)
    return labeled if labeled else value.strip()


def canonicalize_numero_processo(value: str) -> str:
    return normalize_processo_num(value)


def extract_uf_from_text(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\bUF\s*:\s*([A-Z]{2})\b", text)
    if match:
        return match.group(1).upper()
    match = re.search(r"/([A-Z]{2})\b", text)
    if match:
        return match.group(1).upper()
    match = re.search(r"\(([A-Z]{2})\)", text)
    if match:
        return match.group(1).upper()
    match = re.search(
        r"Tribunal Regional Eleitoral d(?:e|o|a)\s+([A-Za-zçÇãõéíóúÁÉÍÓÚ ]+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        state_name = match.group(1).strip().lower()
        uf = STATE_UF.get(state_name)
        if uf:
            return uf
    lower = text.lower()
    for state_name in sorted(STATE_UF.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(state_name)}\b", lower):
            return STATE_UF[state_name]
    return ""


def normalize_origem_value(value: str) -> str:
    if not value:
        return ""
    value = normalize_text(value).strip().rstrip(".")
    if re.search(r"(?i)^tribunal regional eleitoral d(?:e|o|a)\s+", value):
        uf = extract_uf_from_text(value)
        return f"TRE/{uf}" if uf else ""
    tre_match = re.match(r"(?i)^tre[-/\s]?([a-z]{2})$", value)
    if tre_match:
        return f"TRE/{tre_match.group(1).upper()}"
    if re.match(r"^.+\([^)]+\)\s*$", value):
        uf = extract_uf_from_text(value)
        city = re.sub(r"\s*\([^)]+\)\s*$", "", value).strip()
        if city and uf:
            return f"{city}/{uf}"
    match = re.match(r"^(.*?)\s*/\s*([A-Za-z]{2})$", value)
    if match:
        return f"{match.group(1).strip()}/{match.group(2).upper()}"
    match = re.match(r"^(.*?)\s*[-–—]\s*([A-Za-z]{2})$", value)
    if match:
        return f"{match.group(1).strip()}/{match.group(2).upper()}"
    match = re.match(r"^(.*?)\s*,\s*([A-Za-z]{2})$", value)
    if match:
        return f"{match.group(1).strip()}/{match.group(2).upper()}"
    match = re.match(r"^(.*?)\s*,\s*([A-Za-zÀ-ÿ ]+)$", value)
    if match:
        city = match.group(1).strip()
        uf = extract_uf_from_text(match.group(2).strip())
        if city and uf:
            return f"{city}/{uf}"
    return value


def normalize_eleicao_value(value: str) -> str:
    if not value:
        return ""
    value = normalize_text(value).strip()
    lowered = normalize_class_text(value)
    if lowered in {"nao especificada", "nao aplicavel", "geral"}:
        return ""
    match = re.search(r"\b(20\d{2})\b", value)
    if match:
        return match.group(1)
    return value


def normalize_tre(value: str, uf: str) -> str:
    if value:
        match = re.search(r"\bTRE-([A-Z]{2})\b", value)
        if match:
            return f"TRE-{match.group(1).upper()}"
        if normalize_class_text(value) == "tse":
            return "TSE"
    if uf:
        return f"TRE-{uf}"
    return value.strip()


def normalize_ministro_name(name: str) -> str:
    name = re.sub(r"\[\[.*?\]\]", "", name).strip()
    name = re.sub(r"\[[^\]]+\]\([^)]+\)", "", name).strip()
    name = name.replace("*", "")
    name = re.sub(r"\s*\(.*?\)\s*", "", name).strip()
    name = re.sub(r"[\[\]]", "", name).strip()
    name = name.replace("(", "").replace(")", "").strip()
    name = name.rstrip(".;:,").strip()
    if not name:
        return ""
    name = re.sub(r"^(e|E)\s+", "", name).strip()
    name = re.sub(r"^Ministr[oa]s?\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^Ministra\s+", "Min. ", name, flags=re.IGNORECASE)
    name = re.sub(r"^Ministro\s+", "Min. ", name, flags=re.IGNORECASE)
    if not name.startswith("Min."):
        name = f"Min. {name}"
    normalized_key = normalize_class_text(re.sub(r"^Min\.\s*", "", name))
    if re.search(r"\b(kassio|cassio)\b", normalized_key):
        return "Min. Nunes Marques"
    if normalized_key in {"nao especificado", "relator", "ministro", "ministra"}:
        return ""
    alias = MINISTRO_ALIAS_MAP.get(normalized_key)
    if alias:
        return alias
    return name


def normalize_pedido_vista_name(value: str) -> str:
    value = re.sub(r"(?i)^\s*(relator(?:a)?|presidente|vice-?presidente)\s+", "", value).strip()
    name = normalize_ministro_name(value)
    name = re.sub(r"^Min\.\s+(?:Presidente|Vice-Presidente|Relator(?:a)?)\s+", "Min. ", name)
    name = re.sub(r"^Min\.\s+Ministro\s+", "Min. ", name)
    name = re.sub(r"^Min\.\s+Ministra\s+", "Min. ", name)
    if name in {"Min. Presidente", "Min. Relator", "Min. Relatora", "Min. Ministra", "Min. Ministro"}:
        return ""
    return name


def normalize_pedido_vista_value(value: str) -> str:
    return normalize_pedido_vista_name(value) if value else ""


def normalize_composicao(value: str) -> str:
    if not value:
        return ""
    value = value.replace(";", ",")
    normalized: list[str] = []
    seen: set[str] = set()
    for part in [p.strip() for p in value.split(",") if p.strip()]:
        if is_mpe_noise_entry(part):
            continue
        name = normalize_ministro_name(part)
        if name and name not in seen:
            seen.add(name)
            normalized.append(name)
    return ", ".join(normalized)


def normalize_classe_processo(value: str) -> str:
    if not value:
        return ""
    normalized = normalize_class_text(value)
    if not normalized:
        return ""
    data = get_canonization_data()
    canon = data["class_norm_map"].get(normalized)
    if canon:
        return canon
    for pattern, canon_value in CLASSE_PROCESSO_MAP:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return canon_value
    if "agravo regimental" in normalized and "agravo em recurso especial" in normalized:
        return "AgRg-AREspe"
    if "agravo regimental" in normalized and "recurso especial" in normalized:
        return "AgRg-REspe"
    if "criacao de zona eleitoral" in normalized or "remanejamento" in normalized:
        return "Czer"
    if "resolucao" in normalized:
        return "PA"
    if "registro de federacao partidaria" in normalized:
        return "RPP"
    if "inquerito" in normalized:
        return "AgR-HC"
    return value.strip()


def normalize_resultado_piece(value: str, classe_processo: str, allowed: Optional[set[str]]) -> str:
    lowered = normalize_class_text(value)
    if not lowered:
        return ""
    data = get_canonization_data()
    direct = data["result_norm_map"].get(lowered)
    if direct and (not allowed or direct in allowed):
        return direct
    if lowered in {
        "em julgamento",
        "julgamento em curso",
        "julgamento em curso aguardando voto vista",
        "pendente de conclusao",
        "nao especificada",
        "sessao encerrada",
        "precedente citado",
    }:
        return ""
    if "suspens" in lowered and "vista" in lowered:
        return "Suspenso por vista"
    if "nao conhecid" in lowered:
        return "Não conhecida" if "nao conhecida" in lowered else "Não conhecido"
    if re.search(r"\bprovido\b", lowered) and "nao conhec" in lowered and "desprov" not in lowered:
        return "Provido, Não conhecido"
    if "prejudic" in lowered and "desprov" in lowered:
        return "Prejudicado, Desprovido"
    if re.search(r"parcialmente\s+deferid", lowered):
        return "Parcialmente deferido"
    if "homologad" in lowered:
        return "Aprovada"
    if "parcial provimento" in lowered or "provido em parte" in lowered:
        return "Provido em parte"
    if (
        "negado provimento" in lowered
        or "negou provimento" in lowered
        or "nego provimento" in lowered
        or "nega provimento" in lowered
        or "desprovido" in lowered
        or "não provido" in lowered
        or "nao provido" in lowered
    ):
        return "Desprovido"
    if re.search(r"\bprovido\b|\bprovimento\b", lowered):
        return "Provido"
    if (
        "não conhecido" in lowered
        or "nao conhecido" in lowered
        or "não conhecimento" in lowered
        or "nao conhecimento" in lowered
        or "não conhecer" in lowered
        or "nao conhecer" in lowered
    ):
        return "Não conhecido"
    if re.search(r"aprovad[oa]s?\s+com\s+ressalv", lowered):
        return "Aprovada com ressalvas"
    if "aprov" in lowered:
        return "Aprovada"
    if "indeferid" in lowered:
        return "Indeferida" if "indeferida" in lowered else "Indeferido"
    if "deferid" in lowered:
        return "Deferido"
    if "referendad" in lowered:
        return "Referendada" if "referendada" in lowered else "Referendado"
    if "acolh" in lowered and ("em parte" in lowered or "parcial" in lowered):
        return "Acolhido em parte"
    if "acolhido" in lowered or "acolhidos" in lowered or "acolhida" in lowered:
        return "Acolhidos"
    if "rejeitad" in lowered:
        return "Rejeitada" if "rejeitada" in lowered else "Rejeitados"
    if "devolvid" in lowered:
        return "Devolvida"
    if "prejudicado" in lowered:
        return "Prejudicado"
    return value.strip()


def normalize_resultado_final(value: str, classe_processo: str = "") -> str:
    if not value:
        return ""
    text = normalize_text(value).strip()
    if not text:
        return ""
    data = get_canonization_data()
    classe_canon = normalize_classe_processo(classe_processo) if classe_processo else ""
    allowed = data["class_results"].get(classe_canon, set()) if data else set()
    normalized = normalize_class_text(text)
    canonical_direct = data["result_norm_map"].get(normalized)
    if canonical_direct and (not allowed or canonical_direct in allowed):
        return canonical_direct
    if "suspens" in normalized and "vista" in normalized:
        return "Suspenso por vista"
    if re.search(r"\bprovido\b", normalized) and "nao conhec" in normalized and "desprov" not in normalized:
        return "Provido, Não conhecido"
    if "prejudic" in normalized and "desprov" in normalized:
        return "Prejudicado, Desprovido"

    parts = [part.strip() for part in re.split(r"[;,/]", text) if part.strip()]
    if len(parts) > 1:
        normalized_parts: list[str] = []
        for part in parts:
            part_norm = normalize_resultado_piece(part, classe_canon, allowed)
            if part_norm and part_norm not in normalized_parts:
                normalized_parts.append(part_norm)
        if normalized_parts:
            return ", ".join(normalized_parts)

    single = normalize_resultado_piece(text, classe_canon, allowed)
    return single or text.strip()


def normalize_votacao(value: str) -> str:
    lowered = normalize_class_text(value)
    if not lowered:
        return ""
    if lowered in {"nao especificada", "em curso"}:
        return ""
    if "unanim" in lowered or "unanime" in lowered:
        return "Unânime"
    if "maioria" in lowered:
        return "Por maioria"
    if re.search(r"\b\d+\s*a\s*\d+\b", lowered):
        return "Por maioria"
    if "julgamento conjunto" in lowered:
        return ""
    if "pedido de vista" in lowered:
        return "Suspenso"
    if "suspens" in lowered:
        return "Suspenso"
    return value.strip()


def parse_date_from_text(text: str) -> str:
    match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if match:
        a, b, year = map(int, match.groups())
        mdy_ok = True
        dmy_ok = True
        try:
            date(year, a, b)
        except ValueError:
            mdy_ok = False
        try:
            date(year, b, a)
        except ValueError:
            dmy_ok = False
        if a > 12 and dmy_ok:
            return f"{b}/{a}/{year}"
        if b > 12 and mdy_ok:
            return f"{a}/{b}/{year}"
        if dmy_ok:
            return f"{b}/{a}/{year}"
        if mdy_ok:
            return f"{a}/{b}/{year}"
        return ""

    months = {
        "janeiro": "01",
        "fevereiro": "02",
        "março": "03",
        "marco": "03",
        "abril": "04",
        "maio": "05",
        "junho": "06",
        "julho": "07",
        "agosto": "08",
        "setembro": "09",
        "outubro": "10",
        "novembro": "11",
        "dezembro": "12",
    }
    match = re.search(
        r"\b(\d{1,2})\s+de\s+([A-Za-zçÇãõéíóúÁÉÍÓÚ]+)\s+de\s+(\d{4})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    day, month_name, year = match.groups()
    month = months.get(month_name.lower())
    if not month:
        return ""
    day_int = int(day)
    month_int = int(month)
    year_int = int(year)
    try:
        date(year_int, month_int, day_int)
    except ValueError:
        return ""
    return f"{month_int}/{day_int}/{year_int}"


def normalize_session_date_to_iso(value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value
    parsed = parse_date_from_text(value)
    if not parsed:
        return ""
    month, day, year = [int(part) for part in parsed.split("/")]
    return f"{year:04d}-{month:02d}-{day:02d}"


def split_csv_like_text(value: str) -> list[str]:
    if not value:
        return []
    parts = [item.strip() for item in re.split(r"\s*,\s*|\s*;\s*", value) if item.strip()]
    return dedupe_preserve_order(parts)


def parse_multi_value_text(value: str) -> list[str]:
    if isinstance(value, list):
        return dedupe_preserve_order(str(item).strip() for item in value if str(item).strip())
    return split_csv_like_text(str(value or ""))


def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        return parsed.path.lstrip("/").split("/", 1)[0]
    if "youtube.com" in host:
        query = parse_qs(parsed.query)
        video_id = query.get("v", [""])[0]
        if video_id:
            return video_id
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
    return ""


def normalize_youtube_link(value: str) -> str:
    if not value:
        return ""
    video_id = extract_youtube_video_id(value)
    if not video_id:
        return value.strip()
    parsed = urlparse(value.strip())
    query = parse_qs(parsed.query)
    t_value = query.get("t", [""])[0] or query.get("start", [""])[0]
    if not t_value and parsed.fragment.startswith("t="):
        t_value = parsed.fragment.split("=", 1)[1]
    params = {"v": video_id}
    if t_value:
        params["t"] = t_value
    return f"https://www.youtube.com/watch?{urlencode(params)}"


def build_timestamped_youtube_link(value: str, start_seconds: int | None) -> str:
    normalized = normalize_youtube_link(value)
    video_id = extract_youtube_video_id(normalized)
    if not video_id:
        return value.strip()
    params = {"v": video_id}
    if start_seconds is not None and start_seconds >= 0:
        params["t"] = str(int(start_seconds))
    return f"https://www.youtube.com/watch?{urlencode(params)}"
