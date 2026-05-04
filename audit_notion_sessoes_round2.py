from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes import (
    ARTIFACT_ROOT as ROUND1_ARTIFACT_ROOT,
    PageRecord,
    better_theme,
    clean_sentence,
    extract_youtube_timestamp_or_none,
    load_records,
    string_value,
    weak_theme_reason,
)
from local_secrets import get_secret
from tse_normalization import (
    canonicalize_numero_processo,
    dedupe_preserve_order,
    extract_uf_from_text,
    normalize_class_text,
    normalize_origem_value,
    normalize_partes_list,
    parse_multi_value_text,
)
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NOTION_PROPERTY_MAP,
    NotionDataSourceSchema,
    NotionSessoesClient,
    PublishPreviewRow,
    build_fallback_tema,
    infer_theme_from_row_text,
    normalize_party_list,
    normalize_resultado_final,
)


LOGGER = logging.getLogger("audit_notion_sessoes_round2")
ARTIFACT_ROOT = Path("artifacts") / "notion_sessoes_audit_round2"
APPLY_SLEEP_SECONDS = 0.16
SCHEMA_SLEEP_SECONDS = 0.2
REQUEST_RETRIES = 4
REQUEST_RETRY_BASE_SECONDS = 1.6
RELATION_FIELDS = {"materia_semelhante", "materia_semelhante1"}
LIST_FIELDS = {"partes"}
TEXT_FIELDS = {"tema", "punchline"}
SELECT_FIELDS = {"origem", "votacao", "resultado"}
SUPPORTED_UPDATE_FIELDS = TEXT_FIELDS | SELECT_FIELDS | LIST_FIELDS | RELATION_FIELDS
ROUND1_PUNCHLINE_CHANGE_FILES = [
    ROUND1_ARTIFACT_ROOT / "apply_20260430_0914" / "changes.json",
    ROUND1_ARTIFACT_ROOT / "residual_apply_20260430_1000" / "changes.json",
]

UF_CAPITALS = {
    "AC": "Rio Branco/AC",
    "AL": "Maceió/AL",
    "AP": "Macapá/AP",
    "AM": "Manaus/AM",
    "BA": "Salvador/BA",
    "CE": "Fortaleza/CE",
    "DF": "Brasília/DF",
    "ES": "Vitória/ES",
    "GO": "Goiânia/GO",
    "MA": "São Luís/MA",
    "MT": "Cuiabá/MT",
    "MS": "Campo Grande/MS",
    "MG": "Belo Horizonte/MG",
    "PA": "Belém/PA",
    "PB": "João Pessoa/PB",
    "PR": "Curitiba/PR",
    "PE": "Recife/PE",
    "PI": "Teresina/PI",
    "RJ": "Rio de Janeiro/RJ",
    "RN": "Natal/RN",
    "RS": "Porto Alegre/RS",
    "RO": "Porto Velho/RO",
    "RR": "Boa Vista/RR",
    "SC": "Florianópolis/SC",
    "SP": "São Paulo/SP",
    "SE": "Aracaju/SE",
    "TO": "Palmas/TO",
}

STATE_NAME_TO_CAPITAL = {
    "acre": UF_CAPITALS["AC"],
    "alagoas": UF_CAPITALS["AL"],
    "amapa": UF_CAPITALS["AP"],
    "amapá": UF_CAPITALS["AP"],
    "amazonas": UF_CAPITALS["AM"],
    "bahia": UF_CAPITALS["BA"],
    "ceara": UF_CAPITALS["CE"],
    "ceará": UF_CAPITALS["CE"],
    "distrito federal": UF_CAPITALS["DF"],
    "espirito santo": UF_CAPITALS["ES"],
    "espírito santo": UF_CAPITALS["ES"],
    "goias": UF_CAPITALS["GO"],
    "goiás": UF_CAPITALS["GO"],
    "maranhao": UF_CAPITALS["MA"],
    "maranhão": UF_CAPITALS["MA"],
    "mato grosso": UF_CAPITALS["MT"],
    "mato grosso do sul": UF_CAPITALS["MS"],
    "minas gerais": UF_CAPITALS["MG"],
    "para": UF_CAPITALS["PA"],
    "pará": UF_CAPITALS["PA"],
    "paraiba": UF_CAPITALS["PB"],
    "paraíba": UF_CAPITALS["PB"],
    "parana": UF_CAPITALS["PR"],
    "paraná": UF_CAPITALS["PR"],
    "pernambuco": UF_CAPITALS["PE"],
    "piaui": UF_CAPITALS["PI"],
    "piauí": UF_CAPITALS["PI"],
    "rio de janeiro": UF_CAPITALS["RJ"],
    "rio grande do norte": UF_CAPITALS["RN"],
    "rio grande do sul": UF_CAPITALS["RS"],
    "rondonia": UF_CAPITALS["RO"],
    "rondônia": UF_CAPITALS["RO"],
    "roraima": UF_CAPITALS["RR"],
    "santa catarina": UF_CAPITALS["SC"],
    "sao paulo": UF_CAPITALS["SP"],
    "são paulo": UF_CAPITALS["SP"],
    "sergipe": UF_CAPITALS["SE"],
    "tocantins": UF_CAPITALS["TO"],
}
CAPITAL_NAME_TO_VALUE = {normalize_class_text(value.rsplit("/", 1)[0]): value for value in UF_CAPITALS.values()}

CNJ_ELECTORAL_UF_BY_CODE = {
    "01": "AC",
    "02": "AL",
    "03": "AP",
    "04": "AM",
    "05": "BA",
    "06": "CE",
    "07": "DF",
    "08": "ES",
    "09": "GO",
    "10": "MA",
    "11": "MT",
    "12": "MS",
    "13": "MG",
    "14": "PA",
    "15": "PB",
    "16": "PR",
    "17": "PE",
    "18": "PI",
    "19": "RJ",
    "20": "RN",
    "21": "RS",
    "22": "RO",
    "23": "RR",
    "24": "SC",
    "25": "SE",
    "26": "SP",
    "27": "TO",
}

INVALID_ORIGEM_MARKERS = [
    "tribunal regional eleitoral",
    "tribunal superior eleitoral",
    "tre",
    "tse",
    "zona eleitoral",
    "juizo eleitoral",
    "juízo eleitoral",
    "jurisprudencia",
    "jurisprudência",
    "decisao",
    "decisão",
    "decisoes",
    "decisões",
]

OVERBROAD_THEME_KEYS = {
    "abuso de poder",
    "inelegibilidade",
    "lista triplice",
    "lista tríplice",
    "prestacao de contas",
    "prestação de contas",
    "prestacao de contas de campanha",
    "prestação de contas de campanha",
    "prestacao de contas partidarias",
    "prestação de contas partidárias",
    "propaganda eleitoral antecipada",
    "propaganda eleitoral irregular",
}
TRUNCATED_THEME_END_RE = re.compile(
    r"(?i)\b(?:art|artigo|inciso|paragrafo|parágrafo|do|da|dos|das|de|em|no|na|pelo|pela|que|sob|com)$"
)
THEME_SENTENCE_NOISE_RE = re.compile(
    r"(?i)\b(?:"
    r"cerne da controv[eé]rsia|presidente da sess[aã]o|peticionaram|solicitando|"
    r"anuncia o in[ií]cio|vislumbrou|n[aã]o vislumbrou|havia sido prefeito|"
    r"recorreu ao tse|sustentou que|alegou que"
    r")\b"
)
THEME_SENTENCE_START_RE = re.compile(
    r"(?i)^(?:"
    r"a controv[eé]rsia|a proposta|a decis[aã]o|a defesa|a consulente|a alega[cç][aã]o|"
    r"a candidata|a exist[eê]ncia|a falta|a agravante|a embargante|as contas|"
    r"argumentou|alegou|sustentou|considerou|concluiu|afastou|quanto ao|"
    r"ele sustenta|o recurso|o candidato|o partido|o tre|o tribunal|o ministro|"
    r"os ministros|o plen[aá]rio|o pedido"
    r")\b"
)
THEME_PREFIX_RE = re.compile(
    r"(?i)^(?:o processo|o caso|o recurso|a consulta|o julgamento|a ação|a acao)\s+"
    r"(?:trata|discute|versa|cuida)\s+(?:de|sobre)\s+"
)
PARTY_VERSUS_RE = re.compile(r"(?i)\s+(?:x|versus|vs\.?)\s+")


@dataclass
class FieldChange:
    page_id: str
    page_url: str
    data_sessao: str
    video_id: str
    timestamp_seconds: int | None
    numero_processo: str
    field: str
    property_name: str
    old: Any
    new: Any
    reason: str
    confidence: str = "high"

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_url": self.page_url,
            "data_sessao": self.data_sessao,
            "video_id": self.video_id,
            "timestamp_seconds": self.timestamp_seconds,
            "numero_processo": self.numero_processo,
            "field": self.field,
            "property_name": self.property_name,
            "old": self.old,
            "new": self.new,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class PageChangeSet:
    record: PageRecord
    changes: dict[str, FieldChange] = field(default_factory=dict)

    def add(
        self,
        field_name: str,
        property_name: str,
        old_value: Any,
        new_value: Any,
        reason: str,
        confidence: str = "high",
    ) -> None:
        if field_name not in SUPPORTED_UPDATE_FIELDS:
            raise ValueError(f"Campo nao suportado para update: {field_name}")
        normalized_old = normalize_report_value(old_value)
        normalized_new = normalize_report_value(new_value)
        if normalized_old == normalized_new:
            return
        self.changes[field_name] = FieldChange(
            page_id=self.record.page_id,
            page_url=self.record.page_url,
            data_sessao=self.record.row.data_sessao,
            video_id=self.record.video_id,
            timestamp_seconds=self.record.timestamp_seconds,
            numero_processo=self.record.row.numero_processo,
            field=field_name,
            property_name=property_name,
            old=normalized_old,
            new=normalized_new,
            reason=reason,
            confidence=confidence,
        )


def normalize_report_value(value: Any) -> Any:
    if isinstance(value, list):
        return [string_value(item) for item in value if string_value(item)]
    return string_value(value)


def canonical_list(values: Any) -> list[str]:
    if isinstance(values, list):
        raw_values = values
    else:
        raw_values = parse_multi_value_text(string_value(values))
    return dedupe_preserve_order([string_value(value) for value in raw_values if string_value(value)])


def extract_relation_ids(page: dict[str, Any], property_name: str) -> list[str]:
    prop = page.get("properties", {}).get(property_name, {})
    return [
        string_value(item.get("id"))
        for item in prop.get("relation", []) or []
        if string_value(item.get("id"))
    ]


def normalize_id(value: str) -> str:
    return string_value(value).replace("-", "")


def same_relation_set(left: list[str], right: list[str]) -> bool:
    return {normalize_id(value) for value in left} == {normalize_id(value) for value in right}


def notion_request_with_retry(client: NotionSessoesClient, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            return client._request(method, path, **kwargs)
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = any(
                marker in message
                for marker in [
                    "rate_limited",
                    "timeout",
                    "timed out",
                    "502",
                    "503",
                    "504",
                    "connection",
                ]
            )
            if not retryable or attempt == REQUEST_RETRIES:
                raise
            time.sleep(REQUEST_RETRY_BASE_SECONDS ** attempt)
    raise RuntimeError(f"Falha inesperada na API do Notion: {last_error}") from last_error


def previous_punchline_page_ids(extra_paths: list[str] | None = None) -> set[str]:
    paths = list(ROUND1_PUNCHLINE_CHANGE_FILES)
    for item in extra_paths or []:
        paths.append(Path(item))
    page_ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.warning("Nao foi possivel ler historico de punchline em %s", path)
            continue
        for change in payload:
            if change.get("field") == "punchline" and change.get("page_id"):
                page_ids.add(string_value(change.get("page_id")))
    return page_ids


def infer_uf_from_cnj_number(numero_processo: str) -> str:
    match = re.search(r"\.6\.(\d{2})\.\d{4}\b", string_value(numero_processo))
    if not match:
        return ""
    return CNJ_ELECTORAL_UF_BY_CODE.get(match.group(1), "")


def infer_uf_from_tribunal(tribunal: str) -> str:
    match = re.search(r"\bTRE[-/\s]?([A-Z]{2})\b", string_value(tribunal), flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def capital_for_uf(uf: str) -> str:
    return UF_CAPITALS.get(string_value(uf).upper(), "")


def is_state_name_city(city: str, uf: str) -> bool:
    capital = capital_for_uf(uf)
    if not capital:
        return False
    return normalize_class_text(city) in STATE_NAME_TO_CAPITAL and normalize_class_text(city) != normalize_class_text(
        capital.rsplit("/", 1)[0]
    )


def strict_origem_value(row: PublishPreviewRow) -> tuple[str, str]:
    current = string_value(row.origem)
    current_norm = normalize_class_text(current)
    tribunal_uf = infer_uf_from_tribunal(row.tribunal)
    process_uf = infer_uf_from_cnj_number(row.numero_processo)

    if current_norm in {"tse", "tribunal superior eleitoral"}:
        return UF_CAPITALS["DF"], "origem indicava apenas TSE; padronizada pela sede em cidade/UF"

    if re.fullmatch(r"[A-Z]{2}", current, flags=re.IGNORECASE):
        uf = current.upper()
        capital = capital_for_uf(uf)
        if capital:
            return capital, "origem indicava apenas UF; substituida pela capital em cidade/UF"

    if current_norm in STATE_NAME_TO_CAPITAL:
        return STATE_NAME_TO_CAPITAL[current_norm], "origem indicava apenas Estado; substituida pela capital em cidade/UF"
    if current_norm in CAPITAL_NAME_TO_VALUE:
        return CAPITAL_NAME_TO_VALUE[current_norm], "origem trazia capital sem UF; padronizada como cidade/UF"
    if current_norm in {"tribunal regional federal", "trf"}:
        return UF_CAPITALS["DF"], "origem indicava apenas orgao federal; padronizada pela sede em cidade/UF"

    normalized = normalize_origem_value(current)
    normalized_norm = normalize_class_text(normalized)
    if normalized_norm in {"tse", "tribunal superior eleitoral"}:
        return UF_CAPITALS["DF"], "origem indicava apenas TSE; padronizada pela sede em cidade/UF"

    uf = extract_uf_from_text(normalized) or extract_uf_from_text(current)
    if re.fullmatch(r"TRE[/\-\s]?[A-Z]{2}", normalized or current, flags=re.IGNORECASE):
        uf = uf or re.search(r"([A-Z]{2})", normalized or current, flags=re.IGNORECASE).group(1).upper()
        return capital_for_uf(uf), "origem indicava apenas TRE/UF; substituida pela capital em cidade/UF"

    if normalized_norm in STATE_NAME_TO_CAPITAL:
        return STATE_NAME_TO_CAPITAL[normalized_norm], "origem indicava apenas Estado; substituida pela capital em cidade/UF"

    if normalized and "/" in normalized:
        match = re.match(r"^(?P<city>.+?)\s*/\s*(?P<uf>[A-Z]{2})$", normalized)
        if match:
            city = match.group("city").strip()
            uf = match.group("uf").upper()
            city_norm = normalize_class_text(city)
            if city_norm in STATE_NAME_TO_CAPITAL or is_state_name_city(city, uf):
                return capital_for_uf(uf), "origem usava nome do Estado no lugar da cidade; substituida pela capital"
            if any(marker in city_norm for marker in INVALID_ORIGEM_MARKERS):
                return capital_for_uf(uf), "origem continha orgao ou unidade judiciaria, nao cidade; substituida pela capital da UF"
            return f"{city}/{uf}", ""

    if current_norm in {"brasilia", "brasília"}:
        return UF_CAPITALS["DF"], "origem trazia cidade sem UF; padronizada como cidade/UF"

    if normalized and "/" not in normalized:
        normalized_uf = extract_uf_from_text(normalized)
        if normalized_uf:
            return capital_for_uf(normalized_uf), "origem continha apenas referencia estadual; substituida pela capital em cidade/UF"

    fallback_uf = uf or tribunal_uf or process_uf
    if normalized and "/" not in normalized and fallback_uf:
        if not any(marker in normalized_norm for marker in INVALID_ORIGEM_MARKERS):
            return f"{normalized}/{fallback_uf}", "origem trazia cidade sem UF; UF inferida do tribunal ou numero do processo"

    if fallback_uf:
        return capital_for_uf(fallback_uf), "origem sem cidade/UF confiavel; usada capital da UF inferida"

    if not current:
        return UF_CAPITALS["DF"], "origem vazia e sem UF confiavel; usada sede do TSE como fallback"

    return "", "origem sem cidade/UF confiavel e sem UF inferivel"


def clean_theme_phrase(value: str, row: PublishPreviewRow) -> str:
    candidate = string_value(value)
    candidate = THEME_PREFIX_RE.sub("", candidate).strip()
    candidate = re.sub(r"(?i)^trata(?:-|\s+)se de\s+", "", candidate).strip()
    candidate = re.sub(r"\s+", " ", candidate).strip(" .;:-")
    if not candidate:
        return ""
    if re.match(r"^(?:por unanimidade|por maioria|o tribunal|a corte|o plen[aá]rio)\b", normalize_class_text(candidate)):
        return ""
    candidate = clean_sentence(candidate, max_len=130).rstrip(".")
    return candidate


def strip_party_context_from_theme(value: str) -> str:
    candidate = string_value(value)
    candidate = re.split(
        r"(?i)\s+(?:ajuizad[ao]s?|movid[ao]s?|propost[ao]s?|apresentad[ao]s?)\s+"
        r"(?:pelo|pela|por|contra)\b",
        candidate,
        maxsplit=1,
    )[0]
    candidate = re.split(r"(?i)\s+contra\s+(?:o|a|os|as)?\b", candidate, maxsplit=1)[0]
    candidate = re.split(r"(?i)\s+sob\s+relatoria\b", candidate, maxsplit=1)[0]
    return candidate.strip(" ,;:-")


def nominalize_theme_sentence(value: str, row: PublishPreviewRow) -> str:
    text = re.sub(r"\s+", " ", string_value(value)).strip(" .;:-")
    if not text:
        return ""
    patterns = [
        r"(?i)^a controv[eé]rsia (?:gira em torno|reside|est[aá]) (?:da|do|de|na|no|em torno da|em torno do)\s+(.+)$",
        r"(?i)^o foco [eé]\s+(.+)$",
        r"(?i)^discute-se\s+(?:a|o|os|as)?\s*(.+)$",
        r"(?i)^o processo trata (?:da|do|de|sobre)\s+(.+)$",
        r"(?i)^o caso trata (?:da|do|de|sobre)\s+(.+)$",
        r"(?i)^o julgamento trata (?:da|do|de|sobre|de um|de uma)\s+(.+)$",
        r"(?i)^o julgamento refere-se (?:à|a|ao|a uma|a um)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            return clean_theme_phrase(strip_party_context_from_theme(match.group(1)), row)

    lowered = normalize_class_text(text)
    if "prestacao de contas" in lowered or "prestação de contas" in lowered:
        year_match = re.search(r"\b(20\d{2})\b", text)
        if year_match:
            return f"Prestação de contas partidárias do exercício de {year_match.group(1)}"
        if "campanha" in lowered:
            return "Prestação de contas de campanha e regularidade de recursos eleitorais"
        return "Prestação de contas partidárias e regularidade da escrituração contábil"
    if "substituicao de candidato" in lowered or "substituição de candidato" in lowered:
        return "Substituição de candidato após o prazo legal"
    if "abuso de poder economico" in lowered or "abuso de poder econômico" in lowered:
        return "Abuso de poder econômico e gravidade da conduta eleitoral"
    if "desfiliacao" in lowered or "desfiliação" in lowered:
        return "Desfiliação partidária sem justa causa e perda de mandato eletivo"
    if "porte de armas" in lowered and "votacao" in lowered:
        return "Porte de armas em locais de votação"
    if "desinformacao" in lowered or "fake news" in lowered:
        return "Desinformação eleitoral em redes sociais"
    if "plano de midia" in lowered or "plano de mídia" in lowered:
        return "Plano de mídia do horário eleitoral gratuito"
    if "lista triplice" in lowered or "lista tríplice" in lowered:
        tribunal_match = re.search(r"\bTRE[-/ ]?([A-Z]{2})\b", text, flags=re.IGNORECASE)
        suffix = f" do TRE-{tribunal_match.group(1).upper()}" if tribunal_match else ""
        return f"Formação de lista tríplice{suffix}"
    if "fundo especial de financiamento de campanha" in lowered or "fefc" in lowered:
        return "Repasse de recursos do FEFC e do Fundo Partidário entre partidos"
    if "filiação partidária" in text.lower() or "filiacao partidaria" in lowered:
        return "Comprovação de filiação partidária em registro de candidatura"
    if "inelegibilidade" in lowered and "art" in lowered:
        return "Inelegibilidade e incidência de causa legal impeditiva de candidatura"
    return ""


def theme_candidate_is_suspicious(value: str) -> bool:
    candidate = string_value(value)
    normalized = normalize_class_text(candidate)
    if not candidate:
        return True
    if TRUNCATED_THEME_END_RE.search(candidate):
        return True
    if THEME_SENTENCE_NOISE_RE.search(candidate):
        return True
    if THEME_SENTENCE_START_RE.match(candidate):
        return True
    if re.match(r"^(?:o|a|os|as)\s+\w+", normalized):
        return True
    if re.search(r"(?i)\b(?:R\$|nº|n°)\s*$", candidate):
        return True
    return False


def theme_looks_usable(value: str, row: PublishPreviewRow) -> bool:
    candidate = clean_theme_phrase(value, row)
    if not candidate:
        return False
    if theme_candidate_is_suspicious(candidate):
        return False
    if weak_theme_reason(row.model_copy(update={"tema": candidate})):
        return False
    normalized = normalize_class_text(candidate)
    if normalized in {"julgamento adiado por pedido de vista", "adiamento de julgamento por pedido de vista"}:
        return False
    if len(candidate) < 18:
        return False
    return True


def improved_theme(row: PublishPreviewRow) -> str:
    inferred = clean_theme_phrase(infer_theme_from_row_text(row), row)
    fallback = clean_theme_phrase(build_fallback_tema(row), row)
    prior = clean_theme_phrase(row.tema, row)
    nominalized_prior = nominalize_theme_sentence(row.tema, row)
    prior_norm = normalize_class_text(prior)
    prior_usable = theme_looks_usable(prior, row)
    prior_overbroad = prior_norm in OVERBROAD_THEME_KEYS
    if prior_usable and not prior_overbroad:
        return prior

    candidates = [nominalized_prior, inferred, fallback, better_theme(row)]
    for candidate in candidates:
        candidate = clean_theme_phrase(candidate, row)
        candidate_norm = normalize_class_text(candidate)
        if not candidate or not theme_looks_usable(candidate, row):
            continue
        if prior_usable and prior_overbroad and len(candidate) < len(prior) + 8:
            continue
        if candidate_norm != prior_norm:
            return candidate
    return prior if theme_looks_usable(prior, row) else ""


def result_prefix(result: str) -> str:
    normalized = normalize_class_text(result)
    if normalized in {"suspenso por vista", "suspenso mas julgado depois"}:
        return "Pedido de vista suspende"
    if "desprovid" in normalized:
        return "TSE mantém decisão sobre"
    if "provid" in normalized or "deferid" in normalized or "procedente" in normalized:
        return "TSE acolhe tese sobre"
    if "indeferid" in normalized or "improcedente" in normalized or "rejeitad" in normalized:
        return "TSE rejeita pretensão relativa a"
    if "nao conhecid" in normalized or "não conhecid" in normalized:
        return "Óbice processual impede exame de"
    if "aprovad" in normalized:
        return "TSE aprova encaminhamento sobre"
    if "prejudicad" in normalized:
        return "Perda de objeto encerra discussão sobre"
    return "TSE decide controvérsia sobre"


def extract_decisive_ground(row: PublishPreviewRow) -> str:
    combined = " ".join(
        string_value(value)
        for value in [
            row.raciocinio_juridico,
            row.fundamentacao_normativa,
            row.analise_do_conteudo_juridico,
        ]
        if string_value(value)
    )
    combined = re.sub(r"\s+", " ", combined).strip()
    if not combined:
        return ""
    legal_ground_start = (
        r"aus[eê]ncia|falta|insufici[eê]ncia|inexist[eê]ncia|intempestividade|"
        r"preclus[aã]o|perda de objeto|prescri[cç][aã]o|rejei[cç][aã]o de contas|"
        r"prova robusta|documentos? insuficientes?|irregularidade|viol[açc][aã]o|"
        r"configura[cç][aã]o|impossibilidade|incid[eê]ncia|comprova[cç][aã]o"
    )
    patterns = [
        rf"(?i)\b(?:ante|diante de|em razão de|em razao de|por)\s+(?P<ground>(?:{legal_ground_start})[^.;]{{0,95}})",
        r"(?i)\b(?P<ground>(?:aus[eê]ncia|falta|insufici[eê]ncia|inexist[eê]ncia)\s+de\s+[^.;]{12,85})",
        r"(?i)\b(?:porque|pois)\s+(?P<ground>(?:não|nao|houve|ficou|a conduta|a propaganda|as contas|os documentos)[^.;]{18,95})",
    ]
    for pattern in patterns:
        match = re.search(pattern, combined)
        if not match:
            continue
        ground = match.group("ground").strip(" ,;:-'\"()[]")
        ground_norm = normalize_class_text(ground)
        if not ground or any(
            marker in ground_norm
            for marker in [
                "relator",
                "ministro",
                "acompanhou o voto",
                "unanimidade",
                "por maioria",
                "colegiado",
                "recorrente",
                "agravante",
                "candidato",
                "candidata",
                "coligacao",
                "coligação",
            ]
        ):
            continue
        ground = re.sub(r"(?i)\b(?:do|da|dos|das|de)\s*$", "", ground).strip(" ,;:-")
        if "," in ground or re.search(r"(?i)(?:R\$\s*\d+[.,]?|['\"]|,\s*)$", ground):
            continue
        if re.search(r"(?i)\b(?:art|artigo|inciso|paragrafo|parágrafo)\s*$", ground):
            continue
        if len(ground) >= 18:
            return ground[:1].lower() + ground[1:]
    return ""


def editorial_punchline(row: PublishPreviewRow, theme: str) -> str:
    subject = string_value(theme or row.tema).rstrip(".")
    if not subject:
        return ""
    subject = subject[:1].lower() + subject[1:]
    result = string_value(row.resultado)
    prefix = result_prefix(result)
    ground = extract_decisive_ground(row)
    if ground and normalize_class_text(ground) in normalize_class_text(subject):
        ground = ""
    if prefix.startswith("Pedido de vista"):
        if "adiamento de julgamento" in normalize_class_text(subject):
            return clean_sentence("Pedido de vista suspende julgamento posteriormente retomado", max_len=175)
        if row.pedido_vista:
            return clean_sentence(f"Pedido de vista de {row.pedido_vista} suspende exame sobre {subject}", max_len=175)
        return clean_sentence(f"Pedido de vista suspende exame sobre {subject}", max_len=175)
    if ground:
        return clean_sentence(f"{prefix} {subject} por {ground}", max_len=175)
    return clean_sentence(f"{prefix} {subject}", max_len=175)


def add_origem_change(change_set: PageChangeSet) -> None:
    row = change_set.record.row
    proposed, reason = strict_origem_value(row)
    if not proposed:
        return
    if string_value(proposed) != string_value(row.origem):
        confidence = "medium" if "fallback" in reason or "inferida" in reason else "high"
        change_set.add("origem", "origem", row.origem, proposed, reason, confidence=confidence)
        row.origem = proposed


def add_theme_change(change_set: PageChangeSet) -> None:
    row = change_set.record.row
    proposed = improved_theme(row)
    if not proposed:
        return
    if normalize_class_text(proposed) == normalize_class_text(row.tema):
        return
    change_set.add("tema", "tema", row.tema, proposed, "tema refeito para frase nominal juridica e especifica", confidence="medium")
    row.tema = proposed


def add_punchline_change(change_set: PageChangeSet, excluded_page_ids: set[str]) -> None:
    if change_set.record.page_id in excluded_page_ids:
        return
    row = change_set.record.row
    if not theme_looks_usable(row.tema, row):
        return
    proposed = editorial_punchline(row, row.tema)
    if not proposed:
        return
    if normalize_class_text(proposed) == normalize_class_text(row.punchline):
        return
    change_set.add(
        "punchline",
        "punchline",
        row.punchline,
        proposed,
        "punchline refeita nesta rodada, preservando linhas ja corrigidas na rodada anterior",
        confidence="medium",
    )
    row.punchline = proposed


def sort_key_for_judgment(record: PageRecord) -> tuple[str, int, int, str]:
    timestamp = record.timestamp_seconds if record.timestamp_seconds is not None else 10**9
    return (string_value(record.row.data_sessao), timestamp, record.index, record.page_id)


def is_suspended_row(row: PublishPreviewRow) -> bool:
    normalized_result = normalize_class_text(row.resultado)
    normalized_vote = normalize_class_text(row.votacao)
    return normalized_result in {"suspenso por vista", "suspenso"} or normalized_vote == "suspenso"


def is_definitive_row(row: PublishPreviewRow) -> bool:
    normalized_result = normalize_class_text(row.resultado)
    if not normalized_result:
        return False
    if normalized_result.startswith("suspenso"):
        return False
    return True


def add_suspension_resolution_changes(change_sets: dict[str, PageChangeSet]) -> None:
    groups: dict[str, list[PageRecord]] = defaultdict(list)
    for change_set in change_sets.values():
        canonical = canonicalize_numero_processo(change_set.record.row.numero_processo)
        if canonical:
            groups[canonical].append(change_set.record)

    for records in groups.values():
        if len(records) < 2:
            continue
        ordered = sorted(records, key=sort_key_for_judgment)
        for index, record in enumerate(ordered[:-1]):
            if not is_suspended_row(record.row):
                continue
            later_records = ordered[index + 1 :]
            if not any(is_definitive_row(candidate.row) for candidate in later_records):
                continue
            change_set = change_sets[record.page_id]
            if record.row.votacao != "Suspenso*":
                change_set.add(
                    "votacao",
                    "votacao",
                    record.row.votacao,
                    "Suspenso*",
                    "processo suspenso por vista foi julgado definitivamente em sessao posterior",
                )
                record.row.votacao = "Suspenso*"
            if record.row.resultado != "Suspenso mas julgado depois":
                change_set.add(
                    "resultado",
                    "resultado",
                    record.row.resultado,
                    "Suspenso mas julgado depois",
                    "processo suspenso por vista foi julgado definitivamente em sessao posterior",
                )
                record.row.resultado = "Suspenso mas julgado depois"


def split_conjoined_party_values(values: list[str]) -> list[str]:
    split_values: list[str] = []
    for value in values:
        text = string_value(value)
        if not text:
            continue
        pieces = [piece.strip(" ,;") for piece in PARTY_VERSUS_RE.split(text) if piece.strip(" ,;")]
        if len(pieces) > 1:
            split_values.extend(pieces)
        else:
            split_values.append(text)
    return split_values


def normalized_partes_values(values: list[str]) -> list[str]:
    split_values = split_conjoined_party_values(values)
    normalized_text = normalize_partes_list(split_values)
    return normalize_party_list(parse_multi_value_text(normalized_text))


def add_partes_change(change_set: PageChangeSet) -> None:
    current = canonical_list(change_set.record.row.partes)
    if not current:
        return
    proposed = normalized_partes_values(current)
    if not proposed:
        return
    if current != proposed:
        change_set.add(
            "partes",
            "partes",
            current,
            proposed,
            "partes normalizadas e etiquetas encavaladas por 'x' separadas",
        )
        change_set.record.row.partes = proposed


def build_relation_targets(records: list[PageRecord]) -> dict[str, list[str]]:
    groups: dict[str, list[PageRecord]] = defaultdict(list)
    for record in records:
        canonical = canonicalize_numero_processo(record.row.numero_processo)
        if canonical:
            groups[canonical].append(record)
    targets_by_page: dict[str, list[str]] = {}
    for group in groups.values():
        unique_records: list[PageRecord] = []
        seen_page_ids: set[str] = set()
        for record in sorted(group, key=sort_key_for_judgment):
            if record.page_id and record.page_id not in seen_page_ids:
                seen_page_ids.add(record.page_id)
                unique_records.append(record)
        if len(unique_records) < 2:
            continue
        for record in unique_records:
            targets = [
                candidate.page_id
                for candidate in unique_records
                if candidate.page_id != record.page_id
                and (
                    candidate.video_id != record.video_id
                    or candidate.row.data_sessao != record.row.data_sessao
                )
            ]
            if targets:
                targets_by_page[record.page_id] = dedupe_preserve_order(targets)
    return targets_by_page


def add_relation_changes(
    change_sets: dict[str, PageChangeSet],
    schema: NotionDataSourceSchema,
    relation_targets: dict[str, list[str]],
) -> None:
    for page_id, targets in relation_targets.items():
        change_set = change_sets[page_id]
        for property_name in sorted(RELATION_FIELDS):
            if property_name not in schema.properties:
                continue
            current = extract_relation_ids(change_set.record.page, property_name)
            if same_relation_set(current, targets):
                continue
            change_set.add(
                property_name,
                property_name,
                current,
                targets,
                "relation robustecida para vincular sessoes distintas do mesmo numero de processo",
                confidence="high",
            )


def build_audit(
    records: list[PageRecord],
    schema: NotionDataSourceSchema,
    excluded_punchline_pages: set[str],
) -> dict[str, PageChangeSet]:
    change_sets = {record.page_id: PageChangeSet(record=copy.deepcopy(record)) for record in records}
    for change_set in change_sets.values():
        add_origem_change(change_set)
        add_theme_change(change_set)
        add_punchline_change(change_set, excluded_punchline_pages)
        add_partes_change(change_set)
    add_suspension_resolution_changes(change_sets)
    relation_targets = build_relation_targets([change_set.record for change_set in change_sets.values()])
    add_relation_changes(change_sets, schema, relation_targets)
    return change_sets


def property_payload_for_change(
    client: NotionSessoesClient,
    schema: NotionDataSourceSchema,
    change: FieldChange,
) -> dict[str, Any]:
    property_name = change.property_name
    if property_name not in schema.properties:
        raise RuntimeError(f"Propriedade {property_name!r} nao encontrada no schema.")
    new_value = change.new
    if new_value:
        built = client._build_property_value(schema, property_name, new_value)
    else:
        built = client._build_empty_property_value(schema, property_name)
    if built is None:
        raise RuntimeError(f"Nao foi possivel montar payload para {property_name!r}.")
    return built


def apply_page_changes(
    client: NotionSessoesClient,
    schema: NotionDataSourceSchema,
    page_id: str,
    changes: list[FieldChange],
) -> dict[str, Any]:
    payload = {"properties": {}}
    for change in changes:
        payload["properties"][change.property_name] = property_payload_for_change(client, schema, change)
    return notion_request_with_retry(client, "PATCH", f"/pages/{page_id}", json=payload)


def collect_option_usage_from_pages(pages: list[dict[str, Any]], property_name: str, prop_type: str) -> Counter[str]:
    usage: Counter[str] = Counter()
    for page in pages:
        value = page.get("properties", {}).get(property_name, {})
        if prop_type == "select":
            name = string_value((value.get("select") or {}).get("name"))
            if name:
                usage[name] += 1
        elif prop_type == "multi_select":
            for item in value.get("multi_select", []) or []:
                name = string_value(item.get("name"))
                if name:
                    usage[name] += 1
    return usage


def page_option_values(pages: list[dict[str, Any]], property_name: str, prop_type: str) -> list[tuple[str, str | list[str]]]:
    values: list[tuple[str, str | list[str]]] = []
    for page in pages:
        page_id = string_value(page.get("id"))
        if not page_id:
            continue
        prop = page.get("properties", {}).get(property_name, {})
        if prop_type == "select":
            value = string_value((prop.get("select") or {}).get("name"))
            values.append((page_id, value))
        elif prop_type == "multi_select":
            multi_values = [
                string_value(item.get("name"))
                for item in prop.get("multi_select", []) or []
                if string_value(item.get("name"))
            ]
            values.append((page_id, multi_values))
    return values


def create_option_property(client: NotionSessoesClient, property_name: str, prop_type: str) -> None:
    notion_request_with_retry(
        client,
        "PATCH",
        f"/data_sources/{client.data_source_id}",
        json={"properties": {property_name: {prop_type: {}}}},
    )


def restore_option_property_values(
    client: NotionSessoesClient,
    property_name: str,
    prop_type: str,
    page_values: list[tuple[str, str | list[str]]],
) -> tuple[int, list[dict[str, Any]]]:
    page_updates = 0
    failures: list[dict[str, Any]] = []
    for index, (page_id, value) in enumerate(page_values, start=1):
        if not page_id or value in ("", []):
            continue
        if prop_type == "select":
            payload_value = {"select": {"name": string_value(value)}}
        else:
            payload_value = {"multi_select": [{"name": item} for item in canonical_list(value)]}
        try:
            notion_request_with_retry(
                client,
                "PATCH",
                f"/pages/{page_id}",
                json={"properties": {property_name: payload_value}},
            )
            page_updates += 1
        except Exception as exc:
            failures.append({"page_id": page_id, "property": property_name, "error": str(exc)})
        if index % 25 == 0:
            LOGGER.info("Schema cleanup %s: valores restaurados %s/%s", property_name, index, len(page_values))
        time.sleep(SCHEMA_SLEEP_SECONDS)
    return page_updates, failures


def rebuild_option_property(
    client: NotionSessoesClient,
    property_name: str,
    prop_type: str,
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    page_values = page_option_values(pages, property_name, prop_type)
    page_values = [(page_id, value) for page_id, value in page_values if value not in ("", [])]
    temp_name = f"{property_name}__tmp_cleanup"
    legacy_name = f"{property_name}__legacy_cleanup"
    existing_properties = notion_request_with_retry(client, "GET", f"/data_sources/{client.data_source_id}").get("properties", {})
    for residue in [temp_name, legacy_name]:
        if residue in existing_properties:
            notion_request_with_retry(
                client,
                "PATCH",
                f"/data_sources/{client.data_source_id}",
                json={"properties": {residue: None}},
            )
            time.sleep(SCHEMA_SLEEP_SECONDS)

    try:
        create_option_property(client, temp_name, prop_type)
        page_updates, failures = restore_option_property_values(client, temp_name, prop_type, page_values)
        client.rename_property(property_name, legacy_name)
        client.rename_property(temp_name, property_name)
        client.drop_property(legacy_name)
        return {
            "property": property_name,
            "type": prop_type,
            "cleanup_status": "rebuilt_with_temp_property",
            "page_updates": page_updates,
            "failures": failures,
        }
    except Exception as exc:
        message = str(exc)
        LOGGER.warning("Reconstrucao temporaria de %s falhou: %s", property_name, message)
        refreshed = notion_request_with_retry(client, "GET", f"/data_sources/{client.data_source_id}").get("properties", {})
        for residue in [temp_name, legacy_name]:
            if residue in refreshed:
                notion_request_with_retry(
                    client,
                    "PATCH",
                    f"/data_sources/{client.data_source_id}",
                    json={"properties": {residue: None}},
                )
                time.sleep(SCHEMA_SLEEP_SECONDS)
        client.drop_property(property_name)
        create_option_property(client, property_name, prop_type)
        page_updates, failures = restore_option_property_values(client, property_name, prop_type, page_values)
        return {
            "property": property_name,
            "type": prop_type,
            "cleanup_status": "rebuilt_in_place",
            "page_updates": page_updates,
            "failures": failures,
            "temp_error": message,
        }


def cleanup_unused_schema_options(
    client: NotionSessoesClient,
    schema: NotionDataSourceSchema,
    pages: list[dict[str, Any]],
    *,
    apply_changes: bool,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    raw_properties = schema.raw_payload.get("properties", {})
    for property_name, raw_prop in sorted(raw_properties.items()):
        prop_type = raw_prop.get("type")
        if property_name.endswith(("__tmp_cleanup", "__legacy_cleanup")):
            finding = {
                "property": property_name,
                "type": prop_type,
                "options": 0,
                "used_options": 0,
                "unused_options": 0,
                "cleanup_status": "would_drop_cleanup_residue",
            }
            if apply_changes:
                try:
                    notion_request_with_retry(
                        client,
                        "PATCH",
                        f"/data_sources/{client.data_source_id}",
                        json={"properties": {property_name: None}},
                    )
                    finding["cleanup_status"] = "dropped_cleanup_residue"
                except Exception as exc:
                    finding["cleanup_status"] = "drop_residue_failed"
                    finding["error"] = str(exc)
            findings.append(finding)
            continue
        if prop_type not in {"select", "multi_select"}:
            continue
        options = ((raw_prop.get(prop_type) or {}).get("options") or [])
        option_names = [string_value(option.get("name")) for option in options if string_value(option.get("name"))]
        usage = collect_option_usage_from_pages(pages, property_name, prop_type)
        unused = [name for name in option_names if usage[name] == 0]
        if not unused:
            findings.append(
                {
                    "property": property_name,
                    "type": prop_type,
                    "options": len(option_names),
                    "used_options": len(usage),
                    "unused_options": 0,
                    "cleanup_status": "no_unused_options",
                }
            )
            continue
        finding = {
            "property": property_name,
            "type": prop_type,
            "options": len(option_names),
            "used_options": len(usage),
            "unused_options": len(unused),
            "unused_sample": unused[:25],
            "cleanup_status": "would_rebuild_property" if len(usage) > 100 else "would_patch_options",
        }
        if not apply_changes:
            findings.append(finding)
            continue

        if len(usage) <= 100:
            remaining_options = [
                {"name": string_value(option.get("name")), "color": string_value(option.get("color") or "default")}
                for option in options
                if string_value(option.get("name")) and usage[string_value(option.get("name"))] > 0
            ]
            for used_name in usage:
                if used_name not in {option["name"] for option in remaining_options}:
                    remaining_options.append({"name": used_name, "color": "default"})
            try:
                notion_request_with_retry(
                    client,
                    "PATCH",
                    f"/data_sources/{client.data_source_id}",
                    json={"properties": {property_name: {prop_type: {"options": remaining_options}}}},
                )
                finding["cleanup_status"] = "patched_options"
            except Exception as exc:
                finding["cleanup_status"] = "patch_failed"
                finding["error"] = str(exc)
            findings.append(finding)
            continue

        try:
            rebuild_summary = rebuild_option_property(client, property_name, prop_type, pages)
            finding.update(rebuild_summary)
        except Exception as exc:
            finding["cleanup_status"] = "rebuild_failed"
            finding["error"] = str(exc)
        findings.append(finding)
    return findings


def ensure_required_select_options(client: NotionSessoesClient, schema: NotionDataSourceSchema) -> dict[str, Any]:
    missing: dict[str, list[str]] = {}
    for property_name, values in {
        "votacao": ["Suspenso*"],
        "resultado": ["Suspenso mas julgado depois"],
    }.items():
        prop = schema.properties.get(property_name)
        if not prop or prop.type != "select":
            continue
        absent = [value for value in values if value not in prop.options]
        if absent:
            missing[property_name] = absent
    return client.ensure_select_options_default(missing)


def write_reports(
    artifact_dir: Path,
    changes: list[FieldChange],
    schema_findings: list[dict[str, Any]],
    apply_results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    change_dicts = [change.as_dict() for change in changes]
    (artifact_dir / "changes.json").write_text(json.dumps(change_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    with (artifact_dir / "changes.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(change_dicts[0].keys()) if change_dicts else list(
            FieldChange("", "", "", "", None, "", "", "", "", "", "").as_dict().keys()
        )
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(change_dicts)
    (artifact_dir / "schema_cleanup.json").write_text(json.dumps(schema_findings, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "apply_results.json").write_text(json.dumps(apply_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_summary(
    records: list[PageRecord],
    changes: list[FieldChange],
    schema_findings: list[dict[str, Any]],
    apply_results: list[dict[str, Any]],
    apply_mode: bool,
    excluded_punchlines: int,
) -> dict[str, Any]:
    return {
        "mode": "apply" if apply_mode else "dry-run",
        "total_records": len(records),
        "total_field_changes": len(changes),
        "pages_with_changes": len({change.page_id for change in changes}),
        "changes_by_field": dict(sorted(Counter(change.field for change in changes).items())),
        "changes_by_confidence": dict(sorted(Counter(change.confidence for change in changes).items())),
        "previous_round_punchlines_excluded": excluded_punchlines,
        "schema_findings": len(schema_findings),
        "schema_findings_by_status": dict(Counter(item.get("cleanup_status", "unknown") for item in schema_findings)),
        "schema_unused_options": sum(int(item.get("unused_options") or 0) for item in schema_findings),
        "applied_pages": sum(1 for item in apply_results if item.get("status") == "updated"),
        "failed_pages": sum(1 for item in apply_results if item.get("status") == "failed"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditoria rodada 2 da database Notion de sessoes do TSE.")
    parser.add_argument("--apply", action="store_true", help="Aplica as correcoes no Notion. Sem isto, roda em dry-run.")
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--max-pages", type=int, default=0, help="Limita paginas atualizadas no modo apply.")
    parser.add_argument("--only-field", action="append", choices=sorted(SUPPORTED_UPDATE_FIELDS))
    parser.add_argument("--previous-punchline-changes", action="append", default=[])
    parser.add_argument("--skip-schema-cleanup", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    api_key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY/NOTION_TOKEN nao encontrado.")
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")

    client = NotionSessoesClient(api_key=api_key, data_source_id=args.data_source_id)
    LOGGER.info("Carregando schema e paginas do Notion...")
    schema = client.fetch_schema()
    records = load_records(client, schema)
    LOGGER.info("Paginas carregadas: %s", len(records))

    excluded_punchlines = previous_punchline_page_ids(args.previous_punchline_changes)
    change_sets = build_audit(records, schema, excluded_punchlines)
    changes = [change for change_set in change_sets.values() for change in change_set.changes.values()]
    if args.only_field:
        selected = set(args.only_field)
        changes = [change for change in changes if change.field in selected]

    schema_findings: list[dict[str, Any]] = []
    apply_results: list[dict[str, Any]] = []
    if args.apply:
        ensure_summary = ensure_required_select_options(client, schema)
        LOGGER.info("Opcoes obrigatorias verificadas: %s", json.dumps(ensure_summary, ensure_ascii=False))
        schema = client.fetch_schema()
        by_page: dict[str, list[FieldChange]] = defaultdict(list)
        for change in changes:
            by_page[change.page_id].append(change)
        page_items = list(by_page.items())
        if args.max_pages > 0:
            page_items = page_items[: args.max_pages]
        LOGGER.info("Aplicando correcoes em %s paginas...", len(page_items))
        for page_index, (page_id, page_changes) in enumerate(page_items, start=1):
            try:
                apply_page_changes(client, schema, page_id, page_changes)
                apply_results.append(
                    {
                        "page_id": page_id,
                        "status": "updated",
                        "fields": [change.field for change in page_changes],
                    }
                )
            except Exception as exc:
                apply_results.append(
                    {
                        "page_id": page_id,
                        "status": "failed",
                        "fields": [change.field for change in page_changes],
                        "error": str(exc),
                    }
                )
                LOGGER.warning("Falha ao atualizar pagina %s: %s", page_id, exc)
            if page_index % 25 == 0:
                LOGGER.info("Paginas processadas: %s/%s", page_index, len(page_items))
            time.sleep(APPLY_SLEEP_SECONDS)

    if args.skip_schema_cleanup:
        schema_findings = [{"cleanup_status": "skipped_by_flag"}]
    else:
        LOGGER.info("Coletando paginas para limpeza de schema...")
        cleanup_schema = client.fetch_schema()
        cleanup_pages = client.query_data_source()
        schema_findings = cleanup_unused_schema_options(
            client,
            cleanup_schema,
            cleanup_pages,
            apply_changes=args.apply,
        )

    summary = build_summary(records, changes, schema_findings, apply_results, args.apply, len(excluded_punchlines))
    write_reports(artifact_dir, changes, schema_findings, apply_results, summary)
    LOGGER.info("Relatorios gravados em %s", artifact_dir)
    LOGGER.info("Resumo: %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
