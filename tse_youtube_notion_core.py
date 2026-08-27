from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from html import unescape
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import requests
from pydantic import BaseModel, Field, ValidationError

from local_secrets import get_secret, load_local_secrets
from tse_normalization import (
    CNJ_ELECTORAL_UF_BY_CODE,
    STATE_NAME_KEYS,
    STATE_UF,
    UF_CAPITALS,
    build_video_only_youtube_link,
    build_timestamped_youtube_link,
    canonicalize_party_option_label,
    canonicalize_numero_processo,
    cnj_check_digit_is_valid,
    instrucao_autos_originais_cnj,
    resultado_allowed_for_classe,
    dedupe_preserve_order,
    extract_chunk_judgment_process_values,
    extract_full_cnj,
    format_short_process_number_from_digits,
    extract_uf_from_text,
    extract_youtube_video_id,
    identity_overlay_class_key,
    infer_session_date_from_video_title,
    composicao_regimental_issue,
    normalize_advogados_list,
    normalize_class_text,
    normalize_classe_processo,
    normalize_composicao,
    normalize_eleicao_value,
    normalize_ministro_name,
    normalize_mpe_reference,
    normalize_numero_processo_display,
    normalize_origem_value,
    normalize_pedido_vista_value,
    normalize_partes_list,
    normalize_resultado_final,
    normalize_session_date_to_iso,
    normalize_tre,
    normalize_votacao,
    normalize_youtube_link,
    parse_multi_value_text,
    is_regimentally_valid_composicao,
    is_plausible_ministro_name,
    split_csv_like_text,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR / "artifacts" / "tse_youtube_notion"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_NEWS_GEMINI_MODEL = os.getenv("GEMINI_NEWS_MODEL") or DEFAULT_GEMINI_MODEL
GEMINI_REST_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_NOTION_VERSION = os.getenv("NOTION_VERSION") or "2025-09-03"
DEFAULT_NOTION_DATA_SOURCE_ID = (
    os.getenv("NOTION_DATA_SOURCE_ID") or "2eb72195-5c64-80ea-9cd5-000b0e01745d"
)
DEFAULT_NOTION_DATABASE_URL = (
    os.getenv("NOTION_DATABASE_URL")
    or "https://www.notion.so/2eb721955c64809796bec75a81f9555f?v=ffe93c7f3ae4415699545f93f566d152"
)
GEMINI_CALL_RETRIES = 3
GEMINI_RETRY_BASE_DELAY = 2.0
DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS = int(os.getenv("GEMINI_HTTP_TIMEOUT_SECONDS") or "240")
GENERIC_THEME_CLASS_RESULT_REGEX = re.compile(
    r"^(?:"
    r"(?:ed\s+)?(?:agrg\s+)?(?:arespe|respe)"
    r"|pc|pa|ro|ms|rms|rhc|hc|petciv|qo|cta|rpp|rve|ado|adi|adpf|inq"
    r"|lt|lista triplice|lista tríplice"
    r")"
    r"(?:\s+(?:"
    r"provido|desprovido|parcialmente provido|"
    r"deferido|indeferido|aprovado|aprovada|rejeitado|rejeitada|"
    r"nao conhecido|não conhecido|por maioria|unanime|unânime"
    r"))?$",
    flags=re.IGNORECASE,
)
GENERIC_THEME_DECISION_SENTENCE_REGEX = re.compile(
    r"^(?:"
    r"(?:o\s+)?(?:tribunal|plen[aá]rio|colegiado|relator)"
    r"|por\s+unanimidade|por\s+maioria|unanimemente"
    r")\b.*\b(?:"
    r"negou\s+provimento|deu\s+provimento|proveu|desproveu|"
    r"n[aã]o\s+conheceu|conheceu\s+e\s+deu\s+provimento|"
    r"indeferiu|deferiu|rejeitou|acolheu|aprovou|"
    r"julgou\s+(?:procedente|improcedente|prejudicado)|"
    r"manteve|cassou|reformou"
    r")\b",
    flags=re.IGNORECASE,
)
GENERIC_THEME_REPORTING_SENTENCE_REGEX = re.compile(
    r"^(?:(?:al[eé]m disso|ainda|por fim|nesse contexto|nesse ponto)(?:,\s+|\s+))?"
    r"(?:(?:o|a)\s+relator(?:a)?\s+)?"
    r"(?:destacou|assinalou|observou|ressaltou|salientou|pontuou|consignou|"
    r"afirmou|entendeu|registrou|reafirmou|frisou|anotou)\s+que\b",
    flags=re.IGNORECASE,
)
TRUNCATED_PUNCHLINE_ENDING_REGEX = re.compile(
    r"(?i)\b(?:para|com|de|do|da|dos|das|sobre|mediante|visando|quanto|pagamento|utilizacao|utilização|destinacao|destinação|custeio|art)\.?$"
)

OVERBROAD_THEME_VALUES = {
    "fraude a cota de genero",
    "propaganda eleitoral irregular",
    "propaganda eleitoral antecipada",
    "inelegibilidade",
    "prestacao de contas",
    "prestação de contas",
    "prestacao de contas partidarias",
    "prestação de contas partidárias",
    "prestacao de contas de campanha",
    "prestação de contas de campanha",
    "publicidade institucional em periodo vedado",
    "publicidade institucional em período vedado",
}

SAFE_DYNAMIC_SELECT_OPTIONS = {
    "classe_processo": {
        "AgR-HC",
        "AgRg-AI",
        "AgRg-AR",
        "AgRg-AREspe",
        "AgRg-HC",
        "AgRg-MS",
        "AgRg-PC",
        "AgRg-REspe",
        "AgRg-RHC",
        "AgRg-RO",
        "AREspe",
        "AIJE",
        "CTA",
        "Czer",
        "ED-AgRg-AREspe",
        "ED-AREspe",
        "ED-ED-LT",
        "ED-Lista Tríplice",
        "ED-PC",
        "ED-PetCiv",
        "ED-REspe",
        "ED-RO",
        "Lista Tríplice",
        "MS",
        "PA",
        "PC",
        "PetCiv",
        "QO",
        "Ref-TutCautAnt",
        "Ref.-MS",
        "REspe",
        "Rp",
        "RHC",
        "RMS",
        "RO",
        "RPP",
        "RvE",
        "TutCautAnt",
    },
    "resultado": {
        "Procedente",
        "Procedente em parte",
        "Improcedente",
        "Suspenso mas julgado depois",
    },
    "votacao": {"Unânime", "Por maioria", "Suspenso", "Suspenso*"},
    "relator": {
        "Min. Alexandre de Moraes",
        "Min. André Mendonça",
        "Min. Antônio Carlos Ferreira",
        "Min. Benedito Gonçalves",
        "Min. Cármen Lúcia",
        "Min. Edilene Lôbo",
        "Min. Estela Aranha",
        "Min. Floriano de Azevedo Marques",
        "Min. Isabel Gallotti",
        "Min. Nunes Marques",
        "Min. Ramos Tavares",
        "Min. Raul Araújo",
        "Min. Ricardo Villas Bôas Cueva",
        "Min. Vera Lúcia Santana Araújo",
    },
}
SCHEMA_EXPANDED_SELECT_PROPERTIES = {
    "relator",
    "pedido_vista",
}
TIPO_REGISTRO_DYNAMIC_RE = re.compile(r"^Julgamento\s+\d+$", re.IGNORECASE)


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def build_gemini_http_options(types: Any, timeout_seconds: int = DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS) -> Any:
    try:
        timeout_value = int(timeout_seconds)
    except (TypeError, ValueError):
        timeout_value = 0
    if timeout_value <= 0:
        return None
    http_options_cls = getattr(types, "HttpOptions", None)
    if http_options_cls is None:
        return None
    return http_options_cls(timeout=timeout_value)


def create_gemini_client(genai: Any, types: Any, api_key: str, timeout_seconds: int = DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS) -> Any:
    env_google = os.getenv("GOOGLE_API_KEY")
    env_gemini = os.getenv("GEMINI_API_KEY")
    if env_google and env_gemini and env_google == env_gemini:
        os.environ.pop("GEMINI_API_KEY", None)
    http_options = build_gemini_http_options(types, timeout_seconds=timeout_seconds)
    if http_options is None:
        return genai.Client(api_key=api_key)
    return genai.Client(api_key=api_key, http_options=http_options)


def _build_gemini_rest_part(
    *,
    text: str | None = None,
    file_uri: str | None = None,
    mime_type: str | None = None,
    start_seconds: int | None = None,
    end_seconds: int | None = None,
) -> dict[str, Any]:
    if text is not None:
        return {"text": text}
    part: dict[str, Any] = {
        "fileData": {
            "fileUri": str(file_uri or ""),
            "mimeType": str(mime_type or "application/octet-stream"),
        }
    }
    if start_seconds is not None or end_seconds is not None:
        metadata: dict[str, Any] = {}
        if start_seconds is not None:
            metadata["startOffset"] = f"{int(start_seconds)}s"
        if end_seconds is not None:
            metadata["endOffset"] = f"{int(end_seconds)}s"
        part["videoMetadata"] = metadata
    return part


def _extract_generate_content_text(payload: dict[str, Any]) -> str:
    texts: list[str] = []
    for candidate in payload.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = normalize_model_text(part.get("text"))
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def _extract_generate_content_grounding_urls(payload: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for candidate in payload.get("candidates") or []:
        metadata = candidate.get("groundingMetadata") or {}
        for chunk in metadata.get("groundingChunks") or []:
            web = chunk.get("web") or {}
            normalized = normalize_external_url(web.get("uri"))
            if normalized:
                urls.append(normalized)
    return dedupe_preserve_order(urls)


# Teto de saída por chamada. As respostas legítimas deste pipeline não passam de
# poucos milhares de tokens (o maior raw_detail observado tem ~8 KB); o teto existe
# para cortar CEDO a resposta degenerada em loop, que de outro modo só para no
# limite do modelo e volta truncada — chegando ao pydantic como o críptico
# "Invalid JSON: EOF while parsing an object at line 5188". 0 desliga o teto.
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS") or "32768")

_GEMINI_TRUNCATION_REASONS = {"MAX_TOKENS", "RECITATION", "SAFETY", "PROHIBITED_CONTENT", "SPII"}


def _raise_on_truncated_gemini_response(payload: dict[str, Any], *, model_name: str) -> None:
    """Falha com diagnóstico quando a resposta foi CORTADA pelo servidor.

    Sem isto, uma resposta truncada por maxOutputTokens só aparece como erro de
    parsing do JSON, mandando quem lê o log caçar bug de formatação onde o que
    houve foi corte por tamanho (tipicamente: modelo em loop de repetição).
    """
    for candidate in payload.get("candidates") or []:
        reason = str(candidate.get("finishReason") or "").upper()
        if reason and reason in _GEMINI_TRUNCATION_REASONS:
            raise RuntimeError(
                f"Resposta do Gemini interrompida pelo servidor (finishReason={reason}, "
                f"modelo={model_name}). Saída truncada — não é erro de formato do JSON. "
                "MAX_TOKENS aqui costuma indicar o modelo em loop de repetição."
            )


def call_gemini_generate_content_rest(
    *,
    api_key: str,
    model_name: str,
    contents: list[dict[str, Any]],
    system_instruction: str,
    response_model: type[BaseModel],
    temperature: float = 0.1,
    use_google_search: bool = False,
    timeout_seconds: int = DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS,
) -> tuple[BaseModel, str, dict[str, Any]]:
    generation_config: dict[str, Any] = {"temperature": temperature}
    # IMPORTANTE: o Gemini DESABILITA o Grounding with Google Search quando o output
    # é forçado para JSON (responseMimeType). Em chamadas grounded, omitimos o
    # responseMimeType para a busca realmente rodar; os URLs vêm do groundingMetadata
    # e da extração de URLs do texto (ver _coerce_gemini_response_model).
    if not use_google_search:
        generation_config["responseMimeType"] = "application/json"
    payload: dict[str, Any] = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": generation_config,
    }
    if use_google_search:
        payload["tools"] = [{"googleSearch": {}}]

    if GEMINI_MAX_OUTPUT_TOKENS > 0:
        generation_config["maxOutputTokens"] = GEMINI_MAX_OUTPUT_TOKENS

    url = f"{GEMINI_REST_BASE_URL}/models/{model_name}:generateContent?key={api_key}"
    response = requests.post(url, json=payload, timeout=(10, timeout_seconds))
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini REST error {response.status_code}: {response.text[:2000]}")
    response_payload = response.json()
    response_text = _extract_generate_content_text(response_payload)
    _raise_on_truncated_gemini_response(response_payload, model_name=model_name)
    return _coerce_gemini_response_model(response_model, response_text), response_text, response_payload


GLOBAL_SCAN_WINDOW_SECONDS = int(os.getenv("GLOBAL_SCAN_WINDOW_SECONDS") or "300")
GLOBAL_SCAN_OVERLAP_SECONDS = int(os.getenv("GLOBAL_SCAN_OVERLAP_SECONDS") or "30")
GLOBAL_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS = int(os.getenv("GLOBAL_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS") or "3")
GLOBAL_SCAN_FALLBACK_WINDOW_SECONDS = int(os.getenv("GLOBAL_SCAN_FALLBACK_WINDOW_SECONDS") or "120")
GLOBAL_SCAN_FALLBACK_OVERLAP_SECONDS = int(os.getenv("GLOBAL_SCAN_FALLBACK_OVERLAP_SECONDS") or "15")
# Guardas contra DEGENERAÇÃO do modelo no scan global (loop de repetição). Em
# 19/08/2026 o gemini-3.1-flash-lite devolveu, para a janela 540s-840s de um
# vídeo de 3694s, 64 blocos em passos exatos de 100s indo até 8315s, com 63 de
# 64 números CNJ reprovados no dígito verificador — série fabricada por contador.
# A janela pedida é o contrato: bloco que cai muito fora dela é descartado.
SCAN_WINDOW_TOLERANCE_SECONDS = int(os.getenv("SCAN_WINDOW_TOLERANCE_SECONDS") or "300")
SCAN_VIDEO_END_TOLERANCE_SECONDS = int(os.getenv("SCAN_VIDEO_END_TOLERANCE_SECONDS") or "120")
SCAN_DEGENERATION_MIN_BLOCKS = int(os.getenv("SCAN_DEGENERATION_MIN_BLOCKS") or "10")
TRANSCRIPT_SCAN_MAX_CHARS = int(os.getenv("TRANSCRIPT_SCAN_MAX_CHARS") or "4000")
TRANSCRIPT_SCAN_OVERLAP_SNIPPETS = int(os.getenv("TRANSCRIPT_SCAN_OVERLAP_SNIPPETS") or "1")
TRANSCRIPT_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS = int(os.getenv("TRANSCRIPT_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS") or "2")
TRANSCRIPT_DETAIL_PADDING_SECONDS = int(os.getenv("TRANSCRIPT_DETAIL_PADDING_SECONDS") or "45")
# Folga de CAUDA da janela no caminho por VIDEO. O recorte enviado ao Gemini era o
# end_seconds cru do scan, sem folga nenhuma -- so o caminho por transcricao tinha os 45 s
# acima. A proclamacao/anuncio de vista mora depois do ultimo voto e cai fora por poucos
# segundos: na sessao de 25/08/2026 o bloco do RO 0600960-72 terminava em 4465 s e o
# Presidente anunciou o pedido de vista do Min. Andre Mendonca em 4470 s -- 5 s adiante.
# O caso saiu "Unanime"/"Desprovido" no Notion. A extensao nunca invade o bloco seguinte
# (para nao misturar julgamentos) nem encurta a janela original.
VIDEO_DETAIL_TAIL_PADDING_SECONDS = int(os.getenv("VIDEO_DETAIL_TAIL_PADDING_SECONDS") or "60")
TRANSCRIPT_DETAIL_MAX_CHARS = int(os.getenv("TRANSCRIPT_DETAIL_MAX_CHARS") or "6000")
REFINE_START_LOOKBACK_SECONDS = int(os.getenv("REFINE_START_LOOKBACK_SECONDS") or "90")
REFINE_START_LOOKAHEAD_SECONDS = int(os.getenv("REFINE_START_LOOKAHEAD_SECONDS") or "90")
ENABLE_START_REFINEMENT = env_flag("TSE_ENABLE_START_REFINEMENT", True)
ENABLE_TRANSITION_REFINEMENT = env_flag("TSE_ENABLE_TRANSITION_REFINEMENT", True)
CONDITIONAL_START_REFINEMENT = env_flag("TSE_CONDITIONAL_START_REFINEMENT", True)
PREFERRED_GEMINI_MODELS = [DEFAULT_GEMINI_MODEL]
GENERAL_NEWS_LIMIT = 9
NOT_FOUND_TEXT_MARKERS = (
    "pagina nao encontrada",
    "pagina inexistente",
    "page not found",
    "erro 404",
    "error 404",
    "conteudo nao encontrado",
    "conteudo indisponivel",
)

TSE_DOMAIN_RE = re.compile(r"(?:^|\.)tse\.jus\.br$", re.IGNORECASE)
TRE_DOMAIN_RE = re.compile(r"(?:^|\.)tre-[a-z]{2}\.jus\.br$", re.IGNORECASE)

# Convenção do usuário (20/08/2026): caso envolvendo deputado federal ganha 💚 no ícone da
# página; deputado estadual OU distrital ganha 🧡; os dois cargos juntos, verde. O ⏰ (vista
# em aberto) prevalece — por isso vistas/sobrestados não recebem coração, e updates nunca
# tocam o ícone (só a criação), preservando ⏰ e curadoria manual.
ICONE_DEP_FEDERAL = "💚"
ICONE_DEP_ESTADUAL = "🧡"
_DEP_FEDERAL_RE = re.compile(r"deputad[oa]s?\s+federa(?:l|is)", re.IGNORECASE)
_DEP_ESTADUAL_RE = re.compile(r"deputad[oa]s?\s+(?:estadua(?:l|is)|distrita(?:l|is))", re.IGNORECASE)


def suggest_deputado_icon(row: "PublishPreviewRow") -> str:
    """Sugere o coração da convenção para uma linha a publicar; '' quando não se aplica."""
    resultado = str(getattr(row, "resultado", "") or "")
    if resultado in ("Suspenso por vista", "Sobrestado"):
        return ""  # vista em aberto é território do ⏰
    blob = " ".join(
        str(getattr(row, campo, "") or "")
        for campo in ("tema", "punchline", "analise_do_conteudo_juridico", "raciocinio_juridico", "partes")
    )
    if _DEP_FEDERAL_RE.search(blob):
        return ICONE_DEP_FEDERAL
    if _DEP_ESTADUAL_RE.search(blob):
        return ICONE_DEP_ESTADUAL
    return ""
GENERAL_DOMAINS = [
    "folha.uol.com.br",
    "estadao.com.br",
    "gazetadopovo.com.br",
    "cnnbrasil.com.br",
    "cnn.com",
    "conjur.com.br",
    "migalhas.com.br",
    "g1.globo.com",
    "oglobo.globo.com",
    "poder360.com.br",
]
GENERAL_DOMAINS_RE = re.compile(
    r"(?:^|\.)(?:" + "|".join(re.escape(domain) for domain in GENERAL_DOMAINS) + r")$",
    re.IGNORECASE,
)
RECOMPUTED_ERROR_PATTERNS = (
    re.compile(r"^Valor inválido para ", re.IGNORECASE),
    re.compile(r"^Tema/título vazio\.$", re.IGNORECASE),
    re.compile(r"^resultado 'Suspenso por vista' sem pedido_vista", re.IGNORECASE),
)
RECOMPUTED_WARNING_PATTERNS = (
    re.compile(r"^(?:classe_processo|tipo_registro|eleicao|origem|relator|pedido_vista|resultado|votacao) fora das opções do Notion;", re.IGNORECASE),
    re.compile(r"^(?:classe_processo|tipo_registro|eleicao|origem|tribunal|relator|pedido_vista|resultado|votacao|partes|advogados|composicao) com opções novas no Notion:", re.IGNORECASE),
    re.compile(r"^CNJ indica UF ", re.IGNORECASE),
    re.compile(r"^resultado '", re.IGNORECASE),
    re.compile(r"^pedido_vista preenchido por auto-reparo", re.IGNORECASE),
    re.compile(r"^Tema/título vazio\.$", re.IGNORECASE),
    re.compile(r"^Número do processo não identificado;", re.IGNORECASE),
    re.compile(r"^Número do processo incompleto ", re.IGNORECASE),
    re.compile(r"^Data da sessão não identificada em formato ISO\.$", re.IGNORECASE),
    re.compile(r"^Mais de \d+ notícias gerais informadas;", re.IGNORECASE),
    re.compile(r"^noticia_TSE descartada por indisponibilidade da página\.$", re.IGNORECASE),
    re.compile(r"^noticia_TRE descartada por indisponibilidade da página\.$", re.IGNORECASE),
    re.compile(r"^Número corrigido para os autos originais da instrução", re.IGNORECASE),
    re.compile(r"^Dígito verificador do CNJ reprova", re.IGNORECASE),
)

NOTION_PROPERTY_MAP = {
    "tema": "tema",
    "classe_processo": "classe_processo",
    "tipo_registro": "tipo_registro",
    "eleicao": "eleicao",
    "origem": "origem",
    "tribunal": "tribunal",
    "numero_processo": "numero_processo",
    "youtube_link": "youtube_link",
    "relator": "relator",
    "pedido_vista": "pedido_vista",
    "resultado": "resultado",
    "votacao": "votacao",
    "partes": "partes",
    "advogados": "advogados",
    "composicao": "composicao",
    "punchline": "punchline",
    "analise_do_conteudo_juridico": "analise_do_conteudo_juridico",
    "fundamentacao_normativa": "fundamentacao_normativa",
    "precedentes_citados": "precedentes_citados",
    "raciocinio_juridico": "raciocinio_juridico",
    "resolucoes_citadas": "resoluções_citadas",
    "materia_semelhante": "materia_semelhante",
    "data_sessao": "data_sessao",
    "noticia_TSE": "noticia_TSE",
    "noticia_TRE": "noticia_TRE",
}

EXPECTED_NOTION_PROPERTIES = {
    "classe_processo",
    "tipo_registro",
    "eleicao",
    "origem",
    "tribunal",
    "numero_processo",
    "youtube_link",
    "relator",
    "pedido_vista",
    "resultado",
    "votacao",
    "partes",
    "advogados",
    "composicao",
    "punchline",
    "analise_do_conteudo_juridico",
    "fundamentacao_normativa",
    "precedentes_citados",
    "raciocinio_juridico",
    "resoluções_citadas",
    "data_sessao",
}


GLOBAL_SYSTEM_PROMPT = """
Você é um juiz eleitoral incumbido de analisar tecnicamente uma sessão do Tribunal Superior Eleitoral transmitida em vídeo no YouTube.

Use exclusivamente o conteúdo do vídeo fornecido. Não use qualquer fonte externa. Não complete lacunas com suposições.

Nesta primeira etapa, sua função é segmentar a sessão:
- identificar a data da sessão;
- identificar a composição do colegiado presente (os ministros que efetivamente participam dos julgamentos);
- listar cada julgamento com seu timestamp inicial em segundos;
- estimar o timestamp final quando possível;
- marcar explicitamente blocos que devam ser ignorados, especialmente o julgamento em lista ao final da sessão.

ATENÇÃO — "julgamento em lista" NÃO é "lista tríplice". "Julgamento em lista" é o bloco de processos julgados de uma vez em bloco, em regra ao final da sessão ("chamo a julgamento as listas desta pauta"): esse SIM deve ser ignorado. "Lista tríplice" é a formação da lista de três advogados/juristas para uma vaga de membro de TRE: é um JULGAMENTO NORMAL e deve ser extraído como qualquer outro, nunca ignorado nem tratado como bloco cerimonial ou administrativo.

Se houver julgamento conjunto, mantenha um único bloco para o trecho conjunto, mas liste os números de processo percebidos.

Sobre a composição: o TSE julga em colegiado de até 7 ministros (3 oriundos do STF, 2 do STJ e 2 juristas/advogados da classe). Liste cada ministro presente pelo nome, no formato "Min. <Nome>". Pode haver 6 nomes em caso de ausência, ou substitutos. NÃO inclua ministros citados apenas como autores de votos ou precedentes de outros processos, nem partes, advogados ou procuradores. A composição costuma ser lida uma única vez na abertura da sessão e vale para todos os julgamentos daquela mesma sessão.
"""

TRANSCRIPT_GLOBAL_SYSTEM_PROMPT = """
Você é um juiz eleitoral incumbido de analisar tecnicamente uma sessão do Tribunal Superior Eleitoral (TSE) com base exclusivamente na transcrição do próprio vídeo do YouTube.

Use apenas a transcrição fornecida no prompt. Não use qualquer fonte externa. Não complete lacunas com suposições.

Nesta etapa, sua função é segmentar a sessão:
- identificar a data da sessão;
- identificar a composição do colegiado presente (os ministros que efetivamente participam dos julgamentos);
- listar cada julgamento com seu timestamp inicial em segundos;
- estimar o timestamp final quando possível;
- marcar explicitamente blocos que devam ser ignorados, especialmente julgamento em lista, leitura de ata ou trechos meramente cerimoniais/administrativos.

ATENÇÃO — "julgamento em lista" NÃO é "lista tríplice". "Julgamento em lista" é o bloco de processos julgados de uma vez em bloco, em regra ao final da sessão ("chamo a julgamento as listas desta pauta"): esse SIM deve ser ignorado. "Lista tríplice" é a formação da lista de três advogados/juristas para uma vaga de membro de TRE: é um JULGAMENTO NORMAL e deve ser extraído como qualquer outro, nunca ignorado nem tratado como bloco cerimonial ou administrativo.

Se houver julgamento conjunto, mantenha um único bloco para o trecho conjunto, mas liste os números de processo percebidos.

Sobre a composição: o TSE julga em colegiado de até 7 ministros (3 oriundos do STF, 2 do STJ e 2 juristas/advogados da classe). Liste cada ministro presente pelo nome, no formato "Min. <Nome>". Pode haver 6 nomes em caso de ausência, ou substitutos. NÃO inclua ministros citados apenas como autores de votos ou precedentes de outros processos, nem partes, advogados ou procuradores. A composição costuma ser lida uma única vez na abertura da sessão e vale para todos os julgamentos daquela mesma sessão.
"""

DETAIL_SYSTEM_PROMPT = """
Você é um juiz eleitoral incumbido de analisar tecnicamente o conteúdo de uma sessão do Tribunal Superior Eleitoral (TSE) transmitida em vídeo do YouTube.

FONTE:
- Use exclusivamente o conteúdo do próprio vídeo.
- Se a informação não estiver no vídeo, deixe o campo vazio.
- Ignore totalmente julgamento em lista ao final da sessão (o bloco de processos julgados de uma vez). Isso NÃO alcança a lista tríplice: formação de lista tríplice para vaga de TRE é julgamento normal e deve ser extraída integralmente.

TAREFA:
- Analise apenas o julgamento delimitado pelo trecho de vídeo fornecido.
- Se o trecho contiver mais de um processo julgado em conjunto, retorne um item por processo.
- Não crie item para número de processo citado apenas como precedente, comparação, referência jurisprudencial ou exemplo.
- Para cada item, extraia:
  data da sessão, eleição, indicados em lista tríplice se houver, número do processo, origem, UF, partes, advogados, composição, relator, pedido de vista, jurisprudência citada, resoluções, legislação, análise jurídica, fundamentação normativa, precedentes citados, raciocínio jurídico, pontos processuais, efeitos práticos, resultado e votação.
- Produza texto jurídico detalhado e rastreável.

ORIENTAÇÕES OBRIGATÓRIAS POR CAMPO:
- `analise_do_conteudo_juridico`: descreva prioritariamente os FATOS do caso concreto. Informe a conduta atribuída, o contexto fático-eleitoral, o município/UF quando aparecerem no vídeo, a decisão recorrida, a sanção ou consequência discutida e as teses recursais efetivamente mencionadas. Evite fórmulas vagas como "o recurso discute" sem explicar o que aconteceu no caso.
- `analise_do_conteudo_juridico`: o foco deve ser a narrativa factual do litígio e do percurso processual imediato, não uma síntese abstrata da matéria.
- `fundamentacao_normativa`: identifique com precisão os diplomas e dispositivos efetivamente citados no julgamento. Sempre que o vídeo trouxer essa informação com clareza, informe número da lei ou resolução, artigo, parágrafo, inciso, alínea ou súmula mencionados.
- `fundamentacao_normativa`: se o voto reproduzir ou sintetizar a redação do dispositivo, registre essa redação entre parênteses. Não escreva apenas "legislação eleitoral" ou "Lei 9.504/97" sem o dispositivo correspondente.
- `fundamentacao_normativa`: diferencie norma legal ou regulamentar de súmula ou precedente. Se não houver indicação segura do dispositivo, deixe o campo vazio em vez de generalizar.
- `raciocinio_juridico`: reconstrua os argumentos jurídicos aplicados ao caso concreto. Indique quais premissas fáticas ou processuais foram tomadas como dadas, qual norma, súmula ou precedente foi usado, qual argumento da parte foi acolhido ou rejeitado e por que isso levou ao resultado.
- `raciocinio_juridico`: não escreva apenas que o recurso foi provido ou desprovido. Explique o encadeamento lógico entre fatos, norma e conclusão.
- `raciocinio_juridico`: mencione barreiras processuais, como vedação ao reexame de fatos e provas, apenas quando elas forem efetivamente usadas como razão de decidir no voto.
- `tema`: informe uma frase nominal jurídica, específica e indexável, aderente à controvérsia concreta, como "conduta vedada por uso de bens públicos" ou "fraude à cota de gênero em chapa proporcional". Nunca use o nome de ministro, relator ou de quem pediu vista como tema: pedido de vista, leitura de voto e proclamação são incidentes da sessão, não a matéria julgada — o tema tem de sair da controvérsia, mesmo quando o vídeo só mostra a proclamação. "Embargos de declaração" e "Agravo regimental" também descrevem o rito, não a matéria: em vez deles, diga sobre O QUE se embargou ou se agravou. Não descreva o resultado, não repita número do processo, não use nomes das partes como eixo principal e nunca use apenas rótulos genéricos como "Processo", "Julgamento" ou só a classe processual. Se o vídeo não permitir identificar o tema com segurança, deixe o campo vazio.
- `punchline`: escreva uma frase editorial curta, precisa e autônoma, contextualizando o caso, a tese jurídica debatida e a consequência do julgamento. A `punchline` deve complementar o `tema`, não repeti-lo com outras palavras. Evite fórmulas pobres como "recurso provido", "julgamento sobre..." ou simples cópia da ementa.
- `classe_processo`: leia a classe processual exatamente como aparece na autuação/cabeçalho exibido na tela e no pregão do caso (ex.: "AgR-AREspe nº 0601309-60"). Capture a classe COMPLETA, preservando os prefixos de recurso interno, especialmente Agravo Regimental (AgR/AgRg) e Embargos de Declaração (ED), antes da classe-base. Não reduza um "AgR-AREspe" a "AREspe" nem um "ED-REspe" a "REspe". Se houver agravo regimental sendo julgado pelo colegiado contra decisão monocrática, a classe é a forma com AgRg-. Se a tela não exibir a classe com clareza, deixe o campo vazio em vez de adivinhar a classe-base.
- `origem`: informe o MUNICÍPIO de origem do processo no formato "Cidade/UF" (ex.: "Santo Antônio do Tauá/PA"), tal como citado no caso. Não preencha origem com o tribunal ("Tribunal Regional Eleitoral do Pará", "TRE-PA") nem com a capital do estado quando o município específico aparecer no vídeo; o nome do tribunal de origem pertence a outro contexto, não à coluna origem.
- `resultado_final`: registre SEMPRE o desfecho objetivo proclamado para este processo, conforme a classe. Recurso (REspe/AREspe/RO/AgRg-*/RHC/RMS): "Provido", "Desprovido", "Provido em parte", "Não conhecido"/"Não conhecida", "Prejudicado". Consulta: "Aprovada" (consulta respondida). Lista tríplice formada/encaminhada: "Aprovada". Prestação de contas: "Aprovada"/"Aprovada com ressalvas"/"Rejeitada". Registro (RPP/RCand/DRAP): "Deferido"/"Indeferido". Representação/AIJE: "Procedente"/"Procedente em parte"/"Improcedente". Use o gênero correto (recurso=masculino; consulta/contas=feminino). Se o julgamento foi suspenso por pedido de vista, use "Suspenso por vista". Não deixe vazio quando o presidente proclamar o resultado.
- `pedido_vista`: nomeie o ministro que PEDIU VISTA, no formato "Min. <Nome>". Atenção ao pedido de vista ANTECIPADO: é comum o relator ou o presidente ANUNCIAR que outro ministro antecipará a vista ("o Ministro X antecipa o pedido de vista"). Quem pede vista é o ministro NOMEADO, nunca quem anuncia. Se ninguém pediu vista, deixe o campo vazio.
- QUEM FALA NÃO É QUEM DECIDE. Antes de atribuir a alguém uma divergência, um voto-vista ou um pedido de vista, confira o nome dito no julgamento: o presidente conduz a sessão e anuncia atos de outros ministros o tempo todo, e o relator resume votos alheios. Nunca deduza o autor da divergência a partir de quem está com a palavra. Se não der para identificar o ministro com segurança, escreva "houve divergência" sem nome em vez de atribuí-la a quem falava.
- COERÊNCIA entre `pedido_vista`, `analise_do_conteudo_juridico`, `raciocinio_juridico` e `punchline`: quem pediu vista e quem divergiu podem ser pessoas diferentes — não funda os dois papéis num nome só. Se `pedido_vista` ficou vazio, não afirme em prosa quem pediu vista; se o julgamento foi suspenso por vista, a prosa não pode descrevê-lo como decidido.
- `votacao`: registre como o colegiado votou — "Unânime" (decidido sem divergência; ninguém vencido), "Por maioria" (houve voto vencido/divergência aberta/"X votos a Y") ou "Suspenso" (julgamento suspenso por pedido de vista e ainda NÃO decidido). COERÊNCIA com o resultado: se o resultado for um desfecho definitivo, a votação é "Unânime" ou "Por maioria" — nunca "Suspenso". A simples menção a um pedido de vista anterior não torna a votação "Suspenso" se o caso foi efetivamente julgado.
"""

TRANSCRIPT_DETAIL_SYSTEM_PROMPT = """
Você é um juiz eleitoral incumbido de analisar tecnicamente um julgamento do Tribunal Superior Eleitoral (TSE) com base exclusivamente na transcrição do próprio vídeo do YouTube.

FONTE:
- Use apenas a transcrição do vídeo fornecida no prompt.
- Não use qualquer fonte externa.
- Se a informação não estiver claramente na transcrição, deixe o campo vazio.
- Ignore totalmente julgamento em lista ao final da sessão (o bloco de processos julgados de uma vez). Isso NÃO alcança a lista tríplice: formação de lista tríplice para vaga de TRE é julgamento normal e deve ser extraída integralmente.

TAREFA:
- Analise apenas o julgamento delimitado pelos timestamps e pela transcrição fornecidos.
- Se o trecho contiver mais de um processo julgado em conjunto, retorne um item por processo.
- Não crie item para número de processo citado apenas como precedente, comparação, referência jurisprudencial ou exemplo.
- Extraia os mesmos campos exigidos na etapa detalhada do vídeo, preservando fidelidade máxima ao conteúdo efetivamente transcrito.
- `tema`: informe uma frase nominal jurídica, específica e indexável, aderente à controvérsia concreta. Nunca use o nome de ministro, relator ou de quem pediu vista como tema: pedido de vista, leitura de voto e proclamação são incidentes da sessão, não a matéria julgada — o tema tem de sair da controvérsia, mesmo quando o vídeo só mostra a proclamação. "Embargos de declaração" e "Agravo regimental" também descrevem o rito, não a matéria: em vez deles, diga sobre O QUE se embargou ou se agravou. Nunca use número do processo, "Processo", "Julgamento", nomes das partes como eixo principal ou apenas a classe processual. Se não houver base suficiente na transcrição, deixe vazio.
- `punchline`: escreva uma frase editorial curta, precisa e autônoma, contextualizando o caso, a tese jurídica debatida e a consequência do julgamento. A `punchline` deve complementar o `tema`, não repeti-lo com outras palavras. Evite fórmulas pobres como "recurso provido", "julgamento sobre..." ou simples cópia da ementa.
- `classe_processo`: capture a classe COMPLETA como anunciada/transcrita, preservando prefixos de recurso interno, especialmente Agravo Regimental (AgR/AgRg) e Embargos de Declaração (ED). Não reduza "AgR-AREspe" a "AREspe". Se a transcrição não trouxer a classe com clareza, deixe vazio.
- `origem`: informe o MUNICÍPIO no formato "Cidade/UF" como citado no caso, não o tribunal (TRE) nem a capital quando o município específico aparecer.
- `pedido_vista`: nomeie o ministro que PEDIU VISTA, no formato "Min. <Nome>". Atenção ao pedido de vista ANTECIPADO: é comum o relator ou o presidente ANUNCIAR que outro ministro antecipará a vista ("o Ministro X antecipa o pedido de vista"). Quem pede vista é o ministro NOMEADO, nunca quem anuncia. Se ninguém pediu vista, deixe o campo vazio.
- QUEM FALA NÃO É QUEM DECIDE. Antes de atribuir a alguém uma divergência, um voto-vista ou um pedido de vista, confira o nome dito no julgamento: o presidente conduz a sessão e anuncia atos de outros ministros o tempo todo, e o relator resume votos alheios. Nunca deduza o autor da divergência a partir de quem está com a palavra. Se não der para identificar o ministro com segurança, escreva "houve divergência" sem nome em vez de atribuí-la a quem falava.
- COERÊNCIA entre `pedido_vista`, `analise_do_conteudo_juridico`, `raciocinio_juridico` e `punchline`: quem pediu vista e quem divergiu podem ser pessoas diferentes — não funda os dois papéis num nome só. Se `pedido_vista` ficou vazio, não afirme em prosa quem pediu vista; se o julgamento foi suspenso por vista, a prosa não pode descrevê-lo como decidido.
"""

NEWS_ENRICHMENT_SYSTEM_PROMPT = """
Você é um pesquisador jurídico-eleitoral especializado em localizar a COBERTURA jornalística de um julgamento específico do TSE.

TAREFA:
- Use exclusivamente o Grounding with Google Search. Ancore a busca no número do processo, nas partes, no tema e na data da sessão informados.
- Procure ativamente, no MESMO caso, em TRÊS frentes e devolva cada uma no seu campo:
  - noticia_TSE: matéria do portal do próprio TSE (tse.jus.br/comunicacao/noticias) sobre este julgamento.
  - noticia_TRE: matéria do Tribunal Regional Eleitoral de ORIGEM (tre-XX.jus.br, da UF do caso) sobre este caso, quando houver.
  - noticia_geral: matérias da imprensa (G1, Folha, Estadão, O Globo, UOL, ConJur, JOTA, Migalhas, Poder360, Metrópoles, agências regionais) sobre este caso específico.
- Não invente URLs nem retorne páginas genéricas (home, índice de notícias, busca). Só links claramente do MESMO caso/processo.
- Se uma frente não tiver matéria confiável, devolva lista vazia para ela. É melhor vazio do que um link impertinente.
"""

PROCESS_METADATA_SYSTEM_PROMPT = """
Você é um pesquisador jurídico-eleitoral.

TAREFA:
- Use exclusivamente o Grounding with Google Search.
- Complete prioritariamente o número CNJ integral do processo.
- Só informe a cidade/UF de origem quando ela vier de forma clara junto da mesma evidência relevante para o processo. Não faça busca isolada apenas para preencher origem.
- Se os resultados indicarem que o número consultado aparece apenas como precedente citado, e não como processo efetivamente julgado na sessão, marque is_judged_process=false.
- Quando o item for uma LISTA TRÍPLICE (formação de lista para vaga de membro/juiz de TRE pela classe de advogado/jurista), pesquise e devolva em indicados_lista_triplice os nomes COMPLETOS das pessoas indicadas na lista (em regra três advogados/advogadas), exatamente como publicados em fonte oficial (DJe, sites do TSE/TRE/TJ) ou imprensa jurídica confiável. Esses nomes são informação pública. Inclua apenas os nomes que você localizar com segurança para o processo específico; se não encontrar, devolva lista vazia (melhor vazio do que nome errado).
- Não invente número completo, origem, indicados ou classificação do item.
"""

THEME_REPAIR_SYSTEM_PROMPT = """
Você é um pesquisador jurídico-eleitoral encarregado apenas de propor o tema jurídico de um processo já julgado.

REGRAS:
- Use exclusivamente o texto fornecido no prompt.
- Retorne um tema jurídico curto, indexável e aderente ao caso concreto.
- Não repita número do processo.
- Não use apenas rótulos genéricos como "Processo", "Julgamento" ou só a classe processual.
- Se o texto não permitir inferência segura do tema, retorne o campo vazio.

EXEMPLOS:
- Se o texto falar em "propaganda eleitoral irregular na internet", retorne esse núcleo temático.
- Se o texto tratar de correção de ata/proclamação por erro material no valor da multa, retorne um tema como "Retificação de erro material no valor da multa".
- Se o texto só disser que o julgamento foi suspenso por pedido de vista, sem revelar o núcleo jurídico do caso, retorne vazio.
- Se o texto tratar de retotalização de votos e aumento do número de cadeiras da Câmara Municipal, retorne esse núcleo temático.
- Se o texto tratar de uso do Fundo Partidário para custear consultoria jurídica e contábil, retorne esse núcleo temático.
- Se o texto tratar de publicidade institucional em período vedado, retorne esse núcleo temático.
- Se o texto tratar de uso promocional de programa social em campanha, retorne esse núcleo temático.
- Se o texto tratar de inserções de rádio no segundo turno de 2022, retorne esse núcleo temático.
- Se o texto tratar de comprovação de gastos com panfletagem em prestação de contas, retorne esse núcleo temático.
"""

THEME_PUNCHLINE_REPAIR_SYSTEM_PROMPT = """
Você é um editor jurídico-eleitoral encarregado de revisar, em lote, os campos `tema` e `punchline` de julgamentos do TSE.

Use exclusivamente os dados fornecidos no prompt. Não use fonte externa e não invente fatos ausentes.

OBJETIVO:
- `tema`: cabeçalho de ficha de catalogação média, em frase nominal jurídica, específica, sintética e indexável.
- `punchline`: frase editorial autônoma que contextualiza o caso, sintetiza a controvérsia pública/jurídica e registra a consequência do julgamento.

REGRAS PARA `tema`:
- Deve identificar a questão jurídica submetida a julgamento.
- Use preferencialmente 7 a 18 palavras.
- Não inclua número do processo, timestamp, nomes das partes como eixo principal, relator ou resultado.
- Não use tema amplo demais, só classe processual, "Processo", "Julgamento" ou frase de decisão.

REGRAS PARA `punchline`:
- Use uma frase completa, preferencialmente entre 28 e 55 palavras.
- Deve complementar o `tema`, não copiá-lo nem apenas trocar sinônimos.
- Inclua, quando constar do contexto, o cenário fático, a tese debatida, o problema processual e o resultado prático.
- Evite fórmulas pobres como "recurso provido", "julgamento sobre...", "o relator entendeu..." ou repetição da fundamentação.
- Se o julgamento ficou suspenso por pedido de vista, explique qual debate ficou pendente.

RETORNO:
- Devolva exatamente um item para cada `key` recebido.
- Se o contexto for insuficiente para escrever com segurança, mantenha o melhor texto possível a partir dos campos existentes e marque `source_insufficient=true`.
"""

START_REFINEMENT_SYSTEM_PROMPT = """
Você é um analista jurídico-eleitoral encarregado apenas de identificar o instante exato de início de um julgamento em um vídeo do TSE.

REGRAS:
- Considere como início exato o primeiro momento em que o julgamento do item realmente começa, com chamada do caso, anúncio do processo ou lista tríplice, identificação do relator, leitura do relatório, sustentação oral, voto ou discussão jurisdicional correspondente.
- Se houver uma chamada inicial do julgamento antes da identificação completa do número do processo, use a chamada inicial.
- Expressões como "aprego para julgamento", "chamo a julgamento", "passo ao julgamento", "aprego o feito" ou equivalentes marcam o início do julgamento, mesmo que o número completo do processo seja dito alguns segundos depois.
- Não use o começo de falas cerimoniais, administrativas ou introdutórias se ainda não houver início do julgamento.
- Retorne o segundo absoluto do vídeo.
- Se o trecho não mostrar um julgamento, marque should_ignore=true.
"""

TRANSITION_START_SYSTEM_PROMPT = """
Você é um analista jurídico-eleitoral encarregado apenas de identificar o instante exato em que a sessão deixa os atos administrativos iniciais e entra no primeiro julgamento jurisdicional.

REGRAS:
- Ignore leitura de ata, aberturas formais, cumprimentos e falas cerimoniais.
- Considere como início do julgamento o primeiro instante em que aparece uma chamada jurisdicional concreta.
- Expressões como "aprego para julgamento", "chamo a julgamento", "passo ao julgamento", "aprego o feito" ou equivalentes marcam o início, mesmo que o número completo do processo seja dito depois.
- Retorne o segundo absoluto mais cedo que satisfaça essa regra.
- Se o trecho não contiver o começo de um julgamento, marque should_ignore=true.
"""


def load_runtime_secrets() -> None:
    load_local_secrets(base_dir=SCRIPT_DIR)


def get_gemini_api_key() -> str:
    load_runtime_secrets()
    return get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY", base_dir=SCRIPT_DIR)


def get_notion_api_key() -> str:
    load_runtime_secrets()
    return get_secret("NOTION_API_KEY", "NOTION_TOKEN", base_dir=SCRIPT_DIR)


def get_openai_api_key() -> str:
    load_runtime_secrets()
    return get_secret("OPENAI_API_KEY", base_dir=SCRIPT_DIR)


def chunk_rich_text(value: str, chunk_size: int = 1900) -> list[dict[str, Any]]:
    value = value or ""
    if not value:
        return []
    chunks = [value[idx: idx + chunk_size] for idx in range(0, len(value), chunk_size)]
    return [{"type": "text", "text": {"content": chunk}} for chunk in chunks]


def merge_text_blocks(*blocks: tuple[str, str]) -> str:
    merged: list[str] = []
    for title, content in blocks:
        content = (content or "").strip()
        if not content:
            continue
        if title:
            merged.append(f"{title}\n{content}")
        else:
            merged.append(content)
    return "\n\n".join(merged)


LEGACY_RACIOCINIO_HEADINGS = (
    "Raciocínio Jurídico Aplicado ao Caso Concreto",
    "Pontos Processuais Relevantes",
    "Efeitos e Providências Práticas",
)
LEGACY_FUNDAMENTACAO_HEADINGS = (
    "Fundamentação Normativa e Dispositivos Citados",
)


def strip_legacy_section_headings(value: str, headings: tuple[str, ...]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for heading in headings:
        pattern = re.compile(rf"^\s*{re.escape(heading)}\s*\n+", re.IGNORECASE)
        text = pattern.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_legacy_raciocinio_text(value: str) -> str:
    return strip_legacy_section_headings(value, LEGACY_RACIOCINIO_HEADINGS)


def strip_legacy_fundamentacao_text(value: str) -> str:
    return strip_legacy_section_headings(value, LEGACY_FUNDAMENTACAO_HEADINGS)


def build_raciocinio_column_text(
    raciocinio_juridico: str,
    pontos_processuais_relevantes: str,
    efeitos_e_providencias_praticas: str,
) -> str:
    return merge_text_blocks(
        ("", strip_legacy_raciocinio_text(raciocinio_juridico)),
        ("", strip_legacy_raciocinio_text(pontos_processuais_relevantes)),
        ("", strip_legacy_raciocinio_text(efeitos_e_providencias_praticas)),
    )


def build_fundamentacao_column_text(fundamentacao_normativa: str) -> str:
    return strip_legacy_fundamentacao_text(fundamentacao_normativa)


def _normalize_process_digits(value: str) -> str:
    return re.sub(r"\D+", "", str(value or ""))


def _looks_like_process_number_theme(value: str, row: "PublishPreviewRow") -> bool:
    normalized = normalize_class_text(value)
    if not normalized:
        return False
    normalized_digits = _normalize_process_digits(normalized)
    row_digits = _normalize_process_digits(row.numero_processo)
    if normalized_digits and row_digits and normalized_digits == row_digits:
        return True
    return bool(re.fullmatch(r"processo\s+\d[\d.\- ]*", normalized))


def _clean_inferred_theme(value: str) -> str:
    candidate = normalize_model_text(value).strip(" .;,:-")
    candidate = re.sub(r"^(?:um|uma|o|a|os|as)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.split(r"(?i)\b(?:alegando|sustentando|diante|porque|porquanto|quando|embora|exceto)\b", candidate, maxsplit=1)[0].strip(" .;,:-")
    candidate = re.split(r"\s+[—-]\s+", candidate, maxsplit=1)[0].strip(" .;,:-")
    if "," in candidate and len(candidate) > 90:
        candidate = candidate.split(",", 1)[0].strip(" .;,:-")
    if len(candidate) > 120:
        candidate = candidate[:120].rsplit(" ", 1)[0].strip(" .;,:-")
    if not candidate:
        return ""
    if re.search(r"(?i)\b(?:do|da|dos|das)\s+art(?:\.|igo)?$", candidate):
        return ""
    if re.search(r"(?i)\b(?:do|da|dos|das)\s+lei$", candidate):
        return ""
    return candidate[:1].upper() + candidate[1:]


def _is_meta_legal_sentence(normalized: str) -> bool:
    if re.match(
        r"^(?:o|a)\s+(?:raciocinio juridico|raciocínio jurídico|fundamentacao normativa|fundamentação normativa|analise do conteudo juridico|análise do conteúdo jurídico)\b",
        normalized,
    ):
        return True
    return bool(GENERIC_THEME_REPORTING_SENTENCE_REGEX.match(normalized))


def _has_legal_theme_anchor(normalized: str) -> bool:
    return bool(
        re.search(
            r"\b(?:"
            r"fraude|cota de genero|cota de gênero|inelegibilidade|propaganda|prestacao de contas|prestação de contas|"
            r"abuso de poder|lista triplice|lista tríplice|consulta|fundo partidario|fundo partidário|"
            r"modulacao|modulação|representacao feminina|representação feminina|cassacao|cassação|"
            r"desincompatibilizacao|desincompatibilização|conduta vedada|publicidade institucional|"
            r"improbidade|retotalizacao|retotalização|paridade de genero|paridade de gênero"
            r")\b",
            normalized,
        )
    )


def _looks_like_relational_case_stub(normalized: str) -> bool:
    if re.match(
        r"^(?:o\s+)?(?:caso|processo|recurso)\s+(?:do|da|de)\s+(?:municipio|município|cidade|estado|comarca|tre|tse)\b",
        normalized,
    ):
        return True
    if "relator ministro" in normalized or "relatora ministra" in normalized:
        return not _has_legal_theme_anchor(normalized)
    return False


def _looks_like_meta_or_citation_punchline(normalized: str, raw_text: str = "") -> bool:
    if re.match(r"^(?:o|a)\s+relator(?:a)?\b", normalized):
        return True
    if re.match(r"^(?:o|a)\s+voto\s+d[oa]\s+relator(?:a)?\b", normalized):
        return True
    if re.match(
        r"^(?:negad[oa]|desprovid[oa]|provid[oa]|aprovad[oa]|indeferid[oa]|deferid[oa]|rejeitad[oa])\b",
        normalized,
    ):
        return True
    if re.match(r"^(?:o|minist[eé]rio p[uú]blico eleitoral|mpe)\s+recorreu\b", normalized):
        return True
    if re.match(r"^(?:trata(?:-|\s+)se de|precedentes?)\b", normalized):
        return True
    if re.match(r"^(?:art(?:\.|igo)?|lei|resolu[cç][aã]o|c[oó]digo eleitoral|lindb)\b", normalized):
        return True
    if re.search(r"\bacompanhad[oa]s?\s+pelos?\s+ministros?\b", normalized):
        return True
    if re.search(r"\bacompanharam\s+o\s+voto\b", normalized):
        return True
    text = normalize_model_text(raw_text)
    if text.count(";") >= 1 and not re.search(r"\b(?:discute|examina|reconhece|afasta|define|permite|veda|mant[eé]m|cassa)\b", normalized):
        return True
    return False


def _prefer_more_specific_theme(
    row: "PublishPreviewRow",
    current_candidate: str,
    inferred_candidate: str,
) -> str:
    current = normalize_model_text(current_candidate)
    inferred = normalize_model_text(inferred_candidate)
    if not inferred:
        return current
    if not current or _tema_looks_generic(current, row):
        return inferred
    normalized_current = normalize_class_text(current)
    normalized_inferred = normalize_class_text(inferred)
    if normalized_current == normalized_inferred:
        return current
    if normalized_current in OVERBROAD_THEME_VALUES and len(normalized_inferred) > len(normalized_current) + 8:
        return inferred
    return current


def infer_theme_from_row_text(row: "PublishPreviewRow") -> str:
    sources = [
        " ".join(
            value
            for value in [
                normalize_model_text(row.punchline),
                normalize_model_text(row.analise_do_conteudo_juridico),
                normalize_model_text(row.raciocinio_juridico),
                normalize_model_text(row.fundamentacao_normativa),
            ]
            if value
        ).strip(),
        normalize_model_text(row.punchline),
        normalize_model_text(row.analise_do_conteudo_juridico),
        normalize_model_text(row.raciocinio_juridico),
        normalize_model_text(row.fundamentacao_normativa),
    ]
    pattern_builders: list[tuple[re.Pattern[str], Any]] = [
        (
            re.compile(
                r"(?is)fraude [àa] cota de g[eê]nero[\s\S]{0,420}(?:"
                r"modula[cç][aã]o|modular os efeitos|representa[cç][aã]o feminina|"
                r"efeitos contr[aá]rios [àa] pol[ií]tica afirmativa|redu[cç][aã]o da representa[cç][aã]o feminina"
                r")"
            ),
            lambda m: "Fraude à cota de gênero e modulação dos efeitos da cassação",
        ),
        (
            re.compile(
                r"(?is)(?:a[cç][aã]o de impugna[cç][aã]o de mandato eletivo|aime)[\s\S]{0,240}fraude [àa] cota de g[eê]nero|"
                r"fraude [àa] cota de g[eê]nero[\s\S]{0,240}(?:a[cç][aã]o de impugna[cç][aã]o de mandato eletivo|aime)"
            ),
            lambda m: "Fraude à cota de gênero em ação de impugnação de mandato eletivo",
        ),
        (
            re.compile(r"(?i)erro material(?: de [^.,;]+)? no valor da multa"),
            lambda m: "Retificação de erro material no valor da multa",
        ),
        (
            re.compile(r"(?i)fundo partid[aá]rio.*consultoria jur[ií]dica e cont[aá]bil"),
            lambda m: "Uso do Fundo Partidário para custear consultoria jurídica e contábil",
        ),
        (
            re.compile(
                r"(?is)(?:consulta[\s\S]{0,260}conduta vedada|conduta vedada[\s\S]{0,260}consulta)"
                r"[\s\S]{0,260}(?:fatos e provas|casos concretos|aus[êe]ncia de abstra[cç][aã]o|"
                r"incompat[ií]vel com o rito da consulta eleitoral|nao caber ao tse analisar a configuracao|"
                r"não caber ao tse analisar a configuração)"
            ),
            lambda m: "Cabimento de consulta eleitoral para análise abstrata de conduta vedada",
        ),
        (
            re.compile(r"(?i)integridade do sistema eletr[oô]nico de vota[cç][aã]o"),
            lambda m: "Integridade do sistema eletrônico de votação nas eleições de 2022",
        ),
        (
            re.compile(
                r"(?is)prest[aã]?[cç][aã]o de contas[\s\S]{0,120}exerc[ií]cio de (?P<ano>\d{4})"
                r"[\s\S]{0,220}devolu[cç][aã]o ao er[aá]rio"
            ),
            lambda m: f"Prestação de contas partidárias do exercício de {m.group('ano')} com devolução ao erário",
        ),
        (
            re.compile(
                r"(?is)contas? partid[aá]rias?[\s\S]{0,120}exerc[ií]cio de (?P<ano>\d{4})"
                r"[\s\S]{0,220}devolu[cç][aã]o ao er[aá]rio"
            ),
            lambda m: f"Prestação de contas partidárias do exercício de {m.group('ano')} com devolução ao erário",
        ),
        (
            re.compile(r"(?is)prest[aã]?[cç][aã]o de contas[\s\S]{0,120}exerc[ií]cio de (?P<ano>\d{4})"),
            lambda m: f"Prestação de contas partidárias do exercício de {m.group('ano')}",
        ),
        (
            re.compile(r"(?i)inser[cç][õo]es de r[aá]dio.*segundo turno"),
            lambda m: "Distribuição de inserções de rádio no segundo turno de 2022",
        ),
        (
            re.compile(r"(?i)publicidade institucional em per[ií]odo vedado"),
            lambda m: "Publicidade institucional em período vedado",
        ),
        (
            re.compile(r"(?i)(?:utiliza[cç][aã]o|uso) de programas sociais?.*finalidade eleitoral"),
            lambda m: "Uso promocional de programa social como conduta vedada",
        ),
        (
            re.compile(r"(?i)(?:desaprova[cç][aã]o d(?:as|e suas) contas(?: de campanha)?|prest[aã]?[cç][aã]o de contas).*(?:panfletagem|contratos? individuais? de trabalho)"),
            lambda m: "Comprovação de gastos com panfletagem em prestação de contas",
        ),
        (
            re.compile(r"(?i)retifica[cç][aã]o .*?valor da multa"),
            lambda m: "Retificação de erro material no valor da multa",
        ),
        (
            re.compile(r"(?i)propaganda eleitoral irregular(?: na internet)?"),
            lambda m: m.group(0),
        ),
        (
            re.compile(r"(?i)impulsionamento de propaganda eleitoral negativa(?: na internet)?"),
            lambda m: m.group(0),
        ),
        (
            re.compile(r"(?i)propaganda eleitoral antecipada(?: negativa| positiva)?"),
            lambda m: m.group(0),
        ),
        (
            re.compile(r"(?i)fraude [àa] cota de g[eê]nero"),
            lambda m: m.group(0),
        ),
        (
            re.compile(r"(?i)pedido de vista|adiamento do julgamento|julgamento adiado"),
            lambda m: "Adiamento de julgamento por pedido de vista",
        ),
        (
            re.compile(r"(?i)objeto do conv[eê]nio foi executado integralmente"),
            lambda m: "Execução integral de convênio afasta inelegibilidade por rejeição de contas",
        ),
        (
            re.compile(r"(?i)conv[eê]nio.*agricultura familiar.*inelegibilidade"),
            lambda m: "Inelegibilidade por rejeição de contas em convênio da agricultura familiar",
        ),
        (
            re.compile(r"(?i)paridade de g[eê]nero"),
            lambda m: m.group(0),
        ),
        (
            re.compile(r"(?i)al[ií]nea g\b"),
            lambda m: "Inelegibilidade da alínea g da LC 64/1990",
        ),
        (
            re.compile(r"(?i)(?:contagem do )?prazo de inelegibilidade[^.]{0,160}parcelamento da (?:pena de )?multa|parcelamento da (?:pena de )?multa[^.]{0,160}prazo de inelegibilidade"),
            lambda m: "Prazo de inelegibilidade e parcelamento da pena de multa",
        ),
        (
            re.compile(r"(?i)parcelamento da (?:pena de )?multa.*n[aã]o\s+(?:suspende|posterga|afasta).*inelegibilidade|inelegibilidade.*parcelamento da (?:pena de )?multa"),
            lambda m: "Prazo de inelegibilidade e parcelamento da pena de multa",
        ),
        (
            re.compile(
                r"(?i)(?:"
                r"(?:repasse|uso|desvirtuamento) de (?:verba|recursos)[^.]{0,220}(?:cota feminina|fefc)[^.]{0,220}candidat(?:o|ura)s?(?: do g[eê]nero)? mascul(?:ino|ina|inos|inas)"
                r"|fundo especial de financiamento de campanha[^.]{0,220}cota feminina[^.]{0,220}candidat(?:o|ura)s?(?: do g[eê]nero)? mascul(?:ino|ina|inos|inas)"
                r"|desvirtuamento de valores destinados [àa] cota feminina[^.]{0,220}candidat(?:o|ura)s? mascul(?:ino|ina|inos|inas)"
                r")"
            ),
            lambda m: "Desvio de recursos da cota feminina do FEFC para candidatura masculina",
        ),
        (
            re.compile(r"(?i)inelegibilidade(?: [^.,;]+)?"),
            lambda m: m.group(0),
        ),
        (
            re.compile(r"(?i)medidas cautelares? de busca e apreens[aã]o,? sequestro e bloqueio de bens e valores"),
            lambda m: "Manutenção de medidas cautelares patrimoniais",
        ),
        (
            re.compile(r"(?i)busca e apreens[aã]o,? sequestro e bloqueio de bens e valores"),
            lambda m: "Medidas cautelares patrimoniais",
        ),
        (
            re.compile(r"(?i)por suposta ([^.]+?)(?: nas elei[cç][õo]es| no pleito| em [A-ZÁÀÃÂÉÊÍÓÔÕÚÇa-záàãâéêíóôõúç]+/\w{2}|[.,;])"),
            lambda m: m.group(1),
        ),
        (
            re.compile(r"(?i)discute\s+(?:a|o|as|os)?\s*([^.]+?)(?:, alegando|, sustentando|, diante|[.;])"),
            lambda m: m.group(1),
        ),
        (
            re.compile(r"(?i)obje?tivo\s+foi\s+corrigir\s+([^.]+?)(?:[.;]|$)"),
            lambda m: m.group(1),
        ),
        (
            re.compile(r"(?i)organiza[cç][aã]o de servi[cç]os eleitorais"),
            lambda m: "Alteração de resolução sobre organização de serviços eleitorais",
        ),
        (
            re.compile(r"(?i)retotaliza[cç][aã]o dos votos?.*?(?:c[aâ]mara municipal|n[uú]mero de cadeiras)"),
            lambda m: "Retotalização de votos e número de cadeiras na Câmara Municipal",
        ),
        (
            re.compile(r"(?i)trancar uma a[cç][aã]o penal.*organiza[cç][aã]o criminosa"),
            lambda m: "Trancamento de ação penal por organização criminosa e crimes correlatos",
        ),
        (
            re.compile(r"(?i)omiss[aã]o ou retardamento de informa[cç][õo]es?.*presta[cç][aã]o de contas parciais"),
            lambda m: "Omissão ou atraso de informações em prestações de contas parciais",
        ),
    ]
    for source in sources:
        if not source:
            continue
        for pattern, builder in pattern_builders:
            match = pattern.search(source)
            if not match:
                continue
            candidate = _clean_inferred_theme(builder(match))
            if candidate and not _tema_looks_generic(candidate, row):
                return candidate
    return ""


# 25/08/2026 — `tema` que nomeia autoridade em vez de descrever a materia. Nasce quando o
# video traz so a proclamacao ("pedido de vista do Ministro Sergio Banhos") e o modelo toma
# quem falou pelo assunto julgado. Exige CARGO + NOME PROPRIO: assim "nulidade por
# impedimento de Ministro" e "competencia monocratica do relator" continuam validos, porque
# neles a autoridade integra a TESE e nao substitui o tema.
_TEMA_AUTORIDADE_RE = re.compile(
    r"(?:[Mm]in(?:istr[oa])?\.?|[Dd]esembargador(?:a)?)\s+"
    r"(?:[Rr]elator(?:a)?\s+)?"
    r"[A-ZÁÂÃÉÊÍÓÔÕÚÇ][\wÀ-ÿ]+\s+"
    r"[A-ZÁÂÃÉÊÍÓÔÕÚÇ][\wÀ-ÿ]+"
)
# rotulos que dizem o RITO, nao a materia. O prompt ja os proibia em palavras ("nunca use
# apenas ... a classe processual"), mas instrucao nao e garantia: "Embargos de declaracao"
# apareceu como tema em 6 paginas. Aqui o veto e deterministico.
_TEMA_ROTULOS_PROCESSUAIS = {
    "embargos", "embargos de declaracao", "segundos embargos de declaracao",
    "embargos de declaracao em recurso especial", "agravo", "agravo regimental",
    "agravo interno", "agravo de instrumento", "agravo em recurso especial",
    "recurso", "recurso especial", "recurso especial eleitoral", "recurso ordinario",
    "recurso eleitoral", "mandado de seguranca", "acao cautelar", "habeas corpus",
    "peticao", "consulta", "representacao", "reclamacao", "processo administrativo",
    "acao rescisoria", "registro de candidatura", "requerimento de registro de candidatura",
    "direito eleitoral", "materia eleitoral", "questao de ordem",
}


def _tema_nomeia_autoridade(value: str) -> bool:
    """True quando o tema aponta uma pessoa do tribunal como se fosse o assunto."""
    return bool(_TEMA_AUTORIDADE_RE.search(value or ""))


def _tema_looks_generic(value: str, row: "PublishPreviewRow") -> bool:
    normalized = normalize_class_text(value)
    if not normalized:
        return True
    if _looks_like_process_number_theme(value, row):
        return True
    if _is_meta_legal_sentence(normalized):
        return True
    if _looks_like_relational_case_stub(normalized):
        return True
    if _tema_nomeia_autoridade(value):
        return True
    if normalized in _TEMA_ROTULOS_PROCESSUAIS:
        return True
    if re.match(r"^(?:o|a)\s+(?:processo|caso|feito)\s+(?:trata|discute|versa)\s+(?:de|sobre)\b", normalized):
        return True
    if GENERIC_THEME_CLASS_RESULT_REGEX.fullmatch(normalized):
        return True
    if GENERIC_THEME_DECISION_SENTENCE_REGEX.match(normalized):
        return True
    if GENERIC_THEME_REPORTING_SENTENCE_REGEX.match(normalized):
        return True
    generic_candidates = {
        normalize_class_text(row.classe_processo),
        normalize_class_text(row.resultado),
        normalize_class_text(row.votacao),
        normalize_class_text(" ".join(part for part in [row.classe_processo, row.resultado] if part)),
        normalize_class_text(" ".join(part for part in [row.classe_processo, row.votacao] if part)),
        normalize_class_text(" ".join(part for part in [row.classe_processo, row.numero_processo] if part)),
    }
    generic_candidates.discard("")
    if normalized in generic_candidates:
        return True
    return normalized in {
        "processo",
        "julgamento",
        "lista triplice",
        "lista tríplice",
        "provido",
        "desprovido",
        "parcialmente provido",
        "deferido",
        "indeferido",
        "nao conhecido",
        "não conhecido",
        "rejeitado",
        "rejeitados",
        "acolhido",
        "acolhidos",
        "prejudicado",
        "prejudicados",
        "unanime",
        "unânime",
        "por maioria",
        "inelegibilidade",
        "prestacao de contas",
        "prestação de contas",
        "prestacao de contas partidarias",
        "prestação de contas partidárias",
        "prestacao de contas de campanha",
        "prestação de contas de campanha",
        "regularidade das contas",
    }


def tema_looks_generic(value: str, row: "PublishPreviewRow") -> bool:
    return _tema_looks_generic(value, row)


def build_fallback_tema(row: "PublishPreviewRow") -> str:
    inferred = infer_theme_from_row_text(row)
    candidates = [
        _prefer_more_specific_theme(
            row,
            (row.tema or "").strip() if not _tema_looks_generic((row.tema or "").strip(), row) else "",
            inferred,
        ),
        _prefer_more_specific_theme(
            row,
            (row.punchline or "").strip().rstrip(".")
            if not _tema_looks_generic((row.punchline or "").strip(), row)
            else "",
            inferred,
        ),
        inferred,
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def build_theme_repair_context(row: "PublishPreviewRow", artifact_text: str = "") -> str:
    blocks: list[str] = []
    for label, value in [
        ("Classe", row.classe_processo),
        ("Número do processo", row.numero_processo),
        ("Resultado", row.resultado),
        ("Votação", row.votacao),
        ("Relator", row.relator),
        ("Punchline", row.punchline),
        ("Análise factual", row.analise_do_conteudo_juridico),
        ("Fundamentação", row.fundamentacao_normativa),
        ("Raciocínio", row.raciocinio_juridico),
    ]:
        value = normalize_model_text(value)
        if value:
            blocks.append(f"{label}: {value}")
    artifact_excerpt = normalize_model_text(artifact_text)
    if artifact_excerpt:
        blocks.append(f"Trechos de artefato:\n{artifact_excerpt[:4000]}")
    return "\n".join(blocks).strip()


def repair_theme_from_text_context(
    *,
    api_key: str,
    model: str,
    row: "PublishPreviewRow",
    context_text: str,
    artifact_store: Optional["RunArtifacts"] = None,
    logger: Optional[logging.Logger] = None,
    artifact_name: str = "08_theme_repair.txt",
) -> ThemeRepairResult:
    if not context_text.strip():
        return ThemeRepairResult()
    last_error: Optional[Exception] = None
    logger = logger or logging.getLogger(__name__)
    for attempt in range(1, GEMINI_CALL_RETRIES + 1):
        try:
            parsed, response_text, _ = call_gemini_generate_content_rest(
                api_key=api_key,
                model_name=model or DEFAULT_GEMINI_MODEL,
                contents=[{"parts": [_build_gemini_rest_part(text=context_text)]}],
                system_instruction=THEME_REPAIR_SYSTEM_PROMPT,
                response_model=ThemeRepairResult,
                temperature=0.1,
                timeout_seconds=DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS,
            )
            if artifact_store is not None:
                artifact_store.write_text(artifact_name, response_text)
            parsed.tema = build_fallback_tema(row.model_copy(update={"tema": parsed.tema}))
            return parsed
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Falha no reparo textual de tema (tentativa %s/%s): %s",
                attempt,
                GEMINI_CALL_RETRIES,
                exc,
            )
            if should_disable_model(exc):
                break
            if attempt < GEMINI_CALL_RETRIES:
                retry_delay = extract_retry_delay_seconds(exc)
                time.sleep(max(GEMINI_RETRY_BASE_DELAY ** attempt, retry_delay))
    raise RuntimeError(f"Falha definitiva no reparo textual de tema: {last_error}") from last_error


def coerce_seconds(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except Exception:
        return 0


def normalize_model_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "null", "none", "n/a", "na"}:
        return ""
    return text


def normalize_transcript_text(value: Any) -> str:
    text = normalize_model_text(unescape(value))
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _build_row_inference_text(row: "PublishPreviewRow") -> str:
    return "\n".join(
        value
        for value in [
            normalize_model_text(getattr(row, "title_hint", "")),
            normalize_model_text(row.tema),
            normalize_model_text(row.punchline),
            normalize_model_text(row.resultado),
            normalize_model_text(row.analise_do_conteudo_juridico),
            normalize_model_text(row.raciocinio_juridico),
            normalize_model_text(row.fundamentacao_normativa),
            normalize_model_text(row.precedentes_citados),
        ]
        if value
    ).strip()


def infer_full_numero_processo_from_row_text(row: "PublishPreviewRow") -> str:
    text = _build_row_inference_text(row)
    if not text:
        return ""
    expected = canonicalize_numero_processo(row.numero_processo)
    pattern = re.compile(r"\b\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b")
    for match in pattern.finditer(text):
        candidate = match.group(0)
        if not expected or canonicalize_numero_processo(candidate) == expected:
            return candidate
    fallback = extract_full_cnj(text)
    if fallback and (not expected or canonicalize_numero_processo(fallback) == expected):
        return fallback
    return ""


def _merge_special_numero_processo(current: str, candidate: str) -> str:
    current_display = normalize_numero_processo_display(current)
    candidate_display = normalize_numero_processo_display(candidate)
    if not candidate_display:
        return current_display
    if not current_display:
        return candidate_display
    if canonicalize_numero_processo(current_display) == canonicalize_numero_processo(candidate_display):
        return candidate_display
    current_digits = re.sub(r"\D", "", current_display)
    candidate_digits = re.sub(r"\D", "", candidate_display)
    if current_digits and current_digits == candidate_digits:
        current_has_label = bool(re.search(r"\b(?:ADI|ADO)\b", current_display, flags=re.IGNORECASE))
        candidate_has_label = bool(re.search(r"\b(?:ADI|ADO)\b", candidate_display, flags=re.IGNORECASE))
        if candidate_has_label and not current_has_label:
            return candidate_display
        if len(candidate_display) > len(current_display):
            return candidate_display
    return current_display


def infer_special_numero_processo_from_row_text(row: "PublishPreviewRow") -> str:
    text = _build_row_inference_text(row)
    if not text:
        return ""
    current_display = normalize_numero_processo_display(row.numero_processo)
    current_digits = re.sub(r"\D", "", current_display)
    special_matches = [
        f"{match.group('label').upper()} {match.group('number').lstrip('0') or '0'}"
        for match in re.finditer(r"(?i)\b(?P<label>ADI|ADO)\s*(?P<number>\d{1,5})\b", text)
    ]
    for candidate in special_matches:
        candidate_digits = re.sub(r"\D", "", candidate)
        if not current_display or current_digits == candidate_digits or current_display.upper() == candidate.upper():
            return candidate
    for match in re.finditer(r"(?i)\b(?:re?sp(?:e)?|arespe|aresp|rhc|rms|ms|ro)\s*(\d{6,9})\b", text):
        candidate = format_short_process_number_from_digits(match.group(1))
        if not candidate:
            continue
        candidate_digits = re.sub(r"\D", "", candidate)
        if not current_display or current_digits == candidate_digits:
            return candidate
    return ""


def _clean_inferred_punchline(value: str, row: "PublishPreviewRow") -> str:
    candidate = normalize_model_text(value).strip()
    if not candidate:
        return ""
    candidate = re.sub(r"\s+", " ", candidate)
    sentences = [
        sentence.strip(" .;,:-")
        for sentence in re.split(r"(?<=[.!?])\s+", candidate)
        if sentence.strip(" .;,:-")
    ]
    if sentences:
        candidate = sentences[0]
        for sentence in sentences:
            normalized_sentence = normalize_class_text(sentence)
            if GENERIC_THEME_DECISION_SENTENCE_REGEX.match(normalized_sentence):
                continue
            if _is_meta_legal_sentence(normalized_sentence):
                continue
            if _looks_like_relational_case_stub(normalized_sentence):
                continue
            if _looks_like_meta_or_citation_punchline(normalized_sentence, sentence):
                continue
            if re.match(
                r"^(?:o|a)\s+(?:processo|caso|acao|ação|recurso)\s+(?:trata|discute|versa)\s+(?:de|sobre)\b",
                normalized_sentence,
            ):
                continue
            if len(sentence) >= 20:
                candidate = sentence
                break
    else:
        candidate = candidate.strip(" .;,:-")
    candidate = re.sub(
        r"(?i)^o processo\s+\d[\d.\-]*\s+(?:trata|discute|versa)\s+(?:de|sobre)\s+",
        "",
        candidate,
    ).strip(" .;,:-")
    if not candidate:
        return ""
    normalized = normalize_class_text(candidate)
    if not normalized or GENERIC_THEME_DECISION_SENTENCE_REGEX.match(normalized):
        return ""
    if _is_meta_legal_sentence(normalized):
        return ""
    if _looks_like_relational_case_stub(normalized):
        return ""
    if _looks_like_meta_or_citation_punchline(normalized, candidate):
        return ""
    if re.match(
        r"^(?:o processo|o caso|a acao|a ação|trata(?:-|\s+)se de)\b",
        normalized,
    ) and re.search(
        r"\b(?:consulta|acao de investigacao judicial eleitoral|representacao|prestacao de contas|lista triplice|agravo|recurso)\b",
        normalized,
    ):
        return ""
    if _looks_like_process_number_theme(candidate, row):
        return ""
    generic_candidates = {
        normalize_class_text(row.tema),
        normalize_class_text(row.resultado),
        normalize_class_text(row.votacao),
        normalize_class_text(" ".join(part for part in [row.classe_processo, row.resultado] if part)),
    }
    generic_candidates.discard("")
    if normalized in generic_candidates:
        return ""
    if len(candidate) < 12:
        return ""
    if TRUNCATED_PUNCHLINE_ENDING_REGEX.search(candidate) and not re.search(r"[.!?]$", normalize_model_text(value)):
        return ""
    if len(candidate) > 180:
        return ""
    if TRUNCATED_PUNCHLINE_ENDING_REGEX.search(candidate):
        return ""
    if not candidate:
        return ""
    candidate = candidate[:1].upper() + candidate[1:]
    return candidate if candidate.endswith((".", "!", "?")) else f"{candidate}."


def infer_punchline_from_row_text(row: "PublishPreviewRow") -> str:
    combined_text = " ".join(
        value
        for value in [
            normalize_model_text(row.analise_do_conteudo_juridico),
            normalize_model_text(row.raciocinio_juridico),
            normalize_model_text(row.fundamentacao_normativa),
            normalize_model_text(row.precedentes_citados),
        ]
        if value
    ).strip()
    normalized_combined = normalize_class_text(combined_text)
    punchline_patterns: list[tuple[re.Pattern[str], str]] = [
        (
            re.compile(
                r"(?is)fraude [àa] cota de g[eê]nero[\s\S]{0,420}(?:"
                r"modula[cç][aã]o|modular os efeitos|representa[cç][aã]o feminina|"
                r"efeitos contr[aá]rios [àa] pol[ií]tica afirmativa|redu[cç][aã]o da representa[cç][aã]o feminina"
                r")"
            ),
            "",
        ),
        (
            re.compile(r"(?i)fundo partid[aá]rio.*consultoria jur[ií]dica e cont[aá]bil.*defesa de filiados"),
            "Consulta sobre uso do Fundo Partidário para custear consultoria jurídica e contábil em defesa de filiados.",
        ),
        (
            re.compile(
                r"(?i)(?:consulta(?: formulada)?|possibilidade de utiliza[cç][aã]o).*"
                r"fundo partid[aá]rio.*"
                r"(?:despesas?|pagamento).*"
                r"consultoria jur[ií]dica e cont[aá]bil"
            ),
            "Consulta sobre uso do Fundo Partidário para custear consultoria jurídica e contábil.",
        ),
        (
            re.compile(r"(?i)pedido de vista|adiamento do julgamento|julgamento adiado"),
            "Julgamento adiado por pedido de vista.",
        ),
    ]
    for pattern, template in punchline_patterns:
        if pattern.search(combined_text):
            if not template:
                location = normalize_model_text(row.origem)
                if row.resultado == "Suspenso por vista":
                    base = "Julgamento sobre fraude à cota de gênero"
                    if location:
                        base += f" em {location}"
                    candidate = (
                        f"{base} foi suspenso por vista após debate sobre modulação dos efeitos da cassação "
                        "e preservação da representação feminina."
                    )
                    candidate = _clean_inferred_punchline(candidate, row)
                    if candidate:
                        return candidate
                candidate = _clean_inferred_punchline(
                    "Fraude à cota de gênero com debate sobre modulação dos efeitos da cassação e preservação da representação feminina.",
                    row,
                )
                if candidate:
                    return candidate
                continue
            candidate = _clean_inferred_punchline(template, row)
            if candidate:
                return candidate
    sources = [
        row.analise_do_conteudo_juridico,
        row.raciocinio_juridico,
        row.fundamentacao_normativa,
        row.precedentes_citados,
    ]
    for source in sources:
        candidate = _clean_inferred_punchline(source, row)
        if candidate:
            return candidate
    fallback_theme = build_fallback_tema(row)
    if fallback_theme and not _tema_looks_generic(fallback_theme, row):
        normalized_theme = normalize_class_text(fallback_theme)
        theme_subject = fallback_theme[:1].lower() + fallback_theme[1:].rstrip(".")
        location = normalize_model_text(row.origem)
        location_suffix = f" em {location}" if location and location not in {"TSE"} else ""
        if "pedido de vista" in normalized_theme:
            return "Julgamento adiado por pedido de vista."
        if row.resultado:
            if row.resultado == "Aprovada" and row.classe_processo == "CTA":
                return f"Consulta sobre {theme_subject}."
            if row.resultado == "Suspenso por vista":
                return f"Julgamento sobre {theme_subject}{location_suffix} foi suspenso por pedido de vista."
        if normalized_combined:
            return f"Julgamento sobre {theme_subject}{location_suffix}."
        return f"Julgamento sobre {theme_subject}{location_suffix}."
    return ""


def punchline_looks_generic(value: str, row: "PublishPreviewRow") -> bool:
    text = normalize_model_text(value)
    normalized = normalize_class_text(text)
    if not normalized:
        return True
    if normalized in {
        "julgamento adiado por pedido de vista",
        "julgamento suspenso por pedido de vista",
    }:
        return False
    if _is_meta_legal_sentence(normalized):
        return True
    if _looks_like_relational_case_stub(normalized):
        return True
    if _looks_like_meta_or_citation_punchline(normalized, text):
        return True
    if re.match(
        r"^(?:o|a)\s+(?:processo|caso|recurso|acao|ação)\s+(?:trata|discute|versa)\s+(?:de|sobre)\b",
        normalized,
    ):
        return True
    if re.match(
        r"^trata(?:-|\s+)se de\b",
        normalized,
    ):
        return True
    if GENERIC_THEME_DECISION_SENTENCE_REGEX.match(normalized):
        return True
    if _looks_like_process_number_theme(value, row):
        return True
    if len(text.strip()) < 18:
        return True
    if TRUNCATED_PUNCHLINE_ENDING_REGEX.search(text.strip()):
        return True
    generic_candidates = {
        normalize_class_text(row.tema),
        normalize_class_text(row.resultado),
        normalize_class_text(row.votacao),
        normalize_class_text(" ".join(part for part in [row.resultado, row.votacao] if part)),
    }
    generic_candidates.discard("")
    return normalized in generic_candidates


THEME_PUNCHLINE_STOPWORDS = {
    "a",
    "as",
    "ao",
    "aos",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "para",
    "por",
    "que",
    "se",
    "sem",
    "sob",
    "sobre",
}


def _compact_theme_punchline_context(value: Any, limit: int = 1200) -> str:
    text = normalize_model_text(value)
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].strip()
    return f"{clipped}..."


def _strip_process_references_from_text(value: str) -> str:
    text = normalize_model_text(value)
    if not text:
        return ""
    text = re.sub(r"(?i)\bprocesso\s+n?[ºo.]?\s*(?=\d)", "", text)
    text = re.sub(r"\b\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b", "", text)
    text = re.sub(r"\b\d{3,7}-\d{2}\b", "", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,.;:-")


def clean_theme_punchline_theme(value: str, row: "PublishPreviewRow") -> str:
    candidate = _strip_process_references_from_text(value).strip(" \"'`“”‘’.;,:-")
    if not candidate:
        return ""
    candidate = _clean_inferred_theme(candidate)
    if not candidate:
        return ""
    if _tema_looks_generic(candidate, row):
        return ""
    return candidate


def clean_theme_punchline_punchline(value: str, row: "PublishPreviewRow") -> str:
    candidate = normalize_model_text(value).strip(" \"'`“”‘’")
    if not candidate:
        return ""
    candidate = re.sub(r"^\s*[-*]\s*", "", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    candidate = _strip_process_references_from_text(candidate).strip(" \"'`“”‘’")
    if not candidate:
        return ""
    if not re.search(r"[.!?]$", candidate):
        candidate = f"{candidate}."
    if punchline_looks_generic(candidate, row):
        return ""
    return candidate[:1].upper() + candidate[1:]


def _theme_punchline_tokens(value: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", normalize_class_text(value)))
    return {token for token in tokens if len(token) > 2 and token not in THEME_PUNCHLINE_STOPWORDS}


def theme_punchline_pair_too_similar(theme: str, punchline: str) -> bool:
    theme_norm = normalize_class_text(theme)
    punchline_norm = normalize_class_text(punchline)
    if not theme_norm or not punchline_norm:
        return False
    if theme_norm == punchline_norm:
        return True
    if punchline_norm.startswith(theme_norm) and len(punchline_norm) <= len(theme_norm) + 45:
        return True
    theme_tokens = _theme_punchline_tokens(theme)
    punchline_tokens = _theme_punchline_tokens(punchline)
    if len(theme_tokens) < 3 or not punchline_tokens:
        return False
    overlap = len(theme_tokens & punchline_tokens) / len(theme_tokens)
    return overlap >= 0.8 and len(punchline_tokens) <= len(theme_tokens) + 5


def theme_punchline_pair_needs_rewrite(row: "PublishPreviewRow") -> bool:
    theme = normalize_model_text(row.tema)
    punchline = normalize_model_text(row.punchline)
    if _tema_looks_generic(theme, row):
        return True
    if punchline_looks_generic(punchline, row):
        return True
    if len(theme) < 24 or len(theme) > 180:
        return True
    if len(punchline) < 90:
        return True
    if re.match(r"(?i)^julgamento\s+sobre\b", punchline):
        return True
    if theme_punchline_pair_too_similar(theme, punchline):
        return True
    return False


def _first_contextual_sentence_for_punchline(row: "PublishPreviewRow") -> str:
    for source in [
        row.analise_do_conteudo_juridico,
        row.raciocinio_juridico,
        row.fundamentacao_normativa,
        row.precedentes_citados,
    ]:
        text = normalize_model_text(source)
        if not text:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            candidate = sentence.strip(" .;,:-")
            if len(candidate) < 55 or len(candidate) > 220:
                continue
            normalized = normalize_class_text(candidate)
            if _is_meta_legal_sentence(normalized):
                continue
            if _looks_like_meta_or_citation_punchline(normalized, candidate):
                continue
            if _looks_like_relational_case_stub(normalized):
                continue
            return f"{candidate}."
    return ""


def build_editorial_punchline_fallback(row: "PublishPreviewRow", theme: str = "") -> str:
    theme = clean_theme_punchline_theme(theme, row) or build_fallback_tema(row)
    existing = clean_theme_punchline_punchline(row.punchline, row)
    if existing and not theme_punchline_pair_too_similar(theme, existing) and len(existing) >= 90:
        return existing
    inferred = infer_punchline_from_row_text(row)
    inferred = clean_theme_punchline_punchline(inferred, row)
    if inferred and not theme_punchline_pair_too_similar(theme, inferred) and len(inferred) >= 90:
        return inferred
    contextual = _first_contextual_sentence_for_punchline(row)
    if contextual and not theme_punchline_pair_too_similar(theme, contextual):
        result_suffix = ""
        result = normalize_model_text(row.resultado)
        if result:
            result_suffix = f" O desfecho registrado foi {result[:1].lower() + result[1:]}."
        return clean_theme_punchline_punchline(f"{contextual.rstrip('.')}.{result_suffix}", row)
    subject = (theme or "a controvérsia eleitoral").strip().rstrip(".")
    result = normalize_model_text(row.resultado)
    vote = normalize_model_text(row.votacao)
    outcome = ""
    if result:
        outcome = f", com resultado registrado como {result[:1].lower() + result[1:]}"
        if vote:
            outcome += f" e votação {vote[:1].lower() + vote[1:]}"
    return clean_theme_punchline_punchline(
        f"A controvérsia levou o TSE a examinar {subject[:1].lower() + subject[1:]} a partir do caso concreto{outcome}.",
        row,
    )


def build_theme_punchline_repair_payload(row: "PublishPreviewRow", key: str) -> dict[str, Any]:
    return {
        "key": key,
        "classe_processo": normalize_model_text(row.classe_processo),
        "numero_processo": normalize_model_text(row.numero_processo),
        "data_sessao": normalize_model_text(row.data_sessao),
        "origem": normalize_model_text(row.origem),
        "resultado": normalize_model_text(row.resultado),
        "votacao": normalize_model_text(row.votacao),
        "relator": normalize_model_text(row.relator),
        "partes": row.partes[:8],
        "tema_atual": normalize_model_text(row.tema),
        "punchline_atual": normalize_model_text(row.punchline),
        "analise_factual": _compact_theme_punchline_context(row.analise_do_conteudo_juridico, 1400),
        "raciocinio_juridico": _compact_theme_punchline_context(row.raciocinio_juridico, 1100),
        "fundamentacao_normativa": _compact_theme_punchline_context(row.fundamentacao_normativa, 900),
        "precedentes_citados": _compact_theme_punchline_context(row.precedentes_citados, 600),
    }


def _trim_person_capture(value: str) -> str:
    cleaned = normalize_model_text(value)
    cleaned = re.split(
        r"(?i)\s+(?:votou|vota|apresentou|apresenta|prop[oô]s|prop[oõ]e|informou|informa|"
        r"anunciou|anuncia|entendeu|destacou|esclareceu|submeteu|submetendo|concluiu|"
        r"rejeitou|acolheu|negou|deu|proferiu)\b",
        cleaned,
        maxsplit=1,
    )[0]
    cleaned = re.split(r"(?i)\s+e\s+(?:foi|estava|teve|havia)\b", cleaned, maxsplit=1)[0]
    return cleaned.strip(" ,.;:-")


def infer_relator_from_row_text(row: "PublishPreviewRow") -> str:
    text = _build_row_inference_text(row)
    if not text:
        return ""
    patterns = [
        r"(?i)\bo\s+relator,\s*(?:ministro|ministra|min\.)\s+([^,.;:\n]+)",
        r"(?i)\b(?:ministro|ministra|min\.)\s+relator(?:a)?[,:\s]+([^,.;:\n]+)",
        r"(?i)\bsob relatoria d[oa]\s+(?:ministro|ministra|min\.)\s+([^,.;:\n]+)",
        r"(?i)\brelatoria d[oa]\s+(?:ministro|ministra|min\.)\s+([^,.;:\n]+)",
        r"(?i)\bo\s+ministro relator,\s*([^,.;:\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = normalize_ministro_name(_trim_person_capture(match.group(1)))
        if candidate:
            return candidate
    return ""


def infer_pedido_vista_from_row_text(row: "PublishPreviewRow") -> str:
    text = _build_row_inference_text(row)
    if not text:
        return ""
    patterns = [
        r"(?i)\bvoto-?vista d[oa]\s+(?:ministro|ministra|min\.)\s+([^,.;:\n]+)",
        r"(?i)\bpedido de vista d[oa]\s+(?:ministro|ministra|min\.)\s+([^,.;:\n]+)",
        r"(?i)\bpedido de vista pel[oa]\s+(?:ministro|ministra|min\.)\s+([^,.;:\n]+)",
        r"(?i)\b(?:ministro|ministra|min\.)\s+([^,.;:\n]+?)\s+apresentou\s+seu?\s+voto-?vista\b",
        r"(?i)\b(?:ministro|ministra|min\.)\s+([^,.;:\n]+?)\s+pediu\s+vista\b",
        r"(?i)\b(?:ministro|ministra|min\.)\s+([^,.;:\n]+?)\s+proferiu\s+voto-?vista\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = normalize_pedido_vista_value(_trim_person_capture(match.group(1)))
        if candidate:
            return candidate
    return ""


def extract_ministro_roles_from_composition_entries(values: list[str]) -> tuple[str, str]:
    relator = ""
    pedido_vista = ""
    for raw_value in values or []:
        raw_text = str(raw_value or "").strip()
        if not raw_text:
            continue
        normalized = normalize_class_text(raw_text)
        if not relator and re.search(r"(?i)\brelator(?:a)?\b", raw_text):
            candidate = normalize_ministro_name(raw_text)
            if candidate and is_plausible_ministro_name(candidate):
                relator = candidate
        if not pedido_vista and any(marker in normalized for marker in ("voto vista", "votovista", "pedido de vista", "vista")):
            candidate = normalize_pedido_vista_value(raw_text)
            if candidate and is_plausible_ministro_name(candidate):
                pedido_vista = candidate
        if relator and pedido_vista:
            break
    return relator, pedido_vista


def _canonicalize_person_select_value(
    value: str,
    *,
    notion_name: str,
    notion_schema: "NotionDataSourceSchema" | None = None,
) -> str:
    normalizer = normalize_ministro_name if notion_name == "relator" else normalize_pedido_vista_value
    candidate = normalizer(value)
    if not candidate or not is_plausible_ministro_name(candidate):
        return ""
    if notion_schema is None:
        return candidate
    prop = notion_schema.properties.get(notion_name)
    if not prop:
        return candidate
    candidate_key = normalize_class_text(re.sub(r"^Min\.\s*", "", candidate))
    for option in prop.options:
        option_candidate = normalizer(option)
        option_key = normalize_class_text(re.sub(r"^Min\.\s*", "", option_candidate))
        if option_key and option_key == candidate_key:
            return option_candidate
    return candidate


def _controlled_select_value_can_expand(
    notion_name: str,
    value: str,
    row: "PublishPreviewRow",
) -> bool:
    if not value:
        return False
    if notion_name == "tipo_registro":
        return bool(TIPO_REGISTRO_DYNAMIC_RE.match(value))
    if notion_name == "classe_processo":
        return value == normalize_classe_processo(value)
    if notion_name == "resultado":
        return value == normalize_resultado_final(value, row.classe_processo)
    if notion_name == "votacao":
        return value == normalize_votacao(value)
    if notion_name == "eleicao":
        return value == normalize_eleicao_value(value)
    if notion_name == "tribunal":
        return value == normalize_tre(value, extract_uf_from_text(row.origem))
    return False


def infer_votacao_from_row_text(row: "PublishPreviewRow") -> str:
    text = normalize_class_text(_build_row_inference_text(row))
    if not text:
        return ""
    if any(
        marker in text
        for marker in [
            "pedido de vista",
            "julgamento foi suspenso",
            "julgamento suspenso",
            "suspenso devido",
            "impossibilitando a continuidade",
        ]
    ):
        return "Suspenso"
    if any(marker in text for marker in ["por unanimidade", "unanimemente", "decisao unanime", "decisão unânime"]):
        return "Unânime"
    if "por maioria" in text or re.search(r"\b\d+\s*a\s*\d+\b", text):
        return "Por maioria"
    return ""


def infer_resultado_from_row_text(row: "PublishPreviewRow") -> str:
    classe = normalize_classe_processo(row.classe_processo) if row.classe_processo else ""
    candidate_texts = [
        row.punchline,
        row.raciocinio_juridico,
        row.analise_do_conteudo_juridico,
    ]
    for text in candidate_texts:
        normalized = normalize_class_text(text)
        if not normalized:
            continue
        if re.search(r"\bjulgar?\s+improcedente\b|\bimprocedente\b", normalized):
            return "Improcedente"
        if re.search(r"proced[eê]n(?:cia|te)\s+parcial|procedente\s+em\s+parte", normalized):
            return "Procedente em parte"
        if re.search(r"\bjulgar?\s+procedente\b|\bprocedente\b", normalized):
            return "Procedente"
        if re.search(r"\bjulgar?\s+parcialmente\s+procedente\b|\bparcialmente\s+procedente\b", normalized):
            return "Procedente em parte"
        if "consulta respondida" in normalized or "respondeu a consulta" in normalized:
            return "Aprovada"
        if "lista triplice acolhida" in normalized or "lista tríplice acolhida" in normalized:
            return "Acolhidos"
        inferred = normalize_resultado_final(text, classe)
        if inferred and inferred != text.strip() and len(inferred) <= 40:
            return inferred
    return ""


def _classe_processo_specificity(value: str, row: "PublishPreviewRow") -> int:
    classe = normalize_classe_processo(value)
    if not classe:
        return 0
    if classe == "PA":
        return 1
    if classe in {"ADI", "ADO"}:
        return 0
    if classe in {"CTA", "Lista Tríplice"}:
        return 3
    if classe.startswith("AgRg") or classe.startswith("ED-"):
        return 5
    return 4


def should_replace_classe_processo(
    current: str,
    candidate: str,
    row: "PublishPreviewRow",
) -> bool:
    current_norm = normalize_classe_processo(current)
    candidate_norm = normalize_classe_processo(candidate)
    if not candidate_norm or candidate_norm == current_norm:
        return False
    if candidate_norm in {"ADI", "ADO"}:
        return False
    if not current_norm:
        return True
    if current_norm in {"ADI", "ADO"}:
        return True
    current_score = _classe_processo_specificity(current_norm, row)
    candidate_score = _classe_processo_specificity(candidate_norm, row)
    if candidate_score > current_score:
        return True
    if current_norm == "PA" and candidate_norm != "PA":
        return True
    return False


def infer_classe_from_row_text(row: "PublishPreviewRow") -> str:
    text = _build_row_inference_text(row)
    normalized = normalize_class_text(text)
    if not normalized:
        return ""
    arespe_markers = [
        "agravo em recurso especial eleitoral",
        "agravos em recurso especial eleitoral",
        "agravo em recurso especial",
        "agravos em recurso especial",
    ]
    if "agravo regimental" in normalized and (any(marker in normalized for marker in arespe_markers) or "arespe" in normalized):
        return "AgRg-AREspe"
    if "agravo regimental" in normalized and ("recurso especial eleitoral" in normalized or "respe" in normalized):
        return "AgRg-REspe"
    if any(marker in normalized for marker in arespe_markers) or re.search(r"\barespe\b", normalized):
        return "AREspe"
    if "acao de investigacao judicial eleitoral" in normalized or "ação de investigação judicial eleitoral" in normalized:
        return "AIJE"
    if "representacao por propaganda eleitoral irregular" in normalized or "representação por propaganda eleitoral irregular" in normalized:
        return "Rp"
    if "prestacao de contas" in normalized or "prestação de contas" in normalized:
        return "PC"
    if "lista triplice" in normalized or "lista tríplice" in normalized:
        return "Lista Tríplice"
    if "consulta formulada" in normalized or normalized.startswith("consulta ") or "trata de uma consulta" in normalized:
        return "CTA"
    if "recurso ordinario eleitoral" in normalized:
        return "RO"
    if "forca federal" in normalized or "força federal" in normalized:
        return "PA"
    if "agravo regimental" in normalized and "habeas corpus" in normalized:
        return "AgRg-HC"
    if "recurso especial eleitoral" in normalized or "recursos especiais" in normalized or "recurso especial" in normalized:
        return "REspe"
    if any(
        marker in normalized
        for marker in [
            "proposta de resolucao",
            "proposta de resolução",
            "afastamento de magistrado",
            "forca federal",
            "força federal",
            "questao administrativa interna",
            "questão administrativa interna",
            "organizacao de servicos eleitorais",
            "organização de serviços eleitorais",
        ]
    ):
        return "PA"
    candidate = normalize_classe_processo(text)
    if candidate in {"ADI", "ADO"}:
        return ""
    if candidate and candidate != text.strip():
        return candidate
    return ""


# Tokens que denunciam que o trecho capturado "Nome/UF" não é um município, e sim
# uma instituição/órgão (tribunal, autarquia, partido, etc.). Filtrados por TOKEN
# (não substring) para não rejeitar municípios como "Três Rios" (token "tres" != "tre").
ORIGEM_NON_MUNICIPALITY_TOKENS = {
    "tribunal",
    "tre",
    "tse",
    "stf",
    "stj",
    "oab",
    "incra",
    "ibama",
    "instituto",
    "resolucao",
    "camara",
    "diretorio",
    "conselho",
    "ministerio",
    "procuradoria",
    "comarca",
    "partido",
    "federacao",
    "coligacao",
}


def _city_capture_is_institutional(normalized_city: str) -> bool:
    tokens = set(normalized_city.split())
    return bool(tokens & ORIGEM_NON_MUNICIPALITY_TOKENS)


def infer_origin_from_row_text(row: "PublishPreviewRow") -> str:
    text = _build_row_inference_text(row)
    if not text:
        return ""
    normalized_text = normalize_class_text(text)
    if "tribunais regionais eleitorais" in normalized_text:
        return ""
    # Aceita topônimos compostos com palavras capitalizadas e conectores
    # intercalados (ex.: "Santo Antônio do Tauá", "São Gonçalo do Amarante",
    # "Campos dos Goytacazes"). O padrão anterior separava conectores e palavras
    # capitalizadas em grupos distintos e falhava quando duas palavras
    # capitalizadas vinham sem conector entre elas ("Santo Antônio"), truncando o
    # nome para "Antônio do Tauá/PA".
    city_word = r"[A-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ][A-Za-zÁÀÃÂÉÊÍÓÔÕÚÜÇáàãâéêíóôõúüç'`´.\-]+"
    city_pattern = (
        r"\b("
        + city_word
        + r"(?:\s+(?:de|do|da|dos|das|e|" + city_word + r"))*"
        + r")/([A-Z]{2})\b"
    )
    matches = list(re.finditer(city_pattern, text))
    for match in reversed(matches):
        city = match.group(1).strip(" ,.;:-")
        uf = match.group(2).upper()
        normalized_city = normalize_class_text(city)
        if (
            city
            and not city.upper().startswith("TRE")
            and "," not in city
            and normalized_city not in STATE_NAME_KEYS
            and not _city_capture_is_institutional(normalized_city)
        ):
            return f"{city}/{uf}"
    tre_sigla = re.search(r"\bTRE[-/ ]([A-Z]{2})\b", text, flags=re.IGNORECASE)
    if tre_sigla:
        return UF_CAPITALS.get(tre_sigla.group(1).upper(), "")
    tre_extenso = re.search(
        r"(?i)\bTribunal Regional Eleitoral d(?:e|o|a)\s+([A-Za-zÀ-ÿ ]+)",
        text,
    )
    if tre_extenso:
        state_name = normalize_class_text(tre_extenso.group(1))
        for state, uf in STATE_UF.items():
            if state in state_name:
                return UF_CAPITALS.get(uf, "")
    if re.search(r"(?i)\bTribunal Superior Eleitoral\b|\bTSE\b", text):
        return UF_CAPITALS["DF"]
    return ""


def _origem_is_court_reference(value: str) -> bool:
    """True quando a origem informada é apenas uma referência a tribunal (TRE/TSE)
    ou está vazia. Nesses casos a coluna não traz o município de fato, e deve ceder
    ao município efetivamente citado no texto do julgamento (ex.: o modelo devolveu
    "Tribunal Regional Eleitoral do Pará", normalizado para a capital "Belém/PA",
    quando o caso era de "Santo Antônio do Tauá/PA")."""
    key = normalize_class_text(value)
    if not key:
        return True
    return bool(
        re.search(r"\btribunal\b|\btre\b|\btse\b|regional eleitoral|superior eleitoral", key)
    )


def _municipio_from_case_text(row: "PublishPreviewRow") -> str:
    """Município no formato 'Cidade/UF' inferido do texto do julgamento, normalizado.
    Retorna "" quando o texto não traz um município específico (a inferência cai para
    a capital do TRE, que não acrescenta especificidade)."""
    inferred = normalize_origem_value(infer_origin_from_row_text(row))
    if not inferred:
        return ""
    if not re.search(r"/[A-Z]{2}$", inferred) or inferred.upper().startswith("TRE"):
        return ""
    return inferred


def row_indicates_suspension_by_vista(row: "PublishPreviewRow") -> bool:
    if row.resultado == "Suspenso por vista":
        return True
    if row.votacao == "Suspenso":
        combined = normalize_class_text(
            "\n".join(
                value
                for value in [
                    row.tema,
                    row.punchline,
                    row.analise_do_conteudo_juridico,
                    row.raciocinio_juridico,
                ]
                if value
            )
        )
        return bool(row.pedido_vista) or ("vista" in combined)
    return False


DEFINITIVE_VOTACAO = {"Unânime", "Por maioria"}


def compute_suspenso_star_updates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Identifica registros cuja votação deve passar de 'Suspenso' para 'Suspenso*'.

    Regra: quando o MESMO processo (mesmo número canônico) foi julgado de forma
    DEFINITIVA ('Unânime' ou 'Por maioria') em data igual ou posterior à da
    suspensão, a etiqueta 'Suspenso' do registro anterior vira 'Suspenso*'. Assim o
    Notion reflete que aquela suspensão foi posteriormente resolvida quando o
    processo voltou à pauta e foi definitivamente votado.

    Cada registro é um dict com pelo menos: page_id, numero_processo, votacao,
    data_sessao. Retorna a sublista de registros que devem ser atualizados.
    """
    by_proc: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = canonicalize_numero_processo(str(record.get("numero_processo", "")))
        if key:
            by_proc.setdefault(key, []).append(record)

    updates: list[dict[str, Any]] = []
    for group in by_proc.values():
        definitive_dates = sorted(
            date
            for date in (
                normalize_session_date_to_iso(str(r.get("data_sessao", "")))
                for r in group
                if normalize_votacao(str(r.get("votacao", ""))) in DEFINITIVE_VOTACAO
            )
            if date
        )
        has_definitive = any(
            normalize_votacao(str(r.get("votacao", ""))) in DEFINITIVE_VOTACAO for r in group
        )
        if not has_definitive:
            continue
        latest_definitive = definitive_dates[-1] if definitive_dates else ""
        for record in group:
            if normalize_votacao(str(record.get("votacao", ""))) != "Suspenso":
                continue
            suspended_date = normalize_session_date_to_iso(str(record.get("data_sessao", "")))
            # Só rebaixa para 'Suspenso*' quando a suspensão é anterior ou simultânea a
            # um julgamento definitivo; não marca uma suspensão nova que seja posterior
            # a todos os julgamentos definitivos conhecidos do processo.
            if not suspended_date or not latest_definitive or suspended_date <= latest_definitive:
                updates.append(record)
    return updates


def reconcile_suspenso_marks(
    notion_client: "NotionSessoesClient",
    notion_schema: "NotionDataSourceSchema",
    *,
    numero_processos: Optional[set[str]] = None,
    pages: Optional[list[dict[str, Any]]] = None,
    apply: bool = True,
) -> list[dict[str, Any]]:
    """Aplica (ou simula, com apply=False) a reconciliação 'Suspenso' -> 'Suspenso*'.

    Quando `pages` não é fornecido, busca no Notion: se `numero_processos` vier
    preenchido, filtra por esses números (barato, usado na publicação); caso
    contrário varre toda a base (usado no script standalone). Retorna a lista de
    mudanças, cada uma com status 'updated'/'failed' (apply=True) ou 'would_update'
    (apply=False).
    """
    if pages is None:
        pages = []
        if numero_processos:
            seen_ids: set[str] = set()
            for numero in sorted({n for n in numero_processos if n}):
                condition = notion_client.build_filter_condition(notion_schema, "numero_processo", numero)
                if not condition:
                    continue
                for page in notion_client.query_data_source(condition):
                    pid = str(page.get("id", ""))
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        pages.append(page)
        else:
            pages = notion_client.query_data_source()

    records: list[dict[str, Any]] = []
    for page in pages:
        records.append(
            {
                "page_id": str(page.get("id", "")),
                "url": str(page.get("url", "")),
                "numero_processo": notion_client._extract_property_text(page, notion_schema, "numero_processo"),
                "votacao": notion_client._extract_property_text(page, notion_schema, "votacao"),
                "data_sessao": notion_client._extract_property_text(page, notion_schema, "data_sessao"),
            }
        )

    changes: list[dict[str, Any]] = []
    for record in compute_suspenso_star_updates(records):
        change = {
            "page_id": record["page_id"],
            "url": record.get("url", ""),
            "numero_processo": record.get("numero_processo", ""),
            "data_sessao": record.get("data_sessao", ""),
            "old_votacao": "Suspenso",
            "new_votacao": "Suspenso*",
            "status": "would_update",
        }
        if apply and record["page_id"]:
            try:
                notion_client._request(
                    "PATCH",
                    f"/pages/{record['page_id']}",
                    json={"properties": {"votacao": {"select": {"name": "Suspenso*"}}}},
                )
                change["status"] = "updated"
            except Exception as exc:  # pragma: no cover - rede
                change["status"] = "failed"
                change["error"] = str(exc)
        changes.append(change)
    return changes


def audit_label_colors(
    notion_client: "NotionSessoesClient",
    property_names: list[str],
) -> dict[str, Any]:
    """Relata, por coluna (select/multi_select), quantas opções estão com cor
    diferente de 'default' e quais são.

    IMPORTANTE: a API do Notion NÃO permite alterar a cor de uma opção de select já
    existente (responde 400 "Cannot update color of select ..."). Por isso esta
    função é apenas de auditoria; a recoloração de etiquetas existentes precisa ser
    feita manualmente na interface do Notion. Opções novas criadas ao gravar páginas
    recebem cor atribuída pelo próprio Notion (não controlável pela API).
    """
    schema = notion_client.fetch_schema()
    properties = schema.raw_payload.get("properties", {})
    report: dict[str, Any] = {}
    for name in property_names:
        prop = properties.get(name)
        if not prop:
            report[name] = {"status": "missing", "total_options": 0, "non_default": 0, "examples": []}
            continue
        prop_type = prop.get("type")
        if prop_type not in {"select", "multi_select"}:
            report[name] = {"status": "not_select", "total_options": 0, "non_default": 0, "examples": []}
            continue
        options = prop.get(prop_type, {}).get("options", []) or []
        colored = [
            {"name": str(option.get("name", "")), "color": str(option.get("color") or "default")}
            for option in options
            if str(option.get("color") or "default") != "default"
        ]
        report[name] = {
            "status": "already_default" if not colored else "needs_manual_recolor",
            "total_options": len(options),
            "non_default": len(colored),
            "examples": colored[:50],
        }
    return report


_CHAPTER_LINE_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s+(.*\S)\s*$")
_CHAPTER_SKIP_TERMS = (
    "abertura",
    "encerramento",
    "transmiss",
    "julgamento em lista",
    "sessao administrativa",
    "intervalo",
    "leitura de ata",
    "posse",
)


def parse_youtube_chapter_entries(description: str) -> dict[str, dict[str, Any]]:
    """Extrai, da descrição (capítulos) de um vídeo de sessão do TSE, um mapa
    ``{numero_processo_canônico: {"seconds": int, "classe_raw": str, "classe": str}}``.

    As descrições do TSE listam capítulos no formato ``HH:MM:SS <classe> <numero>``
    (ex.: ``00:27:45 AREspe 060078521``, ``01:20:17 AgR no AREspe - 060006171``,
    ``01:33:54 Rp - 060018305 / Rp - 06009479``). Cada segmento separado por "/" é
    tratado isoladamente. Linhas administrativas/cerimoniais são ignoradas. Mantém a
    PRIMEIRA ocorrência de cada processo. ``classe`` é a classe já canonizada.
    """
    result: dict[str, dict[str, Any]] = {}
    for raw_line in (description or "").splitlines():
        match = _CHAPTER_LINE_RE.match(raw_line)
        if not match:
            continue
        if match.group(3) is not None:
            hours, minutes, seconds = int(match.group(1)), int(match.group(2)), int(match.group(3))
        else:
            hours, minutes, seconds = 0, int(match.group(1)), int(match.group(2))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        rest = match.group(4)
        if any(term in normalize_class_text(rest) for term in _CHAPTER_SKIP_TERMS):
            continue
        for segment in rest.split("/"):
            tokens = re.findall(r"\d[\d.\-]{4,}\d", segment)
            if not tokens:
                continue
            numero = canonicalize_numero_processo(tokens[-1])
            if not numero or numero in result:
                continue
            classe_raw = segment[: segment.rfind(tokens[-1])].strip(" -nNºª.")
            result[numero] = {
                "seconds": total_seconds,
                "classe_raw": classe_raw,
                "classe": normalize_classe_processo(classe_raw),
            }
    return result


def parse_youtube_chapter_timestamps(description: str) -> dict[str, int]:
    """Compat: mapa {numero_processo_canônico: segundos_de_início} derivado de
    :func:`parse_youtube_chapter_entries`."""
    return {numero: entry["seconds"] for numero, entry in parse_youtube_chapter_entries(description).items()}


_YOUTUBE_DESC_RE_PRIMARY = re.compile(r'"shortDescription":"(.*?)","isCrawlable"', re.DOTALL)
_YOUTUBE_DESC_RE_FALLBACK = re.compile(r'"shortDescription":"(.*?)"', re.DOTALL)


def fetch_youtube_description(video_id: str, session: Optional["requests.Session"] = None) -> str:
    """Busca a descrição (com os capítulos) de um vídeo do YouTube via HTML público.
    Retorna "" em caso de falha. Usado para enriquecer a extração com a classe e o
    timestamp que o próprio TSE publica nos capítulos da descrição."""
    if not video_id:
        return ""
    getter = session.get if session is not None else requests.get
    url = f"https://www.youtube.com/watch?v={video_id}"
    for attempt in range(1, 4):
        try:
            response = getter(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "pt-BR,pt;q=0.9"},
                timeout=30,
            )
            text = getattr(response, "text", "") or ""
            match = _YOUTUBE_DESC_RE_PRIMARY.search(text) or _YOUTUBE_DESC_RE_FALLBACK.search(text)
            if not match:
                return ""
            return match.group(1).encode("utf-8").decode("unicode_escape", errors="replace")
        except Exception:
            if attempt == 3:
                return ""
            time.sleep(1.0 * attempt)
    return ""


_YOUTUBE_TITLE_RE_META = re.compile(r'<meta name="title" content="([^"]+)"')
_YOUTUBE_TITLE_RE_PLAYER = re.compile(r'"videoDetails":\{[^}]*?"title":"(.*?)","')
_YOUTUBE_TITLE_RE_TAG = re.compile(r"<title>([^<]+)</title>")


def fetch_youtube_title(video_id: str, session: Optional["requests.Session"] = None) -> str:
    """Busca o TÍTULO de um vídeo do YouTube via HTML público. Retorna "" em caso de
    falha. O título publicado pelo TSE traz a data da sessão e é a fonte autoritativa
    dessa data (ver ``infer_session_date_from_video_title``)."""
    if not video_id:
        return ""
    getter = session.get if session is not None else requests.get
    url = f"https://www.youtube.com/watch?v={video_id}"
    for attempt in range(1, 4):
        try:
            response = getter(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "pt-BR,pt;q=0.9"},
                timeout=30,
            )
            text = getattr(response, "text", "") or ""
            # O <meta>/<title> trazem o texto já em UTF-8 com entidades HTML; o bloco
            # "videoDetails" do player traz o título com escapes JSON (\uXXXX, \"). Cada
            # fonte exige um decode diferente — aplicar unicode_escape no UTF-8 do meta
            # corromperia acentos (ex.: 'Sessão' -> 'SessÃ£o').
            html_match = _YOUTUBE_TITLE_RE_META.search(text) or _YOUTUBE_TITLE_RE_TAG.search(text)
            if html_match:
                title = unescape(html_match.group(1))
            else:
                player_match = _YOUTUBE_TITLE_RE_PLAYER.search(text)
                if not player_match:
                    return ""
                try:
                    title = json.loads('"' + player_match.group(1) + '"')
                except Exception:
                    title = player_match.group(1)
            return re.sub(r"\s*-\s*YouTube\s*$", "", str(title)).strip()
        except Exception:
            if attempt == 3:
                return ""
            time.sleep(1.0 * attempt)
    return ""


def enrich_preview_rows_with_session_date_from_title(
    rows: list["PublishPreviewRow"],
    youtube_url: str,
    *,
    title: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> list["PublishPreviewRow"]:
    """Corrige a ``data_sessao`` dos preview rows usando o TÍTULO do vídeo do TSE como
    fonte AUTORITATIVA da data da sessão. O modelo ocasionalmente alucina uma data
    default (ex.: '2024-05-21') que não corresponde à sessão; o título publicado pelo
    próprio tribunal ('Sessão Plenária - 29 de Fevereiro de 2024') é confiável. Só
    sobrescreve quando consegue inferir uma data válida do título e ela difere da atual.
    Faz parte do fluxo principal para que novas extrações já saiam com a data correta."""
    if not rows:
        return rows
    if title is None:
        video_id = extract_youtube_video_id(youtube_url)
        if not video_id:
            return rows
        try:
            title = fetch_youtube_title(video_id)
        except Exception as exc:  # pragma: no cover - rede
            if logger:
                logger.warning("Falha ao buscar título do YouTube: %s", exc)
            return rows
    inferred = infer_session_date_from_video_title(title or "")
    if not inferred:
        return rows
    applied = 0
    for row in rows:
        if normalize_session_date_to_iso(row.data_sessao) != inferred:
            row.data_sessao = inferred
            applied += 1
        else:
            row.data_sessao = inferred
    if logger and applied:
        logger.info(
            "Título do YouTube: data da sessão corrigida para %s em %s linha(s) (título: %r).",
            inferred,
            applied,
            title,
        )
    return rows


def _youtube_link_has_timestamp(url: str) -> bool:
    parsed = urlparse(url or "")
    query = parse_qs(parsed.query)
    if query.get("t") or query.get("start"):
        return True
    return parsed.fragment.startswith("t=")


def classe_is_specificity_downgrade(current: str, chapter: str) -> bool:
    """True quando a classe do capítulo é apenas a BASE de uma classe atual mais
    específica (segmentos separados por '-'; ex.: atual 'ED-AgRg-AREspe' vs capítulo
    'AgRg-AREspe'). Nesses casos o capítulo omite um recurso interno (ED/AgRg) que a
    etiqueta atual carrega — não rebaixamos."""
    current_segments = (current or "").split("-")
    chapter_segments = (chapter or "").split("-")
    return (
        len(chapter_segments) < len(current_segments)
        and current_segments[-len(chapter_segments):] == chapter_segments
    )


def enrich_preview_rows_with_youtube_chapters(
    rows: list["PublishPreviewRow"],
    youtube_url: str,
    notion_schema: Optional["NotionDataSourceSchema"] = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> list["PublishPreviewRow"]:
    """Enriquece os preview rows com os CAPÍTULOS da descrição do vídeo (fonte
    autoritativa do TSE): define a ``classe_processo`` (preenchendo vazio, corrigindo
    'PA' e divergências reais — sem rebaixar) e preenche o marcador de tempo do
    ``youtube_link`` quando ausente. Faz parte do fluxo principal para que novas
    extrações já saiam corretas sem reparos posteriores."""
    video_id = extract_youtube_video_id(youtube_url)
    if not video_id or not rows:
        return rows
    try:
        description = fetch_youtube_description(video_id)
    except Exception as exc:  # pragma: no cover - rede
        if logger:
            logger.warning("Falha ao buscar capítulos do YouTube: %s", exc)
        return rows
    entries = parse_youtube_chapter_entries(description) if description else {}
    if not entries:
        return rows
    valid_classes: Optional[set[str]] = None
    if notion_schema is not None:
        prop = notion_schema.raw_payload.get("properties", {}).get("classe_processo", {})
        valid_classes = {
            str(option.get("name", "")).strip()
            for option in prop.get("select", {}).get("options", []) or []
            if str(option.get("name", "")).strip()
        }
    applied_classe = 0
    applied_ts = 0
    for row in rows:
        entry = entries.get(canonicalize_numero_processo(row.numero_processo))
        if not entry:
            continue
        chapter_classe = str(entry.get("classe", "") or "")
        if chapter_classe and (valid_classes is None or chapter_classe in valid_classes):
            current = str(row.classe_processo or "")
            if (
                not current
                or current == "PA"
                or (current != chapter_classe and not classe_is_specificity_downgrade(current, chapter_classe))
            ):
                if row.classe_processo != chapter_classe:
                    row.classe_processo = chapter_classe
                    applied_classe += 1
        seconds = entry.get("seconds")
        if isinstance(seconds, int) and seconds >= 0 and not _youtube_link_has_timestamp(row.youtube_link):
            row.youtube_link = build_timestamped_youtube_link(row.youtube_link, seconds)
            applied_ts += 1
    if logger and (applied_classe or applied_ts):
        logger.info("Capítulos do YouTube: classe corrigida em %s, timestamp preenchido em %s.", applied_classe, applied_ts)
    return rows


def _carregar_mapa_gabinete(lookup_process, session, *, logger=None) -> dict[str, str]:
    """Mapa {gabinete: ministro} para validar o relator; {} quando indisponivel.

    Reapura da propria base de sessoes quando o cache vence (ver gabinete_relator.py).
    Qualquer falha devolve {} -- sem mapa, nada e corrigido, que e o comportamento seguro.
    """
    try:
        from gabinete_relator import JANELA_MESES, apurar_mapa, carregar_mapa
    except Exception as exc:  # noqa: BLE001 - modulo opcional
        if logger:
            logger.debug("gabinete->relator indisponivel: %s", exc)
        return {}

    def apurador():
        import datetime as _dt

        corte = (_dt.date.today() - _dt.timedelta(days=30 * JANELA_MESES)).isoformat()
        client = NotionSessoesClient(get_notion_api_key())
        paginas_notion = client.query_data_source({
            "and": [
                {"property": "data_sessao", "date": {"on_or_after": corte}},
                {"property": "relator", "select": {"is_not_empty": True}},
            ]
        })
        registros = []
        for page in paginas_notion:
            props = page.get("properties", {}) or {}
            numero = "".join(
                bloco.get("plain_text", "")
                for bloco in (props.get("numero_processo", {}).get("rich_text") or [])
            )
            relator = ((props.get("relator", {}) or {}).get("select") or {}).get("name") or ""
            if numero and relator:
                registros.append({"numero": numero, "relator": relator})
        return apurar_mapa(registros, lookup_process, session=session, logger=logger)

    try:
        return carregar_mapa(apurador, logger=logger)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("gabinete->relator: %s; seguindo sem o mapa.", exc)
        return {}


def _repair_cnj_year(numero: str, lookup_process, session, *, tribunal: str = "",
                     janela: int = 6) -> Optional[str]:
    """Corrige o ANO de um CNJ de 20 digitos cujo DV (mod 97) reprova.

    Mantem nucleo, DV, segmento, TR e zona; varre os anos vizinhos e aceita SO o caso em que
    exatamente UM ano faz o DV fechar E o DataJud confirma o processo. Duas provas
    independentes -- sem qualquer uma delas devolve None e o numero fica como esta (nucleo
    corrompido nao tem conserto aqui; o aviso de DV leva o caso para a vistoria).
    """
    from cnj_datajud import format_cnj_number

    digitos = re.sub(r"\D", "", str(numero or ""))
    if len(digitos) != 20:
        return None
    ano_atual = int(digitos[9:13])
    candidatos: list[str] = []
    for delta in range(-janela, janela + 1):
        if delta == 0:
            continue
        ano = ano_atual + delta
        if not 1990 <= ano <= 2100:
            continue
        tentativa = format_cnj_number(f"{digitos[:9]}{ano:04d}{digitos[13:]}")
        if tentativa and cnj_check_digit_is_valid(tentativa) is True:
            candidatos.append(tentativa)
    if len(candidatos) != 1:
        return None
    try:
        if lookup_process(candidatos[0], tribunal=tribunal, session=session):
            return candidatos[0]
    except Exception:  # pragma: no cover - rede
        return None
    return None



def enrich_preview_rows_with_cnj(
    rows: list["PublishPreviewRow"],
    notion_schema: Optional["NotionDataSourceSchema"] = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> list["PublishPreviewRow"]:
    """Enriquece os preview rows com os dados oficiais do CNJ DataJud (fonte pública dos
    tribunais): completa ``numero_processo`` para o CNJ de 20 dígitos quando consistente
    com o número atual (mesmo prefixo NNNNNNN+DD) e preenche ``classe_processo`` quando
    vazia/'PA' (sem rebaixar AgRg-*/ED-*). Não altera origem (o órgão julgador do CNJ é a
    sede da zona eleitoral, que difere do município do caso) nem partes/advogados
    (indisponíveis na API pública por LGPD). Integrado ao fluxo principal para que novas
    extrações já saiam com o número/classe oficiais sem reparos posteriores."""
    if not rows:
        return rows
    try:
        import requests

        from cnj_datajud import format_cnj_number, lookup_process
        from gabinete_relator import relator_do_gabinete
    except Exception as exc:  # pragma: no cover - dependência opcional
        if logger:
            logger.warning("CNJ DataJud indisponível: %s", exc)
        return rows
    valid_classes: Optional[set[str]] = None
    if notion_schema is not None:
        prop = notion_schema.raw_payload.get("properties", {}).get("classe_processo", {})
        valid_classes = {
            str(option.get("name", "")).strip()
            for option in prop.get("select", {}).get("options", []) or []
            if str(option.get("name", "")).strip()
        }
    session = requests.Session()
    applied_num = 0
    applied_classe = 0
    applied_ano = 0
    applied_relator = 0
    mapa_gabinete = _carregar_mapa_gabinete(lookup_process, session, logger=logger)
    for row in rows:
        # REPARO DE ANO (26/08/2026). O bloco abaixo so COMPLETA numero curto; numero de 20
        # digitos com o ANO errado passava direto -- a CTA 0601908-68 da sessao de 25/08/2026
        # saiu ".2023" quando os autos sao ".2022". O ano nao e um palpite: mantendo nucleo+DV,
        # o modulo 97 fecha para UM unico ano, e so se aplica a correcao quando o DataJud
        # tambem confirma o processo. Duas provas independentes; sem qualquer das duas, o
        # numero fica como esta e o aviso de DV leva o caso para a vistoria.
        if cnj_check_digit_is_valid(row.numero_processo) is False:
            reparado = _repair_cnj_year(row.numero_processo, lookup_process, session,
                                        tribunal=row.tribunal)
            if reparado:
                if logger:
                    logger.info("CNJ DataJud: ano corrigido %s -> %s (DV mod 97 + confirmacao "
                                "no DataJud).", row.numero_processo, reparado)
                row.numero_processo = reparado
                row.add_warning(
                    f"ano do CNJ corrigido para {reparado} (DV mod 97 + DataJud) — conferir na vistoria.")
                applied_ano += 1
        try:
            info = lookup_process(
                row.numero_processo,
                tribunal=row.tribunal,
                year=str(row.data_sessao or "")[:4],
                session=session,
            )
        except Exception as exc:  # pragma: no cover - rede
            if logger:
                logger.warning("Falha no lookup CNJ de %s: %s", row.numero_processo, exc)
            continue
        if not info:
            continue
        cur_digits = re.sub(r"\D", "", str(row.numero_processo or ""))
        cnj_digits = re.sub(r"\D", "", info.numero_completo)
        if len(cnj_digits) == 20 and 9 <= len(cur_digits) < 20 and cnj_digits.startswith(cur_digits[:9]):
            formatted = format_cnj_number(cnj_digits)
            if formatted and formatted != row.numero_processo:
                row.numero_processo = formatted
                applied_num += 1
        # RELATOR pelo GABINETE (26/08/2026). O orgao julgador do DataJud e a CADEIRA do
        # relator -- dado oficial, imune ao ruido do audio. O relator extraido do video nao e:
        # quando o nome nao e dito com clareza, o modelo pega um da lista de "Composição da
        # sessão" que o proprio prompt injeta no bloco, e essa lista sai de uma votacao entre
        # chunks cujo corte tem faixa de empate tecnico. No mesmo julgamento (lista triplice do
        # TRE-AM, janela identica) tres rodadas devolveram tres relatores diferentes.
        # So corrige com gabinete UNANIME na base e amostra minima (ver gabinete_relator.py):
        # gabinete em transicao de ocupante nao entra no mapa.
        esperado = relator_do_gabinete(mapa_gabinete, getattr(info, "orgao_julgador", "") or "")
        if esperado:
            atual = normalize_ministro_name(str(row.relator or ""))
            if not atual:
                row.relator = esperado
                applied_relator += 1
            elif normalize_ministro_name(esperado) != atual:
                if logger:
                    logger.info("Relator corrigido pelo gabinete (%s): %s -> %s [%s]",
                                getattr(info, "orgao_julgador", ""), row.relator, esperado,
                                row.numero_processo)
                row.add_warning(
                    f"relator '{row.relator}' divergia do gabinete {getattr(info, 'orgao_julgador', '')} "
                    f"do DataJud; gravado '{esperado}' — conferir na vistoria.")
                row.relator = esperado
                applied_relator += 1

        sigla = info.classe_sigla
        if sigla and (valid_classes is None or sigla in valid_classes):
            current = str(row.classe_processo or "")
            if (not current or current == "PA") and sigla != current and not classe_is_specificity_downgrade(current, sigla):
                row.classe_processo = sigla
                applied_classe += 1
    if logger and (applied_num or applied_classe or applied_ano or applied_relator):
        logger.info("CNJ DataJud: numero completado em %s, classe preenchida em %s, ano corrigido em %s, "
                    "relator pelo gabinete em %s.", applied_num, applied_classe, applied_ano, applied_relator)
    return rows


def format_transcript_snippet(snippet: TranscriptSnippet) -> str:
    return f"[{snippet.start_seconds}s-{snippet.end_seconds}s] {snippet.text}"


def build_transcript_chunks(
    snippets: list[TranscriptSnippet],
    *,
    max_chars: int = TRANSCRIPT_SCAN_MAX_CHARS,
    overlap_snippets: int = TRANSCRIPT_SCAN_OVERLAP_SNIPPETS,
) -> list[TranscriptChunk]:
    if not snippets:
        return []
    chunks: list[TranscriptChunk] = []
    current: list[TranscriptSnippet] = []
    current_chars = 0
    max_chars = max(1000, int(max_chars or TRANSCRIPT_SCAN_MAX_CHARS))
    overlap_snippets = max(0, int(overlap_snippets or 0))

    def flush() -> None:
        if not current:
            return
        chunk_text = "\n".join(format_transcript_snippet(item) for item in current)
        chunks.append(
            TranscriptChunk(
                start_seconds=current[0].start_seconds,
                end_seconds=current[-1].end_seconds,
                text=chunk_text,
                snippet_count=len(current),
            )
        )

    for snippet in snippets:
        formatted = format_transcript_snippet(snippet)
        formatted_chars = len(formatted) + 1
        if current and current_chars + formatted_chars > max_chars:
            flush()
            current = current[-overlap_snippets:] if overlap_snippets else []
            current_chars = sum(len(format_transcript_snippet(item)) + 1 for item in current)
        current.append(snippet)
        current_chars += formatted_chars

    flush()
    return chunks


def extract_youtube_timestamp_seconds(url: str) -> int:
    normalized = normalize_external_url(url) or str(url or "")
    match = re.search(r"[?&]t=(\d+)", normalized)
    if not match:
        return 0
    return coerce_seconds(match.group(1))


def fetch_youtube_duration_seconds(youtube_url: str) -> int:
    watch_url = normalize_youtube_link(youtube_url)
    response = requests.get(
        watch_url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    match = re.search(r'"lengthSeconds":"(\d+)"', response.text)
    if not match:
        return 0
    return int(match.group(1))


def chunk_video_windows(
    duration_seconds: int,
    window_seconds: int = GLOBAL_SCAN_WINDOW_SECONDS,
    overlap_seconds: int = GLOBAL_SCAN_OVERLAP_SECONDS,
) -> list[tuple[int, int]]:
    if duration_seconds <= 0:
        return [(0, window_seconds)]
    windows: list[tuple[int, int]] = []
    cursor = 0
    while cursor < duration_seconds:
        start = max(0, cursor)
        end = min(duration_seconds, cursor + window_seconds)
        windows.append((start, end))
        if end >= duration_seconds:
            break
        cursor = end - overlap_seconds
    return windows


def normalize_party_list(values: list[str]) -> list[str]:
    cleaned = normalize_partes_list(values)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in split_csv_like_text(cleaned):
        canonical = canonicalize_party_option_label(value)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


def normalize_advogado_list(values: list[str]) -> list[str]:
    normalized = normalize_advogados_list(values)
    return split_csv_like_text(normalized)


def normalize_composition_list(values: list[str]) -> list[str]:
    normalized = normalize_composicao(", ".join(values))
    return split_csv_like_text(normalized)


def _composition_quality(values: list[str]) -> tuple[int, int]:
    normalized = normalize_composition_list(values)
    count = len(normalized)
    regimental_issue = composicao_regimental_issue(normalized)
    if is_regimentally_valid_composicao(normalized):
        return (300, count)
    if regimental_issue in {"category_excess", "distribution"}:
        return (15, -count)
    if count == 7:
        return (100, count)
    if count == 6 and not regimental_issue:
        return (99, count)
    if count == 6:
        return (14, -count)
    if count > 7:
        return (10 - min(count - 8, 9), -count)
    if 1 <= count < 6:
        return (count, count)
    return (0, count)


def _pick_better_composition(primary: list[str], secondary: list[str]) -> list[str]:
    primary_normalized = normalize_composition_list(primary)
    secondary_normalized = normalize_composition_list(secondary)
    primary_score = _composition_quality(primary_normalized)
    secondary_score = _composition_quality(secondary_normalized)
    if secondary_score > primary_score:
        return secondary_normalized
    return primary_normalized


def choose_preferred_composition(item_values: list[str], session_values: list[str]) -> list[str]:
    return _pick_better_composition(item_values, session_values)


def _preview_row_sort_key(item: "PublishPreviewRow") -> tuple[int, int, int, str, str]:
    extracted_ts = extract_youtube_timestamp_seconds(item.youtube_link)
    start_seconds = item.source_start_seconds if item.source_start_seconds >= 0 else extracted_ts
    bundle_index = item.source_bundle_index if item.source_bundle_index >= 0 else 999999
    item_index = item.source_item_index if item.source_item_index >= 0 else 999999
    return (
        max(0, start_seconds),
        bundle_index,
        item_index,
        canonicalize_numero_processo(item.numero_processo),
        normalize_class_text(item.tema),
    )


def preview_row_sort_key(item: "PublishPreviewRow") -> tuple[int, int, int, str, str]:
    return _preview_row_sort_key(item)


def _merge_source_order_fields(primary: "PublishPreviewRow", secondary: "PublishPreviewRow") -> tuple[int, int, int]:
    source = secondary if _preview_row_sort_key(secondary) < _preview_row_sort_key(primary) else primary
    return source.source_start_seconds, source.source_bundle_index, source.source_item_index


def normalize_external_url(url: str) -> str:
    cleaned = (url or "").strip().strip(".,;)]}>\"'")
    if not cleaned:
        return ""
    if not re.match(r"^https?://", cleaned, flags=re.IGNORECASE):
        cleaned = "https://" + cleaned
    if not re.match(r"^https?://", cleaned, flags=re.IGNORECASE):
        return ""
    return cleaned


def normalize_external_url_list(values: list[str], limit: int | None = None) -> list[str]:
    normalized_values: list[str] = []
    for value in values:
        normalized = normalize_external_url(value)
        if not normalized:
            continue
        normalized_values.append(resolve_grounding_redirect_url(normalized))
    normalized = dedupe_preserve_order(normalized_values)
    if limit is not None:
        return normalized[:limit]
    return normalized


def fold_text_for_match(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip()


def domain_from_url(url: str) -> str:
    try:
        host = url.split("://", 1)[1].split("/", 1)[0]
        if "@" in host:
            host = host.split("@", 1)[1]
        if ":" in host:
            host = host.split(":", 1)[0]
        host = host.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def resolve_grounding_redirect_url(url: str) -> str:
    normalized = normalize_external_url(url)
    if not normalized:
        return ""
    if domain_from_url(normalized) != "vertexaisearch.cloud.google.com":
        return normalized
    try:
        response = requests.get(
            normalized,
            timeout=30,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resolved = normalize_external_url(response.url)
        # Só aceita se o redirect realmente resolveu para um destino real; um redirect
        # de grounding não resolvido (continua em vertexaisearch) é lixo, descarta.
        if resolved and domain_from_url(resolved) != "vertexaisearch.cloud.google.com":
            return resolved
        return ""
    except Exception:
        return ""


def _best_effort_response_text(response: Any) -> str:
    text = str(getattr(response, "text", "") or "")
    try:
        content = getattr(response, "content", b"")
        if isinstance(content, (bytes, bytearray)) and content:
            encoding = (
                getattr(response, "encoding", None)
                or getattr(response, "apparent_encoding", None)
                or "utf-8"
            )
            decoded = bytes(content).decode(encoding, errors="ignore")
            if decoded:
                text = decoded
    except Exception:
        pass
    return text


def _extract_visible_html_text(raw_html: str) -> str:
    text = raw_html or ""
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=512)
def fetch_candidate_page_snapshot(url: str) -> tuple[str, int, str, str]:
    normalized = normalize_external_url(url)
    if not normalized:
        return "", 0, "", ""
    try:
        response = requests.get(
            normalized,
            timeout=30,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
    except Exception:
        return normalized, 0, "", ""
    final_url = normalize_external_url(getattr(response, "url", normalized)) or normalized
    status_code = int(getattr(response, "status_code", 0) or 0)
    content_type = (getattr(response, "headers", {}) or {}).get("Content-Type", "") or ""
    return final_url, status_code, content_type.lower(), _best_effort_response_text(response)


def is_html_like_response(content_type: str, text: str) -> bool:
    return "html" in (content_type or "") or "<html" in (text or "").lower()


def page_looks_not_found(*, status_code: int, final_url: str, text: str) -> bool:
    if status_code >= 400:
        return True
    visible_text = fold_text_for_match(_extract_visible_html_text(text))
    if any(marker in visible_text for marker in NOT_FOUND_TEXT_MARKERS):
        return True
    lowered_url = (final_url or "").lower()
    return "/error/" in lowered_url or lowered_url.endswith("/404")


# Subdomínios/sistemas que são base de dados, visualizador de processo (PJe), consulta
# processual ou índice temático de jurisprudência — NUNCA são matéria de imprensa,
# independentemente do path. Casado por FRAGMENTO no host (cobre www./qualquer prefixo).
# Nomes distintivos: valem para qualquer TLD.
_NON_NEWS_SYSTEM_HOST_FRAGMENTS = (
    "temasselecionados",   # índice temático de jurisprudência (TSE)
    "consultaunificada",   # consulta processual unificada
    "pje",                 # Processo Judicial Eletrônico (visualizador de documento)
    "consultapublica",     # consulta pública processual
    "consultaprocessual",
    "divulgacand",         # DivulgaCand (resultados/contas de campanha)
    "sjur",                # sistema de jurisprudência
    "jurisprudencia.",     # subdomínio de jurisprudência
    "inteiroteor",         # visualizador de inteiro teor de acórdão
    "sadp",                # SADP — visualizador de dados do processo (ExibirDadosProcesso)
    "sessoespub",          # visualizador público de votos/sessões
    "seer.",               # plataforma de revista acadêmica (SEER/OJS, ex.: seer.ufrgs.br)
    "periodicos.",         # portal de periódicos acadêmicos
)


def _is_canonical_electoral_news_host(host: str) -> bool:
    """True para os hosts canônicos que publicam imprensa eleitoral: ``tse.jus.br`` e
    ``tre-XX.jus.br`` (com ou sem ``www.``)."""
    bare = host[4:] if host.startswith("www.") else host
    return bare == "tse.jus.br" or bool(re.fullmatch(r"tre-[a-z]{2}\.jus\.br", bare))


def is_non_news_system_url(url: str) -> bool:
    """True quando a URL é de um SISTEMA/base — visualizador de processo/PJe, consulta
    processual, dados de resultado, portal de aplicativo, índice temático de
    jurisprudência ou revista institucional/acadêmica — em vez de matéria de imprensa.
    Em ``.jus.br`` apenas os hosts canônicos (``tse.jus.br``/``tre-XX.jus.br``) publicam
    imprensa; qualquer outro subdomínio (``temasselecionados``, ``consultaunificadapje``,
    ``sadppush``, ``apps``, ``sessoespub``, ``resultados``, ``resenhaeleitoral``...) é
    sistema. Fora de ``.jus.br``, casa fragmentos distintivos (PJe, SEER, periódicos)."""
    host = domain_from_url(normalize_external_url(url))
    if not host:
        return False
    if any(fragment in host for fragment in _NON_NEWS_SYSTEM_HOST_FRAGMENTS):
        return True
    if host.endswith(".jus.br") and not _is_canonical_electoral_news_host(host):
        return True
    return False


def classify_news_urls(urls: list[str]) -> tuple[list[str], list[str], list[str]]:
    tse_urls: list[str] = []
    tre_urls: list[str] = []
    general_urls: list[str] = []

    for url in normalize_external_url_list(urls):
        domain = domain_from_url(url)
        if not domain:
            continue
        if is_non_news_system_url(url):
            continue
        if TSE_DOMAIN_RE.search(domain):
            tse_urls.append(url)
        elif TRE_DOMAIN_RE.search(domain):
            tre_urls.append(url)
        elif GENERAL_DOMAINS_RE.search(domain) or "jus.br" not in domain:
            general_urls.append(url)

    return (
        dedupe_preserve_order(tse_urls),
        dedupe_preserve_order(tre_urls),
        dedupe_preserve_order(general_urls),
    )


def fetch_candidate_page_text(url: str) -> str:
    final_url, status_code, content_type, text = fetch_candidate_page_snapshot(url)
    if not final_url:
        return ""
    if not is_html_like_response(content_type, text):
        return ""
    if page_looks_not_found(status_code=status_code, final_url=final_url, text=text):
        return ""
    return _extract_visible_html_text(text)


def filter_accessible_news_urls(urls: list[str]) -> tuple[list[str], list[str]]:
    accepted: list[str] = []
    dropped: list[str] = []
    for url in normalize_external_url_list(urls):
        final_url, status_code, content_type, text = fetch_candidate_page_snapshot(url)
        candidate_url = final_url or normalize_external_url(url)
        if not candidate_url:
            continue
        if not is_html_like_response(content_type, text):
            dropped.append(candidate_url)
            continue
        if page_looks_not_found(status_code=status_code, final_url=candidate_url, text=text):
            dropped.append(candidate_url)
            continue
        accepted.append(candidate_url)
    return dedupe_preserve_order(accepted), dedupe_preserve_order(dropped)


def _extract_origin_markers(origem: str) -> tuple[str, str]:
    value = normalize_model_text(origem)
    if not value:
        return "", ""
    city = value
    uf = ""
    if "/" in value:
        city, uf = value.split("/", 1)
    elif " - " in value:
        city, uf = value.split(" - ", 1)
    return fold_text_for_match(city), fold_text_for_match(uf)


_NAME_ENTRY_TRIM_REGEX = re.compile(r"^[\s\-–—;,.]+|[\s\-–—;,]+$")


def sanitize_name_entries(entries: list[str]) -> list[str]:
    """Higieniza listas de nomes (partes/advogados) antes de publicar.

    Remove pontuação/travessão sobrando nas pontas de cada entrada (ASR grava
    "Dr. Fulano –") e deduplica case-insensitive preservando a ordem — o mesmo
    nome repetido pelo modelo não pode virar registro duplicado na base.
    """
    seen: set[str] = set()
    cleaned: list[str] = []
    for entry in entries or []:
        value = _NAME_ENTRY_TRIM_REGEX.sub("", re.sub(r"\s{2,}", " ", str(entry or ""))).strip()
        if not value:
            continue
        key = fold_text_for_match(value)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return cleaned


def _extract_party_markers(partes: list[str]) -> list[str]:
    markers: list[str] = []
    generic_terms = {
        "agravante",
        "agravada",
        "agravado",
        "recorrente",
        "recorrido",
        "candidato",
        "candidata",
        "coligacao",
        "partido",
        "prefeito",
        "vice",
        "municipio",
        "cargo",
        "eleicoes",
        "eleicao",
    }
    for value in partes:
        text = re.sub(r"\([^)]*\)", " ", value or "")
        folded = fold_text_for_match(text)
        if not folded:
            continue
        if folded in markers:
            continue
        words = [word for word in folded.split() if word not in generic_terms]
        if len(words) >= 2:
            markers.append(" ".join(words[:3]))
        elif words:
            markers.append(words[0])
    return dedupe_preserve_order([marker for marker in markers if len(marker) >= 4])[:4]


def _extract_theme_markers(row: "PublishPreviewRow") -> list[str]:
    text = fold_text_for_match(" ".join([row.tema, row.punchline, row.analise_do_conteudo_juridico]))
    known_markers = [
        "abuso de poder",
        "capacitacao ilicita de sufragio",
        "captacao ilicita de sufragio",
        "conduta vedada",
        "cota de genero",
        "desinformacao",
        "fundo eleitoral",
        "fundo partidario",
        "inelegibilidade",
        "prestacao de contas",
        "propaganda eleitoral",
        "propaganda irregular",
        "publicidade institucional",
        "registro de candidatura",
        "sobras eleitorais",
        "uso de bens publicos",
    ]
    candidates = [marker for marker in known_markers if marker in text]
    if "campanha eleitoral" in text:
        candidates.append("campanha eleitoral")
    return dedupe_preserve_order(candidates)[:5]


def _process_markers_for_news(row: "PublishPreviewRow") -> tuple[list[str], list[str]]:
    process_values = [
        row.numero_processo,
        extract_full_cnj(row.numero_processo),
        canonicalize_numero_processo(row.numero_processo),
    ]
    folded_markers: list[str] = []
    digit_markers: list[str] = []
    for value in process_values:
        folded = fold_text_for_match(value)
        if folded:
            folded_markers.append(folded)
        digits = re.sub(r"\D", "", str(value or ""))
        if len(digits) >= 6:
            digit_markers.append(digits)
    return dedupe_preserve_order(folded_markers), dedupe_preserve_order(digit_markers)


def _news_page_relevance_evidence(page_text: str, row: "PublishPreviewRow") -> dict[str, Any]:
    folded_page_text = fold_text_for_match(page_text)
    page_digits = re.sub(r"\D", "", page_text or "")
    if not folded_page_text:
        return {"relevant": False, "score": 0}

    process_markers, process_digit_markers = _process_markers_for_news(row)
    full_process = fold_text_for_match(extract_full_cnj(row.numero_processo))
    short_process = fold_text_for_match(canonicalize_numero_processo(row.numero_processo))
    city_marker, uf_marker = _extract_origin_markers(row.origem)
    party_markers = _extract_party_markers(row.partes)
    theme_markers = _extract_theme_markers(row)

    process_hit = (
        bool(full_process and full_process in folded_page_text)
        or bool(short_process and short_process in folded_page_text)
        or any(marker in folded_page_text for marker in process_markers)
        or any(marker and marker in page_digits for marker in process_digit_markers)
    )
    city_hit = bool(city_marker and city_marker in folded_page_text)
    uf_hit = bool(uf_marker and uf_marker in folded_page_text)
    party_hits = sum(1 for marker in party_markers if marker and marker in folded_page_text)
    theme_hits = sum(1 for marker in theme_markers if marker and marker in folded_page_text)
    tribunal_hits = sum(1 for marker in [fold_text_for_match(row.tribunal), "tse"] if marker and marker in folded_page_text)

    score = 0
    if process_hit:
        score += 4
    if city_hit:
        score += 2
    if uf_hit:
        score += 1
    score += min(party_hits, 2) * 2
    score += min(theme_hits, 2) * 2
    score += min(tribunal_hits, 1)

    strong_context_hit = (
        process_hit
        or (city_hit and party_hits >= 1)
        or (city_hit and theme_hits >= 1 and tribunal_hits >= 1)
        or (party_hits >= 1 and theme_hits >= 1 and (city_hit or tribunal_hits >= 1))
    )
    return {
        "relevant": strong_context_hit and score >= 5,
        "score": score,
        "process_hit": process_hit,
        "city_hit": city_hit,
        "uf_hit": uf_hit,
        "party_hits": party_hits,
        "theme_hits": theme_hits,
        "tribunal_hits": tribunal_hits,
    }


def is_news_page_text_relevant(page_text: str, row: "PublishPreviewRow") -> bool:
    return bool(_news_page_relevance_evidence(page_text, row).get("relevant"))


def is_general_news_url_relevant(url: str, row: "PublishPreviewRow") -> bool:
    return is_news_page_text_relevant(fetch_candidate_page_text(url), row)


def filter_general_news_urls(urls: list[str], row: "PublishPreviewRow") -> list[str]:
    accepted: list[str] = []
    for url in normalize_external_url_list(urls):
        if is_non_news_system_url(url):
            continue
        if is_general_news_url_relevant(url, row):
            accepted.append(url)
    return dedupe_preserve_order(accepted)


# Seções (segmentos de caminho) do domínio eleitoral que NÃO são matéria sobre o
# caso (jurisprudência, decisões em destaque/por ano/assunto, busca, agenda, normas,
# transparência...). Casado por SEGMENTO inteiro (não substring) para não pegar
# palavras dentro do slug de uma matéria válida (ex.: ".../tse-aprova-resolucoes").
_GENERIC_INSTITUTIONAL_SECTIONS = {
    "jurisprudencia",
    "institucional",
    "eleicoes",
    "transparencia",
    "servicos",
    "biblioteca",
    "sjur",
    "pesquisa",
    "busca",
    "consulta",
    "acordaos",
    "sumulas",
    "normas",
    "legislacao",
    "agenda",
    "publicacoes",
    "corregedoria",
    "gestao",
}


def is_generic_institutional_news_url(url: str) -> bool:
    """True quando a URL institucional eleitoral (TSE/TRE) é uma página de
    seção/índice (jurisprudência, decisões por ano/assunto, busca, agenda, normas,
    home, listagem por ano) — NÃO uma matéria específica do caso. Demais URLs ficam
    a cargo da checagem de relevância de conteúdo."""
    if is_non_news_system_url(url):
        return True
    parsed = urlparse(normalize_external_url(url))
    host = (parsed.netloc or "").lower()
    path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/").lower()
    if not host.endswith(".jus.br"):
        return False
    is_electoral_domain = host in {"tse.jus.br", "www.tse.jus.br"} or host.startswith("www.tre-") or host.startswith("tre-")
    if not is_electoral_domain:
        return False
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return True
    if any(segment in _GENERIC_INSTITUTIONAL_SECTIONS or segment.startswith("decisoes") for segment in segments):
        return True
    # Índice de notícias/imprensa, seção raiz, ou listagem por ano.
    last_segment = segments[-1]
    if last_segment in {"noticias", "comunicacao", "imprensa", "todas"} or last_segment.isdigit():
        return True
    return False


def filter_relevant_institutional_news_urls(
    urls: list[str],
    row: "PublishPreviewRow",
) -> tuple[list[str], list[str], list[str]]:
    accepted: list[str] = []
    dropped_unavailable: list[str] = []
    dropped_irrelevant: list[str] = []
    for url in normalize_external_url_list(urls):
        if is_generic_institutional_news_url(url):
            dropped_irrelevant.append(url)
            continue
        final_url, status_code, content_type, text = fetch_candidate_page_snapshot(url)
        candidate_url = final_url or normalize_external_url(url)
        if not candidate_url:
            continue
        if is_generic_institutional_news_url(candidate_url):
            dropped_irrelevant.append(candidate_url)
            continue
        if not is_html_like_response(content_type, text):
            dropped_unavailable.append(candidate_url)
            continue
        if page_looks_not_found(status_code=status_code, final_url=candidate_url, text=text):
            dropped_unavailable.append(candidate_url)
            continue
        page_text = f"{candidate_url}\n{_extract_visible_html_text(text)}"
        if not is_news_page_text_relevant(page_text, row):
            dropped_irrelevant.append(candidate_url)
            continue
        accepted.append(candidate_url)
    return (
        dedupe_preserve_order(accepted),
        dedupe_preserve_order(dropped_unavailable),
        dedupe_preserve_order(dropped_irrelevant),
    )


def build_news_enrichment_context(row: "PublishPreviewRow") -> str:
    fields = [
        ("tema", row.tema),
        ("punchline", row.punchline),
        ("numero_processo", row.numero_processo),
        ("classe_processo", row.classe_processo),
        ("tribunal", row.tribunal),
        ("origem", row.origem),
        ("data_sessao", row.data_sessao),
        ("relator", row.relator),
        ("partes", ", ".join(row.partes)),
    ]
    lines: list[str] = []
    for label, value in fields:
        value = (value or "").strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def build_process_metadata_context(row: "PublishPreviewRow") -> str:
    fields = [
        ("tema", row.tema),
        ("numero_processo", row.numero_processo),
        ("classe_processo", row.classe_processo),
        ("tribunal", row.tribunal),
        ("origem", row.origem),
        ("data_sessao", row.data_sessao),
        ("relator", row.relator),
        ("resultado", row.resultado),
        ("partes", ", ".join(row.partes)),
        ("analise_do_conteudo_juridico", row.analise_do_conteudo_juridico),
    ]
    lines: list[str] = []
    for label, value in fields:
        value = normalize_model_text(value)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _row_is_lista_triplice(row: "PublishPreviewRow") -> bool:
    return "lista triplice" in normalize_class_text(getattr(row, "classe_processo", "") or "")


_TRIPLICE_HONORIFIC_RE = re.compile(r"^\s*(?:dr|dra|sr|sra|adv|advogad[oa])\.?\s+", re.IGNORECASE)


def _clean_triplice_indicados(values: list[str]) -> list[str]:
    """Normaliza/dedup nomes de indicados em lista tríplice (page-values seguros). Remove
    honoríficos (Dr./Dra.), que senão fariam o nome ser descartado como advogado."""
    stripped: list[str] = []
    for raw in values or []:
        name = re.sub(r"\s+", " ", str(raw or "")).strip()
        name = _TRIPLICE_HONORIFIC_RE.sub("", name).strip()
        if name:
            stripped.append(name)
    return normalize_party_list(stripped)[:5]


def _metadata_confidence_blocks_write(confidence: str) -> bool:
    """Bloqueia gravação só quando o grounding declara confiança explicitamente baixa."""
    return normalize_class_text(confidence) in {"low", "baixa", "baixo", "none", "nenhuma"}


def _process_metadata_result_is_empty(result: "BaseModel") -> bool:
    """ProcessMetadataResult sem nenhum campo útil (típico de resposta grounded em prosa)."""
    return (
        isinstance(result, ProcessMetadataResult)
        and not result.full_numero_processo
        and not result.origem
        and result.is_judged_process is None
        and not result.indicados_lista_triplice
    )


# O Grounding (Google Search) devolve PROSA, não JSON; esta 2ª passada SEM busca converte o
# texto já pesquisado no schema, com as chaves EXPLÍCITAS (o REST grounded não envia
# responseSchema, então sem isto o modelo devolve JSON com chaves próprias e o parse falha).
PROCESS_METADATA_EXTRACTION_PROMPT = (
    "Converta o texto pesquisado abaixo em JSON com EXATAMENTE estas chaves:\n"
    "- full_numero_processo: número CNJ completo do processo (string; vazio se não houver)\n"
    "- origem: cidade/UF de origem (string; vazio se não houver)\n"
    "- is_judged_process: true se trata de processo efetivamente julgado, false se aparece só como precedente citado, null se indefinido\n"
    "- indicados_lista_triplice: lista com os nomes completos das pessoas indicadas na lista tríplice (vazia se não for lista tríplice ou não houver nomes)\n"
    "- confidence: alta, media ou baixa\n"
    "- rationale: justificativa curta\n"
    "Use EXCLUSIVAMENTE o que está no texto; não invente nomes, números ou origem.\n\n"
    "TEXTO PESQUISADO:\n"
)


def _normalize_local_judgment_probe(value: str) -> str:
    normalized = normalize_class_text(unescape(str(value or "")))
    if not normalized:
        return ""
    return re.sub(r"[^a-z0-9]+", "", normalized)


GROUNDING_INCONCLUSIVE_RATIONALE_RE = re.compile(
    r"data[^.;]{0,60}futur|n[ãa]o (foi |fora )?localizad|n[ãa]o (foi )?encontrad|"
    r"n[úu]mero[^.;]{0,40}incomplet|sem resultado",
    re.IGNORECASE,
)


def _grounding_rationale_is_inconclusive(rationale: str) -> bool:
    """True quando o grounding só diz que NÃO ACHOU o processo (número incompleto,
    nada indexado, data tida como futura) — o que não é evidência de precedente."""
    return bool(GROUNDING_INCONCLUSIVE_RATIONALE_RE.search(normalize_model_text(rationale or "")))


def _row_has_strong_local_judgment_evidence(row: "PublishPreviewRow", artifact_store: "RunArtifacts") -> bool:
    probes = {
        _normalize_local_judgment_probe(row.numero_processo),
        _normalize_local_judgment_probe(normalize_numero_processo_display(row.numero_processo)),
        _normalize_local_judgment_probe(canonicalize_numero_processo(row.numero_processo)),
        _normalize_local_judgment_probe(row.tema),
    }
    special_key = ""
    canonical_process = canonicalize_numero_processo(row.numero_processo)
    overlay_key = identity_overlay_class_key(row.classe_processo) or identity_overlay_class_key(row.tema)
    if canonical_process and overlay_key:
        special_key = f"{overlay_key} {canonical_process}"
    probes.add(_normalize_local_judgment_probe(special_key))
    probes.discard("")
    if not probes:
        return False

    matched_chunks: set[str] = set()
    for chunk_path in sorted(artifact_store.root_dir.glob("raw_global_response_chunk_*.txt")):
        try:
            payload = json.loads(chunk_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for session_payload in payload:
            if not isinstance(session_payload, dict):
                continue
            judgments = session_payload.get("julgamentos") or []
            if not isinstance(judgments, list):
                continue
            for judgment in judgments:
                if not isinstance(judgment, dict) or judgment.get("should_ignore") is True:
                    continue
                for raw_value in extract_chunk_judgment_process_values(judgment):
                    chunk_probe = _normalize_local_judgment_probe(raw_value)
                    if not chunk_probe:
                        continue
                    if any(
                        probe == chunk_probe or probe in chunk_probe or chunk_probe in probe
                        for probe in probes
                    ):
                        matched_chunks.add(chunk_path.name)
                        break
    return len(matched_chunks) >= 2 or (bool(overlay_key) and len(matched_chunks) >= 1)


def require_google_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Biblioteca 'google-genai' não encontrada. Instale com "
            "'.venv/Scripts/python.exe -m pip install google-genai'."
        ) from exc
    return genai, types


def require_youtube_transcript_api():
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise RuntimeError(
            "Biblioteca 'youtube-transcript-api' não encontrada. Instale com "
            "'.venv/Scripts/python.exe -m pip install youtube-transcript-api'."
        ) from exc
    return YouTubeTranscriptApi


def resolve_gemini_model(client: Any, requested_model: str) -> str:
    return DEFAULT_GEMINI_MODEL


def build_gemini_model_candidates(client: Any, requested_model: str) -> list[str]:
    return [DEFAULT_GEMINI_MODEL]


def extract_retry_delay_seconds(exc: Exception) -> float:
    message = str(exc)
    for pattern in (
        r"please retry in ([0-9]+(?:\.[0-9]+)?)s",
        r"'retryDelay': '([0-9]+)s'",
    ):
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return 0.0


def should_disable_model(exc: Exception) -> bool:
    message = str(exc)
    status_code = getattr(exc, "code", None)
    if status_code in (404,):
        return True
    lowered = message.lower()
    if "not found for api version" in lowered or "is not supported" in lowered:
        return True
    if "resource_exhausted" in lowered or "quota exceeded" in lowered:
        return "limit: 0" in lowered
    return False


class SessionWindow(BaseModel):
    title_hint: str = ""
    start_seconds: int = 0
    end_seconds: int | None = None
    mentioned_process_numbers: list[str] = Field(default_factory=list)
    should_ignore: bool = False
    ignore_reason: str = ""


class SessionExtraction(BaseModel):
    data_sessao: str = ""
    composicao: list[str] = Field(
        default_factory=list,
        description=(
            "Ministros que compoem o colegiado e participam dos julgamentos da sessao "
            "(tipicamente 7: 3 do STF, 2 do STJ, 2 juristas; pode haver 6 por ausencia). "
            "Formato 'Min. <Nome>'. Nao inclua ministros citados apenas em votos/precedentes "
            "de outros processos, nem partes, advogados ou procuradores."
        ),
    )
    judgments: list[SessionWindow] = Field(default_factory=list)


@dataclass
class TranscriptSnippet:
    text: str
    start_seconds: int
    end_seconds: int


@dataclass
class TranscriptChunk:
    start_seconds: int
    end_seconds: int
    text: str
    snippet_count: int


# ---------------------------------------------------------------------------
# Rito processual das sessões — camada determinística sobre a transcrição.
#
# O presidente estrutura a pauta com marcadores falados: cada julgamento
# individual começa num apregoamento ("chamo a julgamento...") e termina na
# proclamação ("proclamo o resultado") ou num pedido de vista. Esses eventos
# corrigem o plano de janelas do scan por vídeo (fundem fatias de voto longo,
# resgatam a sessão administrativa marcada como "cerimonial" e desligam os
# julgamentos "em lista", ignorados por decisão de projeto). Nenhuma chamada
# de IA: apenas regex sobre o texto normalizado (sem acentos/pontuação) de
# pares de snippets consecutivos, porque os marcadores quebram entre legendas.
# ---------------------------------------------------------------------------
RITO_EVENT_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    (
        "lista",
        re.compile(
            r"listas? dest[ae] pauta"
            r"|apreg\w+ para (?:o )?julgamento (?:as|das) listas"
            r"|cham[oa]\w*(?: \w+){0,3}? (?:as )?(?:duas |tres )?listas"
            r"|julgamentos? em lista"
        ),
    ),
    (
        "apregoamento",
        re.compile(
            r"cham[oa](?: \w+){0,2}? (?:(?:a|para(?: o)?|em mesa(?: para(?: o)?)?) )?julgamento"
            r"|apreg[ou]\w* (?:para (?:o )?julgamento )?[oa] "
        ),
    ),
    ("proclamacao", re.compile(r"proclam(?:o|amos)(?: \w+){0,2}? (?:o )?resultado")),
    (
        "vista",
        re.compile(r"ped(?:e|iu|ido) (?:de )?vistas?\b|antecip\w+ o pedido de vistas?"),
    ),
    (
        "admin_open",
        re.compile(r"(?:dou |)inicio a sessao administrativa|declaro aberta a sessao administrativa"),
    ),
    ("admin_close", re.compile(r"declaro encerrada a sessao administrativa")),
    ("sessao_open", re.compile(r"declaro aberta a sessao")),
)
RITO_EVENT_DEDUPE_SECONDS = 20


class RitoEvent(BaseModel):
    kind: str
    start_seconds: int
    end_seconds: int
    text: str = ""


def detect_rito_events(snippets: list[TranscriptSnippet]) -> list[RitoEvent]:
    """Varre a transcrição e devolve os eventos do rito em ordem cronológica.

    O matching roda sobre o texto normalizado de cada par de snippets
    consecutivos; eventos do mesmo tipo a menos de RITO_EVENT_DEDUPE_SECONDS
    são deduplicados (efeito do overlap dos pares).
    """
    events: list[RitoEvent] = []
    last_by_kind: dict[str, int] = {}
    for index, snippet in enumerate(snippets):
        nxt = snippets[index + 1] if index + 1 < len(snippets) else None
        raw_text = snippet.text if nxt is None else f"{snippet.text} {nxt.text}"
        normalized = normalize_class_text(raw_text)
        if not normalized:
            continue
        matched: dict[str, str] = {}
        for kind, pattern in RITO_EVENT_PATTERNS:
            found = pattern.search(normalized)
            if found:
                matched[kind] = found.group(0).strip()
        if not matched:
            continue
        # "Chamo a julgamento as listas tríplices" NÃO é o bloco de julgamento em lista:
        # é o apregoamento de julgamentos válidos (formação de lista de três nomes para
        # vaga de TRE). Sem esta ressalva o evento "lista" nasce, engole o apregoamento
        # logo abaixo e o passo 2 de refine_session_windows_with_rito desliga as janelas —
        # o mesmo desfecho que, no caminho SEM transcrição, matou as listas tríplices de
        # Maceió/AL e Florianópolis/SC em 25/08/2026. `normalized` já vem sem acentos.
        if "lista" in matched and re.search(r"\btriplice", normalized):
            chamada = matched.pop("lista")
            # chamar/apregoar listas tríplices É apregoamento. O padrão de apregoamento pede
            # artigo no singular ("apregoo o feito"), então "apregoo ... as listas tríplices"
            # não casa sozinho e o trecho ficaria sem evento algum — sem apregoamento, o passo 1
            # do refine não resgata a janela.
            matched.setdefault("apregoamento", chamada)
        # Chamada das listas não é apregoamento individual; abertura da sessão
        # administrativa não é a abertura genérica da sessão.
        if "lista" in matched:
            matched.pop("apregoamento", None)
        if "admin_open" in matched or "admin_close" in matched:
            matched.pop("sessao_open", None)
        start_seconds = int(snippet.start_seconds)
        end_seconds = int((nxt or snippet).end_seconds)
        for kind, text in matched.items():
            last = last_by_kind.get(kind)
            if last is not None and start_seconds - last < RITO_EVENT_DEDUPE_SECONDS:
                continue
            last_by_kind[kind] = start_seconds
            events.append(
                RitoEvent(kind=kind, start_seconds=start_seconds, end_seconds=end_seconds, text=text)
            )
    events.sort(key=lambda event: (event.start_seconds, event.kind))
    return events


def count_individual_apregoamentos(events: list[RitoEvent]) -> int:
    """Apregoamentos individuais da pauta (julgamentos "em lista" ficam fora)."""
    lista_starts = [event.start_seconds for event in events if event.kind == "lista"]
    first_lista = min(lista_starts) if lista_starts else None
    count = 0
    for event in events:
        if event.kind != "apregoamento":
            continue
        if first_lista is not None and event.start_seconds >= first_lista:
            continue
        count += 1
    return count


def _rito_segments(events: list[RitoEvent], tolerance: int) -> list[tuple[int, int]]:
    """Segmentos [apregoamento, fechamento) — cada um é UM item de pauta.

    O fechamento é a última proclamação/vista antes do apregoamento seguinte;
    sem fechamento detectado, o próprio apregoamento seguinte delimita.
    """
    apregoamentos = [e for e in events if e.kind == "apregoamento"]
    closers = sorted(e.start_seconds for e in events if e.kind in {"proclamacao", "vista"})
    segments: list[tuple[int, int]] = []
    for idx, apreg in enumerate(apregoamentos):
        next_start = apregoamentos[idx + 1].start_seconds if idx + 1 < len(apregoamentos) else None
        candidates = [t for t in closers if t > apreg.start_seconds]
        if next_start is not None:
            candidates = [t for t in candidates if t <= next_start + tolerance]
        if candidates:
            end = max(candidates)
        elif next_start is not None:
            end = next_start
        else:
            end = 10**9
        segments.append((apreg.start_seconds, end))
    return segments


def refine_session_windows_with_rito(
    session: SessionExtraction,
    events: list[RitoEvent],
    logger: Optional[logging.Logger] = None,
    video_duration_seconds: Optional[int] = None,
) -> tuple[SessionExtraction, dict[str, Any]]:
    """Corrige o plano de janelas com os eventos do rito (100% determinístico).

    1) Reativa janelas da sessão administrativa ignoradas como "cerimoniais"
       quando contêm um apregoamento real (PAs e listas tríplices são
       julgamentos válidos).
    2) Desliga janelas do bloco de julgamentos "em lista" (fora do escopo da
       base, por decisão de projeto).
    3) Funde janelas ativas que caem no MESMO segmento apregoamento→fechamento
       (fatias de voto longo); janelas de segmentos diferentes nunca se fundem.
    Não cria janelas novas e não altera nada quando não há eventos.
    """
    log = logger or logging.getLogger(__name__)
    report: dict[str, Any] = {
        "events": [event.model_dump() for event in events],
        "adjustments": [],
        "windows_before": len(session.judgments),
        "apregoamentos_individuais": count_individual_apregoamentos(events),
    }
    if not events or not session.judgments:
        report["windows_after"] = len(session.judgments)
        return session, report

    tolerance = GLOBAL_SCAN_OVERLAP_SECONDS
    updated = session.model_copy(deep=True)
    windows = updated.judgments

    def window_end(window: SessionWindow) -> int:
        if window.end_seconds is not None:
            return int(window.end_seconds)
        return int(window.start_seconds) + GLOBAL_SCAN_WINDOW_SECONDS

    apregoamentos = [e for e in events if e.kind == "apregoamento"]
    listas = [e for e in events if e.kind == "lista"]
    admin_opens = [e for e in events if e.kind == "admin_open"]
    admin_closes = [e for e in events if e.kind == "admin_close"]
    proclamacoes = sorted(e.start_seconds for e in events if e.kind == "proclamacao")

    # 0) Conteúdo pré-abertura: janelas que terminam antes do "declaro aberta a
    # sessão" são vinhetas institucionais/citações do telão — nunca julgamento
    # (origem dos "Julgamento 1" falsos de 23/06 a 01/07/2026).
    #
    # DUAS GUARDAS aprendidas em 20/08/2026 (video fEtIjOrLTH8, sessao 06/08):
    # (a) um `sessao_open` a ate 120 s de um `admin_open` e TRANSICAO para a sessao
    #     administrativa ("declaro aberta a sessao administrativa" partida entre pares
    #     de snippets — o pop de L3589 so pega quando caem no MESMO par), nao a
    #     abertura da sessao;
    # (b) video de CONTINUACAO — julgamentos reais acontecem ANTES da "abertura"
    #     detectada. O sinal e o RITO (apregoamento antes da abertura), nao o
    #     relogio: transmissao normal fica dezenas de minutos em vinheta antes do
    #     "declaro aberta" (18/08/2026: abertura aos 39 min) e o teto por % da
    #     duracao usado ate 21/08 presumia continuacao nesses videos, preservando
    #     vinheta institucional como janela de julgamento.
    # Sem as guardas, o rito descartou 4 julgamentos reais (t=540..1064) porque a
    # unica "abertura" achada estava em t=1480 de 2878 s (51% do video).
    admin_open_starts = [e.start_seconds for e in events if e.kind == "admin_open"]
    opening_starts = []
    for e in events:
        if e.kind == "admin_open":
            opening_starts.append(e.start_seconds)
        elif e.kind == "sessao_open":
            if any(abs(e.start_seconds - a) <= 120 for a in admin_open_starts):
                report["adjustments"].append(
                    {
                        "type": "sessao_open_descartado_transicao_admin",
                        "start_seconds": int(e.start_seconds),
                        "reason": "sessao_open colado em admin_open: transição para a "
                                  "sessão administrativa, não abertura da sessão.",
                    }
                )
                continue
            opening_starts.append(e.start_seconds)
    if opening_starts:
        session_opening = min(opening_starts)
        apregoamentos_antes_da_abertura = [
            e.start_seconds
            for e in apregoamentos
            if e.start_seconds < session_opening - tolerance
        ]
        if apregoamentos_antes_da_abertura:
            report["adjustments"].append(
                {
                    "type": "pre_session_anchor_ignorado_meio_do_video",
                    "start_seconds": int(session_opening),
                    "reason": f"Abertura detectada em t={session_opening}s com "
                              f"{len(apregoamentos_antes_da_abertura)} apregoamento(s) antes "
                              f"dela: vídeo de continuação — janelas anteriores preservadas.",
                }
            )
            log.info(
                "Rito: abertura em t=%ss com %s apregoamento(s) antes dela — vídeo de "
                "continuação; nenhuma janela descartada como pré-abertura.",
                session_opening, len(apregoamentos_antes_da_abertura),
            )
        else:
            for window in windows:
                if window.should_ignore:
                    continue
                if window_end(window) < session_opening - tolerance:
                    window.should_ignore = True
                    window.ignore_reason = "Conteúdo pré-abertura da sessão (rito na transcrição)."
                    report["adjustments"].append(
                        {
                            "type": "ignore_pre_session",
                            "title_hint": window.title_hint,
                            "start_seconds": int(window.start_seconds),
                            "reason": window.ignore_reason,
                        }
                    )

    # 1) Resgate da sessão administrativa.
    if admin_opens:
        admin_start = min(e.start_seconds for e in admin_opens)
        admin_end = min(
            (e.start_seconds for e in admin_closes if e.start_seconds > admin_start),
            default=None,
        )
        for window in windows:
            if not window.should_ignore:
                continue
            w_start, w_end = int(window.start_seconds), window_end(window)
            if w_end < admin_start - tolerance:
                continue
            if admin_end is not None and w_start > admin_end + tolerance:
                continue
            if any(w_start - tolerance <= e.start_seconds <= w_end + tolerance for e in apregoamentos):
                window.should_ignore = False
                window.ignore_reason = ""
                report["adjustments"].append(
                    {
                        "type": "unignore_admin",
                        "title_hint": window.title_hint,
                        "start_seconds": w_start,
                        "reason": "Apregoamento dentro da sessão administrativa (rito na transcrição).",
                    }
                )

    # 2) Julgamentos em lista: ignorados por decisão de projeto.
    for lista_event in listas:
        next_apregoamento = min(
            (
                e.start_seconds
                for e in apregoamentos
                if e.start_seconds > lista_event.start_seconds + tolerance
            ),
            default=None,
        )
        block_end = next_apregoamento if next_apregoamento is not None else 10**9
        for window in windows:
            if window.should_ignore:
                continue
            w_start = int(window.start_seconds)
            if lista_event.start_seconds - tolerance <= w_start < block_end:
                window.should_ignore = True
                window.ignore_reason = "Julgamento em lista (rito na transcrição)."
                report["adjustments"].append(
                    {
                        "type": "mark_lista",
                        "title_hint": window.title_hint,
                        "start_seconds": w_start,
                        "reason": window.ignore_reason,
                    }
                )

    # 3) Fusão de fatias de voto longo por segmento do rito.
    segments = _rito_segments(events, tolerance)

    def segment_index(window: SessionWindow) -> Optional[int]:
        midpoint = (int(window.start_seconds) + window_end(window)) / 2
        for idx, (seg_start, seg_end) in enumerate(segments):
            if seg_start - tolerance <= midpoint < seg_end:
                return idx
        return None

    result_windows: list[SessionWindow] = []
    current_group: list[SessionWindow] = []
    current_segment: Optional[int] = None

    def flush_group() -> None:
        nonlocal current_group
        if not current_group:
            return
        if len(current_group) == 1:
            result_windows.append(current_group[0])
        else:
            merged = current_group[0]
            merged.end_seconds = max(window_end(w) for w in current_group)
            merged.title_hint = next((w.title_hint for w in current_group if w.title_hint.strip()), "")
            merged.mentioned_process_numbers = dedupe_preserve_order(
                number for w in current_group for number in w.mentioned_process_numbers
            )
            result_windows.append(merged)
            report["adjustments"].append(
                {
                    "type": "merge",
                    "title_hint": merged.title_hint,
                    "start_seconds": int(merged.start_seconds),
                    "end_seconds": int(merged.end_seconds or 0),
                    "merged_windows": len(current_group),
                    "reason": "Fatias do mesmo segmento apregoamento-fechamento (rito na transcrição).",
                }
            )
            log.info(
                "Rito: %s janelas fundidas em '%s' (t=%s)",
                len(current_group),
                merged.title_hint or "sem título",
                merged.start_seconds,
            )
        current_group = []

    def has_proclamacao_between(start: int, end: int) -> bool:
        return any(start < t <= end for t in proclamacoes)

    for window in windows:
        if window.should_ignore:
            flush_group()
            current_segment = None
            result_windows.append(window)
            continue
        seg = segment_index(window)
        if seg is not None and seg == current_segment and current_group:
            # Guarda contra apregoamento perdido na legenda: proclamação entre as
            # janelas + números de processo disjuntos = julgamentos distintos,
            # mesmo que o rito detectado sugira um segmento só.
            group_numbers = {n for w in current_group for n in w.mentioned_process_numbers if n}
            new_numbers = {n for n in window.mentioned_process_numbers if n}
            if (
                group_numbers
                and new_numbers
                and group_numbers.isdisjoint(new_numbers)
                and has_proclamacao_between(
                    int(current_group[-1].start_seconds), int(window.start_seconds)
                )
            ):
                flush_group()
                current_group = [window]
                continue
            current_group.append(window)
            continue
        flush_group()
        current_segment = seg
        current_group = [window]
    flush_group()

    # 4) Estende janelas ativas ate a PROCLAMACAO do segmento. Janela que corta antes
    # da proclamacao deixa o Gemini sem o desfecho — em 13/08 e 18/08/2026 isso publicou
    # "Suspenso"/"Suspenso por vista" para julgamentos que terminaram (o 0600015-33 tinha
    # a proclamacao em t=4708 e a janela acabava em 4668). Teto de 600 s de extensao para
    # nao engolir o julgamento vizinho em caso de segmento mal fechado.
    for window in result_windows:
        if window.should_ignore:
            continue
        seg = segment_index(window)
        if seg is None:
            continue
        seg_end = segments[seg][1]
        if seg_end >= 10**9:
            continue
        w_end = window_end(window)
        if w_end < seg_end <= w_end + 600:
            report["adjustments"].append(
                {
                    "type": "extend_to_proclamacao",
                    "title_hint": window.title_hint,
                    "start_seconds": int(window.start_seconds),
                    "reason": f"end estendido de {w_end}s para {seg_end + 15}s "
                              f"(proclamação do segmento em {seg_end}s).",
                }
            )
            window.end_seconds = int(seg_end) + 15
            log.info(
                "Rito: janela '%s' estendida até a proclamação (t=%s).",
                window.title_hint or "sem título", seg_end,
            )

    updated.judgments = result_windows
    report["windows_after"] = len(result_windows)
    return updated, report


class JudgmentItemExtraction(BaseModel):
    data_sessao: str = ""
    eleicao: str = ""
    classe_processo: str = Field(
        default="",
        description=(
            "Classe processual como na autuacao/cabecalho (ex.: 'AgR-AREspe', 'ED-REspe'). "
            "Preserve prefixos de recurso interno (AgR/AgRg, ED) antes da classe-base; "
            "nao reduza 'AgR-AREspe' a 'AREspe'."
        ),
    )
    numero_processo: str = ""
    origem: str = Field(
        default="",
        description=(
            "Municipio de origem no formato 'Cidade/UF' (ex.: 'Santo Antonio do Taua/PA'). "
            "Nao use o tribunal (TRE) nem a capital quando o municipio aparecer no caso."
        ),
    )
    uf: str = ""
    tre: str = ""
    partes: list[str] = Field(default_factory=list)
    advogados: list[str] = Field(default_factory=list)
    composicao: list[str] = Field(
        default_factory=list,
        description=(
            "Ministros que participaram do julgamento deste processo; normalmente igual a "
            "composicao da sessao. Formato 'Min. <Nome>'. Nao deixe vazio quando a composicao "
            "da sessao for conhecida."
        ),
    )
    relator: str = ""
    pedido_vista: str = ""
    indicados_lista_triplice: list[str] = Field(default_factory=list)
    tema: str = ""
    punchline: str = ""
    analise_do_conteudo_juridico: str = ""
    fundamentacao_normativa: str = ""
    precedentes_citados: str = ""
    raciocinio_juridico: str = ""
    pontos_processuais_relevantes: str = ""
    efeitos_e_providencias_praticas: str = ""
    resolucoes_citadas: str = ""
    votacao: str = ""
    resultado_final: str = ""


class JudgmentBundleExtraction(BaseModel):
    title_hint: str = ""
    start_seconds: int = 0
    end_seconds: int | None = None
    should_ignore: bool = False
    ignore_reason: str = ""
    items: list[JudgmentItemExtraction] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    session: SessionExtraction
    bundles: list[JudgmentBundleExtraction] = Field(default_factory=list)


class NewsEnrichmentResult(BaseModel):
    noticia_TSE: list[str] = Field(default_factory=list)
    noticia_TRE: list[str] = Field(default_factory=list)
    noticia_geral: list[str] = Field(default_factory=list)


class InstitutionalRepairResult(BaseModel):
    urls: list[str] = Field(default_factory=list)


class ProcessMetadataResult(BaseModel):
    full_numero_processo: str = ""
    origem: str = ""
    is_judged_process: bool | None = None
    indicados_lista_triplice: list[str] = Field(default_factory=list)
    confidence: str = ""
    rationale: str = ""


class ThemeRepairResult(BaseModel):
    tema: str = ""
    confidence: str = ""
    rationale: str = ""


class ThemePunchlineRepairItem(BaseModel):
    key: str = ""
    tema: str = ""
    punchline: str = ""
    confidence: str = ""
    source_insufficient: bool = False
    reason: str = ""


class ThemePunchlineRepairBatchResult(BaseModel):
    items: list[ThemePunchlineRepairItem] = Field(default_factory=list)


class StartRefinementResult(BaseModel):
    exact_start_seconds: int | None = None
    confidence: str = ""
    reasoning: str = ""
    should_ignore: bool = False


class PublishPreviewRow(BaseModel):
    tema: str = ""
    classe_processo: str = ""
    tipo_registro: str = ""
    eleicao: str = ""
    origem: str = ""
    tribunal: str = ""
    numero_processo: str = ""
    youtube_link: str = ""
    relator: str = ""
    pedido_vista: str = ""
    resultado: str = ""
    votacao: str = ""
    data_sessao: str = ""
    partes: list[str] = Field(default_factory=list)
    advogados: list[str] = Field(default_factory=list)
    composicao: list[str] = Field(default_factory=list)
    punchline: str = ""
    analise_do_conteudo_juridico: str = ""
    fundamentacao_normativa: str = ""
    precedentes_citados: str = ""
    raciocinio_juridico: str = ""
    resolucoes_citadas: str = ""
    materia_semelhante: list[str] = Field(default_factory=list)
    noticia_TSE: str = ""
    noticia_TRE: str = ""
    noticias_gerais: list[str] = Field(default_factory=list)
    page_id: str = ""
    action: str = "create"
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    source_start_seconds: int = -1
    source_bundle_index: int = -1
    source_item_index: int = -1
    force_clear_title: bool = False
    clear_properties: list[str] = Field(default_factory=list)

    def add_warning(self, message: str) -> None:
        if message and message not in self.warnings:
            self.warnings.append(message)

    def add_error(self, message: str) -> None:
        if message and message not in self.errors:
            self.errors.append(message)

    @property
    def blocked(self) -> bool:
        return bool(self.errors)

    def to_editor_record(self) -> dict[str, Any]:
        return {
            "tema": self.tema,
            "classe_processo": self.classe_processo,
            "tipo_registro": self.tipo_registro,
            "eleicao": self.eleicao,
            "origem": self.origem,
            "tribunal": self.tribunal,
            "numero_processo": self.numero_processo,
            "youtube_link": self.youtube_link,
            "relator": self.relator,
            "pedido_vista": self.pedido_vista,
            "resultado": self.resultado,
            "votacao": self.votacao,
            "data_sessao": self.data_sessao,
            "partes": ", ".join(self.partes),
            "advogados": ", ".join(self.advogados),
            "composicao": ", ".join(self.composicao),
            "punchline": self.punchline,
            "analise_do_conteudo_juridico": self.analise_do_conteudo_juridico,
            "fundamentacao_normativa": self.fundamentacao_normativa,
            "precedentes_citados": self.precedentes_citados,
            "raciocinio_juridico": self.raciocinio_juridico,
            "resolucoes_citadas": self.resolucoes_citadas,
            "materia_semelhante": ", ".join(self.materia_semelhante),
            "noticia_TSE": self.noticia_TSE,
            "noticia_TRE": self.noticia_TRE,
            "noticias_gerais": ", ".join(self.noticias_gerais),
            "page_id": self.page_id,
            "action": self.action,
            "warnings": "\n".join(self.warnings),
            "errors": "\n".join(self.errors),
            "blocked": self.blocked,
        }

    @classmethod
    def from_editor_record(cls, record: dict[str, Any]) -> "PublishPreviewRow":
        return cls(
            tema=str(record.get("tema", "") or ""),
            classe_processo=str(record.get("classe_processo", "") or ""),
            tipo_registro=str(record.get("tipo_registro", "") or ""),
            eleicao=str(record.get("eleicao", "") or ""),
            origem=str(record.get("origem", "") or ""),
            tribunal=str(record.get("tribunal", "") or ""),
            numero_processo=str(record.get("numero_processo", "") or ""),
            youtube_link=str(record.get("youtube_link", "") or ""),
            relator=str(record.get("relator", "") or ""),
            pedido_vista=str(record.get("pedido_vista", "") or ""),
            resultado=str(record.get("resultado", "") or ""),
            votacao=str(record.get("votacao", "") or ""),
            data_sessao=str(record.get("data_sessao", "") or ""),
            partes=parse_multi_value_text(record.get("partes", "")),
            advogados=parse_multi_value_text(record.get("advogados", "")),
            composicao=parse_multi_value_text(record.get("composicao", "")),
            punchline=str(record.get("punchline", "") or ""),
            analise_do_conteudo_juridico=coerce_record_text(record.get("analise_do_conteudo_juridico")),
            fundamentacao_normativa=coerce_record_text(record.get("fundamentacao_normativa")),
            precedentes_citados=coerce_record_text(record.get("precedentes_citados")),
            raciocinio_juridico=coerce_record_text(record.get("raciocinio_juridico")),
            resolucoes_citadas=coerce_record_text(record.get("resolucoes_citadas")),
            materia_semelhante=parse_multi_value_text(record.get("materia_semelhante", "")),
            noticia_TSE=str(record.get("noticia_TSE", "") or ""),
            noticia_TRE=str(record.get("noticia_TRE", "") or ""),
            noticias_gerais=parse_multi_value_text(record.get("noticias_gerais", "")),
            page_id=str(record.get("page_id", "") or ""),
            action=str(record.get("action", "") or "create"),
            warnings=split_csv_like_text(str(record.get("warnings", "") or "").replace("\n", ", ")),
            errors=split_csv_like_text(str(record.get("errors", "") or "").replace("\n", ", ")),
        )


def _rename_payload_keys(payload: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    normalized = dict(payload)
    for source, target in mapping.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]
    return normalized


def _normalize_session_window_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _rename_payload_keys(
        payload,
        {
            "titulo": "title_hint",
            "titulo_bloco": "title_hint",
            "nome_bloco": "title_hint",
            "processo": "mentioned_process_numbers",
            "processos": "mentioned_process_numbers",
            "inicio_segundos": "start_seconds",
            "inicio_em_segundos": "start_seconds",
            "segundo_inicial": "start_seconds",
            "timestamp_inicial": "start_seconds",
            "fim_segundos": "end_seconds",
            "fim_em_segundos": "end_seconds",
            "timestamp_final": "end_seconds",
            "processos_mencionados": "mentioned_process_numbers",
            "numeros_processo": "mentioned_process_numbers",
            "numeros_dos_processos": "mentioned_process_numbers",
            "deve_ignorar": "should_ignore",
            "ignorar": "should_ignore",
            "razao_para_ignorar": "ignore_reason",
            "motivo_para_ignorar": "ignore_reason",
            "motivo": "ignore_reason",
            # 26/08/2026 — o motivo escrito pelo modelo e a informacao que distingue "julgamento
            # em lista" (ignorar) de bloco que so parece cerimonial. Sem estes aliases ele chegava
            # VAZIO e o codigo carimbava "Bloco cerimonial ou administrativo sem julgamento" por
            # cima: foi assim que as listas triplices de 25/08/2026 morreram sem deixar rastro do
            # porque. O bloco 4590-4662 do mesmo video trazia motivo_ignore="Julgamento em lista".
            "motivo_ignore": "ignore_reason",
            "motivo_do_ignore": "ignore_reason",
            "razao_ignore": "ignore_reason",
            "razao": "ignore_reason",
            "explicacao_ignore": "ignore_reason",
            "explicacao": "ignore_reason",
            "justificativa": "ignore_reason",
        },
    )
    numbers = normalized.get("mentioned_process_numbers")
    if isinstance(numbers, str):
        normalized["mentioned_process_numbers"] = split_csv_like_text(numbers)
    elif numbers is None:
        normalized["mentioned_process_numbers"] = []
    normalized["title_hint"] = normalize_model_text(normalized.get("title_hint"))
    if not normalized["title_hint"] and normalized["mentioned_process_numbers"]:
        normalized["title_hint"] = ", ".join(normalized["mentioned_process_numbers"])
    normalized["start_seconds"] = coerce_seconds(normalized.get("start_seconds"))
    end_seconds = normalized.get("end_seconds")
    normalized["end_seconds"] = None if end_seconds in {None, "", "null"} else coerce_seconds(end_seconds)
    normalized["mentioned_process_numbers"] = list(normalized.get("mentioned_process_numbers") or [])
    normalized["should_ignore"] = bool(normalized.get("should_ignore"))
    normalized["ignore_reason"] = normalize_model_text(normalized.get("ignore_reason"))
    return normalized


def _normalize_judgment_bundle_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _rename_payload_keys(
        payload,
        {
            "itens": "items",
            "processos": "items",
            "deve_ignorar": "should_ignore",
            "motivo_para_ignorar": "ignore_reason",
        },
    )
    normalized["title_hint"] = normalize_model_text(normalized.get("title_hint"))
    normalized["start_seconds"] = coerce_seconds(normalized.get("start_seconds"))
    end_seconds = normalized.get("end_seconds")
    normalized["end_seconds"] = None if end_seconds in {None, "", "null"} else coerce_seconds(end_seconds)
    normalized["should_ignore"] = bool(normalized.get("should_ignore"))
    normalized["ignore_reason"] = normalize_model_text(normalized.get("ignore_reason"))
    items = normalized.get("items")
    if isinstance(items, dict):
        items = [items]
    normalized_items: list[dict[str, Any]] = []
    for item in list(items or []):
        if not isinstance(item, dict):
            continue
        normalized_item = _rename_payload_keys(
            item,
            {
                "data_da_sessao": "data_sessao",
                "data_da_sessão": "data_sessao",
                "numero_do_processo": "numero_processo",
                "numero_cnj": "numero_processo",
                "cidade_origem": "origem",
                "cidade_de_origem": "origem",
                "estado": "uf",
                "tribunal": "tre",
                "resultado": "resultado_final",
                "resultado_do_julgamento": "resultado_final",
                "lista_triplice": "indicados_lista_triplice",
                "indicados": "indicados_lista_triplice",
                # 26/08/2026 — o DETAIL_SYSTEM_PROMPT pede os campos pelos nomes NATURAIS
                # ("pedido de vista", "pontos processuais", "efeitos praticos", "UF"...) e o
                # modelo devolve exatamente esses. Sem alias, `_rename_payload_keys` nao acha o
                # canonico e o valor e descartado em silencio: uma varredura dos 460 pares
                # raw_detail/02_judgment do acervo achou 50 pedidos de vista COM o nome do
                # ministro jogados fora (o 0601908-68 da sessao de 25/08/2026 virou "(ministro
                # nao identificado)" no Notion tendo "Min. Nunes Marques" no raw), 155 UF,
                # 120 efeitos, 101 pontos processuais, 78 resolucoes e 8 listas de indicados.
                # Mesma classe de bug que "comissao_presente" causou na composicao (ver o mapa
                # de _normalize_session_extraction_payload, logo abaixo).
                "pedido_de_vista": "pedido_vista",
                "pedido_de_vista_ministro": "pedido_vista",
                "ministro_pedido_vista": "pedido_vista",
                "ministro_do_pedido_de_vista": "pedido_vista",
                "vista": "pedido_vista",
                "UF": "uf",
                "sigla_uf": "uf",
                "pontos_processuais": "pontos_processuais_relevantes",
                "pontos_processuais_relevantes_do_caso": "pontos_processuais_relevantes",
                "efeitos_praticos": "efeitos_e_providencias_praticas",
                "efeitos_práticos": "efeitos_e_providencias_praticas",
                "efeitos_e_providencias": "efeitos_e_providencias_praticas",
                "providencias_praticas": "efeitos_e_providencias_praticas",
                "resolucoes": "resolucoes_citadas",
                "resoluções": "resolucoes_citadas",
                "resolucoes_mencionadas": "resolucoes_citadas",
                "indicados_em_lista_triplice": "indicados_lista_triplice",
                "indicados_da_lista_triplice": "indicados_lista_triplice",
            },
        )
        # jurisprudencia_citada/legislacao NAO sao apelidos: o prompt pede os quatro campos e o
        # modelo costuma preencher os dois lados com conteudo DIFERENTE (no raw_detail_07 de
        # 25/08/2026 veio jurisprudencia_citada="ADI 5274 (STF)" ao lado de precedentes_citados=
        # "RO 0600384-25"). Renomear faria o alias perder para o canonico e sumir com metade do
        # dado -- aqui os dois sao costurados, sem repetir o que ja estiver escrito.
        for origem, destino in (
            ("jurisprudencia_citada", "precedentes_citados"),
            ("jurisprudência_citada", "precedentes_citados"),
            ("legislacao", "fundamentacao_normativa"),
            ("legislação", "fundamentacao_normativa"),
        ):
            extra = normalize_model_text(normalized_item.get(origem))
            if not extra:
                continue
            atual = normalize_model_text(normalized_item.get(destino))
            if not atual:
                normalized_item[destino] = extra
            elif fold_text_for_match(extra) not in fold_text_for_match(atual):
                normalized_item[destino] = f"{atual.rstrip('. ')}. {extra}"

        for field_name in ("partes", "advogados", "composicao", "indicados_lista_triplice"):
            value = normalized_item.get(field_name)
            normalized_item[field_name] = parse_multi_value_text(value)
        # `pedido_vista` guarda NOME DE MINISTRO. O modelo as vezes responde a pergunta
        # ("houve pedido de vista?") em vez de nomear quem pediu -- ha "Sim" gravado assim
        # nos artefatos. Melhor vazio (o auto-reparo de apply_rag_consistency_checks tenta
        # garimpar o nome no texto) do que "Sim" virando uma opcao nova no select do Notion.
        vista_bruta = normalize_model_text(normalized_item.get("pedido_vista"))
        if fold_text_for_match(vista_bruta) in {
            "sim", "nao", "true", "false", "s", "n", "na", "n/a", "none", "null",
            "houve", "nenhum", "nenhuma", "sem pedido de vista", "nao houve",
        }:
            normalized_item["pedido_vista"] = ""

        for field_name in (
            "data_sessao",
            "eleicao",
            "classe_processo",
            "numero_processo",
            "origem",
            "uf",
            "tre",
            "relator",
            "pedido_vista",
            "tema",
            "punchline",
            "analise_do_conteudo_juridico",
            "fundamentacao_normativa",
            "precedentes_citados",
            "raciocinio_juridico",
            "pontos_processuais_relevantes",
            "efeitos_e_providencias_praticas",
            "resolucoes_citadas",
            "votacao",
            "resultado_final",
        ):
            normalized_item[field_name] = normalize_model_text(normalized_item.get(field_name))
        normalized_items.append(normalized_item)
    normalized["items"] = normalized_items
    return normalized


def _normalize_session_extraction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _rename_payload_keys(
        payload,
        {
            "data_da_sessao": "data_sessao",
            "data_da_sessão": "data_sessao",
            "composicao_da_sessao": "composicao",
            "composição_da_sessão": "composicao",
            "composicao_do_colegiado": "composicao",
            "composição_do_colegiado": "composicao",
            "composicao_colegiado": "composicao",
            "comissao_colegiado": "composicao",
            # o modelo escolhe livremente a chave (o GLOBAL_SYSTEM_PROMPT não a
            # nomeia); "comissao_presente" já custou a composição inteira de um
            # chunk, lida certa e descartada por não constar deste mapa.
            "comissao_presente": "composicao",
            "comissão_presente": "composicao",
            "composicao_presente": "composicao",
            "composição_presente": "composicao",
            "ministros_presentes": "composicao",
            "composicao_ministros": "composicao",
            "composição_ministros": "composicao",
            "ministros": "composicao",
            "colegiado": "composicao",
            "julgamentos": "judgments",
            "blocos": "judgments",
        },
    )
    normalized["data_sessao"] = normalize_model_text(normalized.get("data_sessao"))
    normalized["composicao"] = parse_multi_value_text(normalized.get("composicao"))
    judgments = normalized.get("judgments")
    if isinstance(judgments, list):
        normalized["judgments"] = [
            _normalize_session_window_payload(item) if isinstance(item, dict) else item
            for item in judgments
        ]
    else:
        normalized["judgments"] = []
    return normalized


def sanitize_scan_chunk_windows(
    judgments: list["SessionWindow"],
    *,
    window_start_seconds: int,
    window_end_seconds: int,
    duration_seconds: int | None = None,
) -> tuple[list["SessionWindow"], dict[str, Any]]:
    """Descarta blocos que o modelo situou FORA da janela pedida (ou do vídeo).

    A janela é o contrato da chamada: o prompt manda devolver timestamps
    ABSOLUTOS dos blocos "iniciados ou claramente identificáveis DENTRO dessa
    janela", tolerando apenas o bloco que atravessa a fronteira. Um bloco a
    milhares de segundos dali — ou depois do fim do vídeo — não é leitura, é
    degeneração. Sem esta guarda o lixo passa: tem `title_hint` e número, então
    sobrevive ao filtro de `_merge_session_chunks` e vira linha no Notion.

    Devolve (blocos_mantidos, relatorio).
    """
    lower_bound = window_start_seconds - SCAN_WINDOW_TOLERANCE_SECONDS
    upper_bound = window_end_seconds + SCAN_WINDOW_TOLERANCE_SECONDS
    if duration_seconds and duration_seconds > 0:
        upper_bound = min(upper_bound, duration_seconds + SCAN_VIDEO_END_TOLERANCE_SECONDS)

    kept: list[SessionWindow] = []
    dropped: list[dict[str, Any]] = []
    for judgment in judgments:
        start = coerce_seconds(judgment.start_seconds)
        if lower_bound <= start <= upper_bound:
            kept.append(judgment)
            continue
        dropped.append(
            {
                "start_seconds": start,
                "end_seconds": judgment.end_seconds,
                "title_hint": judgment.title_hint,
                "mentioned_process_numbers": list(judgment.mentioned_process_numbers),
                "motivo": "start_seconds fora da janela pedida"
                if not (duration_seconds and start > duration_seconds + SCAN_VIDEO_END_TOLERANCE_SECONDS)
                else "start_seconds além da duração do vídeo",
            }
        )

    report = {
        "window": [window_start_seconds, window_end_seconds],
        "bounds_aceitos": [lower_bound, upper_bound],
        "duration_seconds": duration_seconds,
        "blocos_recebidos": len(judgments),
        "blocos_mantidos": len(kept),
        "blocos_descartados": len(dropped),
        "descartados": dropped[:200],
    }
    return kept, report


def scan_chunk_degeneration_report(
    judgments: list["SessionWindow"],
    *,
    window_start_seconds: int,
    window_end_seconds: int,
) -> dict[str, Any]:
    """Assinatura de loop de repetição num chunk do scan, para diagnóstico no log.

    Três sinais independentes, nenhum deles conclusivo sozinho:
    - passo constante entre blocos consecutivos (o modelo virou contador);
    - densidade de blocos incompatível com a janela pedida;
    - números CNJ completos reprovados no dígito verificador (módulo 97).
    """
    starts = [coerce_seconds(item.start_seconds) for item in judgments]
    steps = [b - a for a, b in zip(starts, starts[1:])]
    passo_constante = bool(steps) and len(set(steps)) == 1 and steps[0] > 0

    janela = max(1, window_end_seconds - window_start_seconds)
    densidade = len(judgments) / (janela / 60.0)

    dv_validos = dv_invalidos = 0
    for judgment in judgments:
        for numero in judgment.mentioned_process_numbers:
            veredito = cnj_check_digit_is_valid(numero)
            if veredito is True:
                dv_validos += 1
            elif veredito is False:
                dv_invalidos += 1

    suspeito = len(judgments) >= SCAN_DEGENERATION_MIN_BLOCKS and (
        passo_constante or densidade > 4 or dv_invalidos > max(3, dv_validos)
    )
    return {
        "suspeito": suspeito,
        "blocos": len(judgments),
        "passo_constante_segundos": steps[0] if passo_constante else None,
        "blocos_por_minuto_da_janela": round(densidade, 2),
        "cnj_dv_validos": dv_validos,
        "cnj_dv_invalidos": dv_invalidos,
    }


def _normalize_start_refinement_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _rename_payload_keys(
        payload,
        {
            "inicio_exato_segundos": "exact_start_seconds",
            "segundo_exato_de_inicio": "exact_start_seconds",
            "deve_ignorar": "should_ignore",
            "motivo": "reasoning",
        },
    )
    exact = normalized.get("exact_start_seconds")
    normalized["exact_start_seconds"] = None if exact in {None, "", "null"} else coerce_seconds(exact)
    normalized["confidence"] = normalize_model_text(normalized.get("confidence"))
    normalized["reasoning"] = normalize_model_text(normalized.get("reasoning"))
    normalized["should_ignore"] = bool(normalized.get("should_ignore"))
    return normalized


def _normalize_process_metadata_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _rename_payload_keys(
        payload,
        {
            "numero_processo_completo": "full_numero_processo",
            "numero_cnj_completo": "full_numero_processo",
            "numero_processo_cnj": "full_numero_processo",
            "processo": "full_numero_processo",
            "julgado_na_sessao": "is_judged_process",
            "justificativa": "rationale",
        },
    )
    normalized["full_numero_processo"] = normalize_model_text(normalized.get("full_numero_processo"))
    normalized["origem"] = normalize_model_text(normalized.get("origem"))
    normalized["confidence"] = normalize_model_text(normalized.get("confidence"))
    normalized["rationale"] = normalize_model_text(normalized.get("rationale"))
    judged = normalized.get("is_judged_process")
    if isinstance(judged, str):
        lowered = judged.strip().lower()
        if lowered in {"true", "sim", "yes"}:
            judged = True
        elif lowered in {"false", "nao", "não", "no"}:
            judged = False
        else:
            judged = None
    normalized["is_judged_process"] = judged
    return normalized


def _normalize_news_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _rename_payload_keys(
        payload,
        {
            "noticias_gerais": "noticia_geral",
            "notícia_geral": "noticia_geral",
            "notícia_TSE": "noticia_TSE",
            "notícia_TRE": "noticia_TRE",
        },
    )
    for key in ("noticia_TSE", "noticia_TRE", "noticia_geral"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = split_csv_like_text(value)
        else:
            normalized[key] = list(value or [])
    return normalized


def coerce_record_text(value: Any) -> str:
    """Texto de um campo de registro do editor para colunas rich_text. Junta listas por
    ', ' em vez de gerar o repr da lista — evita ruído tipo "['a', 'b']" gravado na
    célula quando o valor chega como lista."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "")


def _coerce_gemini_response_model(response_model: type[BaseModel], response_text: str) -> BaseModel:
    model_name = response_model.__name__
    if not normalize_model_text(response_text):
        if model_name in {
            "NewsEnrichmentResult",
            "InstitutionalRepairResult",
            "ThemePunchlineRepairBatchResult",
        }:
            return response_model.model_validate({})
    try:
        # modelos às vezes embrulham o JSON em ```json ... ``` (comum no grounding); remove a cerca
        _cleaned = (response_text or "").strip()
        if _cleaned.startswith("```"):
            _cleaned = re.sub(r"^```(?:json)?\s*", "", _cleaned)
            _cleaned = re.sub(r"\s*```$", "", _cleaned).strip()
        payload = json.loads(_cleaned)
    except Exception:
        # Respostas grounded (Google Search) vêm como TEXTO, não JSON. Para os modelos
        # de busca, extrai os URLs do próprio texto (o restante vem do groundingMetadata)
        # em vez de falhar o parsing.
        if model_name in {"NewsEnrichmentResult", "InstitutionalRepairResult"}:
            text_urls = re.findall(r"https?://[^\s)>\]\"'}]+", response_text or "")
            if model_name == "NewsEnrichmentResult":
                return response_model.model_validate({"noticia_geral": text_urls})
            return response_model.model_validate({"urls": text_urls})
        if model_name == "ProcessMetadataResult":
            return response_model.model_validate({})
        # IMPORTANTE: para os demais modelos (SessionExtraction/JudgmentBundleExtraction/etc, via
        # NAO-grounded com responseMimeType=json) um JSON vazio/truncado/malformado DEVE levantar:
        # e o que aciona os retries + fallback de modelo e a salvaguarda allow_transcript_fallback
        # (evita publicar registro raso). NUNCA trocar por model_validate({}) — mascara a falha.
        return response_model.model_validate_json(response_text)

    if isinstance(payload, list):
        if model_name == "SessionExtraction" and all(isinstance(item, dict) for item in payload):
            merged_payload = {"data_sessao": "", "composicao": [], "judgments": []}
            for item in payload:
                normalized_item = _normalize_session_extraction_payload(item)
                if not merged_payload["data_sessao"] and normalized_item["data_sessao"]:
                    merged_payload["data_sessao"] = normalized_item["data_sessao"]
                merged_payload["composicao"] = dedupe_preserve_order(
                    [*merged_payload["composicao"], *normalized_item["composicao"]]
                )
                merged_payload["judgments"].extend(normalized_item["judgments"])
            payload = merged_payload
        elif model_name == "JudgmentBundleExtraction" and all(isinstance(item, dict) for item in payload):
            payload = {"items": payload}
        elif model_name == "ProcessMetadataResult":
            payload = next((item for item in payload if isinstance(item, dict)), {})
        elif model_name == "NewsEnrichmentResult":
            payload = {
                "noticia_TSE": [],
                "noticia_TRE": [],
                "noticia_geral": [str(item) for item in payload if isinstance(item, str)],
            }
        elif model_name == "InstitutionalRepairResult":
            payload = {"urls": [str(item) for item in payload if isinstance(item, str)]}
        elif model_name == "ThemePunchlineRepairBatchResult" and all(isinstance(item, dict) for item in payload):
            payload = {"items": payload}
        elif len(payload) == 1 and isinstance(payload[0], dict):
            payload = payload[0]

    if isinstance(payload, dict):
        if model_name == "SessionExtraction":
            payload = _normalize_session_extraction_payload(payload)
        elif model_name == "JudgmentBundleExtraction":
            payload = _normalize_judgment_bundle_payload(payload)
        elif model_name == "StartRefinementResult":
            payload = _normalize_start_refinement_payload(payload)
        elif model_name == "ProcessMetadataResult":
            payload = _normalize_process_metadata_payload(payload)
        elif model_name == "NewsEnrichmentResult":
            payload = _normalize_news_payload(payload)
        elif model_name == "InstitutionalRepairResult":
            urls = payload.get("urls", payload.get("url", []))
            if isinstance(urls, str):
                payload["urls"] = split_csv_like_text(urls)
            else:
                payload["urls"] = [str(item) for item in list(urls or []) if str(item).strip()]
        elif model_name == "ThemePunchlineRepairBatchResult":
            items = payload.get("items", payload.get("results", payload.get("repairs", [])))
            if isinstance(items, dict):
                items = [items]
            payload["items"] = list(items or [])

    return response_model.model_validate(payload)


@dataclass
class NotionPropertySchema:
    name: str
    type: str
    options: list[str]


@dataclass
class NotionRowMatch:
    page_id: str
    url: str


class RunArtifacts:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_youtube_url(cls, youtube_url: str) -> "RunArtifacts":
        video_id = extract_youtube_video_id(youtube_url) or "unknown-video"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return cls(ARTIFACT_ROOT / f"{timestamp}_{video_id}")

    def write_json(self, filename: str, payload: Any) -> None:
        path = self.root_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def write_text(self, filename: str, payload: str) -> None:
        path = self.root_dir / filename
        path.write_text(payload, encoding="utf-8")

    def exists(self, filename: str) -> bool:
        return (self.root_dir / filename).exists()

    def read_json(self, filename: str) -> Any:
        return json.loads((self.root_dir / filename).read_text(encoding="utf-8"))


class GeminiSessionExtractor:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        artifact_store: Optional[RunArtifacts] = None,
        logger: Optional[logging.Logger] = None,
        client: Any = None,
        enable_start_refinement: bool = ENABLE_START_REFINEMENT,
        enable_transition_refinement: bool = ENABLE_TRANSITION_REFINEMENT,
        conditional_start_refinement: bool = CONDITIONAL_START_REFINEMENT,
        allow_transcript_fallback: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY não encontrado.")
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self.artifact_store = artifact_store or RunArtifacts.for_youtube_url("unknown")
        genai, types = require_google_genai()
        self.types = types
        self.client = client or create_gemini_client(genai, types, api_key)
        self.model = resolve_gemini_model(self.client, model)
        self.model_candidates = build_gemini_model_candidates(self.client, model)
        self.disabled_models: set[str] = set()
        self.enable_start_refinement = enable_start_refinement
        self.enable_transition_refinement = enable_transition_refinement
        self.conditional_start_refinement = conditional_start_refinement
        # Salvaguarda de qualidade: quando False, a extração EXIGE o vídeo. Se o Gemini
        # não consegue processar o vídeo (ex.: vídeo recém-saído de transmissão ao vivo,
        # ainda não disponibilizado como VOD), NÃO cai no fallback de transcrição (que
        # produz registros rasos — sem número/resultado, composição inflada). Em vez disso
        # falha, marcando o vídeo como erro para ser reprocessado quando o VOD estabilizar.
        self.allow_transcript_fallback = allow_transcript_fallback
        self._transcript_snippets_cache: list[TranscriptSnippet] | None = None

    def _fetch_transcript_snippets_best_effort(self, youtube_url: str) -> list[TranscriptSnippet] | None:
        """Transcrição para a camada do rito: ausência NÃO é erro (vídeo sem legenda
        segue o fluxo atual intacto)."""
        try:
            snippets = self._get_transcript_snippets(youtube_url)
        except Exception as exc:
            self.logger.info("Sem transcrição disponível para o refinamento pelo rito: %s", exc)
            return None
        return snippets or None

    def analyze_session(self, youtube_url: str) -> AnalysisResult:
        normalized_url = normalize_youtube_link(youtube_url)
        if self.artifact_store.exists("01_session_windows.json"):
            session = SessionExtraction.model_validate(
                self.artifact_store.read_json("01_session_windows.json")
            )
        else:
            session = self._extract_session_windows(normalized_url)
            # Camada determinística do rito: corrige o plano de janelas ANTES de
            # persistir 01_session_windows.json (retomadas via cache e os índices
            # dos 02_judgment_NN não deslocam). Zero chamadas de IA.
            snippets = self._fetch_transcript_snippets_best_effort(normalized_url)
            rito_report: dict[str, Any] = {"transcript_available": bool(snippets)}
            if snippets:
                events = detect_rito_events(snippets)
                # Duracao do video: essencial para a guarda anti-"abertura no meio do
                # video" (fEtIjOrLTH8, 20/08/2026). Vem do scan; fallback: fim da
                # transcricao.
                duration: Optional[int] = None
                try:
                    if self.artifact_store.exists("00_scan_windows.json"):
                        duration = int(
                            (self.artifact_store.read_json("00_scan_windows.json") or {})
                            .get("duration_seconds") or 0
                        ) or None
                except Exception:  # noqa: BLE001
                    duration = None
                if duration is None:
                    duration = int(snippets[-1].end_seconds) or None
                session, refinement_report = refine_session_windows_with_rito(
                    session, events, logger=self.logger,
                    video_duration_seconds=duration,
                )
                rito_report.update(refinement_report)
            self.artifact_store.write_json("01b_rito_refinement.json", rito_report)
            self.artifact_store.write_json("01_session_windows.json", session.model_dump(mode="json"))

        bundles: list[JudgmentBundleExtraction] = []
        for index, window in enumerate(session.judgments, start=1):
            if window.should_ignore:
                self.logger.info("Ignorando bloco %s: %s", index, window.ignore_reason or "marcado pelo modelo")
                continue
            bundle_filename = f"02_judgment_{index:02d}.json"
            if self.artifact_store.exists(bundle_filename):
                bundle = JudgmentBundleExtraction.model_validate(
                    self.artifact_store.read_json(bundle_filename)
                )
            else:
                try:
                    bundle = self._extract_judgment_bundle(normalized_url, session, window, index)
                except Exception as exc:
                    self.logger.warning(
                        "Falha ao extrair detalhamento do bloco %s (%s); gerando placeholder bloqueável: %s",
                        index,
                        window.title_hint or "sem título",
                        exc,
                    )
                    bundle = self._build_failed_bundle_placeholder(session, window, index, exc)
                self.artifact_store.write_json(
                    bundle_filename,
                    bundle.model_dump(mode="json"),
                )
            bundles.append(bundle)
        return AnalysisResult(session=session, bundles=bundles)

    def _build_failed_bundle_placeholder(
        self,
        session: SessionExtraction,
        window: SessionWindow,
        index: int,
        exc: Exception,
    ) -> JudgmentBundleExtraction:
        placeholder_items: list[JudgmentItemExtraction] = []
        process_numbers = dedupe_preserve_order(
            canonicalize_numero_processo(number) for number in window.mentioned_process_numbers
        )
        if process_numbers:
            for process_number in process_numbers:
                placeholder_items.append(
                    JudgmentItemExtraction(
                        data_sessao=session.data_sessao,
                        numero_processo=process_number,
                        tema=window.title_hint,
                        composicao=session.composicao,
                    )
                )
        else:
            placeholder_items.append(
                JudgmentItemExtraction(
                    data_sessao=session.data_sessao,
                    tema=window.title_hint,
                    composicao=session.composicao,
                )
            )
        self.artifact_store.write_json(
            f"02_judgment_{index:02d}.error.json",
            {
                "title_hint": window.title_hint,
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
                "mentioned_process_numbers": process_numbers,
                "error": str(exc)[:2000],
            },
        )
        return JudgmentBundleExtraction(
            title_hint=window.title_hint,
            start_seconds=window.start_seconds,
            end_seconds=window.end_seconds,
            should_ignore=False,
            ignore_reason=f"Falha de extração detalhada: {str(exc)[:300]}",
            items=placeholder_items,
        )

    def _get_transcript_snippets(self, youtube_url: str) -> list[TranscriptSnippet]:
        cached = getattr(self, "_transcript_snippets_cache", None)
        if cached is not None:
            return cached

        artifact_name = "raw_transcript_fetch.json"
        if self.artifact_store.exists(artifact_name):
            payload = self.artifact_store.read_json(artifact_name)
            snippets = [
                TranscriptSnippet(
                    text=normalize_transcript_text(item.get("text")),
                    start_seconds=coerce_seconds(item.get("start_seconds")),
                    end_seconds=coerce_seconds(item.get("end_seconds")),
                )
                for item in payload.get("snippets", [])
                if normalize_transcript_text(item.get("text"))
            ]
            self._transcript_snippets_cache = snippets
            return snippets

        video_id = extract_youtube_video_id(youtube_url)
        if not video_id:
            raise RuntimeError("Não foi possível extrair o video_id para buscar a transcrição do YouTube.")

        api_cls = require_youtube_transcript_api()
        api = api_cls()
        transcript = None
        last_error: Exception | None = None
        for languages in (("pt-BR", "pt"), ("pt",), ("pt-BR", "pt", "en"), ("en",)):
            try:
                transcript = api.fetch(video_id, languages=languages, preserve_formatting=False)
                break
            except Exception as exc:
                last_error = exc
        if transcript is None:
            raise RuntimeError(f"Falha ao obter transcrição do YouTube: {last_error}") from last_error

        raw_data = transcript.to_raw_data() if hasattr(transcript, "to_raw_data") else list(transcript)
        snippets: list[TranscriptSnippet] = []
        for item in raw_data:
            text = normalize_transcript_text(item.get("text"))
            if not text:
                continue
            start_seconds = coerce_seconds(item.get("start"))
            try:
                end_seconds = max(
                    start_seconds + 1,
                    coerce_seconds(float(item.get("start", 0) or 0) + float(item.get("duration", 0) or 0)),
                )
            except Exception:
                end_seconds = start_seconds + 1
            snippets.append(
                TranscriptSnippet(
                    text=text,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
            )
        if not snippets:
            raise RuntimeError("Transcrição do YouTube vazia ou sem conteúdo útil.")
        self.artifact_store.write_json(
            artifact_name,
            {
                "video_id": video_id,
                "snippet_count": len(snippets),
                "snippets": [
                    {
                        "text": item.text,
                        "start_seconds": item.start_seconds,
                        "end_seconds": item.end_seconds,
                    }
                    for item in snippets
                ],
            },
        )
        self._transcript_snippets_cache = snippets
        return snippets

    def _extract_session_windows_from_transcript(self, youtube_url: str) -> list[SessionExtraction]:
        snippets = self._get_transcript_snippets(youtube_url)
        transcript_chunks = build_transcript_chunks(snippets)
        if not transcript_chunks:
            raise RuntimeError("Não foi possível segmentar a transcrição do vídeo em blocos úteis.")

        self.artifact_store.write_json(
            "00_transcript_windows.json",
            {
                "chunk_count": len(transcript_chunks),
                "snippets": len(snippets),
                "max_chars": TRANSCRIPT_SCAN_MAX_CHARS,
            },
        )

        extracted_chunks: list[SessionExtraction] = []
        consecutive_failures_without_success = 0
        for chunk_index, chunk in enumerate(transcript_chunks, start=1):
            self.artifact_store.write_text(
                f"raw_transcript_chunk_{chunk_index:02d}.txt",
                chunk.text,
            )
            prompt = f"""
Analise apenas a transcrição abaixo, que pertence ao próprio vídeo indicado.

URL do vídeo: {youtube_url}
Janela absoluta aproximada da transcrição: {chunk.start_seconds}s até {chunk.end_seconds}s.

Transcrição com timestamps absolutos:
{chunk.text}

Retorne:
- a data da sessão, se aparecer;
- a composição dos ministros presentes, se aparecer;
- a lista de blocos de julgamento iniciados ou claramente identificáveis nesse trecho;
- para cada bloco, o timestamp inicial em segundos ABSOLUTOS no vídeo;
- um timestamp final aproximado ABSOLUTO quando identificável;
- números de processo mencionados no bloco;
- se deve ser ignorado e por quê.

Marque como should_ignore=true qualquer bloco de julgamento em lista ou equivalente. Não confunda com LISTA TRÍPLICE (formação de lista de três nomes para vaga de TRE), que é julgamento normal: nunca marque should_ignore=true nela. Sempre que marcar should_ignore=true, escreva o motivo em ignore_reason.
"""
            try:
                chunk_result = self._call_gemini_text(
                    prompt=prompt,
                    response_model=SessionExtraction,
                    system_prompt=TRANSCRIPT_GLOBAL_SYSTEM_PROMPT,
                    artifact_name=f"raw_transcript_response_chunk_{chunk_index:02d}.txt",
                )
            except Exception as exc:
                self.logger.warning(
                    "Falha ao extrair chunk da transcrição %s/%s (%ss-%ss): %s",
                    chunk_index,
                    len(transcript_chunks),
                    chunk.start_seconds,
                    chunk.end_seconds,
                    exc,
                )
                self.artifact_store.write_json(
                    f"raw_transcript_response_chunk_{chunk_index:02d}.error.json",
                    {
                        "start_seconds": chunk.start_seconds,
                        "end_seconds": chunk.end_seconds,
                        "error": str(exc)[:2000],
                    },
                )
                consecutive_failures_without_success += 1
                if (
                    not extracted_chunks
                    and consecutive_failures_without_success >= max(TRANSCRIPT_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS, 1)
                ):
                    raise RuntimeError(
                        "Abortando varredura por transcrição após "
                        f"{consecutive_failures_without_success} falhas consecutivas sem nenhum chunk útil."
                    ) from exc
                continue
            consecutive_failures_without_success = 0

            # Mesma guarda anti-degeneração do scan por vídeo: a janela da
            # transcrição é o contrato da chamada, e o texto do trecho está no
            # próprio prompt — bloco situado longe dali não é leitura.
            kept, sanitize_report = sanitize_scan_chunk_windows(
                chunk_result.judgments,
                window_start_seconds=chunk.start_seconds,
                window_end_seconds=chunk.end_seconds,
                duration_seconds=snippets[-1].end_seconds if snippets else None,
            )
            if sanitize_report["blocos_descartados"]:
                self.logger.warning(
                    "Transcrição %s/%s (%ss-%ss): descartados %s de %s blocos fora da janela.",
                    chunk_index,
                    len(transcript_chunks),
                    chunk.start_seconds,
                    chunk.end_seconds,
                    sanitize_report["blocos_descartados"],
                    sanitize_report["blocos_recebidos"],
                )
                self.artifact_store.write_json(
                    f"raw_transcript_response_chunk_{chunk_index:02d}.descartes.json",
                    sanitize_report,
                )
            chunk_result.judgments = kept

            extracted_chunks.append(chunk_result)
            self.artifact_store.write_json(
                f"raw_transcript_response_chunk_{chunk_index:02d}.json",
                chunk_result.model_dump(mode="json"),
            )

        if not extracted_chunks:
            raise RuntimeError("Nenhum chunk da transcrição foi extraído com sucesso.")
        return extracted_chunks

    def _build_transcript_detail_chunk(
        self,
        youtube_url: str,
        *,
        start_seconds: int,
        end_seconds: int | None,
    ) -> TranscriptChunk:
        snippets = self._get_transcript_snippets(youtube_url)
        clip_start = max(0, coerce_seconds(start_seconds) - TRANSCRIPT_DETAIL_PADDING_SECONDS)
        clip_end = (
            coerce_seconds(end_seconds) + TRANSCRIPT_DETAIL_PADDING_SECONDS
            if end_seconds is not None
            else clip_start + max(REFINE_START_LOOKAHEAD_SECONDS * 2, 180)
        )
        selected = [
            item
            for item in snippets
            if item.end_seconds >= clip_start and item.start_seconds <= clip_end
        ]
        if not selected:
            raise RuntimeError("Não foi possível localizar trecho útil na transcrição para o julgamento.")

        chunk_snippets: list[TranscriptSnippet] = []
        chunk_chars = 0
        for item in selected:
            formatted = format_transcript_snippet(item)
            formatted_chars = len(formatted) + 1
            if chunk_snippets and chunk_chars + formatted_chars > TRANSCRIPT_DETAIL_MAX_CHARS:
                break
            chunk_snippets.append(item)
            chunk_chars += formatted_chars

        if not chunk_snippets:
            raise RuntimeError("Trecho da transcrição excede o limite sem conteúdo utilizável.")

        return TranscriptChunk(
            start_seconds=chunk_snippets[0].start_seconds,
            end_seconds=chunk_snippets[-1].end_seconds,
            text="\n".join(format_transcript_snippet(item) for item in chunk_snippets),
            snippet_count=len(chunk_snippets),
        )

    def _extract_session_windows(self, youtube_url: str) -> SessionExtraction:
        duration_seconds = fetch_youtube_duration_seconds(youtube_url)
        primary_windows = chunk_video_windows(duration_seconds)
        fallback_windows = chunk_video_windows(
            duration_seconds,
            window_seconds=GLOBAL_SCAN_FALLBACK_WINDOW_SECONDS,
            overlap_seconds=GLOBAL_SCAN_FALLBACK_OVERLAP_SECONDS,
        )
        self.artifact_store.write_json(
            "00_scan_windows.json",
            {
                "duration_seconds": duration_seconds,
                "plans": [
                    {
                        "label": "primary",
                        "window_seconds": GLOBAL_SCAN_WINDOW_SECONDS,
                        "overlap_seconds": GLOBAL_SCAN_OVERLAP_SECONDS,
                        "windows": primary_windows,
                    },
                    {
                        "label": "fallback",
                        "window_seconds": GLOBAL_SCAN_FALLBACK_WINDOW_SECONDS,
                        "overlap_seconds": GLOBAL_SCAN_FALLBACK_OVERLAP_SECONDS,
                        "windows": fallback_windows,
                    },
                ],
            },
        )
        last_error: Exception | None = None
        for plan_label, windows, artifact_prefix in [
            ("primary", primary_windows, "raw_global_response"),
            ("fallback", fallback_windows, "raw_global_fallback_response"),
        ]:
            try:
                extracted_chunks = self._extract_session_windows_for_plan(
                    youtube_url=youtube_url,
                    windows=windows,
                    artifact_prefix=artifact_prefix,
                    plan_label=plan_label,
                    duration_seconds=duration_seconds,
                )
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Falha na varredura global (%s): %s",
                    plan_label,
                    exc,
                )
                continue
            if extracted_chunks:
                merged = self._merge_session_chunks(extracted_chunks)
                self._preencher_vaos_com_processos_orfaos(
                    merged,
                    getattr(self, "_processos_orfaos_do_scan", []),
                    duration_seconds=duration_seconds,
                )
                return merged
        if not getattr(self, "allow_transcript_fallback", True):
            # Salvaguarda: não rebaixa para transcrição (registros rasos). Falha para que
            # o vídeo seja reprocessado por vídeo quando o Gemini conseguir baixá-lo.
            message = (
                "Extração por VÍDEO falhou em todos os planos e o fallback de transcrição "
                "está desativado (allow_transcript_fallback=False). Provável vídeo recém-saído "
                "de transmissão ao vivo, ainda não processável pelo Gemini — reprocessar por "
                "vídeo quando o VOD estabilizar."
            )
            self.logger.warning("%s", message)
            if last_error is not None:
                raise RuntimeError(message) from last_error
            raise RuntimeError(message)
        try:
            transcript_chunks = self._extract_session_windows_from_transcript(youtube_url)
        except Exception as exc:
            self.logger.warning("Falha na varredura global por transcrição: %s", exc)
            if last_error is None:
                last_error = exc
        else:
            return self._merge_session_chunks(transcript_chunks)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Nenhum chunk global foi extraído com sucesso da sessão.")

    def _extract_session_windows_for_plan(
        self,
        *,
        youtube_url: str,
        windows: list[tuple[int, int]],
        artifact_prefix: str,
        plan_label: str,
        duration_seconds: int | None = None,
    ) -> list[SessionExtraction]:
        extracted_chunks: list[SessionExtraction] = []
        consecutive_failures_without_success = 0
        blocos_descartados_no_plano = 0
        self._processos_orfaos_do_scan: list[tuple[str, int, int]] = []
        for chunk_index, (start_seconds, end_seconds) in enumerate(windows, start=1):
            prompt = f"""
Analise apenas o trecho da sessão delimitado por esta janela.

URL do vídeo: {youtube_url}
Janela absoluta: {start_seconds}s até {end_seconds}s.

Retorne:
- a data da sessão, se aparecer nesse trecho;
- a composição dos ministros presentes, se aparecer nesse trecho;
- a lista de blocos de julgamento iniciados ou claramente identificáveis dentro dessa janela;
- para cada bloco, o timestamp inicial em segundos ABSOLUTOS no vídeo;
- um timestamp final aproximado ABSOLUTO quando identificável;
- números de processo mencionados no bloco;
- se deve ser ignorado e por quê.

Se um bloco atravessar a fronteira da janela, ainda assim devolva o timestamp absoluto que conseguir identificar.
Marque como should_ignore=true qualquer bloco de "julgamento em lista" ou equivalente. Não confunda com LISTA TRÍPLICE (formação de lista de três nomes para vaga de TRE), que é julgamento normal: nunca marque should_ignore=true nela. Sempre que marcar should_ignore=true, escreva o motivo em ignore_reason.
"""
            try:
                chunk_result = self._call_gemini(
                    youtube_url=youtube_url,
                    prompt=prompt,
                    response_model=SessionExtraction,
                    system_prompt=GLOBAL_SYSTEM_PROMPT,
                    artifact_name=f"{artifact_prefix}_chunk_{chunk_index:02d}.txt",
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
            except Exception as exc:
                self.logger.warning(
                    "Falha ao extrair chunk global %s/%s (%ss-%ss); seguindo com os demais: %s",
                    plan_label,
                    chunk_index,
                    start_seconds,
                    end_seconds,
                    exc,
                )
                self.artifact_store.write_json(
                    f"{artifact_prefix}_chunk_{chunk_index:02d}.error.json",
                    {
                        "plan": plan_label,
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                        "error": str(exc)[:2000],
                    },
                )
                consecutive_failures_without_success += 1
                if (
                    not extracted_chunks
                    and consecutive_failures_without_success >= max(GLOBAL_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS, 1)
                ):
                    raise RuntimeError(
                        "Abortando varredura global "
                        f"({plan_label}) após {consecutive_failures_without_success} falhas consecutivas "
                        "sem nenhum chunk útil."
                    ) from exc
                continue
            consecutive_failures_without_success = 0

            # GUARDA ANTI-DEGENERAÇÃO: o modelo pode responder em loop, situando
            # blocos fora da janela pedida e até depois do fim do vídeo, com
            # números CNJ fabricados. Esse lixo tem title_hint e número, logo
            # sobreviveria ao filtro de _merge_session_chunks e viraria linha no
            # Notion. Corta aqui, onde a janela pedida ainda é conhecida.
            degeneration = scan_chunk_degeneration_report(
                chunk_result.judgments,
                window_start_seconds=start_seconds,
                window_end_seconds=end_seconds,
            )
            kept, sanitize_report = sanitize_scan_chunk_windows(
                chunk_result.judgments,
                window_start_seconds=start_seconds,
                window_end_seconds=end_seconds,
                duration_seconds=duration_seconds,
            )
            blocos_descartados_no_plano += sanitize_report["blocos_descartados"]
            # O numero descartado nao se perde: o chunk so viu o recorte [start, end], entao
            # se ele nomeia um processo, o processo esta ali. Guarda para
            # `_preencher_vaos_com_processos_orfaos` decidir depois, com a sessao montada,
            # se sobrou vao para ele. Chunk com assinatura de degeneracao nao entra: ali o
            # proprio numero costuma ser fabricado.
            if not degeneration["suspeito"]:
                for descartado in sanitize_report.get("descartados", []):
                    for numero in descartado.get("mentioned_process_numbers", []):
                        self._processos_orfaos_do_scan.append((numero, start_seconds, end_seconds))
            if sanitize_report["blocos_descartados"]:
                self.logger.warning(
                    "Scan %s/%s (%ss-%ss): descartados %s de %s blocos fora da janela%s.",
                    plan_label,
                    chunk_index,
                    start_seconds,
                    end_seconds,
                    sanitize_report["blocos_descartados"],
                    sanitize_report["blocos_recebidos"],
                    " (assinatura de degeneração do modelo)" if degeneration["suspeito"] else "",
                )
                self.artifact_store.write_json(
                    f"{artifact_prefix}_chunk_{chunk_index:02d}.descartes.json",
                    {"plan": plan_label, "degeneracao": degeneration, **sanitize_report},
                )
            elif degeneration["suspeito"]:
                self.logger.warning(
                    "Scan %s/%s (%ss-%ss): %s blocos dentro da janela, mas com assinatura de "
                    "degeneração (passo=%s, CNJ com DV inválido=%s). Conferir os artifacts.",
                    plan_label,
                    chunk_index,
                    start_seconds,
                    end_seconds,
                    degeneration["blocos"],
                    degeneration["passo_constante_segundos"],
                    degeneration["cnj_dv_invalidos"],
                )
            chunk_result.judgments = kept

            extracted_chunks.append(chunk_result)
            self.artifact_store.write_json(
                f"{artifact_prefix}_chunk_{chunk_index:02d}.json",
                chunk_result.model_dump(mode="json"),
            )
        if not extracted_chunks:
            raise RuntimeError(f"Nenhum chunk global foi extraído com sucesso no plano {plan_label}.")

        # Plano que respondeu, mas cujo conteúdo INTEIRO caiu na guarda de janela, é
        # plano fracassado — não resultado vazio. Sem isto o scan seguiria com zero
        # janelas em silêncio, e nem o plano de janelas curtas nem o fallback por
        # transcrição chegariam a ser tentados (eles só entram por exceção).
        # A condição exige descarte: vídeo legitimamente sem julgamento (sessão
        # solene, por exemplo) devolve zero blocos SEM descarte nenhum e continua
        # sendo um resultado válido.
        blocos_uteis = sum(
            1
            for chunk in extracted_chunks
            for judgment in chunk.judgments
            if normalize_model_text(judgment.title_hint) or judgment.mentioned_process_numbers
        )
        if blocos_uteis == 0 and blocos_descartados_no_plano > 0:
            raise RuntimeError(
                f"Plano {plan_label} não sobrou com nenhum bloco aproveitável: "
                f"{blocos_descartados_no_plano} bloco(s) foram descartados por cair fora da "
                "janela pedida (degeneração do modelo). Escalando para o próximo plano."
            )
        return extracted_chunks

    @staticmethod
    def _chave_de_processo(numero: str) -> str:
        """Chave de comparacao de numero de processo, imune ao zero a esquerda.

        `canonicalize_numero_processo` devolve "600535-60" e "0600535-60" como valores
        DIFERENTES, e comparar por ela criaria a linha fantasma que a revisao de 27/08/2026
        mediu: o mesmo julgamento entrando duas vezes por causa de um zero. O nucleo
        eleitoral tem 7 digitos + 2 de DV, entao zerar a esquerda ate 9 iguala as grafias.
        """
        digitos = re.sub(r"\D", "", canonicalize_numero_processo(numero) or "")
        if not digitos:
            return ""
        return digitos[:9].rjust(9, "0") if len(digitos) <= 9 else digitos[:9]

    def _preencher_vaos_com_processos_orfaos(
        self,
        session: SessionExtraction,
        orfaos: list[tuple[str, int, int]],
        duration_seconds: int | None = None,
        vao_minimo: int = 90,
    ) -> int:
        """Cria janela no VAO da sessao para processo que o scan nomeou e ninguem cobriu.

        A guarda `sanitize_scan_chunk_windows` descarta o bloco cujo timestamp cai fora da
        janela do chunk -- e faz bem, porque timestamp distante e a assinatura da
        degeneracao. Mas o descarte leva junto o CONTEUDO: em 27/08/2026 o chunk da janela
        810-1110 s nomeou a Lista Triplice 0600826-60 situando-a em 1500 s; o julgamento
        estava em 840-1110 s, nenhum outro chunk o viu, e ele sumiu da sessao.

        Aqui a informacao descartada volta pela porta segura. NADA que ja exista e tocado --
        nem timestamp, nem numero, nem ordem: so se cria janela no espaco LIVRE da janela
        daquele chunk, e apenas para processo que sobrou orfao. A alternativa tentada antes
        (reancorar o bloco na janela do chunk) foi descartada numa revisao adversarial que
        mediu o estrago em videos reais: o bloco reancorado puxava o inicio de julgamentos
        VERDADEIROS pelo min() do `_coalesce_windows` (o 0600504-40 de 19/08 ia de 1608 s
        para 270 s, triplicando o recorte enviado ao Gemini) e criava linhas duplicadas
        quando a grafia do numero variava.

        `orfaos`: [(numero, janela_inicio, janela_fim)] -- o numero descartado e a janela do
        chunk que o nomeou, que e onde ele de fato esta.
        """
        if not orfaos:
            return 0
        cobertos = {
            self._chave_de_processo(numero)
            for window in session.judgments
            for numero in window.mentioned_process_numbers
        }
        cobertos.discard("")
        ocupados = sorted(
            (coerce_seconds(w.start_seconds),
             coerce_seconds(w.end_seconds) if w.end_seconds is not None else coerce_seconds(w.start_seconds))
            for w in session.judgments
            if w.start_seconds is not None
        )
        criadas = 0
        for numero, janela_ini, janela_fim in orfaos:
            chave = self._chave_de_processo(numero)
            if not chave or chave in cobertos:
                continue
            limite_ini, limite_fim = coerce_seconds(janela_ini), coerce_seconds(janela_fim)
            if duration_seconds:
                limite_fim = min(limite_fim, coerce_seconds(duration_seconds))
            # MAIOR trecho livre dentro da janela do chunk (nao encurta: pula o ocupado)
            melhor = (0, 0)
            cursor = limite_ini
            for ini_ocupado, fim_ocupado in ocupados + [(limite_fim, limite_fim)]:
                if ini_ocupado >= limite_fim:
                    ini_ocupado = limite_fim
                if ini_ocupado > cursor and ini_ocupado - cursor > melhor[1] - melhor[0]:
                    melhor = (cursor, min(ini_ocupado, limite_fim))
                cursor = max(cursor, fim_ocupado)
                if cursor >= limite_fim:
                    break
            if melhor[1] - melhor[0] < vao_minimo:
                self.logger.info(
                    "Processo %s ficou sem janela propria, mas o trecho do chunk que o nomeou "
                    "ja esta ocupado por outros julgamentos — sem vao para criar.", numero)
                continue
            cobertos.add(chave)
            ocupados = sorted(ocupados + [melhor])
            session.judgments.append(
                SessionWindow(
                    title_hint=numero,
                    start_seconds=melhor[0],
                    end_seconds=melhor[1],
                    mentioned_process_numbers=[numero],
                    should_ignore=False,
                )
            )
            criadas += 1
            self.logger.info(
                "Vao %ss-%ss preenchido com o processo %s, que o scan nomeou e nenhum bloco "
                "cobria (o timestamp devolvido pelo modelo caia fora da janela do chunk).",
                melhor[0], melhor[1], numero)
        if criadas:
            session.judgments.sort(key=lambda item: (item.start_seconds, item.title_hint))
        return criadas

    def _merge_session_chunks(self, chunks: list[SessionExtraction]) -> SessionExtraction:
        merged = SessionExtraction(data_sessao="", composicao=[], judgments=[])
        preferred_dates: list[str] = []
        fallback_dates: list[str] = []
        collected_windows: list[SessionWindow] = []
        composicao_por_chunk: list[list[str]] = []

        for chunk in chunks:
            chunk_date = normalize_session_date_to_iso(normalize_model_text(chunk.data_sessao))
            composicao_por_chunk.append(
                [normalize_model_text(ministro) for ministro in chunk.composicao if normalize_model_text(ministro)]
            )
            cleaned_chunk_windows: list[SessionWindow] = []
            for judgment in chunk.judgments:
                judgment.title_hint = normalize_model_text(judgment.title_hint)
                judgment.ignore_reason = normalize_model_text(judgment.ignore_reason)
                judgment.start_seconds = coerce_seconds(judgment.start_seconds)
                if judgment.end_seconds is not None:
                    judgment.end_seconds = coerce_seconds(judgment.end_seconds)
                judgment.mentioned_process_numbers = dedupe_preserve_order(
                    canonicalize_numero_processo(number) for number in judgment.mentioned_process_numbers
                )
                if self._should_rescue_ignored_window(judgment):
                    self.logger.info(
                        "Bloco '%s' (t=%s) veio marcado para ignorar sem motivo, mas tem processo "
                        "apregoado (%s) e nenhum indicio de julgamento em lista — resgatado.",
                        judgment.title_hint or "sem titulo",
                        judgment.start_seconds,
                        ", ".join(judgment.mentioned_process_numbers),
                    )
                    judgment.should_ignore = False
                    judgment.ignore_reason = ""
                if self._should_force_ignore_window(judgment):
                    judgment.should_ignore = True
                    if not judgment.ignore_reason:
                        judgment.ignore_reason = "Bloco cerimonial ou administrativo sem julgamento."
                if not judgment.title_hint and not judgment.mentioned_process_numbers:
                    continue
                cleaned_chunk_windows.append(judgment)
                collected_windows.append(judgment)

            if chunk_date:
                fallback_dates.append(chunk_date)
                if any(not judgment.should_ignore for judgment in cleaned_chunk_windows):
                    preferred_dates.append(chunk_date)

        if preferred_dates:
            merged.data_sessao = self._pick_session_date(preferred_dates)
        elif fallback_dates:
            merged.data_sessao = self._pick_session_date(fallback_dates)
        merged.composicao = self._vote_composicao(composicao_por_chunk)
        merged.judgments = self._coalesce_windows(collected_windows)
        merged.judgments.sort(key=lambda item: (item.start_seconds, item.title_hint))
        return merged

    @staticmethod
    def _vote_composicao(composicao_por_chunk: list[list[str]]) -> list[str]:
        """Composição por MAIORIA entre os chunks, não por união.

        A composição é lida uma vez na abertura e vale para a sessão inteira, mas
        cada chunk responde à sua janela e o modelo preenche o campo de memória —
        às vezes com ministro que não está ali (numa execução real de 19/08/2026
        a união devolveu 18 nomes para um colegiado de 7, incluindo Alexandre de
        Moraes e Isabel Gallotti, cada um citado por UM único chunk). O elenco
        verdadeiro aparece em quase todas as leituras; o alucinado, em uma.
        `normalize_ministro_name` unifica antes da contagem as grafias que o ASR
        produz para o mesmo ministro (Cássio/Kássio/Kassio Nunes Marques).
        """
        leituras = [nomes for nomes in composicao_por_chunk if nomes]
        if not leituras:
            return []

        # A SAUDACAO DA ABERTURA vence a votacao. O presidente cumprimenta os ministros no
        # comeco da sessao, e a janela que cobre esse trecho traz a composicao LIDA, nao
        # lembrada. Em 25/08/2026 o chunk da abertura devolveu os 7 corretos (3 do STF, 2 do
        # STJ, 2 juristas) e a votacao os diluiu no meio de 17 leituras de memoria,
        # publicando 5 nomes -- dois deles de ministros que nem estavam na Corte.
        # `is_regimentally_valid_composicao` e o filtro: exige o desenho 3+2+2, que leitura
        # de memoria quase nunca acerta (a do chunk 06 daquele lote tinha 7 nomes e reprovou
        # por category_excess). Havendo mais de uma leitura valida, ganha a mais repetida; no
        # empate, a mais cedo no video -- ou seja, a mais proxima da abertura.
        validas: dict[tuple[str, ...], int] = {}
        ordem: dict[tuple[str, ...], int] = {}
        for indice, nomes in enumerate(leituras):
            normalizada = normalize_composition_list(
                [normalize_ministro_name(nome) for nome in nomes if normalize_ministro_name(nome)]
            )
            if not is_regimentally_valid_composicao(normalizada):
                continue
            chave = tuple(normalizada)
            validas[chave] = validas.get(chave, 0) + 1
            ordem.setdefault(chave, indice)
        if validas:
            melhor = min(validas, key=lambda ch: (-validas[ch], ordem[ch]))
            return list(melhor)

        votos: dict[str, int] = {}
        primeira_aparicao: dict[str, int] = {}
        for indice, nomes in enumerate(leituras):
            for ministro in dedupe_preserve_order(normalize_ministro_name(nome) for nome in nomes):
                if not ministro:
                    continue
                votos[ministro] = votos.get(ministro, 0) + 1
                primeira_aparicao.setdefault(ministro, indice)

        ordenados = sorted(votos, key=lambda nome: (-votos[nome], primeira_aparicao[nome]))
        # Com 1 ou 2 leituras não há maioria a apurar: preserva tudo (é o caso do
        # fallback por transcrição, que devolve poucos chunks).
        if len(leituras) < 3:
            return ordenados

        corte = max(2, math.ceil(0.25 * len(leituras)))
        eleitos = [nome for nome in ordenados if votos[nome] >= corte]
        return eleitos or ordenados

    @staticmethod
    def _pick_session_date(candidates: list[str]) -> str:
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        for index, candidate in enumerate(candidates):
            counts[candidate] = counts.get(candidate, 0) + 1
            first_seen.setdefault(candidate, index)
        return min(
            counts,
            key=lambda value: (-counts[value], first_seen[value]),
        )

    @staticmethod
    def _should_rescue_ignored_window(window: SessionWindow) -> bool:
        """O modelo marcou should_ignore SEM motivo num bloco que tem processo apregoado?

        Sessao de 25/08/2026: as listas triplices de Maceio/AL e Florianopolis/SC vieram do
        scan com should_ignore=true e ignore_reason VAZIO -- foi o codigo que carimbou
        "Bloco cerimonial ou administrativo sem julgamento" e as matou. A causa e linguistica:
        o prompt manda ignorar "julgamento em LISTA" e o modelo estende isso a "LISTA
        triplice". A lista triplice de Manaus so escapou porque um outro chunk discordou e o
        _coalesce_windows a resgatou pelo voto de minoria; as duas sem discordancia morreram.

        Resgata apenas quando ha numero de processo apregoado E nenhum indicio, no texto do
        bloco, de julgamento em lista ou de ato cerimonial. Um motivo escrito pelo modelo e
        respeitado: quem sabe por que esta ignorando nao e desautorizado aqui.
        """
        if not window.should_ignore:
            return False
        if not window.mentioned_process_numbers:
            return False
        if normalize_model_text(window.ignore_reason):
            return False
        texto = fold_text_for_match(window.title_hint)
        if not texto:
            return True
        marcadores_lista = ("julgamento em lista", "julgamentos em lista", "em lista",
                            "listas desta pauta", "lista de julgamento", "lista 1", "lista 2")
        if any(m in texto for m in marcadores_lista) and "triplice" not in texto:
            return False
        return not GeminiSessionExtractor._should_force_ignore_window(
            SessionWindow(
                title_hint=window.title_hint,
                start_seconds=window.start_seconds,
                end_seconds=window.end_seconds,
                mentioned_process_numbers=[],
                should_ignore=False,
                ignore_reason="",
            )
        )

    @staticmethod
    def _should_force_ignore_window(window: SessionWindow) -> bool:
        if window.should_ignore:
            return True
        if window.mentioned_process_numbers:
            return False
        text = " ".join(
            part for part in [window.title_hint, window.ignore_reason] if normalize_model_text(part)
        ).lower()
        if not text:
            return False
        ignore_markers = (
            "abertura do ano judiciário",
            "sessão de abertura",
            "sessao de abertura",
            "sessão solene",
            "sessao solene",
            "ano judiciário",
            "discurso",
            "recomendação",
            "recomendacao",
            "diretrizes",
            "normas de conduta",
            "encerramento",
            "leitura da ata",
            "procedimentos administrativos",
        )
        return any(marker in text for marker in ignore_markers)

    def _coalesce_windows(self, windows: list[SessionWindow]) -> list[SessionWindow]:
        merged_windows: list[SessionWindow] = []
        process_index: dict[tuple[str, ...], int] = {}

        for candidate in sorted(windows, key=lambda item: (item.start_seconds, item.title_hint)):
            process_key = tuple(candidate.mentioned_process_numbers)
            merge_index = process_index.get(process_key) if process_key else None

            if process_key and merge_index is not None:
                target = merged_windows[merge_index]
                if self._has_intervening_distinct_process_window(
                    merged_windows,
                    process_key=process_key,
                    previous_start_seconds=target.start_seconds,
                    candidate_start_seconds=candidate.start_seconds,
                ):
                    continue
                target.start_seconds = min(target.start_seconds, candidate.start_seconds)
                end_candidates = [value for value in [target.end_seconds, candidate.end_seconds] if value is not None]
                target.end_seconds = max(end_candidates) if end_candidates else None
                if not target.title_hint and candidate.title_hint:
                    target.title_hint = candidate.title_hint
                if target.should_ignore and not candidate.should_ignore:
                    target.should_ignore = False
                    target.ignore_reason = ""
                elif target.should_ignore and not target.ignore_reason and candidate.ignore_reason:
                    target.ignore_reason = candidate.ignore_reason
                continue

            if self._is_duplicate_window(merged_windows, candidate):
                continue

            merged_windows.append(candidate)
            if process_key:
                process_index[process_key] = len(merged_windows) - 1

        return merged_windows

    @staticmethod
    def _has_intervening_distinct_process_window(
        windows: list[SessionWindow],
        *,
        process_key: tuple[str, ...],
        previous_start_seconds: int,
        candidate_start_seconds: int,
    ) -> bool:
        for window in windows:
            other_key = tuple(window.mentioned_process_numbers)
            if not other_key or other_key == process_key or window.should_ignore:
                continue
            if previous_start_seconds < window.start_seconds < candidate_start_seconds:
                return True
        return False

    @staticmethod
    def _is_duplicate_window(existing: list[SessionWindow], candidate: SessionWindow) -> bool:
        candidate_key = tuple(candidate.mentioned_process_numbers)
        for item in existing:
            same_numbers = tuple(item.mentioned_process_numbers) and tuple(item.mentioned_process_numbers) == candidate_key
            same_title = (
                item.title_hint.strip()
                and candidate.title_hint.strip()
                and item.title_hint.strip().lower() == candidate.title_hint.strip().lower()
            )
            if abs(item.start_seconds - candidate.start_seconds) <= GLOBAL_SCAN_OVERLAP_SECONDS and (same_numbers or same_title):
                return True
        return False

    @staticmethod
    def _detail_end_seconds(session: SessionExtraction, window: SessionWindow) -> int | None:
        """Fim do recorte de video do bloco, com folga de cauda ate a proclamacao.

        O scan fecha a janela no ultimo voto; a proclamacao ("proclamo o resultado")
        e o anuncio de vista vem depois e ficavam FORA do recorte enviado ao Gemini.
        Estende ate VIDEO_DETAIL_TAIL_PADDING_SECONDS, com dois limites: nunca alcanca
        o inicio do proximo bloco (misturaria dois julgamentos) e nunca encurta a janela
        original. Devolve None quando o scan nao deu fim (o recorte segue ate o fim do video).
        """
        if window.end_seconds is None:
            return None
        fim = coerce_seconds(window.end_seconds)
        limite = fim + VIDEO_DETAIL_TAIL_PADDING_SECONDS
        proximos = [
            coerce_seconds(outra.start_seconds)
            for outra in session.judgments
            if outra is not window and outra.start_seconds is not None
            and coerce_seconds(outra.start_seconds) > fim
        ]
        if proximos:
            limite = min(limite, min(proximos))
        return max(fim, limite)

    def _extract_judgment_bundle(
        self,
        youtube_url: str,
        session: SessionExtraction,
        window: SessionWindow,
        index: int,
    ) -> JudgmentBundleExtraction:
        refined_start_seconds = self._refine_bundle_start_seconds(
            youtube_url=youtube_url,
            session=session,
            window=window,
            index=index,
        )
        prompt = f"""
Analise o julgamento do trecho indicado.

Contexto global:
- Data da sessão: {session.data_sessao}
- Composição da sessão: {", ".join(session.composicao)}
- Bloco {index}: {window.title_hint}
- Início em segundos: {refined_start_seconds}
- Fim em segundos: {window.end_seconds if window.end_seconds is not None else ""}
- Processos percebidos no bloco: {", ".join(window.mentioned_process_numbers)}

        Regras:
- Retorne um item por processo julgado.
- Se for julgamento conjunto, duplique as informações comuns em cada item.
- Se o bloco for apenas julgamento em lista (processos julgados em bloco), marque should_ignore=true e explique. Lista tríplice não é julgamento em lista: extraia normalmente.
- Seja fiel ao vídeo e deixe em branco o que não estiver explícito.
- Exceção quanto à composição: preencha o campo composicao de CADA item com a lista de ministros indicada em "Composição da sessão" acima (os que participaram deste julgamento), no formato "Min. <Nome>", mesmo que o trecho não a repita. Só altere se o trecho mostrar entrada, saída, ausência ou substituição de ministro para ESTE processo. Nunca deixe composicao vazia quando a composição da sessão for conhecida.
"""
        try:
            bundle = self._call_gemini(
                youtube_url=youtube_url,
                prompt=prompt,
                response_model=JudgmentBundleExtraction,
                system_prompt=DETAIL_SYSTEM_PROMPT,
                start_seconds=refined_start_seconds,
                end_seconds=self._detail_end_seconds(session, window),
                artifact_name=f"raw_detail_{index:02d}.txt",
            )
        except Exception as exc:
            if not getattr(self, "allow_transcript_fallback", True):
                # Salvaguarda: não rebaixa o bloco para transcrição. Propaga o erro para que
                # vire placeholder bloqueável (não publica registro raso para este processo).
                self.logger.warning(
                    "Falha na extração em vídeo do bloco %s; fallback de transcrição desativado: %s",
                    index,
                    exc,
                )
                raise
            self.logger.warning(
                "Falha na extração detalhada em vídeo do bloco %s; tentando fallback por transcrição: %s",
                index,
                exc,
            )
            bundle = self._extract_judgment_bundle_from_transcript(
                youtube_url=youtube_url,
                session=session,
                window=window,
                index=index,
                refined_start_seconds=refined_start_seconds,
            )
        bundle.title_hint = window.title_hint
        bundle.start_seconds = refined_start_seconds
        bundle.end_seconds = window.end_seconds
        return bundle

    def _extract_judgment_bundle_from_transcript(
        self,
        *,
        youtube_url: str,
        session: SessionExtraction,
        window: SessionWindow,
        index: int,
        refined_start_seconds: int,
    ) -> JudgmentBundleExtraction:
        transcript_chunk = self._build_transcript_detail_chunk(
            youtube_url,
            start_seconds=refined_start_seconds,
            end_seconds=window.end_seconds,
        )
        self.artifact_store.write_text(
            f"raw_detail_transcript_{index:02d}.input.txt",
            transcript_chunk.text,
        )
        transcript_prompt = f"""
Analise o julgamento abaixo com base exclusivamente na transcrição do próprio vídeo.

Contexto global:
- Data da sessão: {session.data_sessao}
- Composição da sessão: {", ".join(session.composicao)}
- Bloco {index}: {window.title_hint}
- Início em segundos: {refined_start_seconds}
- Fim em segundos: {window.end_seconds if window.end_seconds is not None else ""}
- Processos percebidos no bloco: {", ".join(window.mentioned_process_numbers)}
- Faixa efetiva da transcrição usada: {transcript_chunk.start_seconds}s até {transcript_chunk.end_seconds}s

Transcrição com timestamps absolutos:
{transcript_chunk.text}

Regras:
- Retorne um item por processo julgado.
- Se for julgamento conjunto, duplique as informações comuns em cada item.
- Se o trecho for apenas julgamento em lista (processos julgados em bloco), marque should_ignore=true e explique. Lista tríplice não é julgamento em lista: extraia normalmente.
- Seja fiel apenas ao que estiver explicitamente transcrito.
- Exceção quanto à composição: preencha o campo composicao de CADA item com a lista de ministros indicada em "Composição da sessão" acima (os que participaram deste julgamento), no formato "Min. <Nome>", mesmo que a transcrição do trecho não a repita. Só altere se a transcrição mostrar entrada, saída, ausência ou substituição de ministro para ESTE processo. Nunca deixe composicao vazia quando a composição da sessão for conhecida.
"""
        return self._call_gemini_text(
            prompt=transcript_prompt,
            response_model=JudgmentBundleExtraction,
            system_prompt=TRANSCRIPT_DETAIL_SYSTEM_PROMPT,
            artifact_name=f"raw_detail_transcript_{index:02d}.txt",
        )

    def _refine_bundle_start_seconds(
        self,
        *,
        youtube_url: str,
        session: SessionExtraction,
        window: SessionWindow,
        index: int,
    ) -> int:
        previous_admin_window = self._find_previous_administrative_window(session, window)
        if not self._should_refine_bundle_start(
            session=session,
            window=window,
            previous_admin_window=previous_admin_window,
        ):
            return window.start_seconds

        refinement_anchor = self._refinement_anchor_start_seconds(session, window)
        clip_start = max(0, refinement_anchor - REFINE_START_LOOKBACK_SECONDS)
        rough_clip_end = max(
            window.start_seconds + REFINE_START_LOOKAHEAD_SECONDS,
            refinement_anchor + REFINE_START_LOOKBACK_SECONDS + REFINE_START_LOOKAHEAD_SECONDS,
        )
        if window.end_seconds is not None:
            clip_end = min(window.end_seconds, rough_clip_end)
        else:
            clip_end = rough_clip_end
        clip_end = max(clip_start + 1, clip_end)

        prompt = f"""
Analise APENAS o início do bloco {index} para identificar o momento exato em que o julgamento começa.

Contexto:
- Data da sessão: {session.data_sessao}
- Composição da sessão: {", ".join(session.composicao)}
- Descrição do bloco: {window.title_hint}
- Estimativa atual de início: {refinement_anchor}s
- Processos percebidos: {", ".join(window.mentioned_process_numbers)}
- Janela de refinamento: {clip_start}s até {clip_end}s

Retorne o segundo ABSOLUTO exato em que o julgamento efetivamente começa.
Se esse trecho não contiver julgamento, marque should_ignore=true.
"""
        try:
            refined = self._call_gemini(
                youtube_url=youtube_url,
                prompt=prompt,
                response_model=StartRefinementResult,
                system_prompt=START_REFINEMENT_SYSTEM_PROMPT,
                artifact_name=f"raw_start_refinement_{index:02d}.txt",
                start_seconds=clip_start,
                end_seconds=clip_end,
            )
        except Exception as exc:
            self.logger.warning(
                "Falha ao refinar timestamp inicial do bloco %s; usando estimativa original (%ss): %s",
                index,
                window.start_seconds,
                exc,
            )
            return window.start_seconds

        if refined.should_ignore:
            return window.start_seconds

        if refined.exact_start_seconds is None:
            return window.start_seconds

        exact_start_seconds = coerce_seconds(refined.exact_start_seconds)
        if previous_admin_window is not None and self.enable_transition_refinement:
            transition_start = self._refine_transition_from_administrative_window(
                youtube_url=youtube_url,
                session=session,
                window=window,
                previous_window=previous_admin_window,
                index=index,
            )
            if transition_start is not None and transition_start < exact_start_seconds:
                exact_start_seconds = transition_start
        if exact_start_seconds < clip_start or exact_start_seconds > clip_end:
            return window.start_seconds
        return exact_start_seconds

    def _should_refine_bundle_start(
        self,
        *,
        session: SessionExtraction,
        window: SessionWindow,
        previous_admin_window: SessionWindow | None = None,
    ) -> bool:
        if not self.enable_start_refinement:
            return False
        if previous_admin_window is None:
            previous_admin_window = self._find_previous_administrative_window(session, window)
        if previous_admin_window is not None:
            return True
        if not self.conditional_start_refinement:
            return True
        if not window.mentioned_process_numbers:
            return True
        title_text = fold_text_for_match(window.title_hint)
        call_markers = (
            "aprego",
            "chamo a julgamento",
            "passo ao julgamento",
            "aprego o feito",
            "lista triplice",
        )
        return any(marker in title_text for marker in call_markers)

    @staticmethod
    def _refinement_anchor_start_seconds(session: SessionExtraction, window: SessionWindow) -> int:
        windows = sorted(session.judgments, key=lambda item: (item.start_seconds, item.title_hint))
        previous_window: SessionWindow | None = None
        for candidate in windows:
            if candidate is window:
                break
            if candidate.start_seconds <= window.start_seconds:
                previous_window = candidate

        if previous_window is None:
            return window.start_seconds
        if not previous_window.should_ignore:
            return window.start_seconds
        if previous_window.end_seconds is None:
            return window.start_seconds
        if window.start_seconds - previous_window.end_seconds > REFINE_START_LOOKBACK_SECONDS:
            return window.start_seconds

        previous_text = " ".join(
            value.lower()
            for value in [previous_window.title_hint, previous_window.ignore_reason]
            if normalize_model_text(value)
        )
        anchor_markers = (
            "abertura da sessão",
            "leitura da ata",
            "procedimentos administrativos",
            "sessão solene",
            "sessao solene",
            "abertura",
        )
        if any(marker in previous_text for marker in anchor_markers):
            return previous_window.start_seconds
        return window.start_seconds

    @staticmethod
    def _find_previous_administrative_window(
        session: SessionExtraction,
        window: SessionWindow,
    ) -> SessionWindow | None:
        windows = sorted(session.judgments, key=lambda item: (item.start_seconds, item.title_hint))
        previous_window: SessionWindow | None = None
        for candidate in windows:
            if candidate is window:
                break
            if candidate.start_seconds <= window.start_seconds:
                previous_window = candidate
        if previous_window is None or not previous_window.should_ignore or previous_window.end_seconds is None:
            return None
        if window.start_seconds - previous_window.end_seconds > REFINE_START_LOOKBACK_SECONDS:
            return None
        previous_text = " ".join(
            value.lower()
            for value in [previous_window.title_hint, previous_window.ignore_reason]
            if normalize_model_text(value)
        )
        anchor_markers = (
            "abertura da sessão",
            "leitura da ata",
            "procedimentos administrativos",
            "sessão solene",
            "sessao solene",
            "abertura",
        )
        if any(marker in previous_text for marker in anchor_markers):
            return previous_window
        return None

    def _refine_transition_from_administrative_window(
        self,
        *,
        youtube_url: str,
        session: SessionExtraction,
        window: SessionWindow,
        previous_window: SessionWindow,
        index: int,
    ) -> int | None:
        clip_start = previous_window.start_seconds
        rough_clip_end = max(window.start_seconds, previous_window.end_seconds or window.start_seconds)
        if window.end_seconds is not None:
            clip_end = min(window.end_seconds, rough_clip_end + REFINE_START_LOOKAHEAD_SECONDS)
        else:
            clip_end = rough_clip_end + REFINE_START_LOOKAHEAD_SECONDS
        clip_end = max(clip_start + 1, clip_end)

        prompt = f"""
Analise apenas a transição entre os atos administrativos iniciais e o começo do julgamento do bloco {index}.

Contexto:
- Data da sessão: {session.data_sessao}
- Composição da sessão: {", ".join(session.composicao)}
- Janela administrativa anterior: {previous_window.title_hint}
- Bloco jurisdicional seguinte: {window.title_hint}
- Processos percebidos: {", ".join(window.mentioned_process_numbers)}
- Janela absoluta de transição: {clip_start}s até {clip_end}s

Retorne o segundo ABSOLUTO mais cedo em que o julgamento efetivamente começa.
"""
        try:
            refined = self._call_gemini(
                youtube_url=youtube_url,
                prompt=prompt,
                response_model=StartRefinementResult,
                system_prompt=TRANSITION_START_SYSTEM_PROMPT,
                artifact_name=f"raw_transition_refinement_{index:02d}.txt",
                start_seconds=clip_start,
                end_seconds=clip_end,
            )
        except Exception as exc:
            self.logger.warning(
                "Falha ao refinar transição administrativa do bloco %s: %s",
                index,
                exc,
            )
            return None

        if refined.should_ignore or refined.exact_start_seconds is None:
            return None

        exact_start_seconds = coerce_seconds(refined.exact_start_seconds)
        if exact_start_seconds < clip_start or exact_start_seconds > clip_end:
            return None
        return exact_start_seconds

    def _call_gemini(
        self,
        *,
        youtube_url: str,
        prompt: str,
        response_model: type[BaseModel],
        system_prompt: str,
        artifact_name: str,
        start_seconds: int | None = None,
        end_seconds: int | None = None,
    ) -> BaseModel:
        return self._call_gemini_with_contents(
            contents=[
                {
                    "parts": [
                        _build_gemini_rest_part(
                            file_uri=youtube_url,
                            mime_type="video/*",
                            start_seconds=start_seconds,
                            end_seconds=end_seconds,
                        ),
                        _build_gemini_rest_part(text=prompt),
                    ]
                }
            ],
            response_model=response_model,
            system_prompt=system_prompt,
            artifact_name=artifact_name,
        )

    def _call_gemini_text(
        self,
        *,
        prompt: str,
        response_model: type[BaseModel],
        system_prompt: str,
        artifact_name: str,
    ) -> BaseModel:
        return self._call_gemini_with_contents(
            contents=[{"parts": [_build_gemini_rest_part(text=prompt)]}],
            response_model=response_model,
            system_prompt=system_prompt,
            artifact_name=artifact_name,
        )

    def _call_gemini_with_contents(
        self,
        *,
        contents: list[dict[str, Any]],
        response_model: type[BaseModel],
        system_prompt: str,
        artifact_name: str,
    ) -> BaseModel:
        last_error: Optional[Exception] = None
        candidate_models = [
            model_name for model_name in self.model_candidates if model_name not in self.disabled_models
        ]
        if not candidate_models:
            raise RuntimeError(
                "Nenhum modelo Gemini disponível para esta execução. "
                "Todos os candidatos foram desabilitados por quota ou indisponibilidade."
            )

        for model_name in candidate_models:
            for attempt in range(1, GEMINI_CALL_RETRIES + 1):
                try:
                    parsed, response_text, _ = call_gemini_generate_content_rest(
                        api_key=self.api_key,
                        model_name=model_name,
                        contents=contents,
                        system_instruction=system_prompt,
                        response_model=response_model,
                        temperature=0.1,
                        timeout_seconds=DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS,
                    )
                    self.model = model_name
                    self.artifact_store.write_text(artifact_name, response_text)
                    return parsed
                except Exception as exc:
                    last_error = exc
                    self.logger.warning(
                        "Falha ao chamar Gemini (modelo=%s, tentativa %s/%s): %s",
                        model_name,
                        attempt,
                        GEMINI_CALL_RETRIES,
                        exc,
                    )
                    if should_disable_model(exc):
                        self.disabled_models.add(model_name)
                        self.logger.warning("Fazendo fallback de modelo após erro em %s.", model_name)
                        break
                    if attempt < GEMINI_CALL_RETRIES:
                        retry_delay = extract_retry_delay_seconds(exc)
                        sleep_seconds = max(GEMINI_RETRY_BASE_DELAY ** attempt, retry_delay)
                        self.logger.warning(
                            "Aguardando %.1fs antes de nova tentativa no modelo %s.",
                            sleep_seconds,
                            model_name,
                        )
                        time.sleep(sleep_seconds)
        raise RuntimeError(f"Falha definitiva ao chamar Gemini: {last_error}") from last_error


class GeminiNewsEnricher:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_NEWS_GEMINI_MODEL,
        artifact_store: Optional[RunArtifacts] = None,
        logger: Optional[logging.Logger] = None,
        client: Any = None,
        allow_institutional_repair: bool = True,
        max_grounding_attempts: int = GEMINI_CALL_RETRIES,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY não encontrado.")
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self.artifact_store = artifact_store or RunArtifacts.for_youtube_url("unknown")
        genai, types = require_google_genai()
        self.types = types
        self.client = client or create_gemini_client(genai, types, api_key)
        self.model = model or DEFAULT_NEWS_GEMINI_MODEL
        self.allow_institutional_repair = allow_institutional_repair
        self.max_grounding_attempts = max(1, int(max_grounding_attempts or 1))

    def enrich_rows(self, rows: list[PublishPreviewRow]) -> list[PublishPreviewRow]:
        enriched_rows: list[PublishPreviewRow] = []
        for index, row in enumerate(rows, start=1):
            context = build_news_enrichment_context(row)
            cache_filename = f"06_news_enrichment_{index:02d}.json"
            if self.artifact_store.exists(cache_filename):
                cached_payload = self.artifact_store.read_json(cache_filename)
                if cached_payload.get("context") == context and not cached_payload.get("error"):
                    candidate = row.model_copy(deep=True)
                    applied = cached_payload.get("applied") or {}
                    candidate.noticia_TSE = str(applied.get("noticia_TSE", "") or "")
                    candidate.noticia_TRE = str(applied.get("noticia_TRE", "") or "")
                    candidate.noticias_gerais = list(applied.get("noticias_gerais", []) or [])
                    enriched_rows.append(candidate)
                    continue
            if not context:
                candidate = row.model_copy(deep=True)
                candidate.add_warning("Enriquecimento de notícias ignorado por falta de contexto.")
                enriched_rows.append(candidate)
                continue
            reused = self._reuse_existing_news_links(row, context=context, cache_filename=cache_filename)
            if reused is not None:
                enriched_rows.append(reused)
                continue

            uf = extract_uf_from_text(row.origem) or extract_uf_from_text(row.tribunal)
            tre_hint = f"tre-{uf.lower()}.jus.br" if uf else "tre-XX.jus.br"
            prompt = (
                "Pesquise no Google a cobertura jornalística DESTE caso/processo (âncora: número do processo, partes, tema e data).\n"
                "Devolva, quando existirem e forem do MESMO caso:\n"
                "- noticia_TSE: 1 matéria do TSE (tse.jus.br/comunicacao/noticias) sobre este julgamento.\n"
                f"- noticia_TRE: 1 matéria do TRE de origem ({tre_hint}) sobre este caso, se houver.\n"
                "- noticia_geral: até 4 matérias de imprensa (G1, Folha, Estadão, O Globo, UOL, ConJur, JOTA, Migalhas, Poder360, Metrópoles) sobre este caso.\n"
                "Não invente URLs. Não retorne páginas genéricas (home, índice/busca de notícias). Só links claramente do mesmo caso; senão, deixe vazio.\n\n"
                f"Contexto:\n{context}"
            )
            try:
                response, grounding_urls = self._call_grounded_json(
                    prompt=prompt,
                    response_model=NewsEnrichmentResult,
                    artifact_name=f"06_news_enrichment_{index:02d}.txt",
                )
            except Exception as exc:
                candidate = row.model_copy(deep=True)
                candidate.add_warning("Enriquecimento de notícias falhou; mantendo linha sem novas notícias.")
                enriched_rows.append(candidate)
                self.artifact_store.write_json(
                    f"06_news_enrichment_{index:02d}.json",
                    {
                        "context": context,
                        "error": str(exc),
                        "applied": {
                            "noticia_TSE": candidate.noticia_TSE,
                            "noticia_TRE": candidate.noticia_TRE,
                            "noticias_gerais": candidate.noticias_gerais,
                        },
                        "model": getattr(self, "model", DEFAULT_NEWS_GEMINI_MODEL),
                    },
                )
                continue
            tse_urls, tre_urls, general_urls = classify_news_urls(
                response.noticia_TSE + response.noticia_TRE + response.noticia_geral + grounding_urls
            )
            valid_tse_urls, dropped_tse_urls, irrelevant_tse_urls = filter_relevant_institutional_news_urls(tse_urls, row)
            valid_tre_urls, dropped_tre_urls, irrelevant_tre_urls = filter_relevant_institutional_news_urls(tre_urls, row)
            if getattr(self, "allow_institutional_repair", True) and not valid_tse_urls and dropped_tse_urls:
                repaired_tse_urls = self._repair_institutional_urls(
                    context=context,
                    broken_urls=dropped_tse_urls,
                    domain_hint="tse.jus.br",
                    domain_label="TSE",
                    artifact_name=f"06_news_repair_tse_{index:02d}.txt",
                )
                # O reparo pode devolver URL de OUTRO dominio (em 20/08/2026 um link do
                # mpf.mp.br entrou como noticia_TSE por aqui). So tse.jus.br sobrevive.
                repaired_tse_urls = [
                    u for u in repaired_tse_urls
                    if TSE_DOMAIN_RE.search(domain_from_url(u) or "")
                ]
                valid_tse_urls, extra_dropped_tse_urls, extra_irrelevant_tse_urls = filter_relevant_institutional_news_urls(repaired_tse_urls, row)
                dropped_tse_urls = dedupe_preserve_order(dropped_tse_urls + extra_dropped_tse_urls)
                irrelevant_tse_urls = dedupe_preserve_order(irrelevant_tse_urls + extra_irrelevant_tse_urls)
            if getattr(self, "allow_institutional_repair", True) and not valid_tre_urls and dropped_tre_urls:
                repaired_tre_urls = self._repair_institutional_urls(
                    context=context,
                    broken_urls=dropped_tre_urls,
                    domain_hint=domain_from_url(dropped_tre_urls[0]) or "tre-xx.jus.br",
                    domain_label="TRE",
                    artifact_name=f"06_news_repair_tre_{index:02d}.txt",
                )
                # mesma guarda de dominio do ramo TSE: so tre-XX.jus.br sobrevive ao reparo
                repaired_tre_urls = [
                    u for u in repaired_tre_urls
                    if TRE_DOMAIN_RE.search(domain_from_url(u) or "")
                ]
                valid_tre_urls, extra_dropped_tre_urls, extra_irrelevant_tre_urls = filter_relevant_institutional_news_urls(repaired_tre_urls, row)
                dropped_tre_urls = dedupe_preserve_order(dropped_tre_urls + extra_dropped_tre_urls)
                irrelevant_tre_urls = dedupe_preserve_order(irrelevant_tre_urls + extra_irrelevant_tre_urls)
            filtered_general_urls = filter_general_news_urls(general_urls, row)
            candidate = row.model_copy(deep=True)
            candidate.noticia_TSE = valid_tse_urls[0] if valid_tse_urls else ""
            candidate.noticia_TRE = valid_tre_urls[0] if valid_tre_urls else ""
            candidate.noticias_gerais = filtered_general_urls[:GENERAL_NEWS_LIMIT]
            if dropped_tse_urls:
                candidate.add_warning(
                    f"{len(dropped_tse_urls)} link(s) instituciona(is) do TSE descartado(s) por indisponibilidade."
                )
            if dropped_tre_urls:
                candidate.add_warning(
                    f"{len(dropped_tre_urls)} link(s) instituciona(is) de TRE descartado(s) por indisponibilidade."
                )
            if irrelevant_tse_urls:
                candidate.add_warning(
                    f"{len(irrelevant_tse_urls)} link(s) instituciona(is) do TSE descartado(s) por baixa aderência ao caso."
                )
            if irrelevant_tre_urls:
                candidate.add_warning(
                    f"{len(irrelevant_tre_urls)} link(s) instituciona(is) de TRE descartado(s) por baixa aderência ao caso."
                )
            dropped_general_urls = [url for url in general_urls if url not in filtered_general_urls]
            if dropped_general_urls:
                candidate.add_warning(
                    f"{len(dropped_general_urls)} link(s) geral(is) descartado(s) por baixa aderência ao caso."
                )
            if len(filtered_general_urls) > GENERAL_NEWS_LIMIT:
                candidate.add_warning(
                    f"Mais de {GENERAL_NEWS_LIMIT} notícias gerais encontradas; mantendo apenas as primeiras {GENERAL_NEWS_LIMIT}."
                )
            enriched_rows.append(candidate)
            self.artifact_store.write_json(
                f"06_news_enrichment_{index:02d}.json",
                {
                    "context": context,
                    "parsed": response.model_dump(mode="json"),
                    "grounding_urls": grounding_urls,
                    "applied": {
                        "noticia_TSE": candidate.noticia_TSE,
                        "noticia_TRE": candidate.noticia_TRE,
                        "noticias_gerais": candidate.noticias_gerais,
                    },
                    "tse_filtered_out": dropped_tse_urls,
                    "tre_filtered_out": dropped_tre_urls,
                    "tse_irrelevant": irrelevant_tse_urls,
                    "tre_irrelevant": irrelevant_tre_urls,
                    "general_candidates": general_urls,
                    "general_filtered_out": dropped_general_urls,
                    "model": getattr(self, "model", DEFAULT_NEWS_GEMINI_MODEL),
                },
            )
        return enriched_rows

    def _reuse_existing_news_links(
        self,
        row: PublishPreviewRow,
        *,
        context: str,
        cache_filename: str,
    ) -> PublishPreviewRow | None:
        existing_urls = [row.noticia_TSE, row.noticia_TRE] + list(row.noticias_gerais or [])
        if not any(normalize_external_url(value) for value in existing_urls):
            return None
        tse_urls, tre_urls, general_urls = classify_news_urls(existing_urls)
        valid_tse_urls, dropped_tse_urls, irrelevant_tse_urls = filter_relevant_institutional_news_urls(tse_urls, row)
        valid_tre_urls, dropped_tre_urls, irrelevant_tre_urls = filter_relevant_institutional_news_urls(tre_urls, row)
        filtered_general_urls = filter_general_news_urls(general_urls, row)
        if not valid_tse_urls and not valid_tre_urls and not filtered_general_urls:
            return None
        candidate = row.model_copy(deep=True)
        candidate.noticia_TSE = valid_tse_urls[0] if valid_tse_urls else ""
        candidate.noticia_TRE = valid_tre_urls[0] if valid_tre_urls else ""
        candidate.noticias_gerais = filtered_general_urls[:GENERAL_NEWS_LIMIT]
        if dropped_tse_urls or dropped_tre_urls:
            candidate.add_warning("Notícias existentes indisponíveis foram descartadas antes de chamar grounding.")
        if irrelevant_tse_urls or irrelevant_tre_urls:
            candidate.add_warning("Notícias institucionais existentes sem aderência suficiente foram descartadas.")
        self.artifact_store.write_json(
            cache_filename,
            {
                "context": context,
                "skipped_grounding": True,
                "reason": "existing_relevant_news_links",
                "applied": {
                    "noticia_TSE": candidate.noticia_TSE,
                    "noticia_TRE": candidate.noticia_TRE,
                    "noticias_gerais": candidate.noticias_gerais,
                },
                "existing_filtered_out": dedupe_preserve_order(
                    dropped_tse_urls + dropped_tre_urls + irrelevant_tse_urls + irrelevant_tre_urls
                ),
            },
        )
        return candidate

    def _repair_institutional_urls(
        self,
        *,
        context: str,
        broken_urls: list[str],
        domain_hint: str,
        domain_label: str,
        artifact_name: str,
    ) -> list[str]:
        if not broken_urls:
            return []
        prompt = (
            f"Os links abaixo aparentam estar quebrados ou truncados no domínio {domain_hint}.\n"
            f"Usando Google Search, encontre a URL canônica pública e válida do {domain_label} para a mesma notícia institucional.\n"
            "Não invente slug e não retorne páginas de erro.\n"
            f"Retorne apenas URLs do domínio {domain_hint} relacionadas ao mesmo caso.\n\n"
            f"Contexto:\n{context}\n\n"
            "Links quebrados candidatos:\n"
            + "\n".join(f"- {url}" for url in broken_urls)
        )
        try:
            response, grounding_urls = self._call_grounded_json(
                prompt=prompt,
                response_model=InstitutionalRepairResult,
                artifact_name=artifact_name,
            )
        except Exception:
            return []
        return normalize_external_url_list(response.urls + grounding_urls)

    def _call_grounded_json(
        self,
        *,
        prompt: str,
        response_model: type[BaseModel],
        artifact_name: str,
    ) -> tuple[BaseModel, list[str]]:
        last_error: Optional[Exception] = None
        max_attempts = max(1, int(getattr(self, "max_grounding_attempts", GEMINI_CALL_RETRIES) or 1))
        for attempt in range(1, max_attempts + 1):
            try:
                parsed, response_text, response_payload = call_gemini_generate_content_rest(
                    api_key=self.api_key,
                    model_name=self.model,
                    contents=[{"parts": [_build_gemini_rest_part(text=prompt)]}],
                    system_instruction=NEWS_ENRICHMENT_SYSTEM_PROMPT,
                    response_model=response_model,
                    temperature=0.1,
                    use_google_search=True,
                    timeout_seconds=DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS,
                )
                self.artifact_store.write_text(artifact_name, response_text)
                return parsed, _extract_generate_content_grounding_urls(response_payload)
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Falha no enriquecimento com Google Search (tentativa %s/%s): %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                if should_disable_model(exc):
                    break
                if attempt < max_attempts:
                    retry_delay = extract_retry_delay_seconds(exc)
                    time.sleep(max(GEMINI_RETRY_BASE_DELAY ** attempt, retry_delay))
        raise RuntimeError(f"Falha definitiva no enriquecimento de notícias: {last_error}") from last_error

    @staticmethod
    def _extract_grounding_urls(response: Any) -> list[str]:
        urls: list[str] = []
        try:
            candidates = getattr(response, "candidates", None) or []
            first_candidate = candidates[0] if candidates else None
            grounding_metadata = getattr(first_candidate, "grounding_metadata", None)
            grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
            for chunk in grounding_chunks:
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", "") if web is not None else ""
                normalized = normalize_external_url(uri)
                if normalized:
                    urls.append(normalized)
        except Exception:
            return []
        return dedupe_preserve_order(urls)


class GeminiThemePunchlineEnricher:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        artifact_store: Optional[RunArtifacts] = None,
        logger: Optional[logging.Logger] = None,
        batch_size: int = 10,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY não encontrado.")
        self.api_key = api_key
        self.model = model or DEFAULT_GEMINI_MODEL
        self.artifact_store = artifact_store or RunArtifacts.for_youtube_url("unknown")
        self.logger = logger or logging.getLogger(__name__)
        self.batch_size = max(1, int(batch_size or 10))

    def enrich_rows(self, rows: list[PublishPreviewRow]) -> list[PublishPreviewRow]:
        enriched_rows: list[PublishPreviewRow] = []
        for batch_number, start in enumerate(range(0, len(rows), self.batch_size), start=1):
            batch = rows[start: start + self.batch_size]
            payload = [
                build_theme_punchline_repair_payload(row, key=f"row_{start + offset + 1:03d}")
                for offset, row in enumerate(batch)
            ]
            cache_filename = f"04b_theme_punchline_{batch_number:02d}.json"
            if self.artifact_store.exists(cache_filename):
                cached_payload = self.artifact_store.read_json(cache_filename)
                applied = cached_payload.get("applied")
                if cached_payload.get("payload") == payload and applied:
                    enriched_rows.extend(PublishPreviewRow.model_validate(item) for item in applied)
                    continue

            try:
                parsed = self._call_batch(
                    payload=payload,
                    artifact_name=f"04b_theme_punchline_{batch_number:02d}.txt",
                )
                items_by_key = {normalize_model_text(item.key): item for item in parsed.items}
                applied_rows = [
                    self._apply_repair_item(row, items_by_key.get(normalize_model_text(item_payload["key"])))
                    for row, item_payload in zip(batch, payload)
                ]
                self.artifact_store.write_json(
                    cache_filename,
                    {
                        "payload": payload,
                        "parsed": parsed.model_dump(mode="json"),
                        "applied": [row.model_dump(mode="json") for row in applied_rows],
                    },
                )
            except Exception as exc:
                self.logger.warning(
                    "Falha no reparo editorial de tema/punchline; aplicando fallback local: %s",
                    exc,
                )
                applied_rows = [self._apply_repair_item(row, None, error=str(exc)) for row in batch]
                self.artifact_store.write_json(
                    cache_filename,
                    {
                        "payload": payload,
                        "error": str(exc),
                        "applied": [row.model_dump(mode="json") for row in applied_rows],
                    },
                )
            enriched_rows.extend(applied_rows)
        return enriched_rows

    def _call_batch(
        self,
        *,
        payload: list[dict[str, Any]],
        artifact_name: str,
    ) -> ThemePunchlineRepairBatchResult:
        prompt = (
            "Revise os campos `tema` e `punchline` dos itens abaixo.\n"
            "Preserve a fidelidade ao contexto fornecido e retorne JSON no schema solicitado.\n\n"
            f"ITENS:\n{json.dumps({'items': payload}, ensure_ascii=False, indent=2)}"
        )
        last_error: Optional[Exception] = None
        for attempt in range(1, GEMINI_CALL_RETRIES + 1):
            try:
                parsed, response_text, _ = call_gemini_generate_content_rest(
                    api_key=self.api_key,
                    model_name=self.model,
                    contents=[{"parts": [_build_gemini_rest_part(text=prompt)]}],
                    system_instruction=THEME_PUNCHLINE_REPAIR_SYSTEM_PROMPT,
                    response_model=ThemePunchlineRepairBatchResult,
                    temperature=0.2,
                    timeout_seconds=DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS,
                )
                self.artifact_store.write_text(artifact_name, response_text)
                return parsed
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Falha no reparo Gemini de tema/punchline (tentativa %s/%s): %s",
                    attempt,
                    GEMINI_CALL_RETRIES,
                    exc,
                )
                if should_disable_model(exc):
                    break
                if attempt < GEMINI_CALL_RETRIES:
                    retry_delay = extract_retry_delay_seconds(exc)
                    time.sleep(max(GEMINI_RETRY_BASE_DELAY ** attempt, retry_delay))
        raise RuntimeError(f"Falha definitiva no reparo Gemini de tema/punchline: {last_error}") from last_error

    def _apply_repair_item(
        self,
        row: PublishPreviewRow,
        item: ThemePunchlineRepairItem | None,
        *,
        error: str = "",
    ) -> PublishPreviewRow:
        candidate = row.model_copy(deep=True)
        proposed_theme = clean_theme_punchline_theme(item.tema if item else "", candidate)
        if not proposed_theme:
            proposed_theme = build_fallback_tema(candidate)
        if proposed_theme:
            candidate.tema = proposed_theme

        proposed_punchline = clean_theme_punchline_punchline(item.punchline if item else "", candidate)
        if (
            not proposed_punchline
            or len(proposed_punchline) < 90
            or theme_punchline_pair_too_similar(candidate.tema, proposed_punchline)
        ):
            proposed_punchline = build_editorial_punchline_fallback(candidate, candidate.tema)
        if proposed_punchline:
            candidate.punchline = proposed_punchline

        if item and item.source_insufficient:
            candidate.add_warning("Reparo de tema/punchline marcou fonte insuficiente; texto preserva apenas inferências locais.")
        if error:
            candidate.add_warning("Reparo Gemini de tema/punchline falhou; aplicado fallback local.")
        if theme_punchline_pair_needs_rewrite(candidate):
            candidate.add_warning("Tema/punchline ainda requerem revisão editorial manual.")
        return candidate


class GeminiProcessMetadataEnricher:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        artifact_store: Optional[RunArtifacts] = None,
        logger: Optional[logging.Logger] = None,
        client: Any = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY não encontrado.")
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self.artifact_store = artifact_store or RunArtifacts.for_youtube_url("unknown")
        genai, types = require_google_genai()
        self.types = types
        self.client = client or create_gemini_client(genai, types, api_key)
        self.model = model or DEFAULT_GEMINI_MODEL

    def enrich_rows(self, rows: list[PublishPreviewRow]) -> list[PublishPreviewRow]:
        enriched_rows: list[PublishPreviewRow] = []
        for index, row in enumerate(rows, start=1):
            cache_filename = f"04a_process_metadata_{index:02d}.json"
            if self.artifact_store.exists(cache_filename):
                cached_payload = self.artifact_store.read_json(cache_filename)
                applied = cached_payload.get("applied")
                if applied:
                    enriched_rows.append(PublishPreviewRow.model_validate(applied))
                    continue
            has_full_cnj = bool(extract_full_cnj(row.numero_processo))
            has_origem = bool(normalize_model_text(row.origem))
            # Lista tríplice: os indicados (advogados/juristas) são públicos mas não vêm no
            # vídeo nem na API do CNJ; buscar via grounding quando 'partes' está vazio, mesmo
            # que o número CNJ já esteja completo (o gate normal pularia o item).
            needs_triplice_partes = _row_is_lista_triplice(row) and not row.partes
            if not needs_triplice_partes:
                if has_full_cnj and has_origem:
                    enriched_rows.append(row)
                    continue
                if has_full_cnj:
                    enriched_rows.append(row)
                    continue

            context = build_process_metadata_context(row)
            if not context:
                enriched_rows.append(row)
                continue

            prompt = (
                "Analise o item processual abaixo.\n"
                "Complete prioritariamente o número CNJ integral.\n"
                "Só informe a cidade/UF de origem quando ela vier com clareza na mesma evidência relevante do processo.\n"
                "Se o número pesquisado aparecer apenas como precedente citado, marque is_judged_process=false.\n"
            )
            if needs_triplice_partes:
                prompt += (
                    "Este item é uma LISTA TRÍPLICE: pesquise e devolva em indicados_lista_triplice "
                    "os nomes completos das pessoas que compõem a lista (em regra três advogados/"
                    "advogadas), conforme fonte oficial (DJe, TSE/TRE/TJ) ou imprensa jurídica; se "
                    "não localizar com segurança, devolva lista vazia.\n"
                )
            prompt += f"\nContexto:\n{context}"
            candidate = row.model_copy(deep=True)
            try:
                response = self._call_grounded_json(
                    prompt=prompt,
                    response_model=ProcessMetadataResult,
                    artifact_name=f"04a_process_metadata_{index:02d}.txt",
                )
            except Exception as exc:
                candidate.add_warning(
                    "Metadados processuais não enriquecidos por falha no grounding; mantendo dados do vídeo."
                )
                self.artifact_store.write_json(
                    f"04a_process_metadata_{index:02d}.json",
                    {
                        "context": context,
                        "error": str(exc),
                        "applied": candidate.model_dump(mode="json"),
                    },
                )
                enriched_rows.append(candidate)
                continue
            if response.full_numero_processo and not has_full_cnj:
                candidate.numero_processo = response.full_numero_processo
            if response.origem:
                candidate.origem = response.origem
            if needs_triplice_partes:
                indicados = _clean_triplice_indicados(response.indicados_lista_triplice)
                if indicados and not _metadata_confidence_blocks_write(response.confidence):
                    candidate.partes = indicados
                    candidate.add_warning(
                        "partes (indicados da lista tríplice) preenchidas via busca Google; confira os nomes."
                    )
            elif response.is_judged_process is False:
                if _row_has_strong_local_judgment_evidence(candidate, self.artifact_store):
                    candidate.add_warning(
                        "Grounding indicou precedente citado, mas o próprio vídeo traz prova local forte do julgamento; mantendo item."
                    )
                elif _grounding_rationale_is_inconclusive(response.rationale):
                    # A busca não achar o processo NÃO prova que ele não foi julgado:
                    # PAs administrativos quase não têm cobertura e o buscador chegou a
                    # tratar a data da sessão como "futura" (caso do PA 0600904-54 em
                    # 23/06/2026, descartado por engano). Só descartamos quando o
                    # grounding aponta positivamente o número como precedente citado.
                    candidate.add_warning(
                        "Grounding não localizou o processo (número incompleto/data tida como futura); item mantido para vistoria."
                    )
                else:
                    candidate.add_error(
                        "Busca Google indicou que o número consultado aparece como precedente citado, não como processo julgado."
                    )
            enriched_rows.append(candidate)
            self.artifact_store.write_json(
                f"04a_process_metadata_{index:02d}.json",
                {
                    "context": context,
                    "parsed": response.model_dump(mode="json"),
                    "applied": candidate.model_dump(mode="json"),
                },
            )
        return enriched_rows

    def _call_grounded_json(
        self,
        *,
        prompt: str,
        response_model: type[BaseModel],
        artifact_name: str,
    ) -> BaseModel:
        last_error: Optional[Exception] = None
        for attempt in range(1, GEMINI_CALL_RETRIES + 1):
            try:
                parsed, response_text, _ = call_gemini_generate_content_rest(
                    api_key=self.api_key,
                    model_name=self.model,
                    contents=[{"parts": [_build_gemini_rest_part(text=prompt)]}],
                    system_instruction=PROCESS_METADATA_SYSTEM_PROMPT,
                    response_model=response_model,
                    temperature=0.1,
                    use_google_search=True,
                    timeout_seconds=DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS,
                )
                self.artifact_store.write_text(artifact_name, response_text)
                # Grounding responde em prosa: se o schema saiu vazio mas há texto, faz uma
                # 2ª passada SEM busca para estruturar o conteúdo já pesquisado.
                if (
                    response_model is ProcessMetadataResult
                    and _process_metadata_result_is_empty(parsed)
                    and normalize_model_text(response_text)
                ):
                    try:
                        parsed = self._structure_grounded_text(response_text, response_model)
                    except Exception as exc:  # estruturação é best-effort; mantém o vazio
                        self.logger.warning("Falha ao estruturar texto grounded: %s", exc)
                return parsed
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Falha no enriquecimento de metadados processuais com Google Search (tentativa %s/%s): %s",
                    attempt,
                    GEMINI_CALL_RETRIES,
                    exc,
                )
                if should_disable_model(exc):
                    break
                if attempt < GEMINI_CALL_RETRIES:
                    retry_delay = extract_retry_delay_seconds(exc)
                    time.sleep(max(GEMINI_RETRY_BASE_DELAY ** attempt, retry_delay))
        raise RuntimeError(f"Falha definitiva no enriquecimento de metadados processuais: {last_error}") from last_error

    def _structure_grounded_text(self, text: str, response_model: type[BaseModel]) -> BaseModel:
        """2ª passada SEM grounding: estrutura o texto já pesquisado no schema (chaves explícitas)."""
        parsed, _, _ = call_gemini_generate_content_rest(
            api_key=self.api_key,
            model_name=self.model,
            contents=[{"parts": [_build_gemini_rest_part(text=PROCESS_METADATA_EXTRACTION_PROMPT + text)]}],
            system_instruction="Você converte um texto factual já pesquisado em JSON estruturado, sem acrescentar dados ausentes.",
            response_model=response_model,
            temperature=0.0,
            use_google_search=False,
            timeout_seconds=DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS,
        )
        return parsed


class NotionDataSourceSchema:
    def __init__(self, data_source_id: str, raw_payload: dict[str, Any]) -> None:
        self.data_source_id = data_source_id
        self.raw_payload = raw_payload
        self.properties: dict[str, NotionPropertySchema] = {}
        self.title_property_name = ""

        for property_name, prop in (raw_payload.get("properties") or {}).items():
            prop_type = prop.get("type", "")
            options: list[str] = []
            if prop_type in {"select", "status"}:
                options = [opt.get("name", "") for opt in prop.get(prop_type, {}).get("options", [])]
            elif prop_type == "multi_select":
                options = [opt.get("name", "") for opt in prop.get("multi_select", {}).get("options", [])]
            self.properties[property_name] = NotionPropertySchema(
                name=property_name,
                type=prop_type,
                options=[opt for opt in options if opt],
            )
            if prop_type == "title":
                self.title_property_name = property_name

    def ensure_expected_properties(self) -> None:
        missing = sorted(EXPECTED_NOTION_PROPERTIES - set(self.properties))
        if missing:
            raise RuntimeError(
                "O data source do Notion não contém as propriedades esperadas: "
                + ", ".join(missing)
            )
        if not self.title_property_name:
            raise RuntimeError("Não foi encontrada a propriedade title do data source.")

    def property(self, property_name: str) -> NotionPropertySchema:
        if property_name == "tema" and self.title_property_name == "tema":
            return self.properties[self.title_property_name]
        notion_name = NOTION_PROPERTY_MAP.get(property_name, property_name)
        return self.properties[notion_name]


class NotionSessoesClient:
    def __init__(
        self,
        api_key: str,
        data_source_id: str = DEFAULT_NOTION_DATA_SOURCE_ID,
        notion_version: str = DEFAULT_NOTION_VERSION,
        logger: Optional[logging.Logger] = None,
        session: Optional[requests.Session] = None,
        # SEGURANCA: default False. Quando True, a publicacao (_write_row_once) roda
        # ensure_multiselect_options_default (PATCH em options) — que ja apagou etiquetas
        # de advogados uma vez. So scripts de MANUTENCAO que realmente queiram normalizar
        # cores de options devem passar True explicitamente. Fecha o footgun de um
        # entrypoint futuro esquecer de passar False.
        normalize_multiselect_colors_post_write: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("NOTION_API_KEY/NOTION_TOKEN não encontrado.")
        self.data_source_id = data_source_id
        self.logger = logger or logging.getLogger(__name__)
        self.session = session or requests.Session()
        self.normalize_multiselect_colors_post_write = normalize_multiselect_colors_post_write
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Notion-Version": notion_version,
                "Content-Type": "application/json",
            }
        )
        self.base_url = "https://api.notion.com/v1"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response: requests.Response | None = None
        # 6 tentativas com backoff ate ~30 s (total ~60 s): as rajadas de
        # ConnectionResetError 10054 / 504 do Notion duram MINUTOS (medido em
        # 20/08/2026, quando o backoff de ~9 s derrubou uma publicacao no meio).
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                response = self.session.request(method, self.base_url + path, timeout=60, **kwargs)
            except requests.RequestException:
                if attempt >= max_attempts - 1:
                    raise
                time.sleep(min(30.0, 2.0 * (2 ** attempt)))
                continue
            if response.status_code == 429 and attempt < max_attempts - 1:
                retry_after = response.headers.get("Retry-After")
                try:
                    sleep_seconds = float(retry_after) if retry_after else 1.5 * (attempt + 1)
                except ValueError:
                    sleep_seconds = 1.5 * (attempt + 1)
                time.sleep(max(sleep_seconds, 0.5))
                continue
            if response.status_code >= 500 and attempt < max_attempts - 1:
                time.sleep(min(30.0, 2.0 * (2 ** attempt)))
                continue
            break
        if response is None:
            raise RuntimeError("Notion API request failed without response.")
        if response.status_code >= 400:
            raise RuntimeError(f"Notion API error {response.status_code}: {response.text}")
        if not response.content:
            return {}
        return response.json()

    @staticmethod
    def _is_schema_size_limit_error(error: Exception | str) -> bool:
        message = str(error or "").lower()
        return "database schema has exceeded the maximum size" in message

    @staticmethod
    def _extract_saturated_multiselect_properties(
        error: Exception | str,
        candidates: list[str] | None = None,
    ) -> list[str]:
        message = str(error or "").lower()
        ordered_candidates = candidates or ["partes", "advogados"]
        found: list[str] = []
        for property_name in ordered_candidates:
            lowered = property_name.lower()
            if f"'{lowered}'" in message or f'"{lowered}"' in message:
                found.append(property_name)
        return dedupe_preserve_order(found)

    def _compact_multiselect_properties(self, property_names: list[str]) -> list[str]:
        compacted: list[str] = []
        for property_name in dedupe_preserve_order(property_names):
            try:
                summary = self.rebuild_multiselect_property_with_default_colors(property_name)
            except Exception as exc:
                self.logger.warning(
                    "Falha ao compactar schema multiselect de %s no Notion: %s",
                    property_name,
                    exc,
                )
                continue
            if summary.get("updated"):
                compacted.append(property_name)
                self.logger.warning(
                    "Schema multiselect de %s compactado automaticamente para destravar escrita no Notion.",
                    property_name,
                )
        return compacted

    def _drop_unavailable_multiselect_values(
        self,
        schema: "NotionDataSourceSchema",
        row: "PublishPreviewRow",
        property_names: list[str],
    ) -> None:
        property_map = {
            "partes": "partes",
            "advogados": "advogados",
            "composicao": "composicao",
        }
        for property_name in dedupe_preserve_order(property_names):
            internal_name = property_map.get(property_name)
            if not internal_name:
                continue
            prop = schema.properties.get(property_name)
            if not prop or prop.type != "multi_select":
                continue
            current_values = list(getattr(row, internal_name, []) or [])
            if not current_values:
                continue
            kept_values = [value for value in current_values if str(value).strip() in prop.options]
            omitted_values = [value for value in current_values if str(value).strip() not in prop.options]
            if not omitted_values:
                continue
            setattr(row, internal_name, kept_values)
            preview = ", ".join(str(value) for value in omitted_values[:5])
            suffix = " ..." if len(omitted_values) > 5 else ""
            row.add_warning(
                f"{property_name} parcialmente omitido por limite estrutural do schema do Notion: {preview}{suffix}"
            )

    def _prepare_row_schema(self, schema: "NotionDataSourceSchema", row: "PublishPreviewRow") -> "NotionDataSourceSchema":
        missing_select_options = self._collect_missing_select_options(schema, row)
        if missing_select_options:
            self.ensure_select_options_default(missing_select_options)
            self._extend_schema_options(schema, missing_select_options)
        missing_multiselect_options = self._collect_missing_multiselect_options(schema, row)
        if self.normalize_multiselect_colors_post_write and missing_multiselect_options:
            self.ensure_multiselect_options_default(missing_multiselect_options)
            self._extend_schema_options(schema, missing_multiselect_options)
        return schema

    def _write_row_once(
        self,
        schema: "NotionDataSourceSchema",
        row: "PublishPreviewRow",
        *,
        page_id: str = "",
    ) -> dict[str, Any]:
        prepared_schema = self._prepare_row_schema(schema, row)
        if page_id:
            payload = {"properties": self.build_properties_payload(prepared_schema, row)}
            return self._request("PATCH", f"/pages/{page_id}", json=payload)
        payload = {
            "parent": {"type": "data_source_id", "data_source_id": self.data_source_id},
            "properties": self.build_properties_payload(prepared_schema, row),
        }
        # Registros novos nascem aptos ao RAG (tratamento de 02/07/2026: a flag
        # incluir_no_rag passou a filtrar a base 'sessões' no conle_gerador).
        # Só na criação: updates não tocam a flag para não sobrescrever curadoria manual.
        flag_prop = prepared_schema.properties.get("incluir_no_rag")
        if flag_prop is not None and getattr(flag_prop, "type", "") == "checkbox":
            payload["properties"]["incluir_no_rag"] = {"checkbox": True}
        # Convenção dos corações (deputado federal 💚 / estadual-distrital 🧡): só na
        # criação — update nunca toca o ícone, para o ⏰ de vista e a curadoria prevalecerem.
        coracao = suggest_deputado_icon(row)
        if coracao:
            payload["icon"] = {"type": "emoji", "emoji": coracao}
        return self._request("POST", "/pages", json=payload)

    def _write_row_with_schema_recovery(
        self,
        schema: "NotionDataSourceSchema",
        row: "PublishPreviewRow",
        *,
        page_id: str = "",
    ) -> dict[str, Any]:
        try:
            return self._write_row_once(schema, row, page_id=page_id)
        except RuntimeError as exc:
            if not self._is_schema_size_limit_error(exc):
                raise
            candidate_properties = dedupe_preserve_order(
                self._extract_saturated_multiselect_properties(exc)
                + [name for name in self._collect_missing_multiselect_options(schema, row).keys() if name in {"partes", "advogados"}]
            )
            candidate_properties = [name for name in candidate_properties if name in {"partes", "advogados"}]
            if not candidate_properties:
                raise
            self._compact_multiselect_properties(candidate_properties)
            refreshed_schema = self.fetch_schema()
            try:
                return self._write_row_once(refreshed_schema, row, page_id=page_id)
            except RuntimeError as retry_exc:
                if not self._is_schema_size_limit_error(retry_exc):
                    raise
                fallback_properties = self._extract_saturated_multiselect_properties(
                    retry_exc,
                    candidates=candidate_properties,
                ) or candidate_properties
                self._drop_unavailable_multiselect_values(refreshed_schema, row, fallback_properties)
                return self._write_row_once(refreshed_schema, row, page_id=page_id)

    def fetch_schema(self) -> NotionDataSourceSchema:
        payload = self._request("GET", f"/data_sources/{self.data_source_id}")
        schema = NotionDataSourceSchema(self.data_source_id, payload)
        schema.ensure_expected_properties()
        return schema

    def _collect_missing_multiselect_properties(
        self,
        schema: NotionDataSourceSchema,
        row: PublishPreviewRow,
        allowed_properties: Optional[list[str]] = None,
    ) -> list[str]:
        target_properties = set(allowed_properties or ["partes", "advogados"])
        missing_properties: list[str] = []
        for internal_name, notion_name in {
            "partes": "partes",
            "advogados": "advogados",
            "composicao": "composicao",
        }.items():
            if notion_name not in target_properties:
                continue
            prop = schema.properties.get(notion_name)
            if not prop or prop.type != "multi_select":
                continue
            values = getattr(row, internal_name, []) or []
            if any(str(value).strip() and str(value) not in prop.options for value in values):
                missing_properties.append(notion_name)
        return missing_properties

    def _collect_missing_multiselect_options(
        self,
        schema: NotionDataSourceSchema,
        row: PublishPreviewRow,
        allowed_properties: Optional[list[str]] = None,
    ) -> dict[str, list[str]]:
        target_properties = set(allowed_properties or ["partes", "advogados"])
        missing_options: dict[str, list[str]] = {}
        for internal_name, notion_name in {
            "partes": "partes",
            "advogados": "advogados",
            "composicao": "composicao",
        }.items():
            if notion_name not in target_properties:
                continue
            prop = schema.properties.get(notion_name)
            if not prop or prop.type != "multi_select":
                continue
            seen_missing: set[str] = set()
            ordered_missing: list[str] = []
            for value in getattr(row, internal_name, []) or []:
                normalized = str(value).strip()
                if not normalized or normalized in prop.options or normalized in seen_missing:
                    continue
                seen_missing.add(normalized)
                ordered_missing.append(normalized)
            if ordered_missing:
                missing_options[notion_name] = ordered_missing
        return missing_options

    def _collect_missing_select_options(
        self,
        schema: NotionDataSourceSchema,
        row: PublishPreviewRow,
        allowed_properties: Optional[list[str]] = None,
    ) -> dict[str, list[str]]:
        target_properties = set(
            allowed_properties
            or [
                "relator",
                "pedido_vista",
            ]
        )
        missing_options: dict[str, list[str]] = {}
        for internal_name, notion_name in {
            "relator": "relator",
            "pedido_vista": "pedido_vista",
        }.items():
            if notion_name not in target_properties:
                continue
            prop = schema.properties.get(notion_name)
            if not prop or prop.type != "select":
                continue
            value = str(getattr(row, internal_name, "") or "").strip()
            if not value or value in prop.options:
                continue
            missing_options[notion_name] = [value]
        return missing_options

    @staticmethod
    def _merge_select_options(
        live_options_raw: list[dict[str, Any]],
        option_name: str,
    ) -> list[dict[str, Any]]:
        merged_options: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for option in live_options_raw:
            existing_name = str(option.get("name", "")).strip()
            if not existing_name or existing_name in seen_names:
                continue
            seen_names.add(existing_name)
            merged_options.append(
                {
                    "name": existing_name,
                    "color": str(option.get("color") or "default"),
                }
            )
        merged_options.append({"name": option_name, "color": "default"})
        return merged_options

    def ensure_select_options_default(
        self,
        missing_options: dict[str, list[str]],
    ) -> dict[str, Any]:
        if not missing_options:
            return {"updated": False, "properties": []}
        property_summaries: list[dict[str, Any]] = []
        for property_name, option_names in missing_options.items():
            created_count = 0
            skipped_due_to_schema_limit = False
            for option_name in option_names:
                live_payload = self._request("GET", f"/data_sources/{self.data_source_id}")
                live_prop = (live_payload.get("properties") or {}).get(property_name) or {}
                if live_prop.get("type") != "select":
                    continue
                live_options_raw = ((live_prop.get("select") or {}).get("options") or [])
                live_option_names = {
                    str(option.get("name", "")).strip()
                    for option in live_options_raw
                    if str(option.get("name", "")).strip()
                }
                normalized_option_name = str(option_name or "").strip()
                if property_name in {"relator", "pedido_vista"} and normalized_option_name:
                    normalized_option_name = _canonicalize_person_select_value(
                        normalized_option_name,
                        notion_name=property_name,
                    )
                if not normalized_option_name or normalized_option_name in live_option_names:
                    continue
                if len(live_option_names) >= 100:
                    skipped_due_to_schema_limit = True
                    continue
                merged_options = self._merge_select_options(live_options_raw, normalized_option_name)
                try:
                    self._request(
                        "PATCH",
                        f"/data_sources/{self.data_source_id}",
                        json={"properties": {property_name: {"select": {"options": merged_options}}}},
                    )
                except RuntimeError as exc:
                    message = str(exc)
                    if "select.options.length should be ≤" in message:
                        skipped_due_to_schema_limit = True
                        continue
                    if "Cannot update color of select with name:" not in message:
                        raise
                    live_payload = self._request("GET", f"/data_sources/{self.data_source_id}")
                    live_prop = (live_payload.get("properties") or {}).get(property_name) or {}
                    if live_prop.get("type") != "select":
                        continue
                    live_options_raw = ((live_prop.get("select") or {}).get("options") or [])
                    live_option_names = {
                        str(option.get("name", "")).strip()
                        for option in live_options_raw
                        if str(option.get("name", "")).strip()
                    }
                    if normalized_option_name in live_option_names:
                        continue
                    merged_options = [{"name": str(option.get("name", "")).strip()} for option in live_options_raw if str(option.get("name", "")).strip()]
                    merged_options.append({"name": normalized_option_name, "color": "default"})
                    self._request(
                        "PATCH",
                        f"/data_sources/{self.data_source_id}",
                        json={"properties": {property_name: {"select": {"options": merged_options}}}},
                    )
                created_count += 1
            if created_count:
                property_summaries.append({"property": property_name, "created_options": created_count})
            elif skipped_due_to_schema_limit:
                property_summaries.append(
                    {
                        "property": property_name,
                        "created_options": 0,
                        "skipped_schema_update_due_to_limit": True,
                    }
                )
        return {"updated": bool(property_summaries), "properties": property_summaries}

    @staticmethod
    def _extend_schema_options(
        schema: NotionDataSourceSchema,
        new_options: dict[str, list[str]],
    ) -> None:
        for property_name, values in new_options.items():
            prop = schema.properties.get(property_name)
            if not prop:
                continue
            for value in values:
                normalized = str(value or "").strip()
                if normalized and normalized not in prop.options:
                    prop.options.append(normalized)

    def ensure_multiselect_options_default(
        self,
        missing_options: dict[str, list[str]],
    ) -> dict[str, Any]:
        if not missing_options:
            return {"updated": False, "properties": []}
        property_summaries: list[dict[str, Any]] = []
        for property_name, option_names in missing_options.items():
            created_count = 0
            skipped_due_to_schema_limit = False
            for option_name in option_names:
                live_payload = self._request("GET", f"/data_sources/{self.data_source_id}")
                live_prop = (live_payload.get("properties") or {}).get(property_name) or {}
                live_options_raw = ((live_prop.get("multi_select") or {}).get("options") or [])
                live_option_names = {
                    str(option.get("name", "")).strip()
                    for option in live_options_raw
                    if str(option.get("name", "")).strip()
                }
                normalized_option_name = str(option_name or "").strip()
                if not normalized_option_name or normalized_option_name in live_option_names:
                    continue
                # The Notion schema PATCH endpoint rejects option payloads above 100 entries,
                # even when the live property already contains more than 100 options. In that
                # case we skip the schema-level update and let the page write carry the value.
                if len(live_option_names) >= 100:
                    skipped_due_to_schema_limit = True
                    continue
                merged_options: list[dict[str, Any]] = []
                seen_names: set[str] = set()
                for option in live_options_raw:
                    existing_name = str(option.get("name", "")).strip()
                    if not existing_name or existing_name in seen_names:
                        continue
                    seen_names.add(existing_name)
                    merged_options.append(
                        {
                            "name": existing_name,
                            "color": str(option.get("color") or "default"),
                        }
                    )
                merged_options.append({"name": normalized_option_name, "color": "default"})
                try:
                    self._request(
                        "PATCH",
                        f"/data_sources/{self.data_source_id}",
                        json={"properties": {property_name: {"multi_select": {"options": merged_options}}}},
                    )
                except RuntimeError as exc:
                    message = str(exc)
                    if "multi_select.options.length should be ≤" in message:
                        skipped_due_to_schema_limit = True
                        continue
                    if "Cannot update color of select with name:" not in message:
                        raise
                    live_payload = self._request("GET", f"/data_sources/{self.data_source_id}")
                    live_prop = (live_payload.get("properties") or {}).get(property_name) or {}
                    live_options_raw = ((live_prop.get("multi_select") or {}).get("options") or [])
                    live_option_names = {
                        str(option.get("name", "")).strip()
                        for option in live_options_raw
                        if str(option.get("name", "")).strip()
                    }
                    if normalized_option_name in live_option_names:
                        continue
                    merged_options = []
                    seen_names = set()
                    for option in live_options_raw:
                        existing_name = str(option.get("name", "")).strip()
                        if not existing_name or existing_name in seen_names:
                            continue
                        seen_names.add(existing_name)
                        merged_options.append({"name": existing_name})
                    merged_options.append({"name": normalized_option_name, "color": "default"})
                    self._request(
                        "PATCH",
                        f"/data_sources/{self.data_source_id}",
                        json={"properties": {property_name: {"multi_select": {"options": merged_options}}}},
                    )
                created_count += 1
            if created_count:
                property_summaries.append({"property": property_name, "created_options": created_count})
            elif skipped_due_to_schema_limit:
                property_summaries.append(
                    {
                        "property": property_name,
                        "created_options": 0,
                        "skipped_schema_update_due_to_limit": True,
                    }
                )
        return {"updated": bool(property_summaries), "properties": property_summaries}

    def create_multi_select_property(self, property_name: str) -> None:
        self._request("PATCH", f"/data_sources/{self.data_source_id}", json={"properties": {property_name: {"multi_select": {}}}})

    def rename_property(self, current_name: str, new_name: str) -> None:
        payload = self._request("GET", f"/data_sources/{self.data_source_id}")
        prop = payload.get("properties", {}).get(current_name)
        if not prop:
            raise RuntimeError(f"Propriedade {current_name!r} não encontrada para renomear.")
        prop_id = prop.get("id") or current_name
        self._request(
            "PATCH",
            f"/data_sources/{self.data_source_id}",
            json={"properties": {prop_id: {"name": new_name}}},
        )

    def drop_property(self, property_name: str) -> None:
        self._request("PATCH", f"/data_sources/{self.data_source_id}", json={"properties": {property_name: None}})

    def rebuild_multiselect_property_with_default_colors(
        self,
        property_name: str,
        *,
        temp_name: Optional[str] = None,
        sort_options: bool = False,
    ) -> dict[str, Any]:
        payload = self._request("GET", f"/data_sources/{self.data_source_id}")
        properties = payload.get("properties", {})
        prop = properties.get(property_name)
        if not prop or prop.get("type") != "multi_select":
            return {"updated": False, "property": property_name}
        pages = self.query_data_source()
        used_option_names: list[str] = []
        seen_option_names: set[str] = set()
        page_values: list[tuple[str, list[str]]] = []
        for page in pages:
            values = [
                item.get("name", "").strip()
                for item in page.get("properties", {}).get(property_name, {}).get("multi_select", [])
                if item.get("name", "").strip()
            ]
            page_values.append((page.get("id", ""), values))
            for value in values:
                if value not in seen_option_names:
                    seen_option_names.add(value)
                    used_option_names.append(value)
        if sort_options:
            used_option_names = sorted(used_option_names, key=lambda value: normalize_class_text(value))
            option_rank = {value: index for index, value in enumerate(used_option_names)}
            page_values = [
                (
                    page_id,
                    sorted(
                        values,
                        key=lambda value: (option_rank.get(value, len(option_rank)), normalize_class_text(value)),
                    ),
                )
                for page_id, values in page_values
            ]
            page_values.sort(
                key=lambda item: min(
                    (option_rank.get(value, len(option_rank)) for value in item[1]),
                    default=len(option_rank),
                )
            )
        options = prop.get("multi_select", {}).get("options", []) or []
        current_option_names = [option.get("name", "").strip() for option in options if option.get("name", "").strip()]
        nondefault_options = [option for option in options if (option.get("color") or "") != "default"]
        unused_count = len(options) - len(used_option_names)
        if not nondefault_options and unused_count == 0 and current_option_names == used_option_names:
            return {
                "updated": False,
                "property": property_name,
                "used_options": len(used_option_names),
                "page_updates": 0,
            }

        def restore_property_in_place() -> dict[str, Any]:
            self.drop_property(property_name)
            self.create_multi_select_property(property_name)
            page_updates = 0
            for page_id, values in page_values:
                if not page_id or not values:
                    continue
                self._request(
                    "PATCH",
                    f"/pages/{page_id}",
                    json={"properties": {property_name: {"multi_select": [{"name": value} for value in values]}}},
                )
                page_updates += 1
            return {
                "updated": True,
                "property": property_name,
                "nondefault_options": len(nondefault_options),
                "unused_options_observed": unused_count,
                "used_options": len(used_option_names),
                "page_updates": page_updates,
                "recreated_in_place": True,
            }

        working_temp_name = temp_name or f"{property_name}__default_tmp"
        legacy_name = f"{property_name}__legacy_color"
        refreshed_properties = self._request("GET", f"/data_sources/{self.data_source_id}").get("properties", {})
        for residue_name in [working_temp_name, legacy_name]:
            if residue_name in refreshed_properties:
                self.drop_property(residue_name)
        try:
            self.create_multi_select_property(working_temp_name)
            if 0 < len(used_option_names) <= 100:
                self.ensure_multiselect_options_default({working_temp_name: used_option_names})
            page_updates = 0
            for page_id, values in page_values:
                if not page_id or not values:
                    continue
                self._request(
                    "PATCH",
                    f"/pages/{page_id}",
                    json={"properties": {working_temp_name: {"multi_select": [{"name": value} for value in values]}}},
                )
                page_updates += 1
            self.rename_property(property_name, legacy_name)
            self.rename_property(working_temp_name, property_name)
            self.drop_property(legacy_name)
            return {
                "updated": True,
                "property": property_name,
                "nondefault_options": len(nondefault_options),
                "unused_options_observed": unused_count,
                "used_options": len(used_option_names),
                "page_updates": page_updates,
            }
        except RuntimeError as exc:
            if not self._is_schema_size_limit_error(exc):
                raise
            self.logger.warning(
                "Compactação em duas propriedades falhou para %s por limite estrutural do schema; recriando a propriedade em lugar.",
                property_name,
            )
            refreshed_properties = self._request("GET", f"/data_sources/{self.data_source_id}").get("properties", {})
            for residue_name in [working_temp_name, legacy_name]:
                if residue_name in refreshed_properties:
                    self.drop_property(residue_name)
            return restore_property_in_place()

    def set_multi_select_options_color_default(
        self,
        schema: NotionDataSourceSchema,
        property_names: list[str],
    ) -> dict[str, Any]:
        changed_summary: dict[str, int] = {}
        property_summaries: list[dict[str, Any]] = []
        pages = self.query_data_source()
        for property_name in property_names:
            prop = schema.raw_payload.get("properties", {}).get(property_name, {})
            if prop.get("type") != "multi_select":
                continue
            options = prop.get("multi_select", {}).get("options", []) or []
            used_option_names: list[str] = []
            seen_option_names: set[str] = set()
            for page in pages:
                values = [
                    item.get("name", "").strip()
                    for item in page.get("properties", {}).get(property_name, {}).get("multi_select", [])
                    if item.get("name", "").strip()
                ]
                for value in values:
                    if value not in seen_option_names:
                        seen_option_names.add(value)
                        used_option_names.append(value)
            nondefault_options = [option for option in options if (option.get("color") or "") != "default"]
            unused_count = len(options) - len(used_option_names)
            current_option_names = [option.get("name", "").strip() for option in options if option.get("name", "").strip()]
            if not nondefault_options and unused_count == 0 and current_option_names == used_option_names:
                continue
            changed_summary[property_name] = len(nondefault_options)
            property_summaries.append(self.rebuild_multiselect_property_with_default_colors(property_name))
        if not property_summaries:
            return {"updated": False, "changed": changed_summary}
        return {"updated": True, "changed": changed_summary, "properties": property_summaries}

    def query_data_source(self, filter_payload: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        start_cursor: Optional[str] = None
        while True:
            payload: dict[str, Any] = {"page_size": 100}
            if filter_payload:
                payload["filter"] = filter_payload
            if start_cursor:
                payload["start_cursor"] = start_cursor
            page = self._request("POST", f"/data_sources/{self.data_source_id}/query", json=payload)
            results.extend(page.get("results", []))
            if not page.get("has_more"):
                break
            start_cursor = page.get("next_cursor")
        return results

    def build_filter_condition(
        self,
        schema: NotionDataSourceSchema,
        property_name: str,
        value: str,
    ) -> Optional[dict[str, Any]]:
        if not value:
            return None
        prop = schema.properties.get(property_name)
        if not prop:
            return None
        if prop.type == "title":
            return {"property": property_name, "title": {"equals": value}}
        if prop.type == "rich_text":
            return {"property": property_name, "rich_text": {"equals": value}}
        if prop.type == "url":
            return {"property": property_name, "url": {"equals": value}}
        return None

    def _extract_property_text(
        self,
        page: dict[str, Any],
        schema: NotionDataSourceSchema,
        property_name: str,
    ) -> str:
        prop_schema = schema.properties[property_name]
        value = page.get("properties", {}).get(property_name, {})
        if prop_schema.type == "title":
            return "".join(item.get("plain_text", "") for item in value.get("title", []))
        if prop_schema.type == "rich_text":
            return "".join(item.get("plain_text", "") for item in value.get("rich_text", []))
        if prop_schema.type == "url":
            return value.get("url") or ""
        if prop_schema.type == "date":
            return value.get("date", {}).get("start") or ""
        if prop_schema.type in {"select", "status"}:
            payload = value.get(prop_schema.type) or {}
            return payload.get("name") or ""
        if prop_schema.type == "multi_select":
            return ", ".join(item.get("name", "") for item in value.get("multi_select", []))
        if prop_schema.type == "relation":
            return ", ".join(item.get("id", "") for item in value.get("relation", []))
        return ""

    def find_existing_row(
        self,
        schema: NotionDataSourceSchema,
        youtube_link: str,
        numero_processo: str,
        data_sessao: str = "",
    ) -> Optional[NotionRowMatch]:
        candidates: list[dict[str, Any]] = []
        numero_filter = self.build_filter_condition(schema, "numero_processo", numero_processo)
        if numero_filter:
            candidates.extend(self.query_data_source(numero_filter))
        # A igualdade exata NAO enxerga a mesma pagina gravada com o numero em outra
        # forma — nos DOIS sentidos: query curta × base completa (o chamador passa o
        # canonico "0600504-40" e a pagina guarda "0600504-40.2026.6.00.0000"; foi o que
        # duplicou a sessao 03/08 inteira em 20/08/2026) e query completa × base curta.
        # Busca tambem por CONTEM o nucleo curto canonico, SEMPRE que ele exista; a
        # comparacao canonica logo abaixo decide se e a mesma pagina.
        short_core = canonicalize_numero_processo(numero_processo)
        if short_core:
            prop = schema.properties.get("numero_processo")
            if prop is not None and prop.type in {"rich_text", "title"}:
                contains_filter = {
                    "property": "numero_processo",
                    prop.type: {"contains": short_core},
                }
                try:
                    candidates.extend(self.query_data_source(contains_filter))
                except Exception:  # noqa: BLE001
                    pass  # busca auxiliar: a exata e o youtube_link continuam valendo
        if youtube_link:
            youtube_filter = self.build_filter_condition(schema, "youtube_link", youtube_link)
            if youtube_filter:
                candidates.extend(self.query_data_source(youtube_filter))

        normalized_youtube = normalize_youtube_link(youtube_link)
        normalized_numero = canonicalize_numero_processo(numero_processo)
        target_video_id = extract_youtube_video_id(normalized_youtube)
        for candidate in {candidate.get("id", ""): candidate for candidate in candidates}.values():
            candidate_numero = canonicalize_numero_processo(
                self._extract_property_text(candidate, schema, "numero_processo")
            )
            candidate_youtube = normalize_youtube_link(
                self._extract_property_text(candidate, schema, "youtube_link")
            )
            candidate_video_id = extract_youtube_video_id(candidate_youtube)
            same_video = bool(target_video_id and candidate_video_id == target_video_id)
            exact_youtube_match = bool(normalized_youtube) and candidate_youtube == normalized_youtube
            if not normalized_numero:
                # Linha SEM numero (caso antigo/administrativo): a identidade e a URL EXATA
                # do julgamento (com timestamp). So casa pagina tambem SEM numero, para nao
                # colidir com um processo numerado que compartilhe o video.
                if exact_youtube_match and not candidate_numero:
                    return NotionRowMatch(page_id=candidate.get("id", ""), url=candidate.get("url", ""))
                continue
            if candidate_numero == normalized_numero and (
                not target_video_id or exact_youtube_match or same_video
            ):
                return NotionRowMatch(
                    page_id=candidate.get("id", ""),
                    url=candidate.get("url", ""),
                )
            # Mesmo numero + MESMA data de sessao, mas video diferente: e a mesma sessao
            # dividida em partes (transmissao em dois videos). Sem este ramo, o processo
            # publicado pela parte 1 era recriado pela parte 2 — origem das duplicatas da
            # sessao de 13/08/2026. Sessoes DISTINTAS do mesmo processo continuam gerando
            # paginas separadas (a data difere).
            if candidate_numero == normalized_numero and data_sessao:
                candidate_sessao = ""
                if "data_sessao" in schema.properties:
                    candidate_sessao = self._extract_property_text(
                        candidate, schema, "data_sessao")[:10]
                if candidate_sessao and candidate_sessao == str(data_sessao)[:10]:
                    return NotionRowMatch(
                        page_id=candidate.get("id", ""),
                        url=candidate.get("url", ""),
                    )
        return None

    def _build_property_value(
        self,
        schema: NotionDataSourceSchema,
        property_name: str,
        value: Any,
    ) -> Optional[dict[str, Any]]:
        prop = schema.properties[property_name]
        if value in (None, "", []):
            return None
        if prop.type == "title":
            return {"title": chunk_rich_text(str(value))}
        if prop.type == "rich_text":
            # listas (ex.: partes/advogados migrados de multi_select) viram texto " , "-juntado
            text = ", ".join(str(v).strip() for v in value if str(v).strip()) if isinstance(value, list) else str(value)
            return {"rich_text": chunk_rich_text(text)}
        if prop.type == "url":
            return {"url": str(value)}
        if prop.type == "date":
            return {"date": {"start": str(value)}}
        if prop.type == "select":
            return {"select": {"name": str(value)}}
        if prop.type == "status":
            return {"status": {"name": str(value)}}
        if prop.type == "multi_select":
            values = value if isinstance(value, list) else parse_multi_value_text(value)
            return {"multi_select": [{"name": str(item)} for item in values if str(item).strip()]}
        if prop.type == "relation":
            values = value if isinstance(value, list) else parse_multi_value_text(value)
            return {"relation": [{"id": str(item)} for item in values if str(item).strip()]}
        return None

    def _build_empty_property_value(
        self,
        schema: NotionDataSourceSchema,
        property_name: str,
    ) -> Optional[dict[str, Any]]:
        prop = schema.properties[property_name]
        if prop.type == "title":
            return {"title": []}
        if prop.type == "rich_text":
            return {"rich_text": []}
        if prop.type == "url":
            return {"url": None}
        if prop.type == "date":
            return {"date": None}
        if prop.type == "select":
            return {"select": None}
        if prop.type == "status":
            return {"status": None}
        if prop.type == "multi_select":
            return {"multi_select": []}
        if prop.type == "relation":
            return {"relation": []}
        return None

    def build_properties_payload(
        self,
        schema: NotionDataSourceSchema,
        row: PublishPreviewRow,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        title_property = schema.title_property_name
        clear_properties = set(row.clear_properties or [])
        title_value = row.tema or row.punchline
        if title_value:
            payload[title_property] = self._build_property_value(schema, title_property, title_value)
        elif row.force_clear_title:
            payload[title_property] = {"title": []}

        for internal_name, notion_name in NOTION_PROPERTY_MAP.items():
            if notion_name == title_property:
                continue
            if notion_name not in schema.properties:
                continue
            value = getattr(row, internal_name, None)
            if internal_name == "data_sessao":
                value = row.data_sessao
            built = self._build_property_value(schema, notion_name, value)
            if built:
                payload[notion_name] = built
                continue
            if internal_name in clear_properties or notion_name in clear_properties:
                cleared = self._build_empty_property_value(schema, notion_name)
                if cleared is not None:
                    payload[notion_name] = cleared

        for index, url in enumerate(row.noticias_gerais[:GENERAL_NEWS_LIMIT], start=1):
            notion_name = f"noticia_geral_{index}"
            if notion_name not in schema.properties:
                continue
            built = self._build_property_value(schema, notion_name, url)
            if built:
                payload[notion_name] = built
        return payload

    def create_row(self, schema: NotionDataSourceSchema, row: PublishPreviewRow) -> dict[str, Any]:
        return self._write_row_with_schema_recovery(schema, row)

    def update_row(self, schema: NotionDataSourceSchema, page_id: str, row: PublishPreviewRow) -> dict[str, Any]:
        return self._write_row_with_schema_recovery(schema, row, page_id=page_id)


def _extract_vista_minister(texto: str, composicao: list[str]) -> str:
    """Ministro do pedido de vista, garimpado no texto extraido (melhor esforco).

    Primeiro procura um nome da COMPOSICAO a ate ~80 caracteres da palavra "vista"
    (isolada, para nao casar "prevista"/"revista"); depois um padrao generico
    "Min. Fulano ... vista" / "vista ... Min. Fulano". Devolve "" quando nao acha —
    quem chama decide o fallback.
    """
    if not texto:
        return ""
    for membro in composicao or []:
        nome = re.sub(r"^min(?:istr[oa])?\.?\s*", "", str(membro or ""),
                      flags=re.IGNORECASE).strip()
        if len(nome) < 4:
            continue
        alvo = re.escape(nome)
        # janela de ate ~60 caracteres QUAISQUER entre "vista" e o nome ("pedido de
        # vista do Ministro Fulano" tem palavras no meio, nao so pontuacao)
        if re.search(rf"\bvistas?\b[\s\S]{{0,60}}?{alvo}|{alvo}[\s\S]{{0,60}}?\bvistas?\b",
                     texto, re.IGNORECASE):
            return f"Min. {nome}"
    m = re.search(
        r"Min(?:istr[oa])?\.?\s+([A-ZÀ-Ý][\wÀ-ÿ.'-]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÀ-Ý][\wÀ-ÿ.'-]+){0,3})[\s\S]{0,60}?\bvistas?\b"
        r"|\bvistas?\b[\s\S]{0,60}?Min(?:istr[oa])?\.?\s+([A-ZÀ-Ý][\wÀ-ÿ.'-]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÀ-Ý][\wÀ-ÿ.'-]+){0,3})",
        texto)
    if m:
        nome = (m.group(1) or m.group(2) or "").strip().rstrip(".,;")
        if nome:
            return f"Min. {nome}"
    return ""


def apply_rag_consistency_checks(row: "PublishPreviewRow") -> None:
    """Checagens determinísticas de coerência para a base servir de RAG.

    Rodam a cada validate_preview_row; as mensagens estão registradas nos
    padrões RECOMPUTED_*, então revalidações não acumulam duplicatas.
    """
    digits = re.sub(r"\D", "", str(row.numero_processo or ""))
    # Numero sem letras (nao e "PA 19" e afins) com menos digitos que o nucleo curto
    # eleitoral (NNNNNNN-DD = 9): extracao truncada. O zero-pad da canonicalizacao ja
    # evita a duplicata ("312-46" ≡ "0000312-46"), mas o valor gravado segue incompleto
    # e merece vistoria — as duplicatas de 13/08/2026 vieram com as demais colunas
    # erradas junto.
    if 0 < len(digits) < 9 and not re.search(r"[A-Za-z]", str(row.numero_processo or "")):
        row.add_warning(
            f"Número do processo incompleto ({len(digits)} dígito(s): "
            f"'{row.numero_processo}') — conferir na vistoria."
        )
    # Instruções das resoluções "permanentes" tramitam nos autos ORIGINAIS: quando o
    # núcleo+DV casam com uma instrução conhecida e o ano gravado difere (a extração
    # assume o ano da sessão — 0600748-13 saiu ".2026" em 03/08/2026), corrige para o
    # CNJ canônico. Correção determinística: a tabela é curada e núcleo+DV bastam.
    autos_originais = instrucao_autos_originais_cnj(row.numero_processo)
    if autos_originais:
        anterior = row.numero_processo
        row.numero_processo = autos_originais
        digits = re.sub(r"\D", "", autos_originais)
        row.add_warning(
            f"Número corrigido para os autos originais da instrução "
            f"({anterior} → {autos_originais})."
        )
    # DV do CNJ (mod 97): reprovação não prova fabricação (ASR corrompe dígito real),
    # mas todo número fabricado/ano trocado reprova — vistoria decide (auditoria de
    # agosto/2026: 0600748-13.2026 e 0000295-14.2026 teriam sido pegos aqui).
    if len(digits) == 20 and cnj_check_digit_is_valid(row.numero_processo) is False:
        row.add_warning(
            "Dígito verificador do CNJ reprova (mod 97) — possível ano/dígito "
            "corrompido na extração; conferir na vistoria."
        )
    if len(digits) == 20 and digits[13] == "6":
        tr_code = digits[14:16]
        tribunal = str(row.tribunal or "").strip().upper()
        origem_uf = extract_uf_from_text(row.origem)
        if tr_code == "00":
            if tribunal and tribunal != "TSE":
                row.add_warning(
                    f"CNJ indica UF do TSE (TR 00); tribunal aponta {tribunal} — conferir na vistoria."
                )
        else:
            uf = CNJ_ELECTORAL_UF_BY_CODE.get(tr_code, "")
            divergencias: list[str] = []
            if uf and tribunal.startswith("TRE-") and tribunal != f"TRE-{uf}":
                divergencias.append(f"tribunal {tribunal}")
            if uf and origem_uf and origem_uf != uf:
                divergencias.append(f"origem {row.origem}")
            if divergencias:
                row.add_warning(
                    f"CNJ indica UF {uf}; " + " e ".join(divergencias) + " divergem — conferir na vistoria."
                )
    if row.resultado == "Suspenso por vista" and not row.pedido_vista:
        # AUTO-REPARO antes de bloquear (20/08/2026): a inconsistencia e da EXTRACAO, nao
        # do julgamento — o 0600870-47 (Medina/MG, sessao 18/08) foi extraido completo e
        # sumiu do Notion por causa do bloqueio silencioso. Tenta achar o ministro da
        # vista no proprio texto extraido; nao achando, publica com a lacuna sinalizada
        # (a vistoria corrige depois). Bloquear julgamento real e pior que publicar com
        # lacuna visivel.
        texto_busca = " ".join(filter(None, [
            row.votacao, row.analise_do_conteudo_juridico, row.raciocinio_juridico]))
        ministro_vista = _extract_vista_minister(texto_busca, row.composicao)
        if ministro_vista:
            row.pedido_vista = ministro_vista
            row.add_warning(
                f"pedido_vista preenchido por auto-reparo ({ministro_vista}) — conferir na vistoria.")
        elif any(warning.startswith("Aprovado em vistoria") for warning in row.warnings):
            row.add_warning("resultado 'Suspenso por vista' sem pedido_vista (aceito em vistoria).")
        else:
            row.pedido_vista = "(ministro não identificado)"
            row.add_warning(
                "resultado 'Suspenso por vista' sem ministro da vista identificado — "
                "publicado com lacuna; conferir na vistoria.")
    # Espelho da checagem acima: desfecho DEFINITIVO cuja prosa anuncia vista PEDIDA NESTE
    # julgamento. Em 25/08/2026 o RO 0600960-72 saiu "Unânime"/"Desprovido" porque o anúncio da
    # vista do Min. André Mendonça (t=4470) caía 5 s fora da janela do scan; o recorte agora tem
    # folga de cauda (VIDEO_DETAIL_TAIL_PADDING_SECONDS), e este aviso é a segunda barreira para
    # quando ainda assim escapar. Padrões restritos de propósito: "após o voto-vista do Ministro
    # X" e "renovado o pedido de vista" descrevem sessão ANTERIOR e um caso retomado hoje é
    # legitimamente definitivo — esses não podem disparar.
    if row.resultado and row.resultado != "Suspenso por vista" and normalize_model_text(row.votacao) not in {"Suspenso", ""}:
        prosa = " ".join(filter(None, [
            row.analise_do_conteudo_juridico, row.raciocinio_juridico, row.punchline]))
        anuncio_de_vista = re.search(
            r"(?i)(?:pediu vista|pede vista|pedido de vista d[oae]\s|antecip\w+ o pedido de vista"
            r"|antecip\w+ a vista|julgamento (?:foi )?suspenso (?:por|ap[óo]s) (?:o )?(?:pedido de )?vista"
            r"|suspens[ãa]o do julgamento (?:por|ap[óo]s) (?:o )?pedido de vista)",
            prosa)
        retomada = re.search(
            r"(?i)\b(?:voto[- ]vista|ap[óo]s o voto[- ]vista|retomad\w+ o julgamento"
            r"|prosseguimento do julgamento|em sess[ãa]o anterior|na sess[ãa]o de)\b", prosa)
        if anuncio_de_vista and not retomada:
            row.add_warning(
                f"resultado '{row.resultado}' com votação '{row.votacao}', mas o texto extraído "
                f"anuncia pedido de vista neste julgamento ('{anuncio_de_vista.group(0)}') — "
                "conferir na vistoria se o caso foi mesmo concluído."
            )
    allowed = resultado_allowed_for_classe(row.classe_processo)
    if allowed is not None and row.resultado and row.resultado not in allowed:
        row.add_warning(
            f"resultado '{row.resultado}' incompatível com a classe {row.classe_processo} — conferir na vistoria."
        )


def validate_preview_row(
    row: PublishPreviewRow,
    notion_schema: Optional[NotionDataSourceSchema],
) -> PublishPreviewRow:
    row.warnings = [
        message
        for message in row.warnings
        if not any(pattern.search(message) for pattern in RECOMPUTED_WARNING_PATTERNS)
    ]
    row.errors = [
        message
        for message in row.errors
        if not any(pattern.search(message) for pattern in RECOMPUTED_ERROR_PATTERNS)
    ]
    raw_general_news_count = len([value for value in row.noticias_gerais if str(value).strip()])
    original_numero_processo = str(row.numero_processo or "")
    row.data_sessao = normalize_session_date_to_iso(row.data_sessao)
    row.numero_processo = normalize_numero_processo_display(row.numero_processo)
    inferred_full_cnj = infer_full_numero_processo_from_row_text(row)
    if inferred_full_cnj:
        row.numero_processo = normalize_numero_processo_display(inferred_full_cnj)
    inferred_special_numero = infer_special_numero_processo_from_row_text(row)
    if inferred_special_numero:
        row.numero_processo = _merge_special_numero_processo(row.numero_processo, inferred_special_numero)
    row.classe_processo = normalize_classe_processo(row.classe_processo)
    row.eleicao = normalize_eleicao_value(row.eleicao)
    raw_origem_value = str(row.origem or "")
    raw_origem_key = normalize_class_text(raw_origem_value)
    raw_tribunal_value = str(row.tribunal or "").strip().upper()
    federal_origin_hint = (
        raw_origem_key in {"tse", "tribunal superior eleitoral"}
        or "tribunal superior eleitoral" in raw_origem_key
        or raw_tribunal_value == "TSE"
        or row.classe_processo == "CTA"
    )
    row.origem = normalize_origem_value(row.origem)
    uf = extract_uf_from_text(row.origem)
    row.tribunal = "TSE" if federal_origin_hint else normalize_tre(row.tribunal, uf)
    if not row.origem:
        tribunal_value = str(row.tribunal or "").strip().upper()
        tre_match = re.match(r"^TRE-([A-Z]{2})$", tribunal_value)
        if tre_match:
            row.origem = UF_CAPITALS.get(tre_match.group(1), "")
        elif tribunal_value == "TSE":
            row.origem = UF_CAPITALS["DF"]
        elif row.classe_processo == "CTA":
            row.origem = UF_CAPITALS["DF"]
            row.tribunal = "TSE"
    # Quando a origem veio de uma referência a tribunal (TRE/TSE) — e portanto virou a
    # capital por fallback, não o município real — prefira o município efetivamente
    # citado no texto do julgamento. Não sobrescreve uma origem que o modelo já trouxe
    # como município específico.
    if not federal_origin_hint and _origem_is_court_reference(raw_origem_value):
        municipio = _municipio_from_case_text(row)
        if municipio and municipio != row.origem:
            row.origem = municipio
            row.tribunal = normalize_tre(row.tribunal, extract_uf_from_text(municipio))
    row.relator = normalize_ministro_name(row.relator) if row.relator else ""
    row.pedido_vista = normalize_pedido_vista_value(row.pedido_vista)
    row.resultado = normalize_resultado_final(row.resultado, row.classe_processo)
    row.votacao = normalize_votacao(row.votacao)
    if not row.relator:
        row.relator = infer_relator_from_row_text(row)
    if not row.votacao:
        row.votacao = infer_votacao_from_row_text(row)
    inferred_classe = infer_classe_from_row_text(row)
    if inferred_classe and should_replace_classe_processo(row.classe_processo, inferred_classe, row):
        row.classe_processo = inferred_classe
    elif not row.classe_processo:
        row.classe_processo = inferred_classe
    if row.classe_processo in {"ADI", "ADO"}:
        row.add_warning("classe_processo ADI/ADO omitida: TSE não julga ADI/ADO como classe processual.")
        row.classe_processo = ""
    if not row.resultado:
        row.resultado = infer_resultado_from_row_text(row)
    if row.resultado == "Suspenso mas julgado depois" and row.votacao in {"", "Suspenso"}:
        row.votacao = "Suspenso*"
    if row.votacao == "Suspenso*" and row.resultado in {"", "Suspenso por vista"}:
        row.resultado = "Suspenso mas julgado depois"
    if row.resultado == "Suspenso por vista" and not row.votacao:
        row.votacao = "Suspenso"
    if row_indicates_suspension_by_vista(row):
        row.resultado = "Suspenso por vista"
        row.votacao = "Suspenso"
    row.youtube_link = normalize_youtube_link(row.youtube_link)
    row.partes = normalize_party_list(row.partes)
    row.advogados = normalize_advogado_list(row.advogados)
    row.composicao = normalize_composition_list(row.composicao)
    row.fundamentacao_normativa = strip_legacy_fundamentacao_text(normalize_mpe_reference(row.fundamentacao_normativa))
    row.precedentes_citados = normalize_mpe_reference(row.precedentes_citados)
    row.raciocinio_juridico = strip_legacy_raciocinio_text(normalize_mpe_reference(row.raciocinio_juridico))
    row.analise_do_conteudo_juridico = normalize_mpe_reference(row.analise_do_conteudo_juridico)
    if punchline_looks_generic(row.punchline, row):
        row.punchline = infer_punchline_from_row_text(row)
    row.noticia_TSE = normalize_external_url(row.noticia_TSE)
    row.noticia_TRE = normalize_external_url(row.noticia_TRE)
    row.noticias_gerais = normalize_external_url_list(row.noticias_gerais, limit=GENERAL_NEWS_LIMIT)
    row.tema = build_fallback_tema(row)
    if not row.pedido_vista:
        row.pedido_vista = infer_pedido_vista_from_row_text(row)
    apply_rag_consistency_checks(row)
    row.warnings = dedupe_preserve_order(row.warnings)
    row.errors = dedupe_preserve_order(row.errors)

    if original_numero_processo and not row.numero_processo:
        row.add_warning("Número do processo textual inválido removido por falta de identificação confiável.")

    if row.noticia_TSE:
        valid_tse_urls, dropped_tse_urls, irrelevant_tse_urls = filter_relevant_institutional_news_urls(
            [row.noticia_TSE],
            row,
        )
        row.noticia_TSE = valid_tse_urls[0] if valid_tse_urls else ""
        if dropped_tse_urls:
            row.add_warning("noticia_TSE descartada por indisponibilidade da página.")
        if irrelevant_tse_urls:
            row.add_warning("noticia_TSE descartada por baixa aderência ao caso.")
    if row.noticia_TRE:
        valid_tre_urls, dropped_tre_urls, irrelevant_tre_urls = filter_relevant_institutional_news_urls(
            [row.noticia_TRE],
            row,
        )
        row.noticia_TRE = valid_tre_urls[0] if valid_tre_urls else ""
        if dropped_tre_urls:
            row.add_warning("noticia_TRE descartada por indisponibilidade da página.")
        if irrelevant_tre_urls:
            row.add_warning("noticia_TRE descartada por baixa aderência ao caso.")

    if not row.tema:
        row.add_error("Tema/título vazio.")
    if not row.numero_processo:
        row.add_warning("Número do processo não identificado; upsert confiável fica comprometido.")
    if not row.data_sessao:
        row.add_warning("Data da sessão não identificada em formato ISO.")
    if raw_general_news_count > GENERAL_NEWS_LIMIT:
        row.add_warning(
            f"Mais de {GENERAL_NEWS_LIMIT} notícias gerais informadas; apenas as primeiras {GENERAL_NEWS_LIMIT} serão publicadas."
        )

    if notion_schema is None:
        return row

    select_fields = {
        "classe_processo": "classe_processo",
        "tipo_registro": "tipo_registro",
        "eleicao": "eleicao",
        "origem": "origem",
        "tribunal": "tribunal",
        "relator": "relator",
        "pedido_vista": "pedido_vista",
        "resultado": "resultado",
        "votacao": "votacao",
    }
    for internal_name, notion_name in select_fields.items():
        value = getattr(row, internal_name, "")
        if not value:
            continue
        prop = notion_schema.properties.get(notion_name)
        if not prop:
            continue
        if prop.type not in {"select", "status"}:
            continue
        if notion_name in {"relator", "pedido_vista"}:
            canonical_person = _canonicalize_person_select_value(
                value,
                notion_name=notion_name,
                notion_schema=notion_schema,
            )
            if canonical_person:
                setattr(row, internal_name, canonical_person)
                value = canonical_person
        if value not in prop.options:
            if notion_name in {"relator", "pedido_vista"} and value:
                row.add_warning(f"{notion_name} com opção nova no Notion: {value}")
            elif notion_name == "origem":
                row.add_warning(f"origem com opção nova no Notion: {value}")
            elif notion_name == "tipo_registro" and TIPO_REGISTRO_DYNAMIC_RE.match(value):
                row.add_warning(f"{notion_name} com opção nova no Notion: {value}")
            elif value in SAFE_DYNAMIC_SELECT_OPTIONS.get(notion_name, set()):
                row.add_warning(f"{notion_name} com opção nova no Notion: {value}")
            elif notion_name in SCHEMA_EXPANDED_SELECT_PROPERTIES and _controlled_select_value_can_expand(notion_name, value, row):
                row.add_warning(f"{notion_name} com opção nova no Notion: {value}")
            elif notion_name in {"tipo_registro", "eleicao", "classe_processo", "relator", "pedido_vista", "resultado", "votacao"}:
                row.add_warning(f"{notion_name} fora das opções do Notion; valor omitido: {value}")
                setattr(row, internal_name, "")
            else:
                row.add_error(f"Valor inválido para {notion_name}: {value}")

    for internal_name, notion_name in {
        "partes": "partes",
        "advogados": "advogados",
        "composicao": "composicao",
    }.items():
        values = getattr(row, internal_name, [])
        if not values:
            continue
        prop = notion_schema.properties.get(notion_name)
        if not prop or prop.type != "multi_select":
            continue
        missing = [value for value in values if value not in prop.options]
        if missing:
            row.add_warning(
                f"{notion_name} com opções novas no Notion: " + ", ".join(missing)
            )

    return row


def assess_row_publishability(row: PublishPreviewRow) -> tuple[str, list[str]]:
    error_texts = [normalize_model_text(error).lower() for error in row.errors if normalize_model_text(error)]
    if error_texts and all("precedente citado" in error for error in error_texts):
        return "skipped", ["Item descartado: identificado como precedente citado, não como processo julgado."]

    signal_fields = [
        row.numero_processo,
        row.relator,
        row.resultado,
        row.votacao,
        row.pedido_vista,
        row.origem,
        row.tribunal,
        row.analise_do_conteudo_juridico,
        row.punchline,
    ]
    signal_count = sum(1 for value in signal_fields if normalize_model_text(value))
    if signal_count <= 2 and not row.partes and not row.composicao:
        return "skipped", ["Item descartado: densidade informacional insuficiente para representar julgamento autônomo."]

    if row.blocked:
        return "blocked", list(row.errors)

    if row.action != "update":
        create_issues: list[str] = []
        # Numero AUSENTE nao bloqueia mais: casos antigos/administrativos (consultas, PA)
        # efetivamente julgados muitas vezes nao tem numero CNJ. Eles publicam SE passarem
        # pelas guardas de ruido abaixo (resultado/votacao + relator/composicao + densidade
        # + tema). O numero fica em branco (honesto); a identidade/upsert usa a URL do
        # YouTube com timestamp (unica por julgamento) em find_existing_row.
        if not row.data_sessao:
            create_issues.append("Data da sessão ausente para criação.")
        if not (row.resultado or row.votacao or row.pedido_vista):
            create_issues.append("Resultado/votação insuficientes para criação.")
        if not (row.relator or row.composicao):
            create_issues.append("Relator/composição insuficientes para criação.")
        if _tema_looks_generic(row.tema, row) and not row.punchline and not row.analise_do_conteudo_juridico:
            create_issues.append("Tema genérico sem contexto suficiente para criação.")
        if create_issues:
            return "blocked", create_issues

    return "publish", []


def _is_placeholder_numero_processo(numero: str) -> bool:
    digits = re.sub(r"\D", "", str(numero or ""))
    return bool(digits) and set(digits) == {"0"}


INSTITUTIONAL_CONTENT_RE = re.compile(
    r"n[ãa]o apresenta o julgamento|pe[çc]as? (publicit[áa]rias? |)instituciona|"
    r"v[íi]deos? institucionais?|campanhas? de conscientiza[çc][ãa]o",
    re.IGNORECASE,
)


def _looks_like_non_judgment_item(row: PublishPreviewRow) -> bool:
    """Item que não é julgamento e não deve consumir número na sequência do dia.

    Cobre as vinhetas institucionais pré-sessão (biometria, desinformação etc.) que
    a extração devolve com número placeholder (todos os dígitos zero) e os trechos
    cuja própria análise afirma que não há julgamento — casos reais das sessões de
    23/06 a 01/07/2026, em que esses itens viraram "Julgamento 1" e deslocaram a
    numeração de toda a sessão.
    """
    if _is_placeholder_numero_processo(row.numero_processo):
        return True
    if not row.relator and not row.resultado and not row.votacao:
        text = f"{row.tema} {row.analise_do_conteudo_juridico}"
        if INSTITUTIONAL_CONTENT_RE.search(normalize_model_text(text)):
            return True
    return False


def _relatores_da_base(notion_client=None, meses: int = 6) -> list[str]:
    """Relatores gravados na base na janela pedida (um item por julgamento)."""
    import datetime as _dt

    corte = (_dt.date.today() - _dt.timedelta(days=30 * meses)).isoformat()
    paginas = notion_client.query_data_source({"and": [
        {"property": "data_sessao", "date": {"on_or_after": corte}},
        {"property": "relator", "select": {"is_not_empty": True}},
    ]})
    return [
        (((p.get("properties", {}) or {}).get("relator") or {}).get("select") or {}).get("name") or ""
        for p in paginas
    ]


def _elenco_vigente(notion_client=None) -> list[str]:
    """Os 7 titulares do plenario, apurados por relatoria recente; [] se indisponivel.

    A composicao de uma sessao e o colegiado COMPLETO por padrao: em 25/08/2026 os sete
    estavam presentes e a extracao registrou cinco. [] desliga o completamento.
    """
    global _ELENCO_CACHE
    if _ELENCO_CACHE is not None:
        return _ELENCO_CACHE
    _ELENCO_CACHE = []
    # Sem client explicito nao ha I/O: apurar por conta propria faria a suite de testes
    # (e qualquer chamada offline) bater no Notion sem ninguem ter pedido.
    if notion_client is None:
        return _ELENCO_CACHE
    try:
        from gabinete_relator import JANELA_ELENCO_MESES, apurar_elenco

        relatores = _relatores_da_base(notion_client, meses=JANELA_ELENCO_MESES)
        _ELENCO_CACHE = apurar_elenco([{"relator": r} for r in relatores])
    except Exception:  # noqa: BLE001 - sem rede/base, nao completa nada
        _ELENCO_CACHE = []
    return _ELENCO_CACHE


_ELENCO_CACHE: Optional[list[str]] = None


def _membros_por_relatoria(notion_client=None) -> set[str]:
    """Ministros com relatoria na base (janela de gabinete_relator); set() se indisponivel.

    Cache proprio de processo: a consolidacao roda uma vez por lote, mas a funcao e barata
    de chamar e cara de executar (uma query ao Notion). set() vazio desliga a remocao --
    e o comportamento seguro, porque falso positivo aqui APAGA ministro legitimo.
    """
    global _MEMBROS_CACHE
    if _MEMBROS_CACHE is not None:
        return _MEMBROS_CACHE
    _MEMBROS_CACHE = set()
    if notion_client is None:  # ver _elenco_vigente: nada de I/O implicito
        return _MEMBROS_CACHE
    try:
        import datetime as _dt

        from gabinete_relator import JANELA_MESES, apurar_membros

        corte = (_dt.date.today() - _dt.timedelta(days=30 * JANELA_MESES)).isoformat()
        paginas = notion_client.query_data_source({"and": [
            {"property": "data_sessao", "date": {"on_or_after": corte}},
            {"property": "relator", "select": {"is_not_empty": True}},
        ]})
        _MEMBROS_CACHE = apurar_membros([
            {"relator": (((p.get("properties", {}) or {}).get("relator") or {}).get("select") or {}).get("name") or ""}
            for p in paginas
        ])
    except Exception:  # noqa: BLE001 - sem rede/base, segue sem remover ninguem
        _MEMBROS_CACHE = set()
    return _MEMBROS_CACHE


_MEMBROS_CACHE: Optional[set[str]] = None



def consolidate_rows_composicao(
    rows: list[PublishPreviewRow],
    notion_schema: Optional[NotionDataSourceSchema] = None,
    notion_client: Optional["NotionSessoesClient"] = None,
) -> None:
    """Reconstroi a composicao de cada linha a partir do ELENCO vigente do plenario.

    A composicao extraida do video e pouco confiavel: o scan pede a lista em cada uma das
    ~18 janelas, o modelo preenche de memoria e a apuracao por maioria tem uma faixa de
    empate tecnico onde o ruido decide. Ela erra nos dois sentidos --
      - a MENOS: em 25/08/2026 registrou 5 dos 7 presentes, faltando os dois do STJ, que
        nao relataram nada naquela sessao (nenhuma prova interna os alcancava);
      - a MAIS: listava ministros de precedentes historicos, corruptelas do ASR
        ("Min. Diestefali") e quem ja deixou a Corte, chegando a 13+ nomes.

    Por isso a composicao NAO e sanitizada, e sim reconstruida, nesta ordem:
      1. o colegiado vigente (os 7 mais frequentes nas relatorias dos ultimos 3 meses);
      2. mais quem ATUOU nesta sessao (relator ou pedido de vista) e nao esteja nele --
         cobre substituto e convocado, que sao presenca provada;
      3. o que o video trouxe entra so como complemento, e apenas para quem tem relatoria
         na base (ate o cap de 7).
    Ausencia e a excecao e nao aparece no video de forma capturavel: a linha sai com o
    plenario cheio e um aviso para a vistoria corrigir pela ata ou pelo acordao no DJe.

    Sem elenco apurado (base nova, rede fora), cai no comportamento antigo: sanitiza o que
    veio, garante quem atuou e so poda listas maiores que o plenario.
    """
    from gabinete_relator import CADEIRAS_TSE, nao_e_membro

    participantes: list[str] = []
    for row in rows:
        for name in (row.relator, row.pedido_vista):
            canonical = normalize_ministro_name(name) if name else ""
            if canonical and canonical not in participantes:
                participantes.append(canonical)

    allowed: Optional[set[str]] = None
    if notion_schema is not None:
        prop = notion_schema.properties.get("composicao")
        if prop is not None and prop.options:
            allowed = set(prop.options)

    membros = _membros_por_relatoria(notion_client)
    elenco = _elenco_vigente(notion_client)

    for row in rows:
        extraida: list[str] = []
        for name in row.composicao:
            canonical = normalize_ministro_name(name)
            if canonical and canonical not in extraida:
                extraida.append(canonical)

        if elenco:
            comp = list(elenco)
            # substituto/convocado: atuou, logo estava, mesmo fora do elenco titular
            fora_do_elenco = [m for m in participantes if m not in comp]
            if fora_do_elenco:
                comp.extend(fora_do_elenco)
                row.add_warning(
                    "composicao inclui quem atuou fora do colegiado titular (" +
                    ", ".join(fora_do_elenco) + "); confira substituicao/convocacao na ata.")
            # o video so complementa, e so com quem tem relatoria na base
            for nome in extraida:
                if nome in comp or len(comp) >= CADEIRAS_TSE:
                    continue
                if membros and nao_e_membro(membros, nome):
                    continue
                comp.append(nome)
            descartados = [m for m in extraida if m not in comp]
            if descartados:
                row.add_warning(
                    "composicao extraida do video trazia nome fora do colegiado vigente "
                    "(descartado): " + ", ".join(descartados) + ".")
            acrescentados = [m for m in comp if m not in extraida]
            if acrescentados:
                row.add_warning(
                    "composicao completada com o colegiado vigente (" + ", ".join(acrescentados) +
                    "); se houve ausencia nesta sessao, corrija pela ata ou pelo acordao no DJe.")
            row.composicao = comp
            continue

        # --- sem elenco apurado: comportamento conservador de sempre
        comp = list(extraida)
        ausentes = [m for m in participantes if m not in comp]
        if ausentes:
            comp.extend(ausentes)
            row.add_warning(
                "composicao nao trazia quem atuou na sessao (" + ", ".join(ausentes) +
                "); incluidos por relatoria/pedido de vista.")
        # Lista plausível (<=7) fica intacta: um nome legítimo ainda sem opção no
        # Notion não pode ser descartado (o aviso de "opções novas" já denuncia).
        # O saneamento agressivo é só para as listas-salada (>7), que misturam
        # ministros de precedentes históricos e corruptelas de transcrição.
        if len(comp) > CADEIRAS_TSE:
            if allowed is not None:
                dropped = [name for name in comp if name not in allowed]
                if dropped:
                    row.add_warning(
                        "composicao com nomes fora do elenco conhecido (descartados): " + ", ".join(dropped)
                    )
                    comp = [name for name in comp if name in allowed]
            if len(comp) > CADEIRAS_TSE and participantes:
                comp = list(participantes)
                row.add_warning(
                    "composicao excedia o plenário (7); substituída pelos participantes identificados no vídeo (relatores/vistas). Confirme pela ata."
                )
        row.composicao = comp


def build_preview_rows(
    analysis: AnalysisResult,
    youtube_url: str,
    notion_schema: Optional[NotionDataSourceSchema] = None,
    notion_client: Optional[NotionSessoesClient] = None,
) -> list[PublishPreviewRow]:
    rows: list[PublishPreviewRow] = []
    session_composicao = normalize_composition_list(analysis.session.composicao)
    # Aproveita a composicao da sessao como fallback quando ela tiver tamanho plausivel.
    # Antes, QUALQUER divergencia regimental (ex.: um ministro fora do roster, distribuicao
    # institucional atipica) zerava TODO o fallback e deixava a coluna composicao vazia,
    # mesmo havendo os 7 nomes corretos extraidos. O ranking _composition_quality continua
    # preferindo composicoes regimentalmente plenas (3+2+2); a divergencia vira aviso na
    # linha, nao motivo de descarte.
    # O piso desceu de 6 para 5 em 26/08/2026: a extracao da sessao de 25/08 devolveu CINCO
    # nomes (os sete estavam presentes; faltavam no REGISTRO os dois do STJ, que nao
    # relataram nada e por isso nenhuma prova interna os alcancava). Com o piso em 6, uma
    # leitura assim perdia o fallback inteiro e publicava a coluna VAZIA -- e o
    # completamento pelo colegiado vigente, em consolidate_rows_composicao, nem tinha o que
    # completar. Dado incompleto e sinalizado vale mais que dado ausente. Listas com 8+
    # seguem descartadas (exigem reconciliacao/cap) e as de 4 ou menos tambem.
    session_composicao_fallback = session_composicao if 5 <= len(session_composicao) <= 7 else []
    authoritative_session_date = normalize_session_date_to_iso(analysis.session.data_sessao)
    for bundle_index, bundle in enumerate(analysis.bundles, start=1):
        if bundle.should_ignore:
            continue
        for item_index, item in enumerate(_prepare_bundle_items_for_preview(bundle.items), start=1):
            derived_relator, derived_pedido_vista = extract_ministro_roles_from_composition_entries(item.composicao)
            composicao = choose_preferred_composition(item.composicao, session_composicao_fallback)
            origem = item.origem
            item_session_date = normalize_session_date_to_iso(item.data_sessao)
            youtube_link = build_timestamped_youtube_link(youtube_url, bundle.start_seconds)
            if authoritative_session_date and item_session_date and item_session_date != authoritative_session_date:
                youtube_link = build_video_only_youtube_link(youtube_url)
            row = PublishPreviewRow(
                tema=item.tema.strip(),
                classe_processo=item.classe_processo.strip(),
                tipo_registro="",
                eleicao=item.eleicao.strip(),
                origem=origem.strip(),
                tribunal=(item.tre or normalize_tre("", item.uf)).strip(),
                numero_processo=item.numero_processo.strip(),
                youtube_link=youtube_link,
                relator=(item.relator.strip() or derived_relator),
                pedido_vista=(item.pedido_vista.strip() or derived_pedido_vista),
                resultado=item.resultado_final.strip(),
                votacao=item.votacao.strip(),
                data_sessao=analysis.session.data_sessao or item.data_sessao,
                partes=sanitize_name_entries(item.partes + item.indicados_lista_triplice),
                advogados=sanitize_name_entries(item.advogados),
                composicao=composicao,
                punchline=item.punchline.strip(),
                analise_do_conteudo_juridico=item.analise_do_conteudo_juridico.strip(),
                fundamentacao_normativa=build_fundamentacao_column_text(
                    item.fundamentacao_normativa,
                ),
                precedentes_citados=item.precedentes_citados.strip(),
                raciocinio_juridico=build_raciocinio_column_text(
                    item.raciocinio_juridico,
                    item.pontos_processuais_relevantes,
                    item.efeitos_e_providencias_praticas,
                ),
                resolucoes_citadas=item.resolucoes_citadas.strip(),
                source_start_seconds=bundle.start_seconds,
                source_bundle_index=bundle_index,
                source_item_index=item_index,
            )
            row = validate_preview_row(row, notion_schema)
            if _looks_like_non_judgment_item(row):
                # Vinheta institucional/placeholder: não é julgamento e não deve
                # consumir posição na numeração da sessão.
                continue
            row_composition_issue = composicao_regimental_issue(row.composicao)
            if row.composicao and row_composition_issue:
                row.add_warning(
                    "composicao com divergencia regimental ("
                    + row_composition_issue
                    + "); confira os 7 ministros participantes do julgamento. A lista ja "
                      "vem completada com o colegiado vigente — se houve ausencia nesta "
                      "sessao, e aqui que se corrige, pela ata ou pelo acordao no DJe."
                )
            if notion_client and notion_schema and row.youtube_link:
                canon_num = canonicalize_numero_processo(row.numero_processo)
                if canon_num:
                    match = notion_client.find_existing_row(
                        notion_schema,
                        youtube_link=build_video_only_youtube_link(row.youtube_link),
                        numero_processo=canon_num,
                        # mesma sessao em outro video (transmissao em partes) e a MESMA
                        # pagina — ver find_existing_row
                        data_sessao=row.data_sessao,
                    )
                else:
                    # Sem numero: identidade pela URL EXATA (com timestamp) do julgamento,
                    # para o upsert ser idempotente (re-rodar atualiza, nao duplica).
                    match = notion_client.find_existing_row(
                        notion_schema,
                        youtube_link=row.youtube_link,
                        numero_processo="",
                    )
                if match:
                    row.page_id = match.page_id
                    row.action = "update"
            rows.append(row)

    consolidate_rows_composicao(rows, notion_schema, notion_client)
    deduped_rows = _dedupe_preview_rows(rows, youtube_url)
    for index, row in enumerate(deduped_rows, start=1):
        row.tipo_registro = f"Julgamento {index}"
        validate_preview_row(row, notion_schema)
    return deduped_rows


def dedupe_preview_rows(rows: list[PublishPreviewRow], youtube_url: str) -> list[PublishPreviewRow]:
    return _dedupe_preview_rows(rows, youtube_url)


def _prepare_bundle_items_for_preview(items: list[JudgmentItemExtraction]) -> list[JudgmentItemExtraction]:
    prepared: list[JudgmentItemExtraction] = []
    previous_item: JudgmentItemExtraction | None = None

    for item in items:
        candidate = item.model_copy(deep=True)
        if previous_item and _should_inherit_joint_context(previous_item, candidate):
            for field_name in [
                "data_sessao",
                "eleicao",
                "classe_processo",
                "origem",
                "uf",
                "tre",
                "relator",
                "pedido_vista",
                "tema",
                "punchline",
                "analise_do_conteudo_juridico",
                "fundamentacao_normativa",
                "precedentes_citados",
                "raciocinio_juridico",
                "pontos_processuais_relevantes",
                "efeitos_e_providencias_praticas",
                "resolucoes_citadas",
                "votacao",
                "resultado_final",
            ]:
                if not normalize_model_text(getattr(candidate, field_name, "")):
                    setattr(candidate, field_name, getattr(previous_item, field_name, ""))
            if not candidate.partes:
                candidate.partes = list(previous_item.partes)
            if not candidate.advogados:
                candidate.advogados = list(previous_item.advogados)
            if not candidate.composicao:
                candidate.composicao = list(previous_item.composicao)
            if not candidate.indicados_lista_triplice:
                candidate.indicados_lista_triplice = list(previous_item.indicados_lista_triplice)
        prepared.append(candidate)
        previous_item = candidate
    return prepared


def _should_inherit_joint_context(
    previous_item: JudgmentItemExtraction,
    candidate: JudgmentItemExtraction,
) -> bool:
    if canonicalize_numero_processo(previous_item.numero_processo) == canonicalize_numero_processo(candidate.numero_processo):
        return False
    if candidate.partes or candidate.advogados or candidate.composicao:
        return False
    same_class = not candidate.classe_processo or candidate.classe_processo == previous_item.classe_processo
    same_relator = not candidate.relator or candidate.relator == previous_item.relator
    same_result = not candidate.resultado_final or candidate.resultado_final == previous_item.resultado_final
    same_vote = not candidate.votacao or candidate.votacao == previous_item.votacao
    return same_class and same_relator and same_result and same_vote


def _preview_row_signal_score(row: PublishPreviewRow) -> int:
    scalar_fields = [
        row.tema,
        row.classe_processo,
        row.eleicao,
        row.origem,
        row.tribunal,
        row.numero_processo,
        row.relator,
        row.resultado,
        row.votacao,
        row.data_sessao,
        row.punchline,
        row.analise_do_conteudo_juridico,
        row.fundamentacao_normativa,
        row.precedentes_citados,
        row.raciocinio_juridico,
        row.resolucoes_citadas,
    ]
    return sum(1 for value in scalar_fields if normalize_model_text(value)) + len(row.partes) + len(row.advogados)


def _preview_row_numero_specificity(row: PublishPreviewRow) -> int:
    text = normalize_numero_processo_display(row.numero_processo)
    if not text:
        return 0
    if extract_full_cnj(text):
        return 4
    if re.fullmatch(r"\d{3,7}-\d{2}", text):
        return 2
    if re.fullmatch(r"(?:ADO|ADI)\s+\d+", text, flags=re.IGNORECASE):
        return 3
    return 1


def _preview_row_looks_administrative(row: PublishPreviewRow) -> bool:
    text = normalize_class_text(
        " ".join(
            value
            for value in [
                row.tema,
                row.punchline,
                row.analise_do_conteudo_juridico,
                row.raciocinio_juridico,
            ]
            if normalize_model_text(value)
        )
    )
    if not text:
        return False
    administrative_patterns = (
        r"\bresolu[cç][aã]o\b",
        r"\baprova[cç][aã]o\b",
        r"\bminuta\b",
        r"\btexto proposto\b",
        r"\baprovou a resolu[cç][aã]o\b",
        r"\blista triplice\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in administrative_patterns)


def _preview_row_looks_case_specific(row: PublishPreviewRow) -> bool:
    text = normalize_class_text(
        " ".join(
            value
            for value in [
                row.tema,
                row.punchline,
                row.analise_do_conteudo_juridico,
                row.raciocinio_juridico,
                row.origem,
                row.numero_processo,
            ]
            if normalize_model_text(value)
        )
    )
    if not text:
        return False
    judicial_patterns = (
        r"\brecurso especial\b",
        r"\bagr-?respe\b",
        r"\brespe\b",
        r"\badiamento\b",
        r"\bsustentacao oral\b",
        r"\bretirado de pauta\b",
        r"\bproblemas? tecnic",
        r"\bfraude\b",
        r"\bprestacao de contas\b",
        r"\babuso de poder\b",
        r"\bcota de genero\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in judicial_patterns)


def _preview_row_case_specificity_score(row: PublishPreviewRow) -> int:
    score = _preview_row_signal_score(row)
    score += _preview_row_numero_specificity(row) * 3
    if row.origem:
        score += 2
    if row.partes:
        score += 2
    if row.advogados:
        score += 2
    if row.tema and not tema_looks_generic(row.tema, row):
        score += 2
    if row.punchline and not punchline_looks_generic(row.punchline, row):
        score += 1
    if _preview_row_looks_case_specific(row):
        score += 2
    if _preview_row_looks_administrative(row):
        score -= 2
    return score


def _preview_rows_have_semantic_conflict(primary: PublishPreviewRow, secondary: PublishPreviewRow) -> bool:
    if canonicalize_numero_processo(primary.numero_processo) != canonicalize_numero_processo(secondary.numero_processo):
        return False
    if normalize_numero_processo_display(primary.numero_processo) == normalize_numero_processo_display(secondary.numero_processo):
        return False

    primary_admin = _preview_row_looks_administrative(primary)
    secondary_admin = _preview_row_looks_administrative(secondary)
    primary_case = _preview_row_looks_case_specific(primary)
    secondary_case = _preview_row_looks_case_specific(secondary)
    if primary_admin != secondary_admin and primary_case != secondary_case:
        return True

    primary_theme = normalize_class_text(primary.tema)
    secondary_theme = normalize_class_text(secondary.tema)
    if primary_theme and secondary_theme and primary_theme != secondary_theme:
        return True

    return False


def _choose_dedupe_primary(
    existing: PublishPreviewRow,
    candidate: PublishPreviewRow,
) -> tuple[PublishPreviewRow, PublishPreviewRow, bool, bool]:
    same_process = bool(
        canonicalize_numero_processo(existing.numero_processo)
        and canonicalize_numero_processo(existing.numero_processo) == canonicalize_numero_processo(candidate.numero_processo)
    )
    semantic_conflict = same_process and _preview_rows_have_semantic_conflict(existing, candidate)
    if same_process:
        existing_numero_score = _preview_row_numero_specificity(existing)
        candidate_numero_score = _preview_row_numero_specificity(candidate)
        if candidate_numero_score > existing_numero_score:
            return candidate, existing, semantic_conflict, semantic_conflict
        if existing_numero_score > candidate_numero_score:
            return existing, candidate, semantic_conflict, semantic_conflict
        if semantic_conflict:
            existing_score = _preview_row_case_specificity_score(existing)
            candidate_score = _preview_row_case_specificity_score(candidate)
            if candidate_score > existing_score:
                return candidate, existing, True, True
            if existing_score > candidate_score:
                return existing, candidate, True, True

    existing_ts = extract_youtube_timestamp_seconds(existing.youtube_link)
    candidate_ts = extract_youtube_timestamp_seconds(candidate.youtube_link)
    if candidate_ts and (not existing_ts or candidate_ts < existing_ts):
        return candidate, existing, False, False
    if existing_ts and candidate_ts and existing_ts == candidate_ts:
        primary = existing if _preview_row_signal_score(existing) >= _preview_row_signal_score(candidate) else candidate
        secondary = candidate if primary is existing else existing
        return primary, secondary, False, False
    return existing, candidate, False, False


def _merge_preview_row_data(
    primary: PublishPreviewRow,
    secondary: PublishPreviewRow,
    *,
    allow_scalar_backfill: bool = True,
    prefer_primary_source_fields: bool = False,
) -> PublishPreviewRow:
    merged = primary.model_copy(deep=True)

    if allow_scalar_backfill:
        for field_name in [
            "tema",
            "classe_processo",
            "eleicao",
            "origem",
            "tribunal",
            "numero_processo",
            "relator",
            "pedido_vista",
            "resultado",
            "votacao",
            "data_sessao",
            "punchline",
            "analise_do_conteudo_juridico",
            "fundamentacao_normativa",
            "precedentes_citados",
            "raciocinio_juridico",
            "resolucoes_citadas",
            "noticia_TSE",
            "noticia_TRE",
        ]:
            if not normalize_model_text(getattr(merged, field_name, "")):
                setattr(merged, field_name, getattr(secondary, field_name, ""))

    merged.partes = _merge_party_values(merged.partes, secondary.partes)
    merged.advogados = dedupe_preserve_order(merged.advogados + secondary.advogados)
    merged.composicao = _pick_better_composition(merged.composicao, secondary.composicao)
    merged.noticias_gerais = dedupe_preserve_order(merged.noticias_gerais + secondary.noticias_gerais)[:GENERAL_NEWS_LIMIT]
    merged.warnings = dedupe_preserve_order(merged.warnings + secondary.warnings)
    merged.errors = dedupe_preserve_order(merged.errors + secondary.errors)
    if prefer_primary_source_fields:
        (
            merged.source_start_seconds,
            merged.source_bundle_index,
            merged.source_item_index,
        ) = (
            primary.source_start_seconds,
            primary.source_bundle_index,
            primary.source_item_index,
        )
    else:
        (
            merged.source_start_seconds,
            merged.source_bundle_index,
            merged.source_item_index,
        ) = _merge_source_order_fields(primary, secondary)
    if secondary.action == "update" and secondary.page_id:
        merged.action = "update"
        merged.page_id = secondary.page_id
    return merged


def _merge_party_values(primary: list[str], secondary: list[str]) -> list[str]:
    if not primary:
        return list(secondary)
    has_specific_primary = any(not _is_generic_party_label(value) for value in primary)
    merged = list(primary)
    for value in secondary:
        if has_specific_primary and _is_generic_party_label(value):
            continue
        if value and value not in merged:
            merged.append(value)
    return merged


def _is_generic_party_label(value: str) -> bool:
    normalized = normalize_model_text(value).lower()
    if not normalized:
        return False
    generic_markers = (
        "candidato ao cargo de",
        "candidata ao cargo de",
        "prefeito de",
        "vice-prefeita de",
        "vice-prefeito de",
    )
    return any(marker in normalized for marker in generic_markers)


def _preview_row_dedupe_key(row: PublishPreviewRow, youtube_url: str) -> tuple[str, str, str]:
    video_id = extract_youtube_video_id(youtube_url) or normalize_youtube_link(youtube_url)
    process_key = canonicalize_numero_processo(row.numero_processo)
    overlay_classe = identity_overlay_class_key(row.classe_processo)
    if overlay_classe:
        return (video_id, process_key, overlay_classe)
    return (video_id, process_key, "")


def _dedupe_preview_rows(rows: list[PublishPreviewRow], youtube_url: str) -> list[PublishPreviewRow]:
    deduped: dict[tuple[str, str, str], PublishPreviewRow] = {}
    passthrough: list[PublishPreviewRow] = []

    for row in rows:
        if not row.numero_processo:
            passthrough.append(row)
            continue
        key = _preview_row_dedupe_key(row, youtube_url)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
            continue

        primary, secondary, scalar_conflict, prefer_primary_source_fields = _choose_dedupe_primary(existing, row)
        deduped[key] = _merge_preview_row_data(
            primary,
            secondary,
            allow_scalar_backfill=not scalar_conflict,
            prefer_primary_source_fields=prefer_primary_source_fields,
        )

    return sorted(
        passthrough + list(deduped.values()),
        key=_preview_row_sort_key,
    )


def enrich_preview_rows_with_news(
    rows: list[PublishPreviewRow],
    *,
    api_key: str,
    model: str = DEFAULT_NEWS_GEMINI_MODEL,
    artifact_store: Optional[RunArtifacts] = None,
    logger: Optional[logging.Logger] = None,
    enricher: Optional[GeminiNewsEnricher] = None,
    allow_institutional_repair: bool = True,
) -> list[PublishPreviewRow]:
    news_enricher = enricher or GeminiNewsEnricher(
        api_key=api_key,
        model=model,
        artifact_store=artifact_store,
        logger=logger,
        allow_institutional_repair=allow_institutional_repair,
        max_grounding_attempts=1 if not allow_institutional_repair else GEMINI_CALL_RETRIES,
    )
    return news_enricher.enrich_rows(rows)


def enrich_preview_rows_with_process_metadata(
    rows: list[PublishPreviewRow],
    *,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    artifact_store: Optional[RunArtifacts] = None,
    logger: Optional[logging.Logger] = None,
    enricher: Optional[GeminiProcessMetadataEnricher] = None,
    notion_schema: Optional[NotionDataSourceSchema] = None,
) -> list[PublishPreviewRow]:
    metadata_enricher = enricher or GeminiProcessMetadataEnricher(
        api_key=api_key,
        model=model,
        artifact_store=artifact_store,
        logger=logger,
    )
    enriched_rows = metadata_enricher.enrich_rows(rows)
    return [validate_preview_row(row, notion_schema) for row in enriched_rows]


def enrich_preview_rows_with_theme_punchline(
    rows: list[PublishPreviewRow],
    *,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    artifact_store: Optional[RunArtifacts] = None,
    logger: Optional[logging.Logger] = None,
    enricher: Optional[GeminiThemePunchlineEnricher] = None,
    notion_schema: Optional[NotionDataSourceSchema] = None,
) -> list[PublishPreviewRow]:
    text_enricher = enricher or GeminiThemePunchlineEnricher(
        api_key=api_key,
        model=model,
        artifact_store=artifact_store,
        logger=logger,
    )
    enriched_rows = text_enricher.enrich_rows(rows)
    return [validate_preview_row(row, notion_schema) for row in enriched_rows]


def rows_to_editor_records(rows: list[PublishPreviewRow]) -> list[dict[str, Any]]:
    return [row.to_editor_record() for row in rows]


def rows_from_editor_records(
    records: list[dict[str, Any]],
    notion_schema: Optional[NotionDataSourceSchema],
) -> list[PublishPreviewRow]:
    rows: list[PublishPreviewRow] = []
    for record in records:
        row = PublishPreviewRow.from_editor_record(record)
        rows.append(validate_preview_row(row, notion_schema))
    return rows


def _renumber_judgments_after_skips(
    assessed: list[tuple[PublishPreviewRow, str]],
) -> None:
    """Fecha os buracos de numeração deixados por itens descartados (skipped).

    Um item skipped não é julgamento da sessão (ex.: número identificado como
    precedente citado): os "Julgamento N" seguintes do mesmo vídeo/dia descem
    uma posição para a sequência publicada sair contínua. Itens blocked
    preservam o número (o julgamento existiu; pode ser publicado depois na
    posição que reservou) e lotes sem skip não são renumerados — publicações
    avulsas/parciais (scripts de manutenção) ficam intactas.
    """

    def group_key(row: PublishPreviewRow) -> tuple[str, str]:
        return (build_video_only_youtube_link(row.youtube_link), row.data_sessao)

    def judgment_number(row: PublishPreviewRow) -> Optional[int]:
        value = str(row.tipo_registro or "").strip()
        if not TIPO_REGISTRO_DYNAMIC_RE.match(value):
            return None
        digits = re.search(r"\d+", value)
        return int(digits.group(0)) if digits else None

    skipped_by_group: dict[tuple[str, str], list[int]] = {}
    for row, disposition in assessed:
        number = judgment_number(row)
        if disposition == "skipped" and number is not None:
            skipped_by_group.setdefault(group_key(row), []).append(number)
    if not skipped_by_group:
        return

    for row, disposition in assessed:
        if disposition == "skipped":
            continue
        number = judgment_number(row)
        if number is None:
            continue
        shift = sum(1 for skipped in skipped_by_group.get(group_key(row), []) if skipped < number)
        if shift:
            row.tipo_registro = f"Julgamento {number - shift}"


def publish_preview_rows(
    rows: list[PublishPreviewRow],
    notion_client: NotionSessoesClient,
    notion_schema: NotionDataSourceSchema,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    definitive_processos: set[str] = set()
    assessed: list[tuple[PublishPreviewRow, str, list[str]]] = []
    for row in rows:
        row = validate_preview_row(row, notion_schema)
        disposition, reasons = assess_row_publishability(row)
        assessed.append((row, disposition, reasons))
    _renumber_judgments_after_skips([(row, disposition) for row, disposition, _ in assessed])
    for row, disposition, reasons in assessed:
        if disposition == "skipped":
            results.append(
                {
                    "tema": row.tema,
                    "numero_processo": row.numero_processo,
                    "status": "skipped",
                    "errors": [],
                    "warnings": dedupe_preserve_order(row.warnings + reasons),
                }
            )
            continue
        if disposition == "blocked":
            results.append(
                {
                    "tema": row.tema,
                    "numero_processo": row.numero_processo,
                    "status": "blocked",
                    "errors": reasons,
                    "warnings": row.warnings,
                }
            )
            continue
        if row.action == "update" and row.page_id:
            response = notion_client.update_row(notion_schema, row.page_id, row)
            status = "updated"
        else:
            response = notion_client.create_row(notion_schema, row)
            status = "created"
        if row.votacao in DEFINITIVE_VOTACAO and row.numero_processo:
            definitive_processos.add(row.numero_processo)
        results.append(
            {
                "tema": row.tema,
                "numero_processo": row.numero_processo,
                "status": status,
                "page_id": response.get("id", ""),
                "url": response.get("url", ""),
                "errors": [],
                "warnings": row.warnings,
            }
        )

    # Quando um processo foi definitivamente julgado (Unânime/Por maioria) nesta
    # publicação, rebaixa para 'Suspenso*' os registros 'Suspenso' anteriores do
    # mesmo processo. Best-effort: não falha a publicação se a reconciliação falhar.
    if definitive_processos and hasattr(notion_client, "query_data_source"):
        try:
            reconciled = reconcile_suspenso_marks(
                notion_client,
                notion_schema,
                numero_processos=definitive_processos,
                apply=True,
            )
        except Exception as exc:  # pragma: no cover - rede
            results.append({"status": "votacao_reconcile_failed", "errors": [str(exc)], "warnings": []})
        else:
            for change in reconciled:
                results.append(
                    {
                        "numero_processo": change.get("numero_processo", ""),
                        "status": "votacao_reconciled",
                        "page_id": change.get("page_id", ""),
                        "url": change.get("url", ""),
                        "detail": f"{change.get('old_votacao')} -> {change.get('new_votacao')} ({change.get('status')})",
                        "errors": [] if change.get("status") != "failed" else [str(change.get("error", ""))],
                        "warnings": [],
                    }
                )
    return results


def build_runtime_context() -> dict[str, str]:
    return {
        "openai_api_key": get_openai_api_key(),
        "gemini_api_key": get_gemini_api_key(),
        "notion_api_key": get_notion_api_key(),
        "notion_data_source_id": DEFAULT_NOTION_DATA_SOURCE_ID,
        "notion_database_url": DEFAULT_NOTION_DATABASE_URL,
    }
