from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from html import unescape
from pathlib import Path
from typing import Any, Optional

import requests
from pydantic import BaseModel, Field, ValidationError

from local_secrets import get_secret, load_local_secrets
from tse_normalization import (
    build_timestamped_youtube_link,
    canonicalize_numero_processo,
    dedupe_preserve_order,
    extract_full_cnj,
    extract_uf_from_text,
    extract_youtube_video_id,
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
    split_csv_like_text,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR / "artifacts" / "tse_youtube_notion"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_REST_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_NOTION_VERSION = os.getenv("NOTION_VERSION") or "2025-09-03"
DEFAULT_NOTION_DATA_SOURCE_ID = (
    os.getenv("NOTION_DATA_SOURCE_ID") or "2eb72195-5c64-80ea-9cd5-000b0e01745d"
)
DEFAULT_NOTION_DATABASE_URL = (
    os.getenv("NOTION_DATABASE_URL")
    or "https://www.notion.so/2eb721955c64809796bec75a81f9555f?v=6c1e9572d78647038c5dec9e68f688fc"
)
GEMINI_CALL_RETRIES = 3
GEMINI_RETRY_BASE_DELAY = 2.0
DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS = int(os.getenv("GEMINI_HTTP_TIMEOUT_SECONDS") or "90")
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
        "RHC",
        "RMS",
        "RO",
        "RPP",
        "RvE",
        "TutCautAnt",
    },
    "votacao": {"Unânime", "Por maioria", "Suspenso"},
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
    payload: dict[str, Any] = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    if use_google_search:
        payload["tools"] = [{"googleSearch": {}}]

    url = f"{GEMINI_REST_BASE_URL}/models/{model_name}:generateContent?key={api_key}"
    response = requests.post(url, json=payload, timeout=(10, timeout_seconds))
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini REST error {response.status_code}: {response.text[:2000]}")
    response_payload = response.json()
    response_text = _extract_generate_content_text(response_payload)
    return _coerce_gemini_response_model(response_model, response_text), response_text, response_payload


GLOBAL_SCAN_WINDOW_SECONDS = int(os.getenv("GLOBAL_SCAN_WINDOW_SECONDS") or "300")
GLOBAL_SCAN_OVERLAP_SECONDS = int(os.getenv("GLOBAL_SCAN_OVERLAP_SECONDS") or "30")
GLOBAL_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS = int(os.getenv("GLOBAL_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS") or "3")
GLOBAL_SCAN_FALLBACK_WINDOW_SECONDS = int(os.getenv("GLOBAL_SCAN_FALLBACK_WINDOW_SECONDS") or "120")
GLOBAL_SCAN_FALLBACK_OVERLAP_SECONDS = int(os.getenv("GLOBAL_SCAN_FALLBACK_OVERLAP_SECONDS") or "15")
TRANSCRIPT_SCAN_MAX_CHARS = int(os.getenv("TRANSCRIPT_SCAN_MAX_CHARS") or "4000")
TRANSCRIPT_SCAN_OVERLAP_SNIPPETS = int(os.getenv("TRANSCRIPT_SCAN_OVERLAP_SNIPPETS") or "1")
TRANSCRIPT_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS = int(os.getenv("TRANSCRIPT_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS") or "2")
TRANSCRIPT_DETAIL_PADDING_SECONDS = int(os.getenv("TRANSCRIPT_DETAIL_PADDING_SECONDS") or "45")
TRANSCRIPT_DETAIL_MAX_CHARS = int(os.getenv("TRANSCRIPT_DETAIL_MAX_CHARS") or "6000")
REFINE_START_LOOKBACK_SECONDS = int(os.getenv("REFINE_START_LOOKBACK_SECONDS") or "90")
REFINE_START_LOOKAHEAD_SECONDS = int(os.getenv("REFINE_START_LOOKAHEAD_SECONDS") or "90")
ENABLE_START_REFINEMENT = env_flag("TSE_ENABLE_START_REFINEMENT", True)
ENABLE_TRANSITION_REFINEMENT = env_flag("TSE_ENABLE_TRANSITION_REFINEMENT", True)
CONDITIONAL_START_REFINEMENT = env_flag("TSE_CONDITIONAL_START_REFINEMENT", True)
GROUND_PROCESS_METADATA_FOR_ORIGEM_ONLY = env_flag("TSE_GROUND_ORIGEM_WITH_SEARCH", False)
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
)
RECOMPUTED_WARNING_PATTERNS = (
    re.compile(r"^(?:classe_processo|tipo_registro|eleicao|origem|relator|pedido_vista|resultado|votacao) fora das opções do Notion;", re.IGNORECASE),
    re.compile(r"^(?:origem|partes|advogados|composicao) com opções novas no Notion:", re.IGNORECASE),
    re.compile(r"^Tema/título vazio\.$", re.IGNORECASE),
    re.compile(r"^Número do processo não identificado;", re.IGNORECASE),
    re.compile(r"^Data da sessão não identificada em formato ISO\.$", re.IGNORECASE),
    re.compile(r"^Mais de \d+ notícias gerais informadas;", re.IGNORECASE),
    re.compile(r"^noticia_TSE descartada por indisponibilidade da página\.$", re.IGNORECASE),
    re.compile(r"^noticia_TRE descartada por indisponibilidade da página\.$", re.IGNORECASE),
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
- identificar a composição do colegiado presente;
- listar cada julgamento com seu timestamp inicial em segundos;
- estimar o timestamp final quando possível;
- marcar explicitamente blocos que devam ser ignorados, especialmente o julgamento em lista ao final da sessão.

Se houver julgamento conjunto, mantenha um único bloco para o trecho conjunto, mas liste os números de processo percebidos.
"""

TRANSCRIPT_GLOBAL_SYSTEM_PROMPT = """
Você é um juiz eleitoral incumbido de analisar tecnicamente uma sessão do Tribunal Superior Eleitoral (TSE) com base exclusivamente na transcrição do próprio vídeo do YouTube.

Use apenas a transcrição fornecida no prompt. Não use qualquer fonte externa. Não complete lacunas com suposições.

Nesta etapa, sua função é segmentar a sessão:
- identificar a data da sessão;
- identificar a composição do colegiado presente;
- listar cada julgamento com seu timestamp inicial em segundos;
- estimar o timestamp final quando possível;
- marcar explicitamente blocos que devam ser ignorados, especialmente julgamento em lista, leitura de ata ou trechos meramente cerimoniais/administrativos.

Se houver julgamento conjunto, mantenha um único bloco para o trecho conjunto, mas liste os números de processo percebidos.
"""

DETAIL_SYSTEM_PROMPT = """
Você é um juiz eleitoral incumbido de analisar tecnicamente o conteúdo de uma sessão do Tribunal Superior Eleitoral (TSE) transmitida em vídeo do YouTube.

FONTE:
- Use exclusivamente o conteúdo do próprio vídeo.
- Se a informação não estiver no vídeo, deixe o campo vazio.
- Ignore totalmente julgamento em lista ao final da sessão.

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
- `tema`: informe um tema jurídico curto, indexável e aderente ao caso concreto, como "conduta vedada por uso de bens públicos" ou "fraude à cota de gênero". Nunca repita o número do processo. Nunca use apenas rótulos genéricos como "Processo", "Julgamento" ou só a classe processual. Se o vídeo não permitir identificar o tema com segurança, deixe o campo vazio.
"""

TRANSCRIPT_DETAIL_SYSTEM_PROMPT = """
Você é um juiz eleitoral incumbido de analisar tecnicamente um julgamento do Tribunal Superior Eleitoral (TSE) com base exclusivamente na transcrição do próprio vídeo do YouTube.

FONTE:
- Use apenas a transcrição do vídeo fornecida no prompt.
- Não use qualquer fonte externa.
- Se a informação não estiver claramente na transcrição, deixe o campo vazio.
- Ignore totalmente julgamento em lista ao final da sessão.

TAREFA:
- Analise apenas o julgamento delimitado pelos timestamps e pela transcrição fornecidos.
- Se o trecho contiver mais de um processo julgado em conjunto, retorne um item por processo.
- Não crie item para número de processo citado apenas como precedente, comparação, referência jurisprudencial ou exemplo.
- Extraia os mesmos campos exigidos na etapa detalhada do vídeo, preservando fidelidade máxima ao conteúdo efetivamente transcrito.
- `tema`: informe um tema jurídico curto, indexável e aderente ao caso concreto. Nunca use número do processo, "Processo", "Julgamento" ou apenas a classe processual. Se não houver base suficiente na transcrição, deixe vazio.
"""

NEWS_ENRICHMENT_SYSTEM_PROMPT = """
Você é um pesquisador jurídico-eleitoral.

TAREFA:
- Use exclusivamente o Grounding with Google Search para localizar notícias públicas relacionadas ao mesmo caso, decisão ou sessão informada.
- Não invente URLs.
- Só retorne links claramente relacionados ao mesmo item processual, julgamento ou sessão.
- Se não houver notícia confiável, retorne listas vazias.
"""

PROCESS_METADATA_SYSTEM_PROMPT = """
Você é um pesquisador jurídico-eleitoral.

TAREFA:
- Use exclusivamente o Grounding with Google Search.
- Complete prioritariamente o número CNJ integral do processo.
- Só informe a cidade/UF de origem quando ela vier de forma clara junto da mesma evidência relevante para o processo. Não faça busca isolada apenas para preencher origem.
- Se os resultados indicarem que o número consultado aparece apenas como precedente citado, e não como processo efetivamente julgado na sessão, marque is_judged_process=false.
- Não invente número completo, origem ou classificação do item.
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


def infer_theme_from_row_text(row: "PublishPreviewRow") -> str:
    sources = [
        normalize_model_text(row.punchline),
        normalize_model_text(row.analise_do_conteudo_juridico),
        normalize_model_text(row.raciocinio_juridico),
        normalize_model_text(row.fundamentacao_normativa),
    ]
    pattern_builders: list[tuple[re.Pattern[str], Any]] = [
        (
            re.compile(r"(?i)erro material(?: de [^.,;]+)? no valor da multa"),
            lambda m: "Retificação de erro material no valor da multa",
        ),
        (
            re.compile(r"(?i)fundo partid[aá]rio.*consultoria jur[ií]dica e cont[aá]bil"),
            lambda m: "Uso do Fundo Partidário para custear consultoria jurídica e contábil",
        ),
        (
            re.compile(r"(?i)integridade do sistema eletr[oô]nico de vota[cç][aã]o"),
            lambda m: "Integridade do sistema eletrônico de votação nas eleições de 2022",
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
            re.compile(r"(?i)inelegibilidade(?: [^.,;]+)?"),
            lambda m: m.group(0),
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


def _tema_looks_generic(value: str, row: "PublishPreviewRow") -> bool:
    normalized = normalize_class_text(value)
    if not normalized:
        return True
    if _looks_like_process_number_theme(value, row):
        return True
    if GENERIC_THEME_CLASS_RESULT_REGEX.fullmatch(normalized):
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
    }


def tema_looks_generic(value: str, row: "PublishPreviewRow") -> bool:
    return _tema_looks_generic(value, row)


def build_fallback_tema(row: "PublishPreviewRow") -> str:
    candidates = [
        ((row.tema or "").strip() if not _tema_looks_generic((row.tema or "").strip(), row) else ""),
        ((row.punchline or "").strip().rstrip(".") if not _looks_like_process_number_theme((row.punchline or "").strip(), row) else ""),
        infer_theme_from_row_text(row),
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
            normalize_model_text(row.tema),
            normalize_model_text(row.punchline),
            normalize_model_text(row.analise_do_conteudo_juridico),
            normalize_model_text(row.raciocinio_juridico),
            normalize_model_text(row.fundamentacao_normativa),
            normalize_model_text(row.precedentes_citados),
        ]
        if value
    ).strip()


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


def infer_classe_from_row_text(row: "PublishPreviewRow") -> str:
    text = _build_row_inference_text(row)
    normalized = normalize_class_text(text)
    if not normalized:
        return ""
    if "lista triplice" in normalized or "lista tríplice" in normalized:
        return "Lista Tríplice"
    if "consulta formulada" in normalized or normalized.startswith("consulta ") or "trata de uma consulta" in normalized:
        return "CTA"
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
    if candidate and candidate != text.strip():
        return candidate
    return ""


def infer_origin_from_row_text(row: "PublishPreviewRow") -> str:
    text = _build_row_inference_text(row)
    if not text:
        return ""
    city_pattern = (
        r"\b("
        r"[A-ZÁÀÃÂÉÊÍÓÔÕÚÇ][A-Za-zÁÀÃÂÉÊÍÓÔÕÚÇáàãâéêíóôõúç'`´.\-]+"
        r"(?:\s+(?:de|do|da|dos|das|e)\s+"
        r"[A-ZÁÀÃÂÉÊÍÓÔÕÚÇ][A-Za-zÁÀÃÂÉÊÍÓÔÕÚÇáàãâéêíóôõúç'`´.\-]+)*"
        r"(?:\s+[A-ZÁÀÃÂÉÊÍÓÔÕÚÇ][A-Za-zÁÀÃÂÉÊÍÓÔÕÚÇáàãâéêíóôõúç'`´.\-]+)*"
        r")/([A-Z]{2})\b"
    )
    matches = list(re.finditer(city_pattern, text))
    for match in reversed(matches):
        city = match.group(1).strip(" ,.;:-")
        uf = match.group(2).upper()
        if city and not city.upper().startswith("TRE"):
            return f"{city}/{uf}"
    return ""


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
    return split_csv_like_text(cleaned)


def normalize_advogado_list(values: list[str]) -> list[str]:
    normalized = normalize_advogados_list(", ".join(values))
    return split_csv_like_text(normalized)


def normalize_composition_list(values: list[str]) -> list[str]:
    normalized = normalize_composicao(", ".join(values))
    return split_csv_like_text(normalized)


def _composition_quality(values: list[str]) -> tuple[int, int]:
    normalized = normalize_composition_list(values)
    count = len(normalized)
    if count == 7:
        return (100, count)
    if 5 <= count <= 9:
        return (90 - abs(7 - count), count)
    if 1 <= count < 5:
        return (50 + count, count)
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
        return resolved or normalized
    except Exception:
        return normalized


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


def classify_news_urls(urls: list[str]) -> tuple[list[str], list[str], list[str]]:
    tse_urls: list[str] = []
    tre_urls: list[str] = []
    general_urls: list[str] = []

    for url in normalize_external_url_list(urls):
        domain = domain_from_url(url)
        if not domain:
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
    candidates = [
        "conduta vedada" if "conduta vedada" in fold_text_for_match(row.tema + " " + row.punchline) else "",
        "bens publicos" if "bens publicos" in fold_text_for_match(row.tema + " " + row.punchline) else "",
        "campanha eleitoral" if "campanha eleitoral" in fold_text_for_match(row.tema + " " + row.punchline) else "",
    ]
    return [value for value in candidates if value]


def is_general_news_url_relevant(url: str, row: "PublishPreviewRow") -> bool:
    page_text = fold_text_for_match(fetch_candidate_page_text(url))
    if not page_text:
        return False

    full_process = fold_text_for_match(extract_full_cnj(row.numero_processo))
    short_process = fold_text_for_match(canonicalize_numero_processo(row.numero_processo))
    city_marker, uf_marker = _extract_origin_markers(row.origem)
    party_markers = _extract_party_markers(row.partes)
    theme_markers = _extract_theme_markers(row)

    process_hit = bool(full_process and full_process in page_text) or bool(short_process and short_process in page_text)
    city_hit = bool(city_marker and city_marker in page_text)
    uf_hit = bool(uf_marker and uf_marker in page_text)
    party_hits = sum(1 for marker in party_markers if marker and marker in page_text)
    theme_hits = sum(1 for marker in theme_markers if marker and marker in page_text)
    tribunal_hits = sum(1 for marker in [fold_text_for_match(row.tribunal), "tse"] if marker and marker in page_text)

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

    strong_context_hit = process_hit or (city_hit and party_hits >= 1) or (city_hit and theme_hits >= 1 and tribunal_hits >= 1)
    return strong_context_hit and score >= 5


def filter_general_news_urls(urls: list[str], row: "PublishPreviewRow") -> list[str]:
    accepted: list[str] = []
    for url in normalize_external_url_list(urls):
        if is_general_news_url_relevant(url, row):
            accepted.append(url)
    return dedupe_preserve_order(accepted)


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
    composicao: list[str] = Field(default_factory=list)
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


class JudgmentItemExtraction(BaseModel):
    data_sessao: str = ""
    eleicao: str = ""
    classe_processo: str = ""
    numero_processo: str = ""
    origem: str = ""
    uf: str = ""
    tre: str = ""
    partes: list[str] = Field(default_factory=list)
    advogados: list[str] = Field(default_factory=list)
    composicao: list[str] = Field(default_factory=list)
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
    confidence: str = ""
    rationale: str = ""


class ThemeRepairResult(BaseModel):
    tema: str = ""
    confidence: str = ""
    rationale: str = ""


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
            "noticia_TSE": self.noticia_TSE,
            "noticia_TRE": self.noticia_TRE,
            "noticias_gerais": ", ".join(self.noticias_gerais),
            "page_id": self.page_id,
            "action": self.action,
            "warnings": "\n".join(self.warnings),
            "errors": "\n".join(self.errors),
            "blocked": self.blocked,
        }


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
            },
        )
        for field_name in ("partes", "advogados", "composicao", "indicados_lista_triplice"):
            value = normalized_item.get(field_name)
            normalized_item[field_name] = parse_multi_value_text(value)
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


def _coerce_gemini_response_model(response_model: type[BaseModel], response_text: str) -> BaseModel:
    try:
        payload = json.loads(response_text)
    except Exception:
        return response_model.model_validate_json(response_text)

    model_name = response_model.__name__
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

    return response_model.model_validate(payload)

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
            analise_do_conteudo_juridico=str(record.get("analise_do_conteudo_juridico", "") or ""),
            fundamentacao_normativa=str(record.get("fundamentacao_normativa", "") or ""),
            precedentes_citados=str(record.get("precedentes_citados", "") or ""),
            raciocinio_juridico=str(record.get("raciocinio_juridico", "") or ""),
            resolucoes_citadas=str(record.get("resolucoes_citadas", "") or ""),
            noticia_TSE=str(record.get("noticia_TSE", "") or ""),
            noticia_TRE=str(record.get("noticia_TRE", "") or ""),
            noticias_gerais=parse_multi_value_text(record.get("noticias_gerais", "")),
            page_id=str(record.get("page_id", "") or ""),
            action=str(record.get("action", "") or "create"),
            warnings=split_csv_like_text(str(record.get("warnings", "") or "").replace("\n", ", ")),
            errors=split_csv_like_text(str(record.get("errors", "") or "").replace("\n", ", ")),
        )


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
        self._transcript_snippets_cache: list[TranscriptSnippet] | None = None

    def analyze_session(self, youtube_url: str) -> AnalysisResult:
        normalized_url = normalize_youtube_link(youtube_url)
        if self.artifact_store.exists("01_session_windows.json"):
            session = SessionExtraction.model_validate(
                self.artifact_store.read_json("01_session_windows.json")
            )
        else:
            session = self._extract_session_windows(normalized_url)
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

Marque como should_ignore=true qualquer bloco de julgamento em lista ou equivalente.
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
                return self._merge_session_chunks(extracted_chunks)
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
    ) -> list[SessionExtraction]:
        extracted_chunks: list[SessionExtraction] = []
        consecutive_failures_without_success = 0
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
Marque como should_ignore=true qualquer bloco de "julgamento em lista" ou equivalente.
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
            extracted_chunks.append(chunk_result)
            self.artifact_store.write_json(
                f"{artifact_prefix}_chunk_{chunk_index:02d}.json",
                chunk_result.model_dump(mode="json"),
            )
        if not extracted_chunks:
            raise RuntimeError(f"Nenhum chunk global foi extraído com sucesso no plano {plan_label}.")
        return extracted_chunks

    def _merge_session_chunks(self, chunks: list[SessionExtraction]) -> SessionExtraction:
        merged = SessionExtraction(data_sessao="", composicao=[], judgments=[])
        seen_composicao: set[str] = set()
        preferred_dates: list[str] = []
        fallback_dates: list[str] = []
        collected_windows: list[SessionWindow] = []

        for chunk in chunks:
            chunk_date = normalize_session_date_to_iso(normalize_model_text(chunk.data_sessao))
            for ministro in chunk.composicao:
                ministro = normalize_model_text(ministro)
                if not ministro:
                    continue
                if ministro not in seen_composicao:
                    seen_composicao.add(ministro)
                    merged.composicao.append(ministro)
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
        merged.judgments = self._coalesce_windows(collected_windows)
        merged.judgments.sort(key=lambda item: (item.start_seconds, item.title_hint))
        return merged

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
- Se o bloco for apenas julgamento em lista, marque should_ignore=true e explique.
- Seja fiel ao vídeo e deixe em branco o que não estiver explícito.
"""
        try:
            bundle = self._call_gemini(
                youtube_url=youtube_url,
                prompt=prompt,
                response_model=JudgmentBundleExtraction,
                system_prompt=DETAIL_SYSTEM_PROMPT,
                start_seconds=refined_start_seconds,
                end_seconds=window.end_seconds,
                artifact_name=f"raw_detail_{index:02d}.txt",
            )
        except Exception as exc:
            self.logger.warning(
                "Falha na extração detalhada em vídeo do bloco %s; tentando fallback por transcrição: %s",
                index,
                exc,
            )
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
- Se o trecho for apenas julgamento em lista, marque should_ignore=true e explique.
- Seja fiel apenas ao que estiver explicitamente transcrito.
"""
            bundle = self._call_gemini_text(
                prompt=transcript_prompt,
                response_model=JudgmentBundleExtraction,
                system_prompt=TRANSCRIPT_DETAIL_SYSTEM_PROMPT,
                artifact_name=f"raw_detail_transcript_{index:02d}.txt",
            )
        bundle.title_hint = window.title_hint
        bundle.start_seconds = refined_start_seconds
        bundle.end_seconds = window.end_seconds
        return bundle

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
            cache_filename = f"06_news_enrichment_{index:02d}.json"
            if self.artifact_store.exists(cache_filename):
                cached_payload = self.artifact_store.read_json(cache_filename)
                candidate = row.model_copy(deep=True)
                applied = cached_payload.get("applied") or {}
                candidate.noticia_TSE = str(applied.get("noticia_TSE", "") or "")
                candidate.noticia_TRE = str(applied.get("noticia_TRE", "") or "")
                candidate.noticias_gerais = list(applied.get("noticias_gerais", []) or [])
                enriched_rows.append(candidate)
                continue
            context = build_news_enrichment_context(row)
            if not context:
                candidate = row.model_copy(deep=True)
                candidate.add_warning("Enriquecimento de notícias ignorado por falta de contexto.")
                enriched_rows.append(candidate)
                continue

            prompt = (
                "Busque notícias públicas sobre o mesmo item processual, julgamento ou sessão abaixo.\n"
                "Priorize notícias institucionais do TSE, depois TREs, depois imprensa geral.\n"
                "Retorne apenas URLs realmente relacionadas ao mesmo caso.\n\n"
                f"Contexto:\n{context}"
            )
            response, grounding_urls = self._call_grounded_json(
                prompt=prompt,
                response_model=NewsEnrichmentResult,
                artifact_name=f"06_news_enrichment_{index:02d}.txt",
            )
            tse_urls, tre_urls, general_urls = classify_news_urls(
                response.noticia_TSE + response.noticia_TRE + response.noticia_geral + grounding_urls
            )
            valid_tse_urls, dropped_tse_urls = filter_accessible_news_urls(tse_urls)
            valid_tre_urls, dropped_tre_urls = filter_accessible_news_urls(tre_urls)
            if not valid_tse_urls and dropped_tse_urls:
                repaired_tse_urls = self._repair_institutional_urls(
                    context=context,
                    broken_urls=dropped_tse_urls,
                    domain_hint="tse.jus.br",
                    domain_label="TSE",
                    artifact_name=f"06_news_repair_tse_{index:02d}.txt",
                )
                valid_tse_urls, extra_dropped_tse_urls = filter_accessible_news_urls(repaired_tse_urls)
                dropped_tse_urls = dedupe_preserve_order(dropped_tse_urls + extra_dropped_tse_urls)
            if not valid_tre_urls and dropped_tre_urls:
                repaired_tre_urls = self._repair_institutional_urls(
                    context=context,
                    broken_urls=dropped_tre_urls,
                    domain_hint=domain_from_url(dropped_tre_urls[0]) or "tre-xx.jus.br",
                    domain_label="TRE",
                    artifact_name=f"06_news_repair_tre_{index:02d}.txt",
                )
                valid_tre_urls, extra_dropped_tre_urls = filter_accessible_news_urls(repaired_tre_urls)
                dropped_tre_urls = dedupe_preserve_order(dropped_tre_urls + extra_dropped_tre_urls)
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
                    "general_candidates": general_urls,
                    "general_filtered_out": dropped_general_urls,
                },
            )
        return enriched_rows

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
        for attempt in range(1, GEMINI_CALL_RETRIES + 1):
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
                    GEMINI_CALL_RETRIES,
                    exc,
                )
                if should_disable_model(exc):
                    break
                if attempt < GEMINI_CALL_RETRIES:
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


class GeminiProcessMetadataEnricher:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        artifact_store: Optional[RunArtifacts] = None,
        logger: Optional[logging.Logger] = None,
        client: Any = None,
        ground_origem_with_search: bool = GROUND_PROCESS_METADATA_FOR_ORIGEM_ONLY,
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
        self.ground_origem_with_search = ground_origem_with_search

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
            if has_full_cnj and has_origem:
                enriched_rows.append(row)
                continue
            if has_full_cnj and not self.ground_origem_with_search:
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
                "Se o número pesquisado aparecer apenas como precedente citado, marque is_judged_process=false.\n\n"
                f"Contexto:\n{context}"
            )
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
            if response.full_numero_processo:
                candidate.numero_processo = response.full_numero_processo
            if response.origem:
                candidate.origem = response.origem
            if response.is_judged_process is False:
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
    ) -> None:
        if not api_key:
            raise ValueError("NOTION_API_KEY/NOTION_TOKEN não encontrado.")
        self.data_source_id = data_source_id
        self.logger = logger or logging.getLogger(__name__)
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Notion-Version": notion_version,
                "Content-Type": "application/json",
            }
        )
        self.base_url = "https://api.notion.com/v1"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self.session.request(method, self.base_url + path, timeout=60, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"Notion API error {response.status_code}: {response.text}")
        if not response.content:
            return {}
        return response.json()

    def fetch_schema(self) -> NotionDataSourceSchema:
        payload = self._request("GET", f"/data_sources/{self.data_source_id}")
        schema = NotionDataSourceSchema(self.data_source_id, payload)
        schema.ensure_expected_properties()
        return schema

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
        return ""

    def find_existing_row(
        self,
        schema: NotionDataSourceSchema,
        youtube_link: str,
        numero_processo: str,
    ) -> Optional[NotionRowMatch]:
        candidates: list[dict[str, Any]] = []
        numero_filter = self.build_filter_condition(schema, "numero_processo", numero_processo)
        if numero_filter:
            candidates.extend(self.query_data_source(numero_filter))
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
            exact_youtube_match = candidate_youtube == normalized_youtube
            if candidate_numero == normalized_numero and (
                not target_video_id or exact_youtube_match or same_video
            ):
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
            return {"rich_text": chunk_rich_text(str(value))}
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
        payload = {
            "parent": {"type": "data_source_id", "data_source_id": self.data_source_id},
            "properties": self.build_properties_payload(schema, row),
        }
        return self._request("POST", "/pages", json=payload)

    def update_row(self, schema: NotionDataSourceSchema, page_id: str, row: PublishPreviewRow) -> dict[str, Any]:
        payload = {"properties": self.build_properties_payload(schema, row)}
        return self._request("PATCH", f"/pages/{page_id}", json=payload)


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
    row.data_sessao = normalize_session_date_to_iso(row.data_sessao)
    row.numero_processo = normalize_numero_processo_display(row.numero_processo)
    row.classe_processo = normalize_classe_processo(row.classe_processo)
    row.eleicao = normalize_eleicao_value(row.eleicao)
    row.origem = normalize_origem_value(row.origem)
    uf = extract_uf_from_text(row.origem)
    row.tribunal = normalize_tre(row.tribunal, uf)
    row.relator = normalize_ministro_name(row.relator) if row.relator else ""
    row.pedido_vista = normalize_pedido_vista_value(row.pedido_vista)
    row.resultado = normalize_resultado_final(row.resultado, row.classe_processo)
    row.votacao = normalize_votacao(row.votacao)
    row.youtube_link = normalize_youtube_link(row.youtube_link)
    row.partes = normalize_party_list(row.partes)
    row.advogados = normalize_advogado_list(row.advogados)
    row.composicao = normalize_composition_list(row.composicao)
    row.fundamentacao_normativa = strip_legacy_fundamentacao_text(normalize_mpe_reference(row.fundamentacao_normativa))
    row.precedentes_citados = normalize_mpe_reference(row.precedentes_citados)
    row.raciocinio_juridico = strip_legacy_raciocinio_text(normalize_mpe_reference(row.raciocinio_juridico))
    row.analise_do_conteudo_juridico = normalize_mpe_reference(row.analise_do_conteudo_juridico)
    row.noticia_TSE = normalize_external_url(row.noticia_TSE)
    row.noticia_TRE = normalize_external_url(row.noticia_TRE)
    row.noticias_gerais = normalize_external_url_list(row.noticias_gerais, limit=GENERAL_NEWS_LIMIT)
    row.tema = build_fallback_tema(row)
    row.warnings = dedupe_preserve_order(row.warnings)
    row.errors = dedupe_preserve_order(row.errors)

    if row.noticia_TSE:
        valid_tse_urls, dropped_tse_urls = filter_accessible_news_urls([row.noticia_TSE])
        row.noticia_TSE = valid_tse_urls[0] if valid_tse_urls else ""
        if dropped_tse_urls:
            row.add_warning("noticia_TSE descartada por indisponibilidade da página.")
    if row.noticia_TRE:
        valid_tre_urls, dropped_tre_urls = filter_accessible_news_urls([row.noticia_TRE])
        row.noticia_TRE = valid_tre_urls[0] if valid_tre_urls else ""
        if dropped_tre_urls:
            row.add_warning("noticia_TRE descartada por indisponibilidade da página.")

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
        if value not in prop.options:
            if notion_name == "origem":
                row.add_warning(f"origem com opção nova no Notion: {value}")
            elif value in SAFE_DYNAMIC_SELECT_OPTIONS.get(notion_name, set()):
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
        if not row.numero_processo:
            create_issues.append("Número do processo ausente para criação.")
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


def build_preview_rows(
    analysis: AnalysisResult,
    youtube_url: str,
    notion_schema: Optional[NotionDataSourceSchema] = None,
    notion_client: Optional[NotionSessoesClient] = None,
) -> list[PublishPreviewRow]:
    rows: list[PublishPreviewRow] = []
    session_composicao = normalize_composition_list(analysis.session.composicao)
    session_composicao_fallback = session_composicao if 5 <= len(session_composicao) <= 9 else []
    for bundle_index, bundle in enumerate(analysis.bundles, start=1):
        if bundle.should_ignore:
            continue
        for item_index, item in enumerate(_prepare_bundle_items_for_preview(bundle.items), start=1):
            composicao = choose_preferred_composition(item.composicao, session_composicao_fallback)
            origem = item.origem
            row = PublishPreviewRow(
                tema=item.tema.strip(),
                classe_processo=item.classe_processo.strip(),
                tipo_registro="",
                eleicao=item.eleicao.strip(),
                origem=origem.strip(),
                tribunal=(item.tre or normalize_tre("", item.uf)).strip(),
                numero_processo=item.numero_processo.strip(),
                youtube_link=build_timestamped_youtube_link(youtube_url, bundle.start_seconds),
                relator=item.relator.strip(),
                pedido_vista=item.pedido_vista.strip(),
                resultado=item.resultado_final.strip(),
                votacao=item.votacao.strip(),
                data_sessao=item.data_sessao or analysis.session.data_sessao,
                partes=item.partes + item.indicados_lista_triplice,
                advogados=item.advogados,
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
            if notion_client and notion_schema and row.numero_processo and row.youtube_link:
                match = notion_client.find_existing_row(
                    notion_schema,
                    youtube_link=row.youtube_link,
                    numero_processo=canonicalize_numero_processo(row.numero_processo),
                )
                if match:
                    row.page_id = match.page_id
                    row.action = "update"
            rows.append(row)

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


def _merge_preview_row_data(primary: PublishPreviewRow, secondary: PublishPreviewRow) -> PublishPreviewRow:
    merged = primary.model_copy(deep=True)

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


def _dedupe_preview_rows(rows: list[PublishPreviewRow], youtube_url: str) -> list[PublishPreviewRow]:
    video_id = extract_youtube_video_id(youtube_url) or normalize_youtube_link(youtube_url)
    deduped: dict[tuple[str, str], PublishPreviewRow] = {}
    passthrough: list[PublishPreviewRow] = []

    for row in rows:
        if not row.numero_processo:
            passthrough.append(row)
            continue
        key = (video_id, canonicalize_numero_processo(row.numero_processo))
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
            continue

        existing_ts = extract_youtube_timestamp_seconds(existing.youtube_link)
        candidate_ts = extract_youtube_timestamp_seconds(row.youtube_link)
        if candidate_ts and (not existing_ts or candidate_ts < existing_ts):
            primary = row
            secondary = existing
        elif existing_ts and candidate_ts and existing_ts == candidate_ts:
            primary = existing if _preview_row_signal_score(existing) >= _preview_row_signal_score(row) else row
            secondary = row if primary is existing else existing
        else:
            primary = existing
            secondary = row
        deduped[key] = _merge_preview_row_data(primary, secondary)

    return sorted(
        passthrough + list(deduped.values()),
        key=_preview_row_sort_key,
    )


def enrich_preview_rows_with_news(
    rows: list[PublishPreviewRow],
    *,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    artifact_store: Optional[RunArtifacts] = None,
    logger: Optional[logging.Logger] = None,
    enricher: Optional[GeminiNewsEnricher] = None,
) -> list[PublishPreviewRow]:
    news_enricher = enricher or GeminiNewsEnricher(
        api_key=api_key,
        model=model,
        artifact_store=artifact_store,
        logger=logger,
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


def publish_preview_rows(
    rows: list[PublishPreviewRow],
    notion_client: NotionSessoesClient,
    notion_schema: NotionDataSourceSchema,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        row = validate_preview_row(row, notion_schema)
        disposition, reasons = assess_row_publishability(row)
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
    return results


def build_runtime_context() -> dict[str, str]:
    return {
        "gemini_api_key": get_gemini_api_key(),
        "notion_api_key": get_notion_api_key(),
        "notion_data_source_id": DEFAULT_NOTION_DATA_SOURCE_ID,
        "notion_database_url": DEFAULT_NOTION_DATABASE_URL,
    }
