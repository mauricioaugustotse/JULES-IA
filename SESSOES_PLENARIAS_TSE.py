# -*- coding: utf-8 -*-
"""
Converte relatórios `.txt` de sessões plenárias do TSE em CSV via OpenAI.

Fluxo de trabalho:
1. Carrega configuração/logs e resolve arquivos de entrada (CLI ou seletor GUI).
2. Para cada relatório, envia trechos à OpenAI com prompt jurídico e retries.
3. Converte o JSON retornado em linhas tabulares por processo julgado.
4. Aplica normalizações (processo, classe, composição, advogados, TRE, datas e MPE).
5. Grava CSV/checkpoint/report por arquivo na pasta do script.
"""

import os
import csv
import json
import time
import logging
import argparse
import threading
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from openai import OpenAI, APIError
from gui_intuitiva import open_file_panel
from Artefatos.scripts.openai_log_utils import configure_standard_logging
from Artefatos.scripts.openai_progress_utils import (
    build_file_signature,
    make_backup,
    read_json_dict,
    same_file_signature,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
)

OPENAI_DEFAULT_MODEL = "gpt-5-mini"
OPENAI_DEFAULT_TIMEOUT = 45
OPENAI_DEFAULT_RETRIES = 3
OPENAI_DEFAULT_BASE_DELAY = 1.0
OPENAI_DEFAULT_TARGET_RPM = 180
OPENAI_DEFAULT_VISTA_RETRIES = 2
CHECKPOINT_VERSION = 1
SCRIPT_DIR = Path(__file__).resolve().parent


class RequestPacer:
    def __init__(self, target_rpm: int) -> None:
        rpm = max(0, int(target_rpm))
        self._min_interval = (60.0 / float(rpm)) if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait_turn(self) -> None:
        if self._min_interval <= 0.0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_at:
                    self._next_at = now + self._min_interval
                    return
                sleep_for = max(0.0, self._next_at - now)
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.2))


OPENAI_TIMEOUT_SECONDS = OPENAI_DEFAULT_TIMEOUT
OPENAI_RETRY_ATTEMPTS = OPENAI_DEFAULT_RETRIES
OPENAI_BASE_DELAY_SECONDS = OPENAI_DEFAULT_BASE_DELAY
OPENAI_VISTA_RETRIES = OPENAI_DEFAULT_VISTA_RETRIES
OPENAI_REQUEST_PACER = RequestPacer(OPENAI_DEFAULT_TARGET_RPM)

# --- 1. Configuração e Prompts ---

# Carrega as variáveis de ambiente (sua chave da API)
load_dotenv()

class FriendlyFormatter(logging.Formatter):
    LEVEL_MAP = {
        "DEBUG": "DETALHE",
        "INFO": "INFO",
        "WARNING": "AVISO",
        "ERROR": "ERRO",
        "CRITICAL": "ERRO",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.levelname = self.LEVEL_MAP.get(record.levelname, record.levelname)
        return super().format(record)

def configure_logging(args: argparse.Namespace) -> None:
    global SHOW_TECHNICAL_LOGS
    SHOW_TECHNICAL_LOGS = bool(args.debug)
    configure_standard_logging(
        "SESSOES_PLENARIAS_TSE",
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
        debug=bool(args.debug),
        log_file=str(args.log_file or ""),
    )

# Inicializa o cliente da OpenAI
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=0)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("A variável OPENAI_API_KEY não foi encontrada no arquivo .env.")
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível inicializar o cliente da OpenAI. Verifique seu arquivo .env. Detalhes: {e}")
    exit(1)


def _openai_chat_completion(payload: Dict[str, Any]) -> Any:
    args = dict(payload)
    args["timeout"] = max(5, int(OPENAI_TIMEOUT_SECONDS))
    try:
        return client.chat.completions.create(**args)
    except TypeError:
        args.pop("timeout", None)
        return client.chat.completions.create(**args)

# Instrução principal para a IA, focada em precisão e no contexto jurídico.
SYSTEM_PROMPT = """
Você é um assistente de IA especializado em análise de documentos jurídicos, focado na extração de dados de relatórios de sessões do TSE.
Sua tarefa é analisar meticulosamente o relatório fornecido e preencher CADA CAMPO do esquema JSON com a informação correspondente de forma detalhada e completa.
Geralmente são julgados de 1 a 3 processos por sessão. Por isso, a quantidade de linhas no CSV final varia de acordo com cada julgado apreciado no TSE.

REGRAS IMPORTANTES:
1.  **EXTRAÇÃO COMPLETA**: Faça o seu melhor para preencher todos os campos. Se uma informação não estiver, deixe o campo como uma string vazia `""`.
2.  **REGRAS DE FOCO**: Foco na extração de "data_sessao", "numero_processo", "partes", "advogados", "tema", "analise_do_conteudo_juridico" e "raciocinio_juridico", pois são PONTOS CRÍTICOS DE ALTA RELEVÂNCIA.
3.  **PADRÃO DE PREENCHIMENTO**: Para campos que esperam múltiplos valores (como 'composicao', 'partes', 'advogados'), extraia TODOS os itens e separe-os usando vírgula seguido de um espaço (`, `).
4.  **SAÍDA ESTRITA**: O resultado final DEVE SER um único objeto JSON válido, sem nenhum texto, comentário, explicação ou formatação `markdown` (como ```json) antes ou depois.
5.  **ITEM ÚNICO**: Quando o texto fornecido for uma seção individual de um processo, retorne APENAS 1 item no array "itens".
6.  **JULGAMENTO CONJUNTO**: Se houver mais de um número de processo no texto, retorne um item por processo no array "itens". Se algum dado for comum, repita-o nos itens.
7.  **PADRÕES ESPECÍFICOS**:
    - `numero_processo`: use o formato curto `0000000-00` (preserve zeros à esquerda). Se houver CNJ completo, extraia apenas a parte curta.
    - `classe_processo`: use rótulos canônicos (ex.: REspe, AgRg-REspe, AREspe, RO, CTA, QO, PC, PA, Lista Tríplice, RPP, RvE, TutCautAnt, Ref-TutCautAnt, Ref.-MS, MS, RMS, RHC).
    - `resultado_final`: use rótulos canônicos (ex.: Aprovada, Aprovadas, Aprovada com ressalvas, Aprovadas com ressalvas, Desprovido, Desprovida, Desprovadas, Provido, Provido em parte, Não conhecido, Não conhecida, Suspenso por vista, Referendada, Referendado, Indeferido, Indeferida, Deferido, Parcialmente deferido, Rejeitada, Rejeitados, Acolhidos, Acolhido em parte, Acolhida em parte, Acolhidos em parte, Prejudicado, Devolvida, "Provido, Não conhecido", "Prejudicado, Desprovido").
    - `TRE`: use o formato `TRE-UF` quando a UF estiver indicada.
    - `composicao`, `relator` e `pedido_vista`: sempre use o prefixo `Min.` antes do nome.
8.  **RACIOCÍNIO JURÍDICO DETALHADO**: Explique claramente a tese vencedora, a tese divergente (se houver), e indique ministros vencedores e vencidos quando o texto citar. Se a decisão for unânime, deixe isso explícito.
9.  **ADVOGADOS**: Sempre prefixe os nomes com `Dr.` (masculino) ou `Dra.` (feminino).
10. **MINISTÉRIO PÚBLICO ELEITORAL (MPE)**: Quando houver menção ao Ministério Público Eleitoral, use a sigla `MPE`. Não liste representante/procurador do MPE em `advogados` nem em `composicao`, e não inclua MPE em `partes`.
"""

# Esquema que define a estrutura dos dados a serem extraídos.
JSON_SCHEMA = {
    "sessao": {
        "data_sessao": "string (M/D/AAAA)",
        "composicao": "string (nomes dos ministros presentes, separados por ', 'Sempre usar 'Min.' antes do nome",
    },
    "itens": [
        {
            "numero_processo": "string (ex: 0600264-60, 0600758-23, etc.)",
            "eleicao": "string (ex: 2024, 2022, 2020, etc.)",
            "classe_processo": "string (ex: REspe, AgRg-REspe, AREspe, RO, CTA, QO, PC, PA, Lista Tríplice, RPP, RvE, TutCautAnt, Ref-TutCautAnt, Ref.-MS, MS, RMS, RHC).",
            "origem": "string (Município/UF).",
            "TRE": "string (ex: TRE-SP)",
            "partes": "string (NOMES dos candidatos, dos partidos políticos, das coligações envolvidas e dos indicados em lista tríplice, separados por ', ')",
            "advogados": "string (NOMES de todos os advogados, separados por ', ', sempre prefixar 'Dr.' ou 'Dra.' conforme o gênero)",
            "relator": "string (o nome do relator ou relatora da decisão. Sempre usar 'Min.' antes do nome)",
            "tipo_registro": "string (ex: 'Julgamento 1' | 'Julgamento 2' | 'Julgamento 3'",
            "analise_do_conteudo_juridico": "string (resumo robusto do contexto fático julgado e do tema eleitoral)",
            "tema": "string (resume o assunto tratado em até 20 palavras)",
            "punchline": "string (Crie uma frase de efeito, curta e de alto impacto, resumindo o julgado.)",
            "resoluções_citadas": "string (resoluções do TSE citadas, separadas por ', ' com número, ano e artigo.",
            "fundamentacao_normativa": "string (principais artigos, leis e súmulas citados, separados por ', '",
            "precedentes_citados": "string (resumo dos precedentes citados, separados por ', ')",
            "raciocinio_juridico": "string (descrição robusta da tese vencedora, a tese divergente se houver, os argumentos, a conclusão final do colegiado e os ministros vencedores/vencidos quando citados.)",
            "pedido_vista": "string (nome do ministro que pediu vista, se houver. Sempre usar 'Min.' antes do nome. Se não houver, deixar vazio)",
            "resultado_final": "string (somente essas: Aprovada, Aprovadas, Aprovada com ressalvas, Aprovadas com ressalvas, Desprovido, Desprovida, Desprovadas, Provido, Provido em parte, Não conhecido, Não conhecida, Suspenso por vista, Referendada, Referendado, Indeferido, Indeferida, Deferido, Parcialmente deferido, Rejeitada, Rejeitados, Acolhidos, Acolhido em parte, Acolhida em parte, Acolhidos em parte, Prejudicado, Devolvida, \"Provido, Não conhecido\", \"Prejudicado, Desprovido\". Se houver mais de um resultado, separar por ', ')",
            "votacao": "string (somente essas: Unânime, Por maioria, Suspenso)",
            "youtube_link": "string (links do vídeo da sessão no YouTube, separados por ', ' se houver mais de um. UTILIZAR SEMPRE links similares a esse: 'http://www.youtube.com/watch?v=K1QFVHrtzqg)",
        }
    ]
}

# JSON Schema para forçar saída estruturada via response_format.
JSON_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sessao": {
            "type": "object",
            "properties": {
                "data_sessao": {"type": "string"},
                "composicao": {"type": "string"},
            },
            "required": ["data_sessao", "composicao"],
            "additionalProperties": False,
        },
        "itens": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "numero_processo": {"type": "string"},
                    "eleicao": {"type": "string"},
                    "classe_processo": {"type": "string"},
                    "origem": {"type": "string"},
                    "TRE": {"type": "string"},
                    "partes": {"type": "string"},
                    "advogados": {"type": "string"},
                    "relator": {"type": "string"},
                    "tipo_registro": {"type": "string"},
                    "analise_do_conteudo_juridico": {"type": "string"},
                    "tema": {"type": "string"},
                    "punchline": {"type": "string"},
                    "resoluções_citadas": {"type": "string"},
                    "fundamentacao_normativa": {"type": "string"},
                    "precedentes_citados": {"type": "string"},
                    "raciocinio_juridico": {"type": "string"},
                    "pedido_vista": {"type": "string"},
                    "resultado_final": {"type": "string"},
                    "votacao": {"type": "string"},
                    "youtube_link": {"type": "string"},
                },
                "required": [
                    "numero_processo",
                    "eleicao",
                    "classe_processo",
                    "origem",
                    "TRE",
                    "partes",
                    "advogados",
                    "relator",
                    "tipo_registro",
                    "analise_do_conteudo_juridico",
                    "tema",
                    "punchline",
                    "resoluções_citadas",
                    "fundamentacao_normativa",
                    "precedentes_citados",
                    "raciocinio_juridico",
                    "pedido_vista",
                    "resultado_final",
                    "votacao",
                    "youtube_link",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sessao", "itens"],
    "additionalProperties": False,
}

VISTA_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "pedido_vista": {"type": "string"},
    },
    "required": ["pedido_vista"],
    "additionalProperties": False,
}

CSV_COLUMN_ORDER = [
    "data_sessao",
    "TRE",
    "advogados",
    "analise_do_conteudo_juridico",
    "classe_processo",
    "composicao",
    "eleicao",
    "fundamentacao_normativa",
    "numero_processo",
    "origem",
    "partes",
    "pedido_vista",
    "precedentes_citados",
    "punchline",
    "raciocinio_juridico",
    "relator",
    "resoluções_citadas",
    "resultado_final",
    "tema",
    "tipo_registro",
    "votacao",
    "youtube_link",
]

SHOW_TECHNICAL_LOGS = False

CLASSE_PROCESSO_MAP = [
    (r"\bembargos de declaracao\s+nos?\s+embargos de declaracao\b.*\b(lista triplice|lt)\b", "ED-ED-LT"),
    (r"\bed\s+ed\s+lt\b", "ED-ED-LT"),
    (r"\bembargos de declaracao\b.*\b(agravo regimental|agravo interno|agrg)\b.*\b(agravo em recurso especial eleitoral|arespe)\b", "ED-AgRg-AREspe"),
    (r"\bed\s+agrg\s+arespe\b", "ED-AgRg-AREspe"),
    (r"\bembargos de declaracao\b.*\b(agravo em recurso especial eleitoral|arespe)\b", "ED-AREspe"),
    (r"\bembargos de declaracao\b.*\b(recurso especial eleitoral|respe)\b", "ED-REspe"),
    (r"\bembargos de declaracao\b.*\b(recurso ordinario|ro)\b", "ED-RO"),
    (r"\bembargos de declaracao\b.*\b(prestacao de contas|pc)\b", "ED-PC"),
    (r"\bembargos de declaracao\b.*\b(peticao civel|petciv)\b", "ED-PetCiv"),
    (r"\bembargos de declaracao\b.*\b(lista triplice|lt)\b", "ED-Lista Tríplice"),
    (r"\bed\s+lt\b", "ED-Lista Tríplice"),
    (r"\bagr\s+hc\b", "AgR-HC"),
    (r"\bagrhc\b", "AgR-HC"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(habeas corpus|hc)\b", "AgRg-HC"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(recurso em habeas corpus|rhc)\b", "AgRg-RHC"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(recurso especial eleitoral|respe)\b", "AgRg-REspe"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(agravo em recurso especial eleitoral|arespe)\b", "AgRg-AREspe"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(recurso ordinario|ro)\b", "AgRg-RO"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(mandado de seguranca|ms)\b", "AgRg-MS"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(prestacao de contas|pc)\b", "AgRg-PC"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(agravo de instrumento|ai)\b", "AgRg-AI"),
    (r"\b(agravo regimental|agravo interno|agrg)\b.*\b(acao rescisoria|ar)\b", "AgRg-AR"),
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
    (r"\bcri[aã]c[aã]o\s+de\s+zona\s+eleitoral\b", "Czer"),
    (r"\brpp\b", "RPP"),
    (r"\brve\b", "RvE"),
    (r"\bczer\b", "Czer"),
]

CNJ_REGEX = r"\b\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b"
SHORT_PROCESSO_REGEX = r"\b\d{3,7}-\d{2}\b"
LABELED_PROCESSO_REGEX = r"(?i)\bn(?:[º°]|\.)\s*(\d{3,7})\b"
MINISTRO_NAME_PART = (
    r"(?:[A-ZÁÉÍÓÚÂÊÔÃÕÀÜÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÀÜÇáéíóúâêôãõàüç'\-]+"
    r"|de|da|do|dos|das|e|d'|di|del|la|le|van|von)"
)
MINISTRO_NAME_PART_STRICT = r"[A-ZÁÉÍÓÚÂÊÔÃÕÀÜÇ][A-Za-zÁÉÍÓÚÂÊÔÃÕÀÜÇáéíóúâêôãõàüç'\-]+"
MINISTRO_NAME_BODY = rf"{MINISTRO_NAME_PART}(?:\s+{MINISTRO_NAME_PART}){{0,6}}"
MINISTRO_NAME_BODY_STRICT = rf"{MINISTRO_NAME_PART_STRICT}(?:\s+{MINISTRO_NAME_PART}){{0,6}}"
MINISTRO_NAME_WITH_PREFIX_PATTERN = rf"(?:(?i:Min\.)|(?i:Ministra)|(?i:Ministro))\s+{MINISTRO_NAME_BODY}"
LEADERSHIP_NAME_WITHOUT_PREFIX_PATTERN = rf"(?i:(?:Presidente|Vice-?Presidente))\s+{MINISTRO_NAME_BODY}"
MINISTRO_NAME_WITH_PREFIX_STRICT_PATTERN = (
    rf"(?:(?i:Min\.)|(?i:Ministra)|(?i:Ministro))\s+{MINISTRO_NAME_BODY_STRICT}"
)
LEADERSHIP_NAME_WITHOUT_PREFIX_STRICT_PATTERN = (
    rf"(?i:(?:Presidente|Vice-?Presidente))\s+{MINISTRO_NAME_BODY_STRICT}"
)
MINISTRO_NAME_REGEX = re.compile(
    rf"\b{MINISTRO_NAME_WITH_PREFIX_PATTERN}"
)
MINISTRO_NAME_STRICT_REGEX = re.compile(
    rf"\b{MINISTRO_NAME_WITH_PREFIX_STRICT_PATTERN}"
)
LEADERSHIP_NAME_REGEX = re.compile(
    rf"\b{LEADERSHIP_NAME_WITHOUT_PREFIX_STRICT_PATTERN}"
)
EMPTY_ADVOGADOS_REGEX = re.compile(
    r"(?i)\b("
    r"n[ãa]o\s+(?:citad\w*|mencionad\w*|informad\w*|consta\w*|houve|h[áa]|aplic[aá]vel)"
    r"|sem\s+sustenta[cç][aã]o"
    r"|sem\s+advogad\w*"
    r"|n/?a"
    r")\b"
)
MPE_REFERENCE_REGEX = re.compile(
    r"(?i)\bminist[ée]rio\s+p[úu]blico\s+eleitoral\b"
)
MINISTERIO_PUBLICO_REGEX = re.compile(
    r"(?i)\bminist[ée]rio\s+p[úu]blico\b"
)
MPE_ABBREV_REGEX = re.compile(
    r"(?i)\bmp\s*eleitoral\b|\bm\.?\s*p\.?\s*e\.?\b|\bmpe\b"
)
MPE_REPRESENTATIVE_REGEX = re.compile(
    r"(?i)\b(vice-?procurador(?:-geral)?\s+eleitoral|"
    r"procurador(?:-geral)?(?:\s+regional)?\s+eleitoral|"
    r"procuradoria\s+geral\s+eleitoral|pge)\b"
)
PROCURADOR_GENERIC_REGEX = re.compile(
    r"(?i)\b(vice-?procurador(?:-geral)?|procurador(?:-geral)?|procuradoria)\b"
)

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
    cleaned = []
    for part in parts:
        if part == "MPE" or is_mpe_noise_entry(part):
            continue
        cleaned.append(part)
    return ", ".join(dedupe_preserve_order(cleaned))

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

def get_user_prompt(
    report_text: str,
    base_context: Optional[Dict[str, Any]] = None,
    prefilled_fields: Optional[Dict[str, Any]] = None,
    single_item: bool = False,
    process_numbers: Optional[List[str]] = None
) -> str:
    """Cria o prompt do usuário com as instruções, o esquema e o texto do relatório."""
    base_context = base_context or {}
    prefilled_fields = prefilled_fields or {}

    context_block = ""
    if base_context:
        context_block += "\nCONTEXTO BASE (use se estiver coerente com o texto):\n"
        context_block += json.dumps(base_context, indent=2, ensure_ascii=False)
    if prefilled_fields:
        context_block += "\n\nCAMPOS PRE-PREENCHIDOS (use como base e corrija se necessário):\n"
        context_block += json.dumps(prefilled_fields, indent=2, ensure_ascii=False)

    single_item_note = "\nRETORNE APENAS 1 item no array \"itens\".\n" if single_item else ""
    joint_note = ""
    if process_numbers and len(process_numbers) > 1 and not single_item:
        joined = ", ".join(process_numbers)
        joint_note = f"\nPROCESSOS IDENTIFICADOS (retorne um item por processo):\n{joined}\n"

    return f"""
Por favor, extraia os dados do relatório de sessão do TSE abaixo.
Siga RIGOROSAMENTE o esquema JSON fornecido para estruturar sua resposta.
{single_item_note}
{joint_note}
{context_block}

ESQUEMA JSON DE SAÍDA:
{json.dumps(JSON_SCHEMA, indent=2, ensure_ascii=False)}

RELATÓRIO PARA ANÁLISE:
---
{report_text}
---
"""

# --- 2. Extração e Processamento de Dados ---

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

def extract_full_cnj(text: str) -> str:
    match = re.search(CNJ_REGEX, text)
    return match.group(0) if match else ""

def extract_short_processo(text: str) -> str:
    match = re.search(SHORT_PROCESSO_REGEX, text)
    return match.group(0) if match else ""

def extract_labeled_short_processo(text: str) -> str:
    match = re.search(LABELED_PROCESSO_REGEX, text)
    return match.group(1) if match else ""

def clean_label_value(value: str) -> str:
    value = value.replace("**", "").strip()
    value = re.sub(r"^\*+\s*", "", value).strip()
    value = re.sub(r"\s*\*+$", "", value).strip()
    return value.rstrip(".").strip()

def normalize_token(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")

CANON_CSV_FILENAME = "padrões para canonizar.csv"
CANON_DATA: Optional[Dict[str, Any]] = None

def normalize_class_text(value: str) -> str:
    if not value:
        return ""
    value = normalize_text(value)
    value = normalize_token(value)
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def get_canonization_data() -> Dict[str, Any]:
    global CANON_DATA
    if CANON_DATA is not None:
        return CANON_DATA
    data: Dict[str, Any] = {
        "classes": set(),
        "results": set(),
        "class_results": {},
        "class_norm_map": {},
        "result_norm_map": {},
    }
    path = os.path.join(os.path.dirname(__file__), CANON_CSV_FILENAME)
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            class_results: Dict[str, set] = {}
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

FEMALE_NAME_HINTS = {
    "ana", "maria", "mariana", "marina", "carla", "claudia", "clara", "camila",
    "carolina", "caroline", "beatriz", "bianca", "renata", "fernanda", "patricia",
    "luciana", "lucia", "marcia", "sandra", "silvana", "viviane", "oneida",
    "andreia", "adriana", "tatiana", "vanessa", "aline", "leticia", "raquel",
    "cristina", "cristiane", "gisele", "giovana", "gabriela", "isabel", "marta",
    "claudia", "carmen", "estela", "juliana", "julia", "amanda", "bruna",
    "daniela", "elaine", "eliane", "joana", "laura", "luana", "marcela", "patricia",
}

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

def is_empty_advogados_value(value: str) -> bool:
    if not value:
        return True
    return bool(EMPTY_ADVOGADOS_REGEX.search(value))

def normalize_advogado_name(name: str, label_hint: str = "") -> str:
    name = name.strip()
    if not name:
        return ""
    if is_mpe_noise_entry(name):
        return ""
    if is_empty_advogados_value(name):
        return ""
    suffix = ""
    if "(" in name:
        base, extra = name.split("(", 1)
        name = base.strip()
        suffix = " (" + extra.strip()
    name = name.rstrip(".;:,").strip()
    if not name:
        return ""
    if is_empty_advogados_value(name):
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

def normalize_advogados_list(value: str, label_hint: str = "") -> str:
    if not value:
        return ""
    if is_empty_advogados_value(value) and not re.search(r"(?i)[,;]|\bdr\.|\bdra\.|\be\b", value):
        return ""
    text = normalize_text(value)
    parts = split_advogados_entries(text)
    normalized = []
    for part in parts:
        part = re.sub(r"(?i)^(advogad[oa]s?|defensor[oa]s?)\s*:?\s*", "", part).strip()
        if not part:
            continue
        if is_mpe_noise_entry(part):
            continue
        if is_empty_advogados_value(part):
            continue
        normalized_name = normalize_advogado_name(part, label_hint)
        if normalized_name:
            normalized.append(normalized_name)
    return ", ".join(dedupe_preserve_order(normalized))

def split_advogados_entries(text: str) -> List[str]:
    parts = []
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

def dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    unique = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique

def extract_process_numbers_from_line(line: str) -> List[str]:
    line = normalize_text(line)
    fulls = re.findall(CNJ_REGEX, line)
    if fulls:
        return dedupe_preserve_order(fulls)
    shorts = re.findall(SHORT_PROCESSO_REGEX, line)
    if shorts:
        return dedupe_preserve_order(shorts)
    labeled = re.findall(LABELED_PROCESSO_REGEX, line)
    return dedupe_preserve_order(labeled)

def extract_ministro_names_from_line(line: str) -> List[str]:
    line = normalize_text(line).strip()
    if not line:
        return []
    line = re.sub(r"^\s*[*\-•]+\s*", "", line)
    line = re.sub(r"^\s*\d+\.\s*", "", line)
    line = re.sub(r"^\s*\d+\)\s*", "", line)
    line = line.replace("**", "").strip()
    line = re.sub(r"^\*{1,2}|\*{1,2}$", "", line).strip()
    line = re.split(r"(?i)\b(?:nota|obs|observa[cç][aã]o|não votam|nao votam)\s*:", line)[0].strip()
    line = re.sub(r"\s+e\s+", ", ", line)
    lower_line = line.lower()
    if (
        ("procurador" in lower_line or "pge" in lower_line or "mpe" in lower_line
         or "ministério público" in lower_line or "ministerio publico" in lower_line)
        and not re.search(r"(?i)\bmin\.|\bministro\b|\bministra\b", line)
    ):
        return []
    parts = re.split(r"\s*[;,]\s*", line)
    names = []
    label_prefix = (
        r"(ministros?\s+presentes?|"
        r"ministers?\s+present(?:es)?|"
        r"ministros?\s+que\s+comp[oó]em\s+a\s+sess[aã]o|"
        r"composi[cç][aã]o\s+da\s+(?:mesa(?:\s+julgadora)?|"
        r"corte(?:\s+na\s+sess[aã]o)?|plen[aá]rio)|"
        r"presid[êe]ncia(?:\s*\(.*?\))?|"
        r"presidente(?:\s+da\s+sess[aã]o|\s+do\s+julgamento)?|"
        r"vice-?presidente|"
        r"ministros?\s+substitutos?(?:\s+mencionados?)?|"
        r"ministras?\s+substitutas?(?:\s+mencionadas?)?|"
        r"aus[eê]nci\w*|"
        r"ausent\w*)"
    )
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = re.sub(rf"(?i)^{label_prefix}\s*:?\s*", "", part).strip()
        part = part.lstrip(":-*• ").strip()
        part = re.sub(r"\s*\(.*?\)\s*", "", part).strip()
        part = part.lstrip(":-*• ").strip()
        if not part:
            continue
        lower = part.lower()
        if is_mpe_noise_entry(part):
            continue
        if "composi" in lower and ("completa" in lower or "mesma" in lower or "vide item" in lower):
            continue
        if "presidente da república" in lower or "presidente da republica" in lower:
            continue
        if "ministério" in lower or "ministerio" in lower:
            continue
        if "secretário" in lower or "secretario" in lower:
            continue
        if not re.search(r"[A-Za-zÁÉÍÓÚÂÊÔÃÕÇ]", part):
            continue
        names.append(normalize_ministro_name(part))
    return [name for name in names if name]

def is_ministros_block_terminator(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("---"):
        return True
    if re.match(r"^#{2,4}\s+", stripped):
        return True
    if re.search(r"(?i)^\s*\*{0,2}(MINIST[ÉE]RIO|SECRET[ÁA]RIO|PRESID[ÊE]NCIA|DATA DA SESS[ÃA]O)\b", stripped):
        return True
    if re.search(r"(?i)^\s*\*{0,2}[A-ZÁÉÍÓÚÃÕÇ][A-ZÁÉÍÓÚÃÕÇ\s\-]+:\s*", stripped):
        return True
    return False

def extract_dados_processo_block(text: str) -> str:
    match = re.search(
        r"(?im)^\s*.*\bDADOS\s+DO(?:S)?\s+(?:PROCESSO(?:S)?|CASO)\b.*$",
        text
    )
    if not match:
        return ""
    start = match.end()
    next_match = re.search(
        r"(?im)^\s*(?:#{2,4}\s+|"
        r"\*{0,2}(?:ANÁLISE DO CONTEÚDO JURÍDICO|ANÁLISE|"
        r"RACIOCÍNIO DESENVOLVIDO|RACIOCÍNIO|CONCLUSÃO|"
        r"PONTOS PROCESSUAIS|EFEITOS|OBS|OBSERVA)\b.*)$",
        text[start:]
    )
    end = start + next_match.start() if next_match else len(text)
    return text[start:end].strip()

def extract_process_numbers_from_section(text: str) -> Tuple[List[str], bool]:
    numbers = []
    plural_label = False
    retificacao_present = False
    header_lines = text.splitlines()[:5]
    for line in header_lines:
        if re.search(r"retific", line, flags=re.IGNORECASE):
            retificacao_present = True
        line_numbers = extract_process_numbers_from_line(line)
        if line_numbers:
            numbers.extend(line_numbers)
    for match in re.finditer(
        r"(?im)^\s*[*-]?\s*\*{0,2}Processos?\*{0,2}\s*:\s*([^\n]+)",
        text
    ):
        label = match.group(0)
        if re.search(r"Processos", label, flags=re.IGNORECASE):
            plural_label = True
        line = clean_label_value(match.group(1))
        line = normalize_text(line)
        if re.search(r"retific", line, flags=re.IGNORECASE):
            retificacao_present = True
        line_numbers = extract_process_numbers_from_line(line)
        if line_numbers:
            numbers.extend(line_numbers)

    dados_block = extract_dados_processo_block(text)
    if dados_block:
        skip_labels = (
            "jurisprud",
            "precedent",
            "legisla",
            "fundamenta",
            "resolu",
            "súmula",
            "sumula",
        )
        for raw_line in dados_block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if any(label in line.lower() for label in skip_labels):
                continue
            line_numbers = extract_process_numbers_from_line(line)
            if line_numbers:
                numbers.extend(line_numbers)

    numbers = dedupe_preserve_order(numbers)
    has_joint_keyword = bool(
        re.search(
            r"(julgament[oa]s?\s+conjunt[oa]s?|"
            r"julgad[oa]s?\s+em\s+conjunto|"
            r"julgamento\s+simult[aâ]neo|"
            r"processos?\s+conexos|"
            r"apensad[oa]s?|"
            r"reunid[oa]s?\s+para\s+julgamento|"
            r"conjunto\s+de\s+agravos)",
            text,
            flags=re.IGNORECASE
        )
    )

    if retificacao_present and not plural_label and not has_joint_keyword and len(numbers) > 1:
        numbers = [numbers[-1]]

    is_joint = (len(numbers) > 1) or has_joint_keyword or plural_label
    return numbers, is_joint

def extract_process_numbers_from_value(value: str) -> List[str]:
    if not value:
        return []
    value = normalize_text(value)
    fulls = re.findall(CNJ_REGEX, value)
    if fulls:
        return dedupe_preserve_order(fulls)
    shorts = re.findall(SHORT_PROCESSO_REGEX, value)
    if shorts:
        return dedupe_preserve_order(shorts)
    labeled = re.findall(LABELED_PROCESSO_REGEX, value)
    return dedupe_preserve_order(labeled)

def extract_process_numbers_from_items(items: List[Dict[str, Any]]) -> Tuple[List[str], bool]:
    numbers = []
    multi_found = False
    for item in items:
        item_numbers = extract_process_numbers_from_value(item.get("numero_processo", ""))
        if len(item_numbers) > 1:
            multi_found = True
        numbers.extend(item_numbers)
    return dedupe_preserve_order(numbers), multi_found

def align_items_to_process_numbers(
    items: List[Dict[str, Any]],
    process_numbers: List[str],
    is_joint: bool
) -> List[Dict[str, Any]]:
    if not is_joint:
        return items

    expanded = []
    for item in items:
        item_numbers = extract_process_numbers_from_value(item.get("numero_processo", ""))
        if len(item_numbers) <= 1:
            expanded.append(item)
            continue
        for number in item_numbers:
            clone = item.copy()
            clone["numero_processo"] = number
            expanded.append(clone)

    if not process_numbers or len(process_numbers) <= 1:
        return expanded

    if not expanded:
        return items

    if len(expanded) > len(process_numbers):
        used = set()
        for item in expanded:
            for number in extract_process_numbers_from_value(item.get("numero_processo", "")):
                used.add(number)
        missing = [num for num in process_numbers if num not in used]
        if missing:
            for item in expanded:
                if not item.get("numero_processo") and missing:
                    item["numero_processo"] = missing.pop(0)
            if missing:
                template = expanded[-1]
                for number in missing:
                    clone = template.copy()
                    clone["numero_processo"] = number
                    expanded.append(clone)
        return expanded

    aligned = []
    unused = expanded[:]
    for number in process_numbers:
        match_idx = None
        for idx, item in enumerate(unused):
            item_numbers = extract_process_numbers_from_value(item.get("numero_processo", ""))
            if number in item_numbers:
                match_idx = idx
                break
        if match_idx is not None:
            item = unused.pop(match_idx)
        elif unused:
            item = unused.pop(0)
        else:
            item = expanded[-1].copy()
        item["numero_processo"] = number
        aligned.append(item)
    return aligned

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
    if match:
        day, month_name, year = match.groups()
        month = months.get(month_name.lower())
        if month:
            day_int = int(day)
            month_int = int(month)
            year_int = int(year)
            try:
                date(year_int, month_int, day_int)
            except ValueError:
                return ""
            return f"{month_int}/{day_int}/{year_int}"
    return ""

YOUTUBE_LINK_REGEX = (
    r"https?://(?:www\.)?"
    r"(?:youtube\.com/watch\?v=[\w-]+(?:&\S*)?|youtu\.be/[\w-]+(?:\?\S*)?)"
)

def extract_youtube_links(text: str) -> List[str]:
    cleaned = []
    for match in re.finditer(YOUTUBE_LINK_REGEX, text):
        link = match.group(0).rstrip(").,;\"']")
        if link not in cleaned:
            cleaned.append(link)
    return cleaned

def extract_first_youtube_link(text: str) -> str:
    for match in re.finditer(YOUTUBE_LINK_REGEX, text):
        return match.group(0).rstrip(").,;\"']")
    return ""

def normalize_youtube_link(value: str) -> str:
    if not value:
        return ""
    match = re.search(YOUTUBE_LINK_REGEX, value)
    if match:
        return match.group(0).rstrip(").,;\"']")
    if "," in value:
        return value.split(",", 1)[0].strip()
    return value.strip()

def find_label_value(text: str, labels: List[str]) -> str:
    for label in labels:
        pattern = rf"(?im)^\s*[*-]?\s*\*{{0,2}}{re.escape(label)}\*{{0,2}}\s*(?:[:\t]| {{2,}})\s*([^\n]+)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_label_value(match.group(1))
        pattern = rf"{re.escape(label)}\s*(?:[:\t]| {{2,}})\s*([^\n]+)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_label_value(match.group(1))
    return ""

def extract_uf_from_text(text: str) -> str:
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
        r"Tribunal Regional Eleitoral d[oa]\s+([A-Za-zçÇãõéíóúÁÉÍÓÚ ]+)",
        text,
        flags=re.IGNORECASE
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

def normalize_tre(value: str, uf: str) -> str:
    if value:
        match = re.search(r"\bTRE-([A-Z]{2})\b", value)
        if match:
            return f"TRE-{match.group(1).upper()}"
    if uf:
        return f"TRE-{uf}"
    return value

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
    return name

def normalize_pedido_vista_name(value: str) -> str:
    name = normalize_ministro_name(value)
    name = re.sub(r"^Min\.\s+(?:Presidente|Vice-Presidente|Relator(?:a)?)\s+", "Min. ", name)
    if name in {"Min. Presidente", "Min. Relator", "Min. Relatora", "Min. Ministra", "Min. Ministro"}:
        return ""
    return name

def extract_first_ministro_name(text: str) -> str:
    if not text:
        return ""
    cleaned = normalize_text(text)
    cleaned = re.sub(r"\[\[.*?\]\]", "", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", "", cleaned)
    cleaned = cleaned.replace("*", "")
    match = MINISTRO_NAME_STRICT_REGEX.search(cleaned)
    if match:
        return normalize_pedido_vista_name(match.group(0))
    return ""

def normalize_pedido_vista_value(value: str) -> str:
    if not value:
        return ""
    name = extract_first_ministro_name(value)
    if name:
        return name
    return normalize_pedido_vista_name(value)

def normalize_composicao(value: str) -> str:
    if not value:
        return ""
    value = value.replace(";", ",")
    parts = [p.strip() for p in value.split(",") if p.strip()]
    normalized = []
    seen = set()
    for part in parts:
        if is_mpe_noise_entry(part):
            continue
        name = normalize_ministro_name(part)
        if name and name not in seen:
            seen.add(name)
            normalized.append(name)
    return ", ".join(normalized)

def parse_composicao_labeled_value(value: str) -> str:
    if not value:
        return ""
    value = normalize_text(value)
    label_pattern = (
        r"(?:presidente|vice-?presidente|ministros?\s+titulares|ministros?\s+presentes?|"
        r"ministros?\s+que\s+comp[oó]em\s+a\s+sess[aã]o|ministra\s+substituta|ministros?\s+substitutos?)"
    )
    skip_pattern = r"(?i)procurador|minist[ée]rio p[úu]blico|pge|mpe|mp\s*eleitoral"
    segments = [seg.strip() for seg in re.split(r"\s*;\s*", value) if seg.strip()]
    names = []
    for segment in segments:
        segment = re.sub(rf"(?i)^{label_pattern}\s*:\s*", "", segment).strip()
        if not segment:
            continue
        if re.search(skip_pattern, segment):
            continue
        segment = re.sub(r"\s+e\s+", ", ", segment)
        for part in [p.strip() for p in segment.split(",") if p.strip()]:
            if re.search(skip_pattern, part):
                continue
            if re.search(r"(?i)\bdr[a]?\.\b", part):
                continue
            name = normalize_ministro_name(part)
            if name:
                names.append(name)
    return ", ".join(dedupe_preserve_order(names))

def is_valid_composicao(value: str) -> bool:
    if not value:
        return False
    if re.search(r"(?i)composi[cç][aã]o completa|vide item|mesma compos", value):
        return False
    matches = [m.group(0) for m in MINISTRO_NAME_REGEX.finditer(value)]
    if not matches:
        return False
    noise = re.compile(
        r"(?i)\b(advogad|relator|jurisprud|resolu|legisla|fundamenta|an[aá]lise|"
        r"pontos?|processo|partes?|origem|elei[cç][aã]o|classe|proced[eê]ncia|"
        r"resultado|efeitos?|conclus[aã]o|procurador|pge|mpe|mp\s*eleitoral|minist[ée]rio)\b"
    )
    valid = []
    for raw in matches:
        name = normalize_ministro_name(raw)
        base = re.sub(r"^Min\.\s+", "", name, flags=re.IGNORECASE).strip()
        if not base:
            continue
        if noise.search(base):
            continue
        valid.append(name)
    return len(valid) >= 2

def parse_composicao(text: str, base_composicao: str) -> str:
    value = find_label_value(
        text,
        [
            "Ministros presentes",
            "Ministros que compõem a sessão",
            "Ministros que compoem a sessao",
            "Ministros",
            "Composição da Mesa Julgadora",
            "Composição da Mesa",
            "Composição do Plenário",
            "Composição da Corte na sessão",
            "Composição da Corte na sessao",
            "Composição da Corte",
            "Composicao do Plenario",
        ]
    )
    if not value:
        extracted = extract_composicao_from_text(text, max_lines=None, include_leadership=False)
        return extracted or base_composicao
    if re.search(r"composi[cç][aã]o completa|vide item anterior|mesma compos", value, flags=re.IGNORECASE):
        return base_composicao
    if re.search(r"(?i)(presidente|vice-?presidente|ministros?\s+titulares|ministra\s+substituta)\s*:", value):
        labeled = parse_composicao_labeled_value(value)
        if labeled:
            return labeled
    normalized = normalize_composicao(value)
    if base_composicao and normalized and len(normalized.split(", ")) < 2:
        return base_composicao
    return normalized

def parse_relator(text: str) -> str:
    value = find_label_value(text, ["Ministro Relator", "Ministra Relatora", "Relator", "Relatora"])
    return normalize_ministro_name(value) if value else ""

def extract_advogados_block(text: str) -> Tuple[str, str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.search(
            r"(?im)^\s*[*-]?\s*\*{0,2}\d*\.?\)?\s*(Advogados?|Advogada)"
            r"(?:\s+[^:]+)?\*{0,2}(?:\s*\([^)]*\))?\s*:\s*(.*)$",
            line
        )
        if not match:
            continue
        label = match.group(1)
        remainder = clean_label_value(match.group(2))
        names = []
        if remainder:
            names.append(remainder)
        for next_line in lines[index + 1:]:
            if not next_line.strip():
                if names:
                    break
                continue
            if is_composicao_block_terminator(next_line):
                break
            if re.search(r"(?i)^\s*(?:#{1,4}\s+|\d+\.|\d+\))", next_line.strip()):
                break
            cleaned = clean_label_value(next_line)
            cleaned = re.sub(r"^\s*[*-]\s*", "", cleaned)
            cleaned = re.sub(r"(?i)^advogad[oa]s?\s*:?\s*", "", cleaned).strip()
            if cleaned:
                names.append(cleaned)
        if names:
            return ", ".join(names), label
    return "", ""

def parse_advogados(text: str) -> str:
    value, label = extract_advogados_block(text)
    if value:
        return normalize_advogados_list(value, label)
    return ""

def extract_partes(text: str) -> str:
    parties = []
    base = find_label_value(text, ["Nome das partes/interessados", "Nome das partes", "Partes"])
    if base:
        parties.append(base)
    extra_labels = [
        "Recorrentes",
        "Recorrente",
        "Recorridos",
        "Recorrido",
        "Recorrida",
        "Agravantes/Recorrentes",
        "Agravados/Recorridos",
        "Agravantes",
        "Agravados",
        "Embargante",
        "Embargantes",
        "Embargado",
        "Embargados",
        "Interessado",
        "Interessados",
        "Indicados",
        "Indicado",
    ]
    for label in extra_labels:
        value = find_label_value(text, [label])
        if value:
            parties.append(value)
    for match in re.finditer(
        r"(?im)^\s*\*?\s*\*\*(?:Parties|Partes)[^:]*\*\*\s*:\s*(.+)$",
        text
    ):
        value = clean_label_value(match.group(1))
        if value:
            parties.append(value)
    unique = []
    seen = set()
    for party in parties:
        normalized = normalize_mpe_reference(party)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return remove_mpe_from_partes(", ".join(unique))

def parse_eleicao(text: str) -> str:
    value = find_label_value(text, ["Eleição a que se refere", "Eleição"])
    if not value:
        return ""
    if re.search(r"não se aplica", value, flags=re.IGNORECASE):
        return ""
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return match.group(0) if match else value

def extract_section_block(text: str, titles: List[str]) -> str:
    for title in titles:
        pattern = re.compile(
            rf"(?im)^\s*(?:#{{1,4}}\s*)?\*{{0,2}}(?:\d+(?:\.\d+)*\.)?\s*{re.escape(title)}\*{{0,2}}(?:\s*:\s*.*|\s+.*)?\s*$",
            flags=re.IGNORECASE | re.MULTILINE
        )
        match = pattern.search(text)
        if not match:
            continue
        start = match.end()
        next_match = re.search(
            r"(?im)^\s*(?:#{1,4}\s+|\*{0,2}\d+(?:\.\d+)*\.\s*[A-ZÁÉÍÓÚ][A-ZÁÉÍÓÚ\sÇÃÕÂÊÔÍÓÚ\-]*:?\s*$|\*\*Obs:|OBS:|OBSERVA)",
            text[start:]
        )
        end = start + next_match.start() if next_match else len(text)
        return text[start:end].strip()
    return ""

def collapse_block(block: str, separator: str = ", ") -> str:
    if not block:
        return ""
    lines = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\*+\s*", "", line)
        line = re.sub(r"^\d+\.\s*", "", line)
        line = line.strip("-•")
        line = clean_label_value(line)
        if line:
            lines.append(line)
    unique = []
    seen = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return separator.join(unique)

def parse_origem(text: str) -> str:
    city = find_label_value(text, ["Cidade de origem", "Origem"])
    if not city:
        return ""
    city = normalize_text(city)
    city = city.strip().rstrip(".")
    if "/" in city:
        match = re.match(r"^(.*?)/\s*([A-Za-z]{2})$", city.strip())
        if match:
            return f"{match.group(1).strip()}/{match.group(2).upper()}"
        return city
    uf = extract_uf_from_text(city) or extract_uf_from_text(text)
    city = re.sub(r"\s*\(.*?\)\s*", "", city).strip()
    if re.search(r"\s[-–—]\s", city):
        city = re.split(r"\s[-–—]\s", city, maxsplit=1)[0].strip()
    if city and uf:
        return f"{city}/{uf}"
    return city

def normalize_origem_value(value: str) -> str:
    if not value:
        return ""
    value = normalize_text(value).strip().rstrip(".")
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
    return value

def extract_processo_e_classe(text: str) -> Dict[str, str]:
    result = {"numero_processo": "", "classe_processo": ""}
    header_line = ""
    for raw_line in text.splitlines():
        if raw_line.strip():
            header_line = raw_line
            break
    line = find_label_value(text, ["Processo", "Processos"])
    if line:
        line = normalize_text(line)
        line = line.split("(")[0].strip()
        full_cnj = extract_full_cnj(line)
        if full_cnj:
            result["numero_processo"] = full_cnj
        else:
            short_num = extract_short_processo(line)
            if short_num:
                result["numero_processo"] = short_num
            if not result["numero_processo"]:
                labeled_num = extract_labeled_short_processo(line)
                if labeled_num:
                    result["numero_processo"] = labeled_num
        splitter = re.compile(r"\bn[ºo°]\.?\b", flags=re.IGNORECASE)
        parts = splitter.split(line, maxsplit=1)
        result["classe_processo"] = parts[0].strip()
    else:
        result["numero_processo"] = extract_full_cnj(text) or extract_short_processo(text)
        if not result["numero_processo"] and header_line:
            result["numero_processo"] = extract_labeled_short_processo(header_line)
    if not result["classe_processo"]:
        tipo_value = find_label_value(
            text,
            ["Tipo", "Classe", "Classe do Processo", "Classe/Número", "Classe/Numero"]
        )
        if tipo_value:
            result["classe_processo"] = tipo_value
    return result

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
        if re.search(pattern, normalized):
            return canon_value
    return value.strip()

def select_canonical_result(preferred: List[str], allowed: Optional[set]) -> str:
    if allowed:
        for candidate in preferred:
            if candidate in allowed:
                return candidate
    return preferred[0]

def normalize_resultado_piece(value: str, classe_processo: str, allowed: Optional[set]) -> str:
    if not value:
        return ""
    normalized = normalize_class_text(value)

    if "suspens" in normalized and "vista" in normalized:
        return "Suspenso por vista"
    if "prejudic" in normalized:
        return select_canonical_result(["Prejudicado"], allowed)
    if "nao conhec" in normalized:
        return select_canonical_result(["Não conhecido", "Não conhecida"], allowed)
    if (
        "desprov" in normalized
        or "improv" in normalized
        or re.search(r"nega\w*\s+provimento", normalized)
    ):
        return select_canonical_result(["Desprovido", "Desprovida", "Desprovadas"], allowed)
    if "provido" in normalized or "provimento" in normalized:
        if "parcial" in normalized or "em parte" in normalized:
            return select_canonical_result(["Provido em parte"], allowed)
        return select_canonical_result(["Provido"], allowed)
    if "deferid" in normalized and "parcial" in normalized:
        return select_canonical_result(["Parcialmente deferido"], allowed)
    if "indefer" in normalized:
        return select_canonical_result(["Indeferido", "Indeferida"], allowed)
    if re.search(r"\bdeferid", normalized):
        return select_canonical_result(["Deferido"], allowed)
    if "referend" in normalized:
        return select_canonical_result(["Referendado", "Referendada"], allowed)
    if "aprovad" in normalized:
        if "ressalv" in normalized:
            return select_canonical_result(
                ["Aprovada com ressalvas", "Aprovadas com ressalvas"],
                allowed,
            )
        return select_canonical_result(["Aprovada", "Aprovadas"], allowed)
    if "acolhid" in normalized:
        if "em parte" in normalized or "parcial" in normalized:
            return select_canonical_result(
                ["Acolhido em parte", "Acolhida em parte", "Acolhidos em parte"],
                allowed,
            )
        return select_canonical_result(["Acolhidos"], allowed)
    if "rejeitad" in normalized:
        return select_canonical_result(["Rejeitados", "Rejeitada"], allowed)
    if "devolvid" in normalized:
        return select_canonical_result(["Devolvida"], allowed)
    if "anulad" in normalized:
        return "Anulado"
    return ""

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
    if canonical_direct:
        if not allowed or canonical_direct in allowed:
            return canonical_direct
    lowered = normalize_class_text(text)
    if "suspens" in lowered and "vista" in lowered:
        return "Suspenso por vista"
    if re.search(r"\bprovido\b", lowered) and "nao conhec" in lowered and "desprov" not in lowered:
        return "Provido, Não conhecido"
    if "prejudic" in lowered and "desprov" in lowered:
        return "Prejudicado, Desprovido"

    parts = [part.strip() for part in re.split(r"[;,/]", text) if part.strip()]
    if len(parts) > 1:
        normalized_parts = []
        for part in parts:
            part_norm = normalize_resultado_piece(part, classe_canon, allowed)
            if part_norm and part_norm not in normalized_parts:
                normalized_parts.append(part_norm)
        if normalized_parts:
            return ", ".join(normalized_parts)

    single = normalize_resultado_piece(text, classe_canon, allowed)
    return single or text.strip()

def extract_resolucoes(text: str) -> str:
    matches = re.findall(
        r"Resolu[cç][aã]o(?:-TSE)?\s*n[ºo]?\s*[\d\.]+(?:/\d{4})?(?:\s*\([^)]+\))?",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = []
    for match in matches:
        value = match.strip().rstrip(".")
        if value not in cleaned:
            cleaned.append(value)
    return ", ".join(cleaned)

def extract_resultado_votacao(text: str) -> Dict[str, str]:
    lower = text.lower()
    resultado = ""
    votacao = ""

    if re.search(r"por unanimidade|decis[aã]o un[aâ]nime|aprova[cç][aã]o un[aâ]nime|un[aâ]nime|unanimemente", lower):
        votacao = "Unânime"
    if "por maioria" in lower:
        votacao = "Por maioria"
    if "suspenso" in lower:
        votacao = "Suspenso"
        if "vista" in lower:
            resultado = "Suspenso por vista"

    if re.search(r"parcialmente\s+deferid", lower):
        resultado = "Parcialmente deferido"
    elif "parcial provimento" in lower or "provido em parte" in lower:
        resultado = "Provido em parte"
    elif "parcialmente" in lower and ("provimento" in lower or "provido" in lower):
        resultado = "Provido em parte"
    elif (
        "negado provimento" in lower
        or "negou provimento" in lower
        or "nego provimento" in lower
        or "nega provimento" in lower
        or "desprovido" in lower
        or "não provido" in lower
        or "nao provido" in lower
    ):
        resultado = "Desprovido"
    elif re.search(r"\bprovido\b|\bprovimento\b", lower):
        resultado = "Provido"
    elif (
        "não conhecido" in lower
        or "nao conhecido" in lower
        or "não conhecimento" in lower
        or "nao conhecimento" in lower
        or "não conhecer" in lower
        or "nao conhecer" in lower
    ):
        resultado = "Não conhecido"
    elif re.search(r"aprovad[oa]s?\s+com\s+ressalv", lower):
        resultado = "Aprovada com ressalvas"
    elif "aprov" in lower and not resultado:
        resultado = "Aprovada"
    elif "indeferid" in lower:
        resultado = "Indeferida" if "indeferida" in lower else "Indeferido"
    elif "deferid" in lower:
        resultado = "Deferido"
    elif "referendad" in lower:
        resultado = "Referendada" if "referendada" in lower else "Referendado"
    elif "acolh" in lower and ("em parte" in lower or "parcial" in lower):
        resultado = "Acolhido em parte"
    elif "acolhido" in lower or "acolhidos" in lower or "acolhida" in lower:
        resultado = "Acolhidos"
    elif "rejeitad" in lower:
        resultado = "Rejeitada" if "rejeitada" in lower else "Rejeitados"
    elif "devolvid" in lower:
        resultado = "Devolvida"
    elif "prejudicado" in lower:
        resultado = "Prejudicado"
    elif "anulado" in lower:
        resultado = "Anulado"

    return {"resultado_final": resultado, "votacao": votacao}

VISTA_NOUN_REGEX = r"pedido\s+de\s+vistas?|vista\s+regimental"
VISTA_VERB_REGEX = r"(?:pediu|pede|solicitou|requereu|requisitou|formulou|apresentou|antecipou|renovou|reiterou)"
VISTA_REQUEST_REGEX = re.compile(
    rf"(?i)\b(?:{VISTA_NOUN_REGEX}|{VISTA_VERB_REGEX}\s+(?:um\s+)?(?:pedido\s+de\s+)?vistas?)\b"
)
VISTA_NAME_CAPTURE_PATTERN = (
    rf"(?:{MINISTRO_NAME_WITH_PREFIX_STRICT_PATTERN}|{LEADERSHIP_NAME_WITHOUT_PREFIX_STRICT_PATTERN})"
)
VISTA_NAME_AFTER_NOUN_REGEX = re.compile(
    rf"(?i:\b(?:{VISTA_NOUN_REGEX})\b)"
    rf"(?P<between>[^\n]{{0,80}}?)"
    rf"(?:\b(?:do|da|de|pelo|pela|ao|a|o|à)\b\s*)?"
    rf"(?P<name>{VISTA_NAME_CAPTURE_PATTERN})"
)
VISTA_NAME_AFTER_VERB_REGEX = re.compile(
    rf"(?i:\b(?:{VISTA_VERB_REGEX})\b)"
    rf"[^\n]{{0,40}}?"
    rf"(?i:\b(?:pedido\s+de\s+)?vistas?\b)"
    rf"(?P<between>[^\n]{{0,20}}?)"
    rf"(?:\b(?:ao|a|o|à|do|da|de|pelo|pela)\b\s+)"
    rf"(?P<name>{VISTA_NAME_CAPTURE_PATTERN})"
)
VISTA_NAME_BEFORE_VERB_REGEX = re.compile(
    rf"(?P<name>{VISTA_NAME_CAPTURE_PATTERN})"
    rf"(?P<between>[^\n]{{0,80}}?)"
    rf"(?i:\b(?:{VISTA_VERB_REGEX})\b)"
    rf"[^\n]{{0,20}}?"
    rf"(?i:\b(?:pedido\s+de\s+)?vistas?\b)"
)
VISTA_NAME_BEFORE_NOUN_REGEX = re.compile(
    rf"(?P<name>{VISTA_NAME_CAPTURE_PATTERN})"
    rf"(?P<between>[^\n]{{0,40}}?)"
    rf"(?i:\b(?:{VISTA_NOUN_REGEX})\b)"
)
RELATOR_VISTA_REGEX = re.compile(
    rf"(?i)\brelator(a)?\b[^\n]{{0,40}}\b(?:{VISTA_VERB_REGEX})\b[^\n]{{0,20}}\b(?:pedido\s+de\s+)?vistas?\b"
)
RELATOR_VISTA_NOUN_REGEX = re.compile(
    rf"(?i)\b(?:{VISTA_NOUN_REGEX})\b[^\n]{{0,40}}\b(?:pelo|pela)\s+relator(a)?\b"
)
NEGATED_VISTA_REGEX = re.compile(
    r"(?i)\b("
    r"n[aã]o\s+houve|n[aã]o\s+h[aá]|n[aã]o\s+ha|n[aã]o\s+se\s+pediu|"
    r"n[aã]o\s+foi\s+solicitad[ao]|sem|aus[eê]ncia\s+de|"
    r"inexist[eê]ncia\s+de|inexistiu|dispensad[ao]|desnecess[aá]ri[ao]"
    r")\s+(?:pedido\s+de\s+)?vistas?\b"
)

def line_has_negated_vista(line: str) -> bool:
    if not line:
        return False
    if re.search(r"(?i)\bvista\s+dos?\s+memoriais\b", line):
        return True
    return bool(NEGATED_VISTA_REGEX.search(line))

def has_pedido_vista_marker(text: str) -> bool:
    if not text:
        return False
    for line in text.splitlines():
        if not VISTA_REQUEST_REGEX.search(line):
            continue
        if line_has_negated_vista(line):
            continue
        return True
    return False

def clean_vista_span(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"\[\[.*?\]\]", "", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", "", cleaned)
    cleaned = cleaned.replace("*", "")
    return cleaned

def extract_pedido_vista_name_from_span(span: str) -> str:
    if not span:
        return ""
    cleaned = clean_vista_span(span)
    best_name = ""
    best_distance = None
    for regex in (
        VISTA_NAME_AFTER_NOUN_REGEX,
        VISTA_NAME_AFTER_VERB_REGEX,
        VISTA_NAME_BEFORE_VERB_REGEX,
        VISTA_NAME_BEFORE_NOUN_REGEX,
    ):
        for match in regex.finditer(cleaned):
            name = normalize_pedido_vista_name(match.group("name"))
            if not name:
                continue
            between = match.groupdict().get("between") or ""
            distance = len(between.strip())
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_name = name
    if best_name:
        return best_name
    if not VISTA_REQUEST_REGEX.search(cleaned):
        return ""
    names = []
    for match in MINISTRO_NAME_STRICT_REGEX.finditer(cleaned):
        raw = match.group(0)
        if re.search(r"(?i)\brelator(a)?\b", raw):
            continue
        name = normalize_pedido_vista_name(raw)
        if name:
            names.append(name)
    if len(names) == 1:
        return names[0]
    leader_names = []
    for match in LEADERSHIP_NAME_REGEX.finditer(cleaned):
        name = normalize_pedido_vista_name(match.group(0))
        if name:
            leader_names.append(name)
    if len(leader_names) == 1 and not names:
        return leader_names[0]
    return ""

def build_vista_span(lines: List[str], idx: int, lookahead: int = 2) -> str:
    if idx >= len(lines):
        return ""
    base = lines[idx].strip()
    if not base:
        return ""
    if MINISTRO_NAME_STRICT_REGEX.search(base) or LEADERSHIP_NAME_REGEX.search(base):
        return base
    parts = [base]
    for offset in range(1, lookahead + 1):
        if idx + offset >= len(lines):
            break
        part = lines[idx + offset].strip()
        if not part:
            continue
        if MINISTRO_NAME_STRICT_REGEX.search(part) or LEADERSHIP_NAME_REGEX.search(part):
            parts.append(part)
            break
    return " ".join(parts)

def extract_pedido_vista_map(
    text: str,
    search_back: int = 8,
    search_forward: int = 2
) -> Dict[str, str]:
    if not text:
        return {}
    lines = text.splitlines()
    vista_map: Dict[str, str] = {}
    for idx, line in enumerate(lines):
        if not VISTA_REQUEST_REGEX.search(line):
            continue
        if line_has_negated_vista(line):
            continue
        span = build_vista_span(lines, idx, lookahead=2)
        name = extract_pedido_vista_name_from_span(span)
        if not name:
            continue
        candidates = extract_process_numbers_from_line(line)
        if not candidates:
            for back in range(1, search_back + 1):
                if idx - back < 0:
                    break
                candidates = extract_process_numbers_from_line(lines[idx - back])
                if candidates:
                    break
        if not candidates:
            for forward in range(1, search_forward + 1):
                if idx + forward >= len(lines):
                    break
                candidates = extract_process_numbers_from_line(lines[idx + forward])
                if candidates:
                    break
        if len(candidates) != 1:
            continue
        normalized = normalize_processo_num(candidates[0])
        if normalized and normalized not in vista_map:
            vista_map[normalized] = name
    return vista_map

def apply_pedido_vista_map(items: List[Dict[str, Any]], vista_map: Dict[str, str]) -> None:
    if not items or not vista_map:
        return
    for item in items:
        mapped = ""
        for number in extract_process_numbers_from_value(item.get("numero_processo", "")):
            normalized = normalize_processo_num(number)
            if normalized in vista_map:
                mapped = vista_map[normalized]
                break
        item["pedido_vista"] = mapped or ""

def extract_pedido_vista(text: str, relator: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if not VISTA_REQUEST_REGEX.search(line):
            continue
        if line_has_negated_vista(line):
            continue
        span = build_vista_span(lines, idx, lookahead=2)
        name = extract_pedido_vista_name_from_span(span)
        if name:
            return name
        if relator and (RELATOR_VISTA_REGEX.search(span) or RELATOR_VISTA_NOUN_REGEX.search(span)):
            return relator
    return ""

def extract_pedido_vista_via_api(section_text: str, model: str, max_retries: Optional[int] = None) -> str:
    prompt = (
        "Extraia APENAS o nome do ministro que pediu vista do processo (ou cujo pedido de vista anterior "
        "tenha sido mencionado). Se não houver menção clara de pedido de vista, retorne string vazia.\n"
        "Regras:\n"
        "- Retorne SOMENTE o nome do ministro no formato 'Min. Nome Sobrenome'.\n"
        "- Se houver mais de um nome, retorne apenas o ministro que efetivamente pediu vista.\n"
        "- Ignore expressões não relacionadas (ex.: 'vista dos memoriais', 'revista', etc.).\n"
        "Responda com JSON estrito no esquema fornecido.\n\n"
        "TEXTO:\n---\n"
        f"{section_text}\n"
        "---"
    )

    retries = max(1, int(max_retries if max_retries is not None else OPENAI_VISTA_RETRIES))
    for attempt in range(retries):
        try:
            logging.info("Solicitando pedido_vista via API (tentativa %d/%d)...", attempt + 1, retries)
            OPENAI_REQUEST_PACER.wait_turn()
            response = _openai_chat_completion(
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Você extrai apenas o pedido de vista do texto."},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "pedido_vista_extractor",
                            "schema": VISTA_OUTPUT_SCHEMA,
                            "strict": True,
                        },
                    },
                }
            )
            content = response.choices[0].message.content
            if SHOW_TECHNICAL_LOGS:
                logging.debug("Resposta pedido_vista (JSON): %s", content)
            parsed = json.loads(content)
            value = parsed.get("pedido_vista", "")
            return normalize_pedido_vista_value(value)
        except Exception as e:
            logging.warning(
                "Falha ao extrair pedido_vista via API (tentativa %d/%d).",
                attempt + 1,
                retries,
            )
            if SHOW_TECHNICAL_LOGS:
                logging.debug("Detalhes do erro: %s", e)
            if attempt < retries - 1:
                wait_time = min(10.0, max(0.2, float(OPENAI_BASE_DELAY_SECONDS)) * (2 ** attempt))
                time.sleep(wait_time)
    return ""

def merge_composicao_values(values: List[str]) -> str:
    names = []
    seen = set()
    for value in values:
        if not value:
            continue
        normalized = normalize_composicao(value)
        for name in [part.strip() for part in normalized.split(",") if part.strip()]:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return ", ".join(names)

COMPOSICAO_SCAN_LIMIT = 240

def find_composicao_header_limit(lines: List[str]) -> int:
    for index, line in enumerate(lines):
        if re.search(r"(?i)relat[óo]rio de julgamentos", line):
            return index
        if re.search(r"(?i)^\s*#{1,4}\s*\d*\.?\s*(processo|julgamento)", line):
            return index
        if re.search(r"(?i)^\s*\d+\)\s*(processo|julgamento)", line):
            return index
        if re.search(r"(?i)^\s*\d+\.\s*(processo|julgamento)", line):
            return index
        if re.search(r"(?i)^\s*dados do processo\b", line):
            return index
    return len(lines)

def detect_composicao_label(line: str) -> str:
    if re.search(r"(?i)\bministros?\s+presentes?\b", line):
        return "presentes"
    if re.search(r"(?i)\bministers?\s+present(?:es)?\b", line):
        return "presentes"
    if re.search(r"(?i)\bministros?\s+que\s+comp[oó]em\s+a\s+sess[aã]o\b", line):
        return "presentes"
    if re.search(r"(?i)\bcomposi[cç][aã]o\s+da\s+(mesa|corte|mesa\s+julgadora|plen[aá]rio)\b", line):
        return "composicao"
    if re.match(r"(?i)^\s*[*-]?\s*\*{0,2}\d*\.?\s*ministros?\s*:", line):
        return "ministros"
    return ""

def is_composicao_block_terminator(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("---"):
        return True
    if re.match(r"^#{2,4}\s+", stripped):
        return True
    if re.search(
        r"(?i)^\s*\*{0,2}\d*\.?\)?\s*\*{0,2}(data da sess[aã]o|sess[aã]o|"
        r"relat[oó]rio|processo|ministro relator|relator|elei[cç][aã]o|"
        r"classe|origem|partes|advogad|jurisprud|fundamenta|racioc[ií]nio|"
        r"conclus[aã]o|resolu[cç][aã]o|legisla[cç][aã]o|resultado|vota[cç][aã]o)\b",
        stripped
    ):
        return True
    if re.search(r"(?i)minist[ée]rio p[uú]blico|vice-procurador|procurador|pge|mpe", stripped):
        return True
    return False

def extract_composicao_block(lines: List[str], start_index: int) -> List[str]:
    block_names = []
    for line in lines[start_index + 1:]:
        if not line.strip():
            continue
        names = extract_ministro_names_from_line(line)
        if names:
            block_names.extend(names)
            continue
        if is_composicao_block_terminator(line):
            break
    return dedupe_preserve_order(block_names)

def extract_composicao_candidates(
    text: str,
    max_lines: Optional[int] = None
) -> List[Tuple[List[str], int, str]]:
    lines = text.splitlines()
    if max_lines:
        lines = lines[:max_lines]
    candidates = []
    for idx, line in enumerate(lines):
        label_type = detect_composicao_label(line)
        if not label_type:
            continue
        inline_names = extract_ministro_names_from_line(line)
        block_names = extract_composicao_block(lines, idx)
        if inline_names and block_names:
            combined = dedupe_preserve_order(inline_names + block_names)
            candidates.append((combined, idx, label_type))
            continue
        if inline_names:
            candidates.append((dedupe_preserve_order(inline_names), idx, label_type))
            continue
        if block_names:
            candidates.append((block_names, idx, label_type))
    return candidates

def score_composicao_candidate(names: List[str], line_index: int, label_type: str) -> int:
    size = len(names)
    score = 0
    if 6 <= size <= 7:
        score += 6
    elif size == 5 or size == 8:
        score += 3
    elif size == 4:
        score += 1
    if size > 7:
        score -= (size - 7)
    label_bonus = {"presentes": 3, "composicao": 2, "ministros": 1}
    score += label_bonus.get(label_type, 0)
    score += max(0, 3 - (line_index // 40))
    return score

def select_best_composicao_candidate(
    candidates: List[Tuple[List[str], int, str]]
) -> List[str]:
    preferred = [
        (names, line_index, label_type)
        for names, line_index, label_type in candidates
        if label_type == "composicao" and len(names) >= 5
    ]
    if preferred:
        candidates = preferred
    best = None
    best_score = None
    best_index = None
    for names, line_index, label_type in candidates:
        score = score_composicao_candidate(names, line_index, label_type)
        if best is None or score > best_score:
            best = names
            best_score = score
            best_index = line_index
            continue
        if score == best_score:
            if abs(7 - len(names)) < abs(7 - len(best)):
                best = names
                best_index = line_index
            elif abs(7 - len(names)) == abs(7 - len(best)) and line_index < best_index:
                best = names
                best_index = line_index
    return best or []

def extract_absent_ministros(text: str) -> List[str]:
    names = []
    for line in text.splitlines():
        match = re.search(r"(?i)aus[eê]nci\w*.*", line)
        if match:
            names.extend(extract_ministro_names_from_line(match.group(0)))
            continue
        if re.match(r"(?i)^\s*[*-]?\s*\*{0,2}ausent", line):
            match = re.search(r"(?i)ausent\w*.*", line)
            if match:
                names.extend(extract_ministro_names_from_line(match.group(0)))
    return dedupe_preserve_order(names)

def extract_presidencia_name(text: str) -> str:
    value = find_label_value(text, ["Presidência", "Presidente", "Presidente da Sessão"])
    return normalize_ministro_name(value) if value else ""

def extract_leadership_names(lines: List[str]) -> List[str]:
    leaders = []
    for line in lines:
        if re.search(r"(?i)presidente\s+do\s+julgamento", line):
            continue
        if re.match(
            r"(?i)^\s*[*-]?\s*\*{0,2}(presid[êe]ncia(?:\s*\(.*?\))?|"
            r"presidente(?:\s+da\s+sess[aã]o)?|vice-?presidente)\b",
            line
        ):
            leaders.extend(extract_ministro_names_from_line(line))
    return dedupe_preserve_order(leaders)

def extract_composicao_from_text(
    text: str,
    max_lines: Optional[int] = COMPOSICAO_SCAN_LIMIT,
    include_leadership: bool = True
) -> str:
    lines = text.splitlines()
    if max_lines:
        lines = lines[:max_lines]
    header_limit = find_composicao_header_limit(lines)
    header_text = "\n".join(lines[:header_limit])
    candidates = extract_composicao_candidates(header_text)
    if not candidates and max_lines:
        candidates = extract_composicao_candidates(text, None)
    if not candidates:
        return ""
    selected = select_best_composicao_candidate(candidates)
    absent = extract_absent_ministros(text)
    names = [name for name in selected if name not in absent]
    presidencia = extract_presidencia_name(header_text)
    if presidencia and presidencia not in names and presidencia not in absent:
        names.insert(0, presidencia)
    if include_leadership and len(names) < 7:
        lines = text.splitlines()
        if max_lines:
            lines = lines[:max_lines]
        leaders = extract_leadership_names(lines)
        for leader in leaders:
            if leader not in names and leader not in absent:
                names.append(leader)
                if len(names) >= 7:
                    break
    return ", ".join(dedupe_preserve_order(names))

def extract_composicao_from_header(text: str) -> str:
    return extract_composicao_from_text(text, max_lines=COMPOSICAO_SCAN_LIMIT, include_leadership=False)

def extract_global_context(text: str) -> Dict[str, Any]:
    base = {}
    base_date = find_label_value(
        text,
        ["Data da Sessão", "Data da sessão", "Data e Hora de Início", "Data e Hora de Inicio"]
    )
    base["data_sessao"] = parse_date_from_text(base_date) or parse_date_from_text(text)
    composicao_candidates = []
    label_composicao = parse_composicao(text, "")
    if label_composicao:
        composicao_candidates.append(label_composicao)
    header_composicao = extract_composicao_from_header(text)
    if header_composicao:
        composicao_candidates.append(header_composicao)
    base["composicao"] = merge_composicao_values(composicao_candidates)
    first_youtube_link = extract_first_youtube_link(text)
    if first_youtube_link:
        base["youtube_link"] = first_youtube_link
    return base

def extract_raw_fields(section_text: str, base_context: Dict[str, Any]) -> Dict[str, Any]:
    raw = {}
    raw["tema"] = extract_section_title(section_text)
    raw["data_sessao"] = parse_date_from_text(section_text) or base_context.get("data_sessao", "")
    raw["composicao"] = parse_composicao(section_text, base_context.get("composicao", ""))
    raw["relator"] = parse_relator(section_text)
    raw["advogados"] = parse_advogados(section_text)
    raw["partes"] = extract_partes(section_text)
    raw["precedentes_citados"] = find_label_value(
        section_text,
        ["Jurisprudência citada", "Jurisprudência do TSE mencionada", "Jurisprudência mencionada"]
    )
    raw["fundamentacao_normativa"] = find_label_value(section_text, ["Legislação", "Fundamentação Normativa"])
    if not raw["fundamentacao_normativa"]:
        block = extract_section_block(section_text, ["FUNDAMENTAÇÃO NORMATIVA", "FUNDAMENTAÇÃO"])
        raw["fundamentacao_normativa"] = collapse_block(block, separator=", ")
    if not raw["precedentes_citados"]:
        block = extract_section_block(section_text, ["PRECEDENTES CITADOS", "PRECEDENTES"])
        raw["precedentes_citados"] = collapse_block(block, separator="; ")

    raw["analise_do_conteudo_juridico"] = collapse_block(
        extract_section_block(section_text, ["ANÁLISE DO CONTEÚDO JURÍDICO", "ANÁLISE"]),
        separator=" "
    )
    raciocinio_base = collapse_block(
        extract_section_block(section_text, ["RACIOCÍNIO DESENVOLVIDO", "RACIOCÍNIO"]),
        separator=" "
    )
    pontos_processuais = collapse_block(
        extract_section_block(section_text, ["PONTOS PROCESSUAIS RELEVANTES", "PONTOS PROCESSUAIS"]),
        separator=" "
    )
    efeitos_providencias = collapse_block(
        extract_section_block(section_text, ["EFEITOS E PROVIDÊNCIAS PRÁTICAS", "CONCLUSÃO E EFEITOS", "EFEITOS"]),
        separator=" "
    )
    extras = []
    if pontos_processuais:
        extras.append(f"Pontos processuais: {pontos_processuais}.")
    if efeitos_providencias:
        extras.append(f"Efeitos e providências: {efeitos_providencias}.")
    raw["raciocinio_juridico"] = " ".join([raciocinio_base, *extras]).strip()

    processo_info = extract_processo_e_classe(section_text)
    raw.update(processo_info)
    raw["numero_processo"] = normalize_processo_num(raw.get("numero_processo", ""))
    if not raw.get("classe_processo"):
        tipo_value = find_label_value(
            section_text,
            ["Tipo", "Classe", "Classe do Processo", "Classe/Número", "Classe/Numero"]
        )
        if tipo_value:
            raw["classe_processo"] = tipo_value
    raw["classe_processo"] = normalize_classe_processo(raw.get("classe_processo", ""))

    raw["eleicao"] = parse_eleicao(section_text)
    raw["origem"] = parse_origem(section_text)

    uf = extract_uf_from_text(section_text)
    if not uf and raw.get("origem"):
        uf = extract_uf_from_text(raw["origem"])
    tre_from_text = normalize_tre("", uf)
    if tre_from_text:
        raw["TRE"] = tre_from_text

    raw["resoluções_citadas"] = extract_resolucoes(section_text)
    resultado_votacao = extract_resultado_votacao(section_text)
    raw.update(resultado_votacao)
    raw["pedido_vista"] = extract_pedido_vista(section_text, raw.get("relator", ""))

    first_youtube_link = extract_first_youtube_link(section_text)
    if first_youtube_link:
        raw["youtube_link"] = first_youtube_link

    return raw

def split_report_into_sections(text: str) -> List[str]:
    def section_has_process_marker(section: str) -> bool:
        if re.search(r"DADOS DO(?:S)? PROCESSO(?:S)?|DADOS DO CASO", section, flags=re.IGNORECASE):
            return True
        if re.search(r"(?i)\bProposta de\s+(Resolu[cç][aã]o|altera[cç][aã]o)\b", section):
            return True
        if re.search(r"(?im)^\s*\*{0,2}PROCESSOS?\b", section):
            return True
        if re.search(r"(?i)n[úu]mero do processo", section):
            return True
        header_line = ""
        for raw_line in section.splitlines():
            if raw_line.strip():
                header_line = raw_line
                break
        if header_line and re.search(r"(?i)\bprocesso\b", header_line):
            return True
        if header_line and re.search(LABELED_PROCESSO_REGEX, header_line):
            return True
        if re.search(CNJ_REGEX, section) or re.search(SHORT_PROCESSO_REGEX, section):
            return True
        return False

    headings = list(re.finditer(r"(?m)^(#{2,4})\s+.+$", text))
    sections = []
    if headings:
        for index, heading in enumerate(headings):
            level = len(heading.group(1))
            start = heading.start()
            end = len(text)
            for next_heading in headings[index + 1:]:
                next_level = len(next_heading.group(1))
                if next_level <= level:
                    end = next_heading.start()
                    break
            section = text[start:end].strip()
            obs_match = re.search(r"(?im)^\s*(?:\*\*)?OBSERVA", section)
            if obs_match:
                section = section[:obs_match.start()].rstrip()
            if section_has_process_marker(section):
                sections.append(section)
        return sections

    lines = text.splitlines()
    start_indices = [
        idx for idx, line in enumerate(lines)
        if re.match(r"(?i)^\s*\d+\.\s+Julgamento\b", line)
    ]
    if not start_indices:
        return []

    roman_indices = [
        idx for idx, line in enumerate(lines)
        if re.match(r"(?i)^\s*[IVXLCDM]+\.\s+", line)
    ]
    separator_indices = [
        idx for idx, line in enumerate(lines)
        if re.match(r"^\s*[-–—]{4,}\s*$", line)
    ]
    boundary_indices = sorted(set(roman_indices + separator_indices + [len(lines)]))

    def next_boundary(after_index: int) -> int:
        for idx in boundary_indices:
            if idx > after_index:
                return idx
        return len(lines)

    for idx, start in enumerate(start_indices):
        next_start = start_indices[idx + 1] if idx + 1 < len(start_indices) else None
        end_candidates = [next_boundary(start)]
        if next_start is not None:
            end_candidates.append(next_start)
        end_idx = min(end_candidates)
        section = "\n".join(lines[start:end_idx]).strip()
        obs_match = re.search(r"(?im)^\s*(?:\*\*)?OBSERVA", section)
        if obs_match:
            section = section[:obs_match.start()].rstrip()
        if section_has_process_marker(section):
            sections.append(section)
    return sections

def extract_section_title(section_text: str) -> str:
    match = re.search(r"(?m)^\s*\d+\.\s+(Julgamento\b.+)$", section_text, flags=re.IGNORECASE)
    if match:
        title = clean_label_value(match.group(1))
        title = re.sub(r"^\s*\d+(?:\.\d+)*\s*[\)\.\-–—]*\s*", "", title)
        title = title.strip()
        return title or "Seção"
    match = re.search(r"(?m)^#{2,4}\s+\*{0,2}(.+?)\*{0,2}\s*$", section_text)
    if match:
        title = clean_label_value(match.group(1))
        title = re.sub(r"^\s*\d+(?:\.\d+)*\s*[\)\.\-–—]*\s*", "", title)
        title = title.strip()
        return title or "Seção"
    return "Seção"

def merge_session_info(
    extracted_session: Dict[str, Any],
    base_context: Dict[str, Any],
    raw_fields: Dict[str, Any]
) -> Dict[str, Any]:
    data_sessao = (
        extracted_session.get("data_sessao")
        or raw_fields.get("data_sessao")
        or base_context.get("data_sessao")
        or ""
    )
    composicao = (
        extracted_session.get("composicao")
        or raw_fields.get("composicao")
        or base_context.get("composicao")
        or ""
    )
    return {
        "data_sessao": data_sessao,
        "composicao": normalize_composicao(composicao),
    }

def normalize_processo_num(value: str) -> str:
    if not value:
        return ""
    value = normalize_text(value)
    full_cnj = extract_full_cnj(value)
    if full_cnj:
        return full_cnj
    short_num = extract_short_processo(value)
    if short_num:
        return short_num
    labeled_num = extract_labeled_short_processo(value)
    return labeled_num if labeled_num else value.strip()

def canonicalize_numero_processo(value: str) -> str:
    if not value:
        return ""
    value = normalize_text(value)
    short = extract_short_processo(value)
    if short:
        return short
    labeled = extract_labeled_short_processo(value)
    return labeled if labeled else value.strip()

def apply_item_fallbacks(
    item: Dict[str, Any],
    raw_fields: Dict[str, Any],
    base_context: Dict[str, Any],
    registro_index: int
) -> Dict[str, Any]:
    for key, value in raw_fields.items():
        if key in item and not item.get(key) and value:
            item[key] = value

    if not item.get("youtube_link") and base_context.get("youtube_link"):
        item["youtube_link"] = base_context["youtube_link"]

    if "partes" in item:
        item["partes"] = remove_mpe_from_partes(item.get("partes", ""))

    item["tipo_registro"] = f"Julgamento {registro_index}"

    if raw_fields.get("composicao") or base_context.get("composicao"):
        if not is_valid_composicao(item.get("composicao", "")):
            item["composicao"] = raw_fields.get("composicao") or base_context.get("composicao", "")

    raw_full_cnj = extract_full_cnj(raw_fields.get("numero_processo", ""))
    item_full_cnj = extract_full_cnj(item.get("numero_processo", ""))
    if raw_full_cnj and not item_full_cnj:
        item["numero_processo"] = raw_full_cnj
    item["numero_processo"] = normalize_processo_num(item.get("numero_processo", ""))

    uf = extract_uf_from_text(item.get("origem", "")) or extract_uf_from_text(raw_fields.get("origem", ""))
    item["TRE"] = normalize_tre(item.get("TRE", ""), uf)

    if item.get("classe_processo"):
        item["classe_processo"] = normalize_classe_processo(item["classe_processo"])
    elif raw_fields.get("classe_processo"):
        item["classe_processo"] = normalize_classe_processo(raw_fields["classe_processo"])

    item["resultado_final"] = normalize_resultado_final(
        item.get("resultado_final", ""),
        item.get("classe_processo", "")
    )

    if item.get("relator"):
        item["relator"] = normalize_ministro_name(item["relator"])
    if item.get("pedido_vista"):
        item["pedido_vista"] = normalize_ministro_name(item["pedido_vista"])

    return item

def apply_sequential_tipo_registro(rows: List[Dict[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        row["tipo_registro"] = f"Julgamento {index}"

def process_report_text(
    report_text: str,
    model: str,
    use_api_for_vista: bool = False,
    resume_rows: Optional[List[Dict[str, Any]]] = None,
    resume_done_sections: Optional[Set[int]] = None,
    progress_callback: Optional[Callable[[List[Dict[str, Any]], Set[int], int], None]] = None,
) -> List[Dict[str, Any]]:
    normalized_text = normalize_text(report_text)
    base_context = extract_global_context(normalized_text)
    sections = split_report_into_sections(normalized_text)

    if not sections:
        process_numbers, is_joint = extract_process_numbers_from_section(normalized_text)
        structured_data = extract_structured_data(
            normalized_text,
            model,
            process_numbers=process_numbers if is_joint else None
        )
        items = structured_data.get("itens", []) or []
        derived_numbers, multi_found = extract_process_numbers_from_items(items)
        if not process_numbers and derived_numbers:
            process_numbers = derived_numbers
        if multi_found:
            is_joint = True
        if is_joint:
            items = align_items_to_process_numbers(items, process_numbers, is_joint)
            structured_data = {**structured_data, "itens": items}
        if items:
            vista_marker = has_pedido_vista_marker(normalized_text)
            if len(items) == 1:
                regex_vista = extract_pedido_vista(normalized_text, "")
                if regex_vista:
                    if not items[0].get("pedido_vista"):
                        items[0]["pedido_vista"] = regex_vista
                elif use_api_for_vista and vista_marker and (len(process_numbers) <= 1):
                    needs_vista = any(not item.get("pedido_vista") for item in items)
                    if needs_vista:
                        vista_api = extract_pedido_vista_via_api(normalized_text, model)
                        if vista_api:
                            items[0]["pedido_vista"] = vista_api
                if not vista_marker:
                    items[0]["pedido_vista"] = ""
            else:
                vista_map = extract_pedido_vista_map(normalized_text) if vista_marker else {}
                if vista_map:
                    apply_pedido_vista_map(items, vista_map)
                else:
                    for item in items:
                        item["pedido_vista"] = ""
        rows = flatten_data_for_csv(structured_data)
        apply_sequential_tipo_registro(rows)
        if progress_callback is not None:
            progress_callback(rows, {0}, 1)
        return rows

    logging.info(f"Encontrei {len(sections)} processo(s). Vou analisar um por um.")

    total_sections = len(sections)
    done_sections: Set[int] = set()
    for raw_idx in (resume_done_sections or set()):
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        if idx >= 0:
            done_sections.add(idx)
    rows = list(resume_rows or [])
    registro_index = len(rows) + 1
    for section_idx, section in enumerate(sections):
        if section_idx in done_sections:
            continue
        section_title = extract_section_title(section)
        logging.info(f"Analisando: {section_title}")
        process_numbers, is_joint = extract_process_numbers_from_section(section)
        if is_joint and process_numbers:
            logging.info(
                "Detectei julgamento conjunto com %d processo(s).",
                len(process_numbers)
            )
        raw_fields = extract_raw_fields(section, base_context)
        vista_marker = has_pedido_vista_marker(section)
        vista_map = extract_pedido_vista_map(section) if vista_marker else {}
        if is_joint and len(process_numbers) > 1:
            raw_fields["pedido_vista"] = ""
            item_prefill = {}
            fallback_fields = {
                k: v for k, v in raw_fields.items()
                if k not in {
                    "numero_processo",
                    "classe_processo",
                    "partes",
                    "advogados",
                    "origem",
                    "eleicao",
                    "TRE",
                }
            }
        else:
            item_prefill = {k: v for k, v in raw_fields.items() if k not in ("data_sessao", "composicao")}
            fallback_fields = raw_fields
        if not vista_marker:
            raw_fields["pedido_vista"] = ""
            fallback_fields["pedido_vista"] = ""
        structured_data = extract_structured_data(
            section,
            model,
            base_context=base_context,
            prefilled_fields=item_prefill,
            single_item=not is_joint,
            process_numbers=process_numbers if (is_joint and process_numbers) else None
        )
        items = structured_data.get("itens", []) or []
        derived_numbers, multi_found = extract_process_numbers_from_items(items)
        if not process_numbers and derived_numbers:
            process_numbers = derived_numbers
        if multi_found:
            is_joint = True
        if is_joint:
            items = align_items_to_process_numbers(items, process_numbers, is_joint)
        if not items:
            logging.warning("Esta parte não gerou dados. Vou seguir para a próxima.")
            done_sections.add(section_idx)
            if progress_callback is not None:
                progress_callback(rows, done_sections, total_sections)
            continue
        if len(items) == 1:
            if use_api_for_vista and vista_marker and not raw_fields.get("pedido_vista"):
                needs_vista = any(not item.get("pedido_vista") for item in items)
                if needs_vista:
                    vista_api = extract_pedido_vista_via_api(section, model)
                    if vista_api:
                        raw_fields["pedido_vista"] = vista_api
                        fallback_fields["pedido_vista"] = vista_api
        else:
            raw_fields["pedido_vista"] = ""
            fallback_fields["pedido_vista"] = ""
            if vista_map:
                apply_pedido_vista_map(items, vista_map)
            else:
                for item in items:
                    item["pedido_vista"] = ""

        sessao_info = merge_session_info(structured_data.get("sessao", {}), base_context, raw_fields)
        for item in items:
            item = apply_item_fallbacks(item, fallback_fields, base_context, registro_index)
            if not vista_marker:
                item["pedido_vista"] = ""
            rows.extend(flatten_data_for_csv({"sessao": sessao_info, "itens": [item]}))
            registro_index += 1
        done_sections.add(section_idx)
        apply_sequential_tipo_registro(rows)
        if progress_callback is not None:
            progress_callback(rows, done_sections, total_sections)

    apply_sequential_tipo_registro(rows)
    return rows

def select_inputs_via_gui(args: argparse.Namespace) -> Tuple[List[str], Dict[str, bool]]:
    """Abre um painel GUI completo para selecionar entradas e flags de execucao."""
    gui = open_file_panel(
        title="Sessoes Plenarias TSE - TXT para CSV",
        subtitle="Selecione os relatorios .txt e ajuste as opcoes de execucao.",
        filetypes=[("Relatorios TXT", "*.txt"), ("Todos os arquivos", "*.*")],
        extensions=[".txt"],
        initial_files=[],
        allow_add_dir=True,
        recursive_dir=True,
        min_files=1,
        output_label="",
        extra_bools=[
            ("verbose", "Verbose", bool(args.verbose)),
            ("debug", "Debug tecnico", bool(args.debug)),
            ("quiet", "Quiet", bool(args.quiet)),
            ("vista_api", "Ativar fallback vista via API", bool(args.vista_api)),
        ],
    )
    if not gui or not gui.get("confirmed"):
        return [], {}
    files = list(gui.get("files") or [])
    bools = {k: bool(v) for k, v in (gui.get("bools") or {}).items()}
    return files, bools

def normalize_input_files(file_paths: List[str]) -> List[str]:
    """Filtra e normaliza a lista de arquivos selecionados."""
    normalized = []
    for path in file_paths:
        if not path.lower().endswith(".txt"):
            logging.warning(f"Ignorando arquivo que não é .txt: {path}")
            continue
        if not os.path.isfile(path):
            logging.warning(f"Arquivo não encontrado no caminho informado: {path}")
            continue
        normalized.append(os.path.abspath(path))
    return normalized

def extract_structured_data(
    report_text: str,
    model: str,
    max_retries: Optional[int] = None,
    base_context: Optional[Dict[str, Any]] = None,
    prefilled_fields: Optional[Dict[str, Any]] = None,
    single_item: bool = False,
    process_numbers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Envia o texto do relatório para a API da OpenAI e retorna os dados estruturados.
    Implementa uma estratégia de retentativa com backoff exponencial para lidar com
    erros de API.
    """
    prompt = get_user_prompt(
        report_text,
        base_context=base_context,
        prefilled_fields=prefilled_fields,
        single_item=single_item,
        process_numbers=process_numbers
    )
    logging.debug("Texto preparado para envio à IA.")
    logging.debug(f"Tamanho do relatório: {len(report_text)} caracteres. Tamanho do pedido à IA: {len(prompt)} caracteres.")

    retries = max(1, int(max_retries if max_retries is not None else OPENAI_RETRY_ATTEMPTS))
    for attempt in range(retries):
        try:
            logging.info(f"Enviando o relatório para a IA (tentativa {attempt + 1}/{retries})...")
            start_time = time.perf_counter()
            stop_event = threading.Event()
            progress_thread = None
            progress_interval = 5 if logging.getLogger().isEnabledFor(logging.DEBUG) else 10

            def log_api_progress():
                elapsed = 0
                interval = progress_interval
                while not stop_event.wait(interval):
                    elapsed += interval
                    logging.info(f"A IA ainda está analisando... {elapsed}s")

            progress_thread = threading.Thread(target=log_api_progress, daemon=True)
            progress_thread.start()

            try:
                OPENAI_REQUEST_PACER.wait_turn()
                response = _openai_chat_completion(
                    {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "tse_relatorio_sessao",
                                "schema": JSON_OUTPUT_SCHEMA,
                                "strict": True,
                            },
                        },
                    }
                )
            finally:
                stop_event.set()
                if progress_thread is not None:
                    progress_thread.join(timeout=1)

            elapsed = time.perf_counter() - start_time
            usage = getattr(response, "usage", None)
            logging.info("Resposta da IA recebida em %.2fs.", elapsed)
            if usage:
                logging.debug(
                    "Detalhes de uso: tokens prompt=%s, completion=%s, total=%s",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                )

            content = response.choices[0].message.content
            if SHOW_TECHNICAL_LOGS:
                logging.debug(f"Resposta bruta da IA (JSON, início): {content[:400]}...")

            # Validação final para garantir que é um JSON válido
            parsed_json = json.loads(content)
            logging.info("Resposta validada. Vamos preparar o CSV.")
            return parsed_json

        except json.JSONDecodeError as e:
            logging.warning(
                f"A resposta da IA não veio no formato esperado (tentativa {attempt + 1}). Vamos tentar novamente."
            )
            if SHOW_TECHNICAL_LOGS:
                logging.debug(f"Detalhes do erro: {e}. Conteúdo problemático: {content}")

        except APIError as e:
            logging.warning(
                f"A comunicação com a IA falhou nesta tentativa ({attempt + 1}). Vamos tentar novamente."
            )
            if SHOW_TECHNICAL_LOGS:
                logging.debug(f"Detalhes do erro da API: {e}")

        except Exception as e:
            logging.error(
                f"Ocorreu um problema inesperado nesta tentativa ({attempt + 1}).",
                exc_info=True
            )

        # Lógica de espera (backoff exponencial) antes da próxima tentativa
        if attempt < retries - 1:
            wait_time = min(10.0, max(0.2, float(OPENAI_BASE_DELAY_SECONDS)) * (2 ** attempt))
            logging.info(f"Aguardando {wait_time}s para tentar novamente.")
            time.sleep(wait_time)

    raise ConnectionError(f"Não foi possível obter uma resposta JSON válida da API após {retries} tentativas.")

def flatten_data_for_csv(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Converte o JSON aninhado em uma lista de dicionários planos para o CSV."""
    sessao_info = json_data.get("sessao", {})
    itens_julgamento = json_data.get("itens", [])

    if not itens_julgamento:
        logging.warning("Não encontrei julgamentos nesta resposta da IA.")
        return []

    logging.debug(f"Transformando {len(itens_julgamento)} julgamento(s) em linhas de CSV.")
    
    csv_rows = []
    for item in itens_julgamento:
        row = {
            "data_sessao": sessao_info.get("data_sessao", ""),
            "composicao": sessao_info.get("composicao", ""),
            **item
        }
        
        # Limpa e padroniza os valores para o CSV
        for key, value in row.items():
            if value is None:
                row[key] = ""
            elif isinstance(value, list):
                # Converte listas em string, embora o prompt já peça para a IA fazer isso
                row[key] = ", ".join(map(str, value))
            else:
                row[key] = str(value).strip()
            if row[key]:
                row[key] = normalize_mpe_reference(row[key])

        row["data_sessao"] = parse_date_from_text(row.get("data_sessao", "")) or row.get("data_sessao", "")
        row["numero_processo"] = canonicalize_numero_processo(row.get("numero_processo", ""))
        row["origem"] = normalize_origem_value(row.get("origem", ""))
        row["TRE"] = normalize_tre(row.get("TRE", ""), extract_uf_from_text(row.get("origem", "")))
        row["advogados"] = normalize_advogados_list(row.get("advogados", ""))
        row["classe_processo"] = normalize_classe_processo(row.get("classe_processo", ""))
        row["resultado_final"] = normalize_resultado_final(
            row.get("resultado_final", ""),
            row.get("classe_processo", "")
        )
        row["composicao"] = normalize_composicao(row.get("composicao", ""))
        row["partes"] = remove_mpe_from_partes(row.get("partes", ""))
        row["youtube_link"] = normalize_youtube_link(row.get("youtube_link", ""))
        if row.get("relator"):
            row["relator"] = normalize_ministro_name(row["relator"])
        if row.get("pedido_vista"):
            row["pedido_vista"] = normalize_pedido_vista_value(row["pedido_vista"])
        
        csv_rows.append(row)
    
    return csv_rows

# --- 3. Execução Principal ---

def main():
    """
    Função principal que orquestra a leitura, processamento e gravação dos dados.
    """
    global OPENAI_TIMEOUT_SECONDS
    global OPENAI_RETRY_ATTEMPTS
    global OPENAI_BASE_DELAY_SECONDS
    global OPENAI_VISTA_RETRIES
    global OPENAI_REQUEST_PACER

    parser = argparse.ArgumentParser(description="Converte relatórios .txt do TSE para .csv usando a API da OpenAI.")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Força modo detalhado (padrão quando não há --quiet)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mostra detalhes técnicos (conteúdo bruto e erros detalhados)."
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Exibe apenas avisos e erros."
    )
    parser.add_argument("--log-file", default="", help="Arquivo opcional para salvar logs.")
    parser.add_argument(
        "--vista-api",
        action="store_true",
        help="Se pedido_vista não for encontrado, tenta uma chamada extra à API para extrair apenas esse campo."
    )
    parser.add_argument("--openai-model", default=OPENAI_DEFAULT_MODEL, help="Modelo OpenAI.")
    parser.add_argument("--openai-timeout", type=int, default=OPENAI_DEFAULT_TIMEOUT, help="Timeout por chamada (s).")
    parser.add_argument("--openai-retries", type=int, default=OPENAI_DEFAULT_RETRIES, help="Tentativas da extração principal.")
    parser.add_argument("--openai-base-delay", type=float, default=OPENAI_DEFAULT_BASE_DELAY, help="Delay base do backoff exponencial.")
    parser.add_argument("--openai-target-rpm", type=int, default=OPENAI_DEFAULT_TARGET_RPM, help="Limite alvo de requests por minuto.")
    parser.add_argument("--openai-vista-retries", type=int, default=OPENAI_DEFAULT_VISTA_RETRIES, help="Tentativas da extração de pedido_vista.")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Desativa painel GUI e usa apenas os argumentos CLI.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Caminhos para arquivos .txt. Se omitido, abre o Windows Explorer para seleção."
    )
    args = parser.parse_args()

    if args.files:
        selected_files = args.files
    elif not args.no_gui:
        gui_files, gui_flags = select_inputs_via_gui(args)
        selected_files = gui_files
        if gui_flags:
            args.verbose = bool(gui_flags.get("verbose", args.verbose))
            args.debug = bool(gui_flags.get("debug", args.debug))
            args.quiet = bool(gui_flags.get("quiet", args.quiet))
            args.vista_api = bool(gui_flags.get("vista_api", args.vista_api))
    else:
        selected_files = []

    configure_logging(args)
    OPENAI_TIMEOUT_SECONDS = max(5, int(args.openai_timeout))
    OPENAI_RETRY_ATTEMPTS = max(1, int(args.openai_retries))
    OPENAI_BASE_DELAY_SECONDS = max(0.2, float(args.openai_base_delay))
    OPENAI_VISTA_RETRIES = max(1, int(args.openai_vista_retries))
    OPENAI_REQUEST_PACER = RequestPacer(max(0, int(args.openai_target_rpm)))

    if args.quiet:
        logging.info("Modo silencioso ativado. Logs reduzidos.")
    else:
        logging.info("Modo detalhado ativado (padrão).")

    logging.info(
        "OpenAI runtime: model=%s timeout=%ss retries=%d vista_retries=%d rpm=%d base_delay=%.2fs",
        str(args.openai_model or OPENAI_DEFAULT_MODEL).strip() or OPENAI_DEFAULT_MODEL,
        OPENAI_TIMEOUT_SECONDS,
        OPENAI_RETRY_ATTEMPTS,
        OPENAI_VISTA_RETRIES,
        int(args.openai_target_rpm),
        OPENAI_BASE_DELAY_SECONDS,
    )

    MODELO_OPENAI = str(args.openai_model or OPENAI_DEFAULT_MODEL).strip() or OPENAI_DEFAULT_MODEL

    if args.vista_api:
        logging.info("Fallback via API para pedido_vista ativado.")

    files_to_process = normalize_input_files(selected_files)

    if not files_to_process:
        logging.warning("Nenhum arquivo selecionado. Encerrando.")
        return

    logging.info(f"Vamos processar {len(files_to_process)} arquivo(s).")

    for input_file_path in files_to_process:
        filename = os.path.basename(input_file_path)
        logging.info(f"Iniciando análise do arquivo: {filename}")
        try:
            input_path_obj = Path(input_file_path).resolve()
            with input_path_obj.open('r', encoding='utf-8') as f:
                report_text = f.read()
            
            if not report_text.strip():
                logging.warning(f"O arquivo '{filename}' está vazio. Não há nada para processar.")
                continue
            
            logging.debug("Texto do relatório carregado com sucesso.")

            final_columns = CSV_COLUMN_ORDER
            base_name = input_path_obj.stem
            output_csv_path = (SCRIPT_DIR / f"{base_name}.csv").resolve()
            checkpoint_path = (SCRIPT_DIR / f".{base_name}.openai.checkpoint.json").resolve()
            report_path = (SCRIPT_DIR / f".{base_name}.openai.report.json").resolve()
            source_sig = build_file_signature(input_path_obj)
            started_file_at = time.time()

            if output_csv_path.exists() and not checkpoint_path.exists():
                backup = make_backup(output_csv_path, label="startup_backup")
                if backup is not None:
                    logging.info("Backup inicial criado: %s", backup)

            resume_rows: List[Dict[str, Any]] = []
            resume_done_sections: Set[int] = set()
            cp = read_json_dict(checkpoint_path)
            cp_sig = cp.get("source_signature", {})
            cp_rows = cp.get("rows", [])
            cp_done = cp.get("done_sections", [])
            cp_model = str(cp.get("openai_model", "")).strip()
            cp_vista = bool(cp.get("use_api_for_vista", False))
            if (
                int(cp.get("version", 0) or 0) == CHECKPOINT_VERSION
                and same_file_signature(cp_sig, source_sig)
                and cp_model == MODELO_OPENAI
                and cp_vista == bool(args.vista_api)
                and isinstance(cp_rows, list)
                and isinstance(cp_done, list)
            ):
                resume_rows = [r for r in cp_rows if isinstance(r, dict)]
                for raw_idx in cp_done:
                    try:
                        idx = int(raw_idx)
                    except Exception:
                        continue
                    if idx >= 0:
                        resume_done_sections.add(idx)
                if resume_rows or resume_done_sections:
                    logging.info(
                        "Retomando checkpoint: secoes_concluidas=%d | linhas=%d",
                        len(resume_done_sections),
                        len(resume_rows),
                    )

            progress_state: Dict[str, Any] = {
                "done_sections": set(resume_done_sections),
                "total_sections": 0,
            }

            def _rows_to_csv(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                out_rows: List[Dict[str, Any]] = []
                for row in rows:
                    out_rows.append(
                        {
                            col: ("" if row.get(col) is None else str(row.get(col, "")))
                            for col in final_columns
                        }
                    )
                return out_rows

            def _save_progress(status: str, rows: List[Dict[str, Any]], done_sections: Set[int], total_sections: int) -> None:
                progress_state["done_sections"] = set(done_sections)
                progress_state["total_sections"] = int(total_sections)
                csv_rows = _rows_to_csv(rows)
                write_csv_atomic(output_csv_path, final_columns, csv_rows)
                write_json_atomic(
                    checkpoint_path,
                    {
                        "version": CHECKPOINT_VERSION,
                        "source_signature": source_sig,
                        "openai_model": MODELO_OPENAI,
                        "use_api_for_vista": bool(args.vista_api),
                        "input_file": str(input_path_obj),
                        "output_csv": str(output_csv_path),
                        "total_sections": int(total_sections),
                        "done_sections": sorted(int(x) for x in done_sections),
                        "rows": rows,
                        "status": status,
                        "updated_at": utc_now_iso(),
                    },
                )
                write_json_atomic(
                    report_path,
                    {
                        "script": "SESSOES_PLENARIAS_TSE.py",
                        "input_file": str(input_path_obj),
                        "output_csv": str(output_csv_path),
                        "checkpoint_file": str(checkpoint_path),
                        "status": status,
                        "openai_model": MODELO_OPENAI,
                        "use_api_for_vista": bool(args.vista_api),
                        "total_sections": int(total_sections),
                        "done_sections": len(done_sections),
                        "rows_written": len(csv_rows),
                        "elapsed_seconds": round(max(0.0, time.time() - started_file_at), 2),
                        "updated_at": utc_now_iso(),
                    },
                )

            flattened_rows = process_report_text(
                report_text,
                MODELO_OPENAI,
                use_api_for_vista=args.vista_api,
                resume_rows=resume_rows,
                resume_done_sections=resume_done_sections,
                progress_callback=lambda rows, done, total: _save_progress("running", rows, done, total),
            )

            if not flattened_rows:
                logging.warning(
                    f"Não consegui extrair dados de '{filename}'. Por isso, o CSV não será criado."
                )
                _save_progress(
                    "empty_output",
                    [],
                    set(progress_state.get("done_sections", set())),
                    int(progress_state.get("total_sections", 0)),
                )
                continue

            total_sections = int(progress_state.get("total_sections", 0))
            done_sections = set(progress_state.get("done_sections", set()))
            _save_progress("completed", flattened_rows, done_sections, total_sections)
            if checkpoint_path.exists():
                checkpoint_path.unlink()

            if output_csv_path.is_file():
                logging.info(f"Pronto! CSV gerado com {len(flattened_rows)} linha(s).")
            else:
                logging.warning("Tentei salvar o CSV, mas não encontrei o arquivo no disco.")

        except Exception as e:
            logging.error(
                f"Ocorreu um problema ao processar o arquivo {filename}.",
                exc_info=not args.quiet
            )
    
    logging.info("Processamento finalizado.")

if __name__ == "__main__":
    main()
