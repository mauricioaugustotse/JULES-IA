# -*- coding: utf-8 -*-
"""
Enrich sessoes_all_2024_2025.csv with news links using the Gemini API.

Usage:
  python3 SESSOES_TSE_noticias_viaAPI.py
  python3 SESSOES_TSE_noticias_viaAPI.py --input sessoes_all_2024_2025.csv --output sessoes_all_2024_2025_noticias.csv

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types, errors
except ImportError as exc:
    raise SystemExit("ERRO: google-genai nao encontrado. Execute: pip install google-genai") from exc


DEFAULT_INPUT = "sessoes_all_2024_2025.csv"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL") or "gemini-2.0-flash"

NEWS_COLUMNS = ["noticia_TSE", "noticia_TRE", "noticia_geral"]

CONTEXT_FIELDS = [
    "tema",
    "punchline",
    "numero_processo",
    "classe_processo",
    "tribunal",
    "origem",
    "data_sessao",
    "relator",
    "partes",
]

SYSTEM_PROMPT = (
    "You are a research assistant. Use web search when available. "
    "Return JSON only, without any extra text. "
    "Strictly adhere to the requested JSON schema."
)

USER_PROMPT_TEMPLATE = (
    "Find news articles related to the following Brazilian electoral court session item. "
    "Only include links if the article is clearly about the same case/decision/session. "
    "If no relevant news exists, return empty arrays.\n\n"
    "Return ONLY a JSON object with keys:\n"
    "- noticia_TSE: list of URLs from domains that end with tse.jus.br\n"
    "- noticia_TRE: list of URLs from domains that match tre-XX.jus.br (any subdomain)\n"
    "- noticia_geral: list of URLs from Folha (folha.uol.com.br), Estadao (estadao.com.br), "
    "Gazeta do Povo (gazetadopovo.com.br), CNN (cnnbrasil.com.br or cnn.com), "
    "ConJur (conjur.com.br), Migalhas (migalhas.com.br)\n\n"
    "Context:\n{context}\n"
)

URL_RE = re.compile(r"https?://[^\s\]\)>,;\"']+", re.IGNORECASE)
TRE_DOMAIN_RE = re.compile(r"(?:^|\.)tre-[a-z]{2}\.jus\.br$", re.IGNORECASE)

GENERAL_DOMAINS = [
    "folha.uol.com.br",
    "estadao.com.br",
    "gazetadopovo.com.br",
    "cnnbrasil.com.br",
    "cnn.com",
    "conjur.com.br",
    "migalhas.com.br",
]


def _build_context(row: Dict[str, str], max_len: int = 240) -> str:
    lines: List[str] = []
    for field in CONTEXT_FIELDS:
        raw = (row.get(field) or "").strip()
        if not raw:
            continue
        if len(raw) > max_len:
            raw = raw[:max_len].rstrip() + "..."
        lines.append(f"{field}: {raw}")
    return "\n".join(lines).strip()


def _output_text_from_response(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    try:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            texts = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    texts.append(part_text)
            if texts:
                return "\n".join(texts).strip()
    except Exception:
        return ""
    return ""


def _build_search_tools() -> Optional[List[types.Tool]]:
    if types is None:
        return None
    for cls_name in ("GoogleSearchRetrieval", "GoogleSearch"):
        tool_cls = getattr(types, cls_name, None)
        if not tool_cls:
            continue
        try:
            return [types.Tool(google_search=tool_cls())]
        except Exception:
            continue
    return None


def _call_gemini_with_web_search(
    client: genai.Client,
    model: str,
    prompt: str,
    max_retries: int,
    search_tools: Optional[List[types.Tool]],
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            config_kwargs = {
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
            }
            if search_tools:
                config_kwargs["tools"] = search_tools
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            text = _output_text_from_response(response)
            if not text:
                raise ValueError("Resposta vazia da API.")
            return text
        except (errors.ClientError, errors.ServerError) as exc:
            last_err = exc
            logging.warning("Erro da API (tentativa %d/%d): %s", attempt + 1, max_retries, exc)
            # Exponential backoff with jitter
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(sleep_time)
        except Exception as exc:
            last_err = exc
            logging.warning("Erro inesperado na API (tentativa %d/%d): %s", attempt + 1, max_retries, exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Falha na chamada da API apos {max_retries} tentativas: {last_err}")


def _extract_json(text: str) -> Dict[str, object]:
    # Remove markdown code blocks if present
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _normalize_url(url: str) -> str:
    cleaned = (url or "").strip().strip(".,;)]}>\"'")
    if not cleaned:
        return ""
    if not re.match(r"^https?://", cleaned, re.IGNORECASE):
        cleaned = "https://" + cleaned
    return cleaned


def _domain_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _is_tse_domain(domain: str) -> bool:
    return domain == "tse.jus.br" or domain.endswith(".tse.jus.br")


def _is_tre_domain(domain: str) -> bool:
    return bool(TRE_DOMAIN_RE.search(domain))


def _is_general_domain(domain: str) -> bool:
    for base in GENERAL_DOMAINS:
        if domain == base or domain.endswith("." + base):
            return True
    return False


def _urls_from_value(value: object) -> List[str]:
    urls: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                urls.append(item)
    elif isinstance(value, str):
        urls.extend(URL_RE.findall(value))
    return urls


def _classify_urls(urls: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    tse: List[str] = []
    tre: List[str] = []
    geral: List[str] = []
    seen = set()
    for raw in urls:
        normalized = _normalize_url(raw)
        if not normalized or normalized in seen:
            continue
        domain = _domain_from_url(normalized)
        if not domain:
            continue
        if _is_tse_domain(domain):
            tse.append(normalized)
        elif _is_tre_domain(domain):
            tre.append(normalized)
        elif _is_general_domain(domain):
            geral.append(normalized)
        else:
            continue
        seen.add(normalized)
    return tse, tre, geral


def _combine_urls_from_response(text: str) -> Tuple[List[str], List[str], List[str]]:
    data = _extract_json(text)
    urls: List[str] = []
    for key in NEWS_COLUMNS:
        urls.extend(_urls_from_value(data.get(key)))
    if not urls:
        urls = URL_RE.findall(text)
    return _classify_urls(urls)


def _join_urls(urls: List[str]) -> str:
    return ", ".join(urls) if urls else ""


def _derive_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".csv"
    return f"{base}_noticias{ext}"


def _read_existing_output(path: str) -> Tuple[List[str], int]:
    if not os.path.exists(path):
        return [], 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0
        count = 0
        for _ in reader:
            count += 1
    return header, count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Busca noticias por linha usando API e adiciona colunas ao CSV."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Caminho do CSV de entrada.")
    parser.add_argument("--output", default="", help="Caminho do CSV de saida.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo Gemini (opcional).")
    parser.add_argument("--limit", type=int, default=0, help="Processa apenas as primeiras N linhas.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Pausa entre chamadas da API (segundos).")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximo de tentativas da API.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma a partir do CSV de saida existente, mantendo as linhas ja geradas.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log detalhado.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("ERRO: GEMINI_API_KEY ou GOOGLE_API_KEY nao definido no ambiente.")
    client = genai.Client(api_key=api_key)

    search_tools = _build_search_tools()
    if search_tools is None:
        logging.warning("Ferramenta de web search indisponivel; seguindo sem busca.")

    input_path = args.input
    output_path = args.output or _derive_output_path(input_path)

    if not os.path.exists(input_path):
        raise SystemExit(f"ERRO: arquivo de entrada nao encontrado: {input_path}")

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        logging.warning("Nenhuma linha encontrada no CSV de entrada.")
        return

    output_fields = fieldnames[:]
    for col in NEWS_COLUMNS:
        if col not in output_fields:
            output_fields.append(col)

    cache: Dict[str, Tuple[List[str], List[str], List[str]]] = {}

    existing_header: List[str] = []
    processed_rows = 0
    write_mode = "w"
    if args.resume and os.path.exists(output_path):
        existing_header, processed_rows = _read_existing_output(output_path)
        if existing_header:
            if existing_header != output_fields:
                raise SystemExit(
                    "ERRO: o cabecalho do CSV de saida nao corresponde ao esperado. "
                    "Use --output para um novo arquivo ou remova o arquivo existente."
                )
            write_mode = "a"
            logging.info("Retomando a partir da linha %d do CSV de saida.", processed_rows + 1)

    with open(output_path, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        if write_mode == "w":
            writer.writeheader()

        total = len(rows) if args.limit <= 0 else min(len(rows), args.limit)
        if processed_rows >= total:
            logging.info("Nada a fazer: %d linhas ja processadas.", processed_rows)
            return

        for idx, row in enumerate(rows[:total], start=1):
            if idx <= processed_rows:
                continue
            context = _build_context(row)
            if not context:
                row["noticia_TSE"] = ""
                row["noticia_TRE"] = ""
                row["noticia_geral"] = ""
                writer.writerow(row)
                continue

            cache_key = context
            if cache_key in cache:
                tse_urls, tre_urls, geral_urls = cache[cache_key]
            else:
                prompt = USER_PROMPT_TEMPLATE.format(context=context)
                try:
                    raw_text = _call_gemini_with_web_search(
                        client=client,
                        model=args.model,
                        prompt=prompt,
                        max_retries=args.max_retries,
                        search_tools=search_tools,
                    )
                    tse_urls, tre_urls, geral_urls = _combine_urls_from_response(raw_text)
                    cache[cache_key] = (tse_urls, tre_urls, geral_urls)
                except Exception as exc:
                    logging.error("Falha ao consultar API na linha %d: %s", idx, exc)
                    tse_urls, tre_urls, geral_urls = [], [], []

            row["noticia_TSE"] = _join_urls(tse_urls)
            row["noticia_TRE"] = _join_urls(tre_urls)
            row["noticia_geral"] = _join_urls(geral_urls)
            writer.writerow(row)

            if idx % 10 == 0 or idx == total:
                logging.info("Processado %d/%d", idx, total)
            if args.sleep:
                time.sleep(args.sleep)

    logging.info("Arquivo gerado: %s", output_path)


if __name__ == "__main__":
    main()
