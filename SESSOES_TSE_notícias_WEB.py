# -*- coding: utf-8 -*-
"""
Compat layer for the legacy news-enrichment script.

The original file in this repository had a partially merged async refactor and no
longer compiled. This replacement keeps the helper API stable for the existing
tests and for lightweight CSV enrichment runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

from local_secrets import get_secret, load_local_secrets

load_local_secrets(base_dir=os.path.dirname(os.path.abspath(__file__)))

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = SimpleNamespace(Client=None)

    class _FallbackGoogleSearch:
        pass

    class _FallbackTool:
        def __init__(self, google_search=None):
            self.google_search = google_search

    class _FallbackGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    types = SimpleNamespace(
        Tool=_FallbackTool,
        GoogleSearch=_FallbackGoogleSearch,
        GenerateContentConfig=_FallbackGenerateContentConfig,
    )


DEFAULT_INPUT = ""
DEFAULT_MODEL = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
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
CONTEXT_PREFIXES = [(field, f"{field}: ") for field in CONTEXT_FIELDS]

SYSTEM_PROMPT = (
    "You are a research assistant. Use Google Search to find real news articles. "
    "Return ONLY valid JSON, without markdown formatting or extra text. "
    "Strictly adhere to the requested JSON schema."
)

USER_PROMPT_TEMPLATE = (
    "Find news articles related to the following Brazilian electoral court session item. "
    "Only include links if the article is clearly about the same case/decision/session. "
    "If no relevant news exists, return empty arrays.\n\n"
    "Return ONLY a JSON object with keys:\n"
    "- noticia_TSE: list of URLs from domains that end with tse.jus.br\n"
    "- noticia_TRE: list of URLs from domains that match tre-XX.jus.br\n"
    "- noticia_geral: list of URLs from major Brazilian news outlets (Folha, Estadao, CNN, G1, Conjur, Migalhas, etc)\n\n"
    "Context:\n{context}\n"
)

URL_RE = re.compile(r"https?://[^\s\]\)>,;\"']+", re.IGNORECASE)
TRE_DOMAIN_RE = re.compile(r"(?:^|\.)tre-[a-z]{2}\.jus\.br$", re.IGNORECASE)
TSE_DOMAIN_RE = re.compile(r"(?:^|\.)tse\.jus\.br$", re.IGNORECASE)
NORMALIZE_PROTOCOL_RE = re.compile(r"^https?://", re.IGNORECASE)

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
    r"(?:^|\.)(?:" + "|".join(re.escape(d) for d in GENERAL_DOMAINS) + r")$",
    re.IGNORECASE,
)


def _build_context(row: Dict[str, str], max_len: int = 300) -> str:
    lines: List[str] = []
    for field, prefix in CONTEXT_PREFIXES:
        raw = (row.get(field) or "").strip()
        if not raw:
            continue
        if len(raw) > max_len:
            raw = raw[:max_len].rstrip() + "..."
        lines.append(prefix + raw)
    return "\n".join(lines)


def _get_api_key_securely() -> str:
    return get_secret(
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        base_dir=os.path.dirname(os.path.abspath(__file__)),
    )


def _call_gemini_with_web_search(
    client,
    model: str,
    prompt: str,
    max_retries: int,
    verbose: bool = False,
) -> str:
    if client is None:
        raise ValueError("Client Gemini não fornecido.")
    if types is None:
        raise RuntimeError("Biblioteca google-genai não está disponível.")

    google_search_tool = types.Tool(google_search=types.GoogleSearch())
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        if verbose:
            logging.info(
                "Chamando API Gemini (modelo=%s, tentativa %d/%d)...",
                model,
                attempt + 1,
                max_retries,
            )
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[google_search_tool],
                    temperature=0.1,
                ),
            )
            if getattr(response, "text", ""):
                return response.text
            return "{}"
        except Exception as exc:  # pragma: no cover - behavior validated via mocks
            last_err = exc
            status_code = getattr(exc, "code", None)
            if status_code in (401, 403):
                logging.error("Erro fatal de autenticação/permissão (%s): %s", status_code, exc)
                raise
            error_msg = str(exc)
            if "429" in error_msg or "ResourceExhausted" in error_msg or status_code == 429:
                logging.warning("Cota atingida (429). Aguardando 60s...")
                time.sleep(60)
                continue
            logging.warning("Erro na API (tentativa %d/%d): %s", attempt + 1, max_retries, exc)
            time.sleep(2 ** attempt)
    logging.error("Falha definitiva após %d tentativas: %s", max_retries, last_err)
    return "{}"


def _extract_json(text: str) -> Dict[str, object]:
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


def _normalize_url(url: str) -> str:
    cleaned = (url or "").strip().strip(".,;)]}>\"'")
    if not cleaned:
        return ""
    if not NORMALIZE_PROTOCOL_RE.match(cleaned):
        cleaned = "https://" + cleaned
    return cleaned


def _domain_from_url(url: str) -> str:
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


def _classify_urls(urls: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    tse: List[str] = []
    tre: List[str] = []
    geral: List[str] = []
    seen: set[str] = set()

    for raw in urls:
        normalized = _normalize_url(raw)
        if not normalized or normalized in seen:
            continue
        domain = _domain_from_url(normalized)
        if not domain:
            continue
        if TSE_DOMAIN_RE.search(domain):
            tse.append(normalized)
        elif TRE_DOMAIN_RE.search(domain):
            tre.append(normalized)
        elif GENERAL_DOMAINS_RE.search(domain):
            geral.append(normalized)
        elif "jus.br" not in domain:
            geral.append(normalized)
        seen.add(normalized)
    return tse, tre, geral


def _process_response_text(text: str) -> Tuple[str, str, List[str]]:
    data = _extract_json(text)
    urls: List[str] = []
    for key in NEWS_COLUMNS:
        value = data.get(key)
        if isinstance(value, list):
            urls.extend(str(item) for item in value if isinstance(item, str))
    if not urls:
        urls = URL_RE.findall(text)
    tse_list, tre_list, geral_list = _classify_urls(urls)
    return ", ".join(tse_list), ", ".join(tre_list), geral_list


def _apply_geral_links(row: Dict[str, str], geral_links: List[str]) -> None:
    row["noticia_geral"] = geral_links[0] if geral_links else ""
    for idx in range(1, 10):
        row[f"noticia_geral_{idx}"] = geral_links[idx] if idx < len(geral_links) else ""


def enrich_rows(rows: List[Dict[str, str]], model: str, verbose: bool = False) -> List[Dict[str, str]]:
    api_key = _get_api_key_securely()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY não encontrado.")
    if genai is None:
        raise RuntimeError("Biblioteca google-genai não instalada.")

    client = genai.Client(api_key=api_key)
    enriched: List[Dict[str, str]] = []
    for row in rows:
        context = _build_context(row)
        if not context:
            enriched.append(row)
            continue
        raw_text = _call_gemini_with_web_search(
            client=client,
            model=model,
            prompt=USER_PROMPT_TEMPLATE.format(context=context),
            max_retries=3,
            verbose=verbose,
        )
        noticia_tse, noticia_tre, noticia_geral_links = _process_response_text(raw_text)
        row = dict(row)
        row["noticia_TSE"] = noticia_tse
        row["noticia_TRE"] = noticia_tre
        _apply_geral_links(row, noticia_geral_links)
        enriched.append(row)
    return enriched


def read_csv_rows(input_path: str) -> List[Dict[str, str]]:
    with open(input_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(output_path: str, rows: List[Dict[str, str]]) -> None:
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca notícias relacionadas a julgados do TSE via Gemini.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV de entrada.")
    parser.add_argument("--output", default="", help="CSV de saída.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo Gemini.")
    parser.add_argument("--verbose", action="store_true", help="Exibe logs detalhados.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    input_path = (args.input or "").strip()
    if not input_path:
        raise SystemExit("Informe --input com o caminho do CSV.")
    if not os.path.exists(input_path):
        raise SystemExit(f"Arquivo não encontrado: {input_path}")

    output_path = (args.output or "").strip()
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base} - com notícias da WEB{ext}"

    rows = read_csv_rows(input_path)
    enriched = enrich_rows(rows, model=args.model, verbose=args.verbose)
    write_csv_rows(output_path, enriched)
    logging.info("Arquivo salvo em: %s", output_path)


if __name__ == "__main__":
    main()
