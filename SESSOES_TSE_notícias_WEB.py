# -*- coding: utf-8 -*-
"""
Enriquece o arquivo com links de notícias usando a API Gemini (Google GenAI SDK).
Lê a chave de API do arquivo .env para segurança.

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

# --- IMPORTAÇÃO SEGURA DE VARIÁVEIS DE AMBIENTE ---
try:
    from dotenv import load_dotenv
    # Carrega o arquivo .env que está na mesma pasta
    load_dotenv()
except ImportError:
    print("AVISO: python-dotenv não instalado. Se der erro de chave, instale: pip install python-dotenv")

# --- IMPORTAÇÃO DA NOVA BIBLIOTECA GOOGLE GENAI ---
try:
    from google import genai
    from google.genai import types, errors
except ImportError as exc:
    raise SystemExit("ERRO CRÍTICO: Biblioteca 'google-genai' não encontrada.\nExecute: pip install google-genai") from exc

# Configurações Padrão
DEFAULT_INPUT = ""
DEFAULT_MODEL = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

NEWS_COLUMNS = ["noticia_TSE", "noticia_TRE", "noticia_geral"]

CONTEXT_FIELDS = [
    "tema", "punchline", "numero_processo", "classe_processo",
    "tribunal", "origem", "data_sessao", "relator", "partes",
]

# Prompt do Sistema (Instrução fixa)
SYSTEM_PROMPT = (
    "You are a research assistant. Use Google Search to find real news articles. "
    "Return ONLY valid JSON, without markdown formatting or extra text. "
    "Strictly adhere to the requested JSON schema."
)

# Prompt do Usuário (Template)
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

GENERAL_DOMAINS = [
    "folha.uol.com.br", "estadao.com.br", "gazetadopovo.com.br",
    "cnnbrasil.com.br", "cnn.com", "conjur.com.br", "migalhas.com.br",
    "g1.globo.com", "oglobo.globo.com", "poder360.com.br"
]

def _build_context(row: Dict[str, str], max_len: int = 300) -> str:
    """Cria um resumo do caso para enviar ao Gemini."""
    lines: List[str] = []
    for field in CONTEXT_FIELDS:
        raw = (row.get(field) or "").strip()
        if not raw:
            continue
        # Trunca campos muito longos para economizar tokens
        if len(raw) > max_len:
            raw = raw[:max_len].rstrip() + "..."
        lines.append(f"{field}: {raw}")
    return "\n".join(lines).strip()

def _call_gemini_with_web_search(
    client: genai.Client,
    model: str,
    prompt: str,
    max_retries: int,
    verbose: bool = False
) -> str:
    """Chama a API com suporte a Google Search e tratamento de erro."""
    
    # Configuração da ferramenta de busca
    google_search_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        if verbose:
            logging.info(
                "Chamando API Gemini (modelo=%s, tentativa %d/%d)...",
                model,
                attempt + 1,
                max_retries,
            )
            logging.info("Prompt enviado (chars=%d).", len(prompt))
        call_started_at = time.monotonic()
        try:
            # CORREÇÃO: Removemos response_mime_type="application/json" pois conflita com Tools
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[google_search_tool], # Uso de ferramenta ativado
                    temperature=0.1
                )
            )
            
            if response.text:
                if verbose:
                    elapsed = time.monotonic() - call_started_at
                    logging.info(
                        "Resposta recebida em %.2fs (chars=%d).",
                        elapsed,
                        len(response.text),
                    )
                return response.text
            else:
                logging.warning("Resposta vazia (possível bloqueio de segurança).")
                return "{}"

        except Exception as exc:
            last_err = exc
            error_msg = str(exc)
            if verbose:
                elapsed = time.monotonic() - call_started_at
                logging.warning("Falha na chamada (%.2fs): %s", elapsed, exc)

            # Tratamento para Cota Excedida (Erro 429)
            if "429" in error_msg or "ResourceExhausted" in error_msg:
                wait_time = 60
                logging.warning(f"Cota atingida (429). Aguardando {wait_time}s... (Tentativa {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            
            # Outros erros
            logging.warning(f"Erro na API (tentativa {attempt+1}): {exc}")
            time.sleep(2 ** attempt)

    logging.error(f"Falha definitiva após {max_retries} tentativas.")
    return "{}"

def _extract_json(text: str) -> Dict[str, object]:
    """Limpa a resposta e converte para dicionário Python."""
    # Remove blocos de código markdown (```json ... ```)
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Tenta encontrar JSON dentro do texto se houver lixo em volta
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
            
    return {}

def _normalize_url(url: str) -> str:
    cleaned = (url or "").strip().strip(".,;)]}>\"'")
    if not cleaned: return ""
    if not re.match(r"^https?://", cleaned, re.IGNORECASE):
        cleaned = "https://" + cleaned
    return cleaned

def _domain_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."): host = host[4:]
        return host
    except Exception: return ""

def _classify_urls(urls: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    tse, tre, geral = [], [], []
    seen = set()

    for raw in urls:
        normalized = _normalize_url(raw)
        if not normalized or normalized in seen: continue
        
        domain = _domain_from_url(normalized)
        if not domain: continue

        if domain == "tse.jus.br" or domain.endswith(".tse.jus.br"):
            tse.append(normalized)
        elif bool(TRE_DOMAIN_RE.search(domain)):
            tre.append(normalized)
        elif any(domain.endswith(gd) for gd in GENERAL_DOMAINS):
            geral.append(normalized)
        else:
            if "jus.br" not in domain:
                geral.append(normalized)
                
        seen.add(normalized)
    return tse, tre, geral

def _process_response_text(text: str) -> Tuple[str, str, List[str]]:
    data = _extract_json(text)
    
    urls = []
    for key in NEWS_COLUMNS:
        val = data.get(key)
        if isinstance(val, list):
            urls.extend([str(v) for v in val if isinstance(v, str)])
            
    if not urls:
        urls = URL_RE.findall(text)

    tse_list, tre_list, geral_list = _classify_urls(urls)
    return ", ".join(tse_list), ", ".join(tre_list), geral_list

def _apply_geral_links(row: Dict[str, str], geral_links: List[str]) -> List[str]:
    for key in list(row.keys()):
        if re.match(r"^noticia_geral_\d+$", key):
            row.pop(key, None)

    row["noticia_geral"] = geral_links[0] if geral_links else ""
    new_columns: List[str] = []
    for idx, link in enumerate(geral_links[1:], start=1):
        col_name = f"noticia_geral_{idx}"
        row[col_name] = link
        new_columns.append(col_name)

    return new_columns

def _rewrite_output_file(output_path: str, fieldnames: List[str]) -> None:
    temp_path = f"{output_path}.tmp"
    with open(output_path, "r", newline="", encoding="utf-8") as src, open(
        temp_path, "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            writer.writerow(row)
    os.replace(temp_path, output_path)

def _open_output_writer(output_path: str, fieldnames: List[str], mode: str):
    f = open(output_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if mode == "w":
        writer.writeheader()
    return f, writer

def _select_input_file() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        logging.error("Falha ao abrir caixa de seleção (tkinter): %s", exc)
        return ""

    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        file_path = filedialog.askopenfilename(
            title="Selecione o arquivo CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
    finally:
        root.destroy()

    return file_path or ""

def _get_api_key_securely():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        key = os.getenv("GOOGLE_API_KEY")
    return key

def main() -> None:
    parser = argparse.ArgumentParser(description="Busca noticias TSE via Gemini API (google-genai).")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV de entrada. Se omitido, abre caixa de seleção.")
    parser.add_argument("--output", default="", help="CSV de saida.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo Gemini.")
    parser.add_argument("--limit", type=int, default=0, help="Limite de linhas.")
    parser.add_argument("--sleep", type=float, default=5.0, help="Pausa (segundos) entre chamadas.")
    parser.add_argument("--verbose", action="store_true", help="Exibe detalhes do chamamento à API.")
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    api_key = _get_api_key_securely()
    if not api_key:
        logging.error("ERRO FATAL: Chave API não encontrada.")
        logging.error("Certifique-se de que o arquivo .env existe e contem GEMINI_API_KEY=...")
        return

    logging.info(f"Iniciando com o modelo: {args.model}")

    client = genai.Client(api_key=api_key)

    input_path = (args.input or "").strip()
    if not input_path:
        logging.info("Selecione o arquivo CSV de entrada.")
        input_path = _select_input_file()
        if not input_path:
            logging.error("Nenhum arquivo selecionado.")
            return

    if not os.path.exists(input_path):
        logging.error(f"Arquivo não encontrado: {input_path}")
        return

    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(input_path)
        suffix = " - com notícias da WEB"
        output_path = f"{base}{suffix}{ext}"

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    existing_header = None
    if os.path.exists(output_path):
        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)

    output_fields = list(existing_header) if existing_header else list(fieldnames)
    header_changed = False
    for col in fieldnames:
        if col not in output_fields:
            output_fields.append(col)
            header_changed = True
    for col in NEWS_COLUMNS:
        if col not in output_fields:
            output_fields.append(col)
            header_changed = True

    # PERFORMANCE OPTIMIZATION: Pre-allocate extra columns to avoid expensive file rewrites inside the loop.
    # We anticipate up to 50 extra news links.
    for i in range(1, 51):
        col = f"noticia_geral_{i}"
        if col not in output_fields:
            output_fields.append(col)
            header_changed = True

    if existing_header and header_changed:
        _rewrite_output_file(output_path, output_fields)

    processed_count = 0
    write_mode = "w"
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) > 1:
                processed_count = len(lines) - 1
                write_mode = "a"
                logging.info(f"Retomando processamento. {processed_count} linhas já existem.")

    total_to_process = len(rows) if args.limit == 0 else min(len(rows), args.limit)

    f, writer = _open_output_writer(output_path, output_fields, write_mode)
    try:
        for idx, row in enumerate(rows):
            if idx >= total_to_process:
                break
            
            if idx < processed_count:
                continue

            logging.info(f"Processando linha {idx+1}/{total_to_process}...")
            
            context = _build_context(row)
            
            if context:
                raw_text = _call_gemini_with_web_search(
                    client=client, 
                    model=args.model, 
                    prompt=USER_PROMPT_TEMPLATE.format(context=context), 
                    max_retries=3,
                    verbose=args.verbose
                )
                noticia_tse, noticia_tre, noticia_geral_links = _process_response_text(raw_text)
            else:
                noticia_tse, noticia_tre, noticia_geral_links = "", "", []

            row["noticia_TSE"] = noticia_tse
            row["noticia_TRE"] = noticia_tre
            new_columns = _apply_geral_links(row, noticia_geral_links)

            if new_columns:
                header_updated = False
                for col in new_columns:
                    if col not in output_fields:
                        output_fields.append(col)
                        header_updated = True
                if header_updated:
                    f.flush()
                    f.close()
                    _rewrite_output_file(output_path, output_fields)
                    f, writer = _open_output_writer(output_path, output_fields, "a")
            
            writer.writerow(row)
            f.flush()
            
            time.sleep(args.sleep)
    finally:
        f.close()

    logging.info(f"Concluído! Arquivo salvo em: {output_path}")

if __name__ == "__main__":
    main()
