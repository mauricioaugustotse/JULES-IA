from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional


SECRET_FILE_CANDIDATES = {
    "OPENAI_API_KEY": ("Chave_secreta_OpenAI.txt", "CHAVE_SECRETA_API_Mauricio_local.txt", "Chave_OpenAI.txt"),
    "NOTION_API_KEY": ("Chave_Notion.txt",),
    "PERPLEXITY_API_KEY": ("Chave_secreta_Perplexity.txt",),
    "GEMINI_API_KEY": ("Chave_Gemini.txt", "Chave_Google_API.txt"),
    "GOOGLE_API_KEY": ("Chave_Google_API.txt", "Chave_Gemini.txt"),
}

SECRET_ENV_ALIASES = (
    ("NOTION_API_KEY", "NOTION_TOKEN"),
    ("PERPLEXITY_API_KEY", "PPLX_API_KEY"),
    ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
)


def _resolve_base_dir(base_dir: Optional[os.PathLike[str] | str] = None) -> Path:
    if base_dir is None:
        return Path(__file__).resolve().parent
    return Path(base_dir).resolve()


def _load_dotenv_if_available(base_dir: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(base_dir / ".env")


def _read_secret_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    return ""


def _set_aliases() -> None:
    for primary, alias in SECRET_ENV_ALIASES:
        primary_value = os.getenv(primary)
        alias_value = os.getenv(alias)

        if primary_value and not alias_value:
            os.environ[alias] = primary_value
        elif alias_value and not primary_value:
            os.environ[primary] = alias_value


def load_local_secrets(
    base_dir: Optional[os.PathLike[str] | str] = None,
    candidates: Optional[Dict[str, Iterable[str]]] = None,
) -> Dict[str, str]:
    resolved_base_dir = _resolve_base_dir(base_dir)
    candidate_map = candidates or SECRET_FILE_CANDIDATES

    _set_aliases()
    loaded_from_files: Dict[str, str] = {}

    for env_name, filenames in candidate_map.items():
        if os.getenv(env_name):
            continue

        for filename in filenames:
            secret_path = resolved_base_dir / filename
            if not secret_path.is_file():
                continue

            secret_value = _read_secret_file(secret_path)
            if not secret_value:
                continue

            os.environ[env_name] = secret_value
            loaded_from_files[env_name] = filename
            break

    _load_dotenv_if_available(resolved_base_dir)
    _set_aliases()
    return loaded_from_files


def get_secret(*env_names: str, base_dir: Optional[os.PathLike[str] | str] = None) -> str:
    load_local_secrets(base_dir=base_dir)

    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return value.strip()

    return ""
