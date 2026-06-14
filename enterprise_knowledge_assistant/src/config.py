"""Centralised configuration.

Reads settings from Streamlit secrets first, then environment variables.
This lets the app run both locally (with ``.env``) and on Streamlit Cloud
(with the Secrets manager) without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Optional

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def _streamlit_secrets() -> Mapping[str, str]:
    """Return Streamlit secrets as a plain mapping, or an empty dict.

    Accessing ``st.secrets`` emits a UI warning when no ``secrets.toml``
    file is present, so we check the well-known paths first and skip the
    import entirely if none of them exist.
    """
    candidate_paths = [
        os.path.expanduser("~/.streamlit/secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    if not any(os.path.isfile(path) for path in candidate_paths):
        return {}

    try:
        import streamlit as st

        return {key: str(value) for key, value in st.secrets.items()}
    except Exception:
        return {}


def _read(key: str, default: Optional[str] = None) -> Optional[str]:
    secrets = _streamlit_secrets()
    if key in secrets and secrets[key]:
        return secrets[key]
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: Optional[str]
    openrouter_api_key: Optional[str]
    llm_provider: str
    gemini_model: str
    openrouter_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)


def load_settings() -> Settings:
    return Settings(
        gemini_api_key=_read("GEMINI_API_KEY"),
        openrouter_api_key=_read("OPENROUTER_API_KEY"),
        llm_provider=(_read("LLM_PROVIDER", "gemini") or "gemini").lower(),
        gemini_model=_read("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash",
        openrouter_model=_read("OPENROUTER_MODEL", "google/gemini-2.5-flash") or "google/gemini-2.5-flash",
        embedding_model=_read("EMBEDDING_MODEL", "gemini-embedding-001") or "gemini-embedding-001",
        chunk_size=int(_read("CHUNK_SIZE", "900") or 900),
        chunk_overlap=int(_read("CHUNK_OVERLAP", "150") or 150),
        top_k=int(_read("TOP_K", "5") or 5),
    )
