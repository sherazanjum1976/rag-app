"""
config.py

Centralized configuration for the RAG application.

Configuration values are resolved in this order of precedence:
    1. Streamlit Secrets (st.secrets)      -> used on Streamlit Community Cloud
    2. Environment variables (os.environ)  -> used for local development (.env)
    3. Hard-coded defaults defined below

No secrets are ever hard-coded in this file.
"""

import os

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:  # pragma: no cover - streamlit is always available at runtime
    _STREAMLIT_AVAILABLE = False

# Load a local .env file if python-dotenv is available. This is a no-op on
# Streamlit Community Cloud (there is no .env file there); secrets are
# supplied via st.secrets instead.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _get_config_value(key: str, default: str | None = None) -> str | None:
    """
    Resolve a configuration value, checking Streamlit secrets first,
    then environment variables, then falling back to a default.
    """
    if _STREAMLIT_AVAILABLE:
        try:
            # st.secrets raises if no secrets.toml exists at all in some
            # environments, so this is wrapped defensively.
            if key in st.secrets:
                return str(st.secrets[key])
        except Exception:
            pass

    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Groq / LLM configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY: str | None = _get_config_value("GROQ_API_KEY")
LLM_MODEL: str = _get_config_value("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE: float = float(_get_config_value("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(_get_config_value("LLM_MAX_TOKENS", "1024"))

# ---------------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------------
# all-MiniLM-L6-v2 is chosen as the default open-source embedding model
# because it produces 384-dimensional embeddings, runs comfortably on CPU,
# has a small (~90MB) download footprint, and offers a strong balance of
# retrieval quality and speed for general-purpose document Q&A. This makes
# it well suited to the limited memory/CPU available on Streamlit
# Community Cloud's free tier.
EMBEDDING_MODEL: str = _get_config_value(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION: int = int(_get_config_value("EMBEDDING_DIMENSION", "384"))

# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------
# Sizes are expressed in approximate tokens (estimated at ~4 characters per
# token, a common rule of thumb for English text) so they are meaningful
# regardless of the exact tokenizer used downstream.
CHUNK_SIZE: int = int(_get_config_value("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(_get_config_value("CHUNK_OVERLAP", "150"))

# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------
TOP_K: int = int(_get_config_value("TOP_K", "4"))
MIN_SIMILARITY_SCORE: float = float(_get_config_value("MIN_SIMILARITY_SCORE", "0.15"))

# ---------------------------------------------------------------------------
# Misc application configuration
# ---------------------------------------------------------------------------
MAX_PDF_SIZE_MB: int = int(_get_config_value("MAX_PDF_SIZE_MB", "50"))
APP_TITLE: str = _get_config_value("APP_TITLE", "📄 RAG Document Q&A")


def is_groq_configured() -> bool:
    """Return True if a (non-empty) Groq API key is configured."""
    return bool(GROQ_API_KEY and GROQ_API_KEY.strip())
