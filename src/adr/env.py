"""Load local secrets and expose Azure OpenAI as OpenAI-compatible env vars.

Gym judges construct ``AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)``.
They cannot take Azure-specific kwargs, so we alias the Azure resource onto those
two names and point ``OPENAI_BASE_URL`` at the Azure ``/openai/v1`` route.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def azure_v1_base_url(endpoint: str | None) -> str:
    """Microsoft's Azure OpenAI v1 route: ``https://{resource}.openai.azure.com/openai/v1/``."""
    raw = (endpoint or "").strip()
    if not raw:
        return ""
    if "/openai/v1" in raw:
        return raw if raw.endswith("/") else f"{raw}/"
    return f"{raw.rstrip('/')}/openai/v1/"


def apply_azure_openai_compat_env() -> None:
    """Copy Azure credentials into OPENAI_* when those are unset."""
    azure_key = (os.environ.get("AZURE_OPENAI_API_KEY") or "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or ""
    if azure_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = azure_key
    if endpoint and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = azure_v1_base_url(endpoint)


def load_env(dotenv_path: Path | None = None) -> None:
    """Read ``.env`` then alias Azure so ``adr`` and Gym judges share one key."""
    load_dotenv(dotenv_path or (ROOT / ".env"))
    apply_azure_openai_compat_env()
