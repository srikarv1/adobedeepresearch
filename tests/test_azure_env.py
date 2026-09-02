from __future__ import annotations

import os

import pytest

from adr.env import apply_azure_openai_compat_env, azure_v1_base_url
from adr.llm.factory import build_llm


def test_azure_v1_base_url_appends_openai_v1() -> None:
    assert azure_v1_base_url("https://example.openai.azure.com") == (
        "https://example.openai.azure.com/openai/v1/"
    )


def test_azure_env_aliases_openai_compat_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    apply_azure_openai_compat_env()
    assert os.environ["OPENAI_API_KEY"] == "azure-secret"
    assert os.environ["OPENAI_BASE_URL"].rstrip("/") == (
        "https://example.openai.azure.com/openai/v1"
    )


def test_factory_builds_azure_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    client = build_llm({"provider": "azure", "model": "gpt-5.6-sol"})
    assert client.name == "azure"
    assert client.model == "gpt-5.6-sol"
    assert client.azure_endpoint == "https://example.openai.azure.com/"
