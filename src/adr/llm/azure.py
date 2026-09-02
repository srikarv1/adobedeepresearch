"""Azure OpenAI chat client matching the Foundry deployment snippet.

``model`` is the Azure *deployment* name. gpt-5 family uses
``max_completion_tokens`` and does not take ``temperature``.
"""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncAzureOpenAI

from adr.core.types import TokenUsage
from adr.env import azure_v1_base_url
from adr.llm.base import LLMResponse, Message

DEFAULT_API_VERSION = "2024-12-01-preview"


def _gpt5_family(model: str) -> bool:
    return "gpt-5" in (model or "").lower()


class AzureOpenAILLM:
    name = "azure"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key_env: str = "AZURE_OPENAI_API_KEY",
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        key = (
            api_key
            or os.environ.get(api_key_env)
            or os.environ.get("AZURE_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        endpoint = (azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
        version = (
            api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or DEFAULT_API_VERSION
        )
        if not key:
            raise ValueError("Azure OpenAI key missing: set AZURE_OPENAI_API_KEY")
        if not endpoint:
            raise ValueError("Azure OpenAI endpoint missing: set AZURE_OPENAI_ENDPOINT")
        self.azure_endpoint = endpoint + "/"
        self.api_version = version
        self.base_url = azure_v1_base_url(endpoint)
        self._client = AsyncAzureOpenAI(
            api_key=key,
            azure_endpoint=self.azure_endpoint,
            api_version=version,
        )

    async def complete(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload = [
            {"role": m.role, "content": m.content} if isinstance(m, Message) else dict(m)
            for m in messages
        ]
        limit = self.default_max_tokens if max_tokens is None else max_tokens
        kwargs: dict[str, Any] = {"model": self.model, "messages": payload}
        if _gpt5_family(self.model):
            kwargs["max_completion_tokens"] = limit
        else:
            kwargs["max_tokens"] = limit
            kwargs["temperature"] = (
                self.default_temperature if temperature is None else temperature
            )
        if extra:
            kwargs.update(extra)
        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0].message
        text = choice.content or ""
        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )
        return LLMResponse(text=text, usage=usage, raw=response.model_dump())
