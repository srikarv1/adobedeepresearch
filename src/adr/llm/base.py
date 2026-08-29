from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from adr.core.types import TokenUsage


class Message(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    text: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """Any chat model. Implementations should be OpenAI-compatible when possible."""

    name: str
    model: str

    async def complete(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse: ...
