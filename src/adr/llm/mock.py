from __future__ import annotations

from typing import Any

from adr.core.types import TokenUsage
from adr.llm.base import LLMResponse, Message


class MockLLM:
    """Deterministic stand-in so the harness can run without a model."""

    name = "mock"

    def __init__(self, model: str = "mock", replies: list[str] | None = None) -> None:
        self.model = model
        self.replies = list(replies or ["OK"])
        self.calls: list[list[Message]] = []
        self._i = 0

    async def complete(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse:
        normalized = [
            m if isinstance(m, Message) else Message(role=m["role"], content=m["content"])
            for m in messages
        ]
        self.calls.append(normalized)
        text = self.replies[min(self._i, len(self.replies) - 1)]
        self._i += 1
        prompt_tokens = sum(len(m.content.split()) for m in normalized)
        completion_tokens = len(text.split())
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
