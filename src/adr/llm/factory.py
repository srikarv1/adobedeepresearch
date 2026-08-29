from __future__ import annotations

from adr.llm.base import LLMClient
from adr.llm.mock import MockLLM
from adr.llm.openai_compat import OpenAICompatLLM


def build_llm(cfg: dict) -> LLMClient:
    provider = str(cfg.get("provider", "mock")).lower()
    model = cfg.get("model", "gpt-4.1-mini")
    if provider in {"mock", "none"}:
        return MockLLM(model=model, replies=cfg.get("replies"))
    if provider in {"openai", "openai_compat", "openrouter", "vllm", "ollama"}:
        return OpenAICompatLLM(
            model=model,
            base_url=cfg.get("base_url"),
            api_key=cfg.get("api_key"),
            api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
            default_temperature=float(cfg.get("temperature", 0.0)),
            default_max_tokens=int(cfg.get("max_output_tokens", 2048)),
        )
    raise ValueError(f"Unknown LLM provider: {provider}")
