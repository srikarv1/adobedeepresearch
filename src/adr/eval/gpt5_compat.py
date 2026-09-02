"""Strip gpt-5-incompatible kwargs from Flitternie Gym judge OpenAI calls.

Upstream eval_quality_async.py hardcodes ``temperature=0`` and ``seed=42``.
Azure gpt-5.6-sol rejects those. We do not edit the official rubrics; we wrap
the OpenAI client the judges construct.
"""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable
from typing import Any


def scrub_gpt5_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    model = str(kwargs.get("model") or "")
    if "gpt-5" not in model.lower():
        return kwargs
    out = dict(kwargs)
    out.pop("temperature", None)
    out.pop("seed", None)
    if "max_tokens" in out and "max_completion_tokens" not in out:
        out["max_completion_tokens"] = out.pop("max_tokens")
    else:
        out.pop("max_tokens", None)
    return out


def _wrap_async(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **scrub_gpt5_kwargs(kwargs))

    return wrapped


def _patch_completions(completions: Any) -> None:
    if getattr(completions, "_adr_gpt5_patched", False):
        return
    orig_create = completions.create
    completions.create = _wrap_async(orig_create)
    orig_parse = getattr(completions, "parse", None)
    if orig_parse is not None:
        completions.parse = _wrap_async(orig_parse)
    completions._adr_gpt5_patched = True


def _patch_client(client: Any) -> None:
    chat = getattr(client, "chat", None)
    if chat is not None and hasattr(chat, "completions"):
        _patch_completions(chat.completions)
    beta = getattr(client, "beta", None)
    if beta is not None:
        beta_chat = getattr(beta, "chat", None)
        if beta_chat is not None and hasattr(beta_chat, "completions"):
            _patch_completions(beta_chat.completions)


def install_gpt5_openai_compat() -> None:
    """Patch AsyncOpenAI so Gym judges can call gpt-5 deployments."""
    import openai

    if getattr(openai, "_adr_gpt5_compat", False):
        _patch_already_imported()
        return

    orig_init = openai.AsyncOpenAI.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        orig_init(self, *args, **kwargs)
        _patch_client(self)

    openai.AsyncOpenAI.__init__ = patched_init  # type: ignore[method-assign]
    openai._adr_gpt5_compat = True
    _patch_already_imported()


def _patch_already_imported() -> None:
    for name, mod in list(sys.modules.items()):
        if name.startswith("eval_") and hasattr(mod, "client"):
            _patch_client(mod.client)
