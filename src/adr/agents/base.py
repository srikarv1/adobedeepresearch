from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from adr.core.types import ResearchTask, Trajectory
from adr.llm.base import LLMClient
from adr.tools.search import SearchBackend


@dataclass
class AgentContext:
    """Dependencies the harness injects. The agent should not construct these."""

    llm: LLMClient
    search: SearchBackend
    extra: dict[str, Any]


@runtime_checkable
class ResearchAgent(Protocol):
    """Turns a query + injected tools into a report trajectory.

    Real researchers use ``InternAgent`` and swap the orchestrator. Register a
    new name in ``registry.py``. The eval stack only needs ``report.article``.
    """

    name: str

    async def run(self, task: ResearchTask, ctx: AgentContext) -> Trajectory: ...
