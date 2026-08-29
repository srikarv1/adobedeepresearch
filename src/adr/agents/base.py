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
    """Anything that turns a query + tools into a report trajectory.

    Implement `run` in `deep_research.py` or `pilot.py` and register the class
    in `registry.py`. The rest of the eval stack does not care which model you
    use; it only needs a `Trajectory` with `report.article` set.
    """

    name: str

    async def run(self, task: ResearchTask, ctx: AgentContext) -> Trajectory: ...
