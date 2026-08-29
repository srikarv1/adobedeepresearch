"""Your deep research agent goes here.

The evaluation harness only needs `async def run(...)` to return a Trajectory
whose `report.article` is the markdown/text report (with citations if you have
them). Use `ctx.llm` and `ctx.search` — do not hard-code a model vendor.
"""

from __future__ import annotations

from adr.agents.base import AgentContext, ResearchAgent
from adr.core.types import ResearchTask, Trajectory


class DeepResearchAgent:
    name = "deep_research"

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    async def run(self, task: ResearchTask, ctx: AgentContext) -> Trajectory:
        raise NotImplementedError(
            "Implement DeepResearchAgent.run in src/adr/agents/deep_research.py"
        )


def build(config: dict | None = None) -> ResearchAgent:
    return DeepResearchAgent(config)
