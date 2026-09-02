"""Frozen intern + FixedOrchestrator (constant breadth/depth, top-k keep)."""

from __future__ import annotations

from adr.agents.base import AgentContext, ResearchAgent
from adr.agents.intern import run_intern
from adr.core.types import ResearchTask, Trajectory
from adr.orchestrate import build_orchestrator


class DeepResearchAgent:
    name = "deep_research"

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        orch_name = str(self.config.get("orchestrator", "fixed"))
        self.orchestrator = build_orchestrator(orch_name, self.config)

    async def run(self, task: ResearchTask, ctx: AgentContext) -> Trajectory:
        return await run_intern(
            task,
            ctx,
            max_subquestions=int(self.config.get("max_subquestions", 4)),
            top_k=int(self.config.get("top_k", 5)),
            reads_per_round=int(self.config.get("reads_per_round", 1)),
            orchestrator=self.orchestrator,
        )


def build(config: dict | None = None) -> ResearchAgent:
    return DeepResearchAgent(config)
