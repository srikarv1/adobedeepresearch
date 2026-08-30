"""PILOT orchestrator goes here.

Suggested inputs (already implemented in `adr.core.state.ResearchState`):
  - evidence pool with retain / prune / superseded
  - open subtask frontier
  - remaining budget
  - compact_stats() for a small observation instead of raw passages

Suggested outputs: plan / search / read / prune / allocate / write / terminate.
"""

from __future__ import annotations

from adr.agents.base import AgentContext, ResearchAgent
from adr.core.types import ResearchTask, Trajectory


class PilotAgent:
    name = "pilot"

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    async def run(self, task: ResearchTask, ctx: AgentContext) -> Trajectory:
        raise NotImplementedError("Implement PilotAgent.run in src/adr/agents/pilot.py")


def build(config: dict | None = None) -> ResearchAgent:
    return PilotAgent(config)
