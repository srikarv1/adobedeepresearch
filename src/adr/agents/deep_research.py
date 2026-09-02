"""Intern + fixed orchestrator (constant breadth / top-k keep)."""

from adr.agents.intern import InternAgent, build_deep_research as build

__all__ = ["DeepResearchAgent", "build"]


class DeepResearchAgent(InternAgent):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config, name="deep_research")
