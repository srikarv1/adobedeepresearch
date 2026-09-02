"""Intern + learned orchestrator stub. Train the policy later."""

from adr.agents.intern import InternAgent, build_learned as build

__all__ = ["LearnedAgent", "build"]


class LearnedAgent(InternAgent):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config, name="learned")
