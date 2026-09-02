"""Intern + prompted orchestrator (thin keep / allocate / stop baseline)."""

from adr.agents.intern import InternAgent, build_pilot as build

__all__ = ["PilotAgent", "build"]


class PilotAgent(InternAgent):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config, name="pilot")
