"""Swappable orchestrator: decide(state) -> (u, m, w)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from adr.core.state import ResearchState
from adr.core.types import PilotDecision


@runtime_checkable
class Orchestrator(Protocol):
    name: str

    async def decide(self, state: ResearchState, ctx: Any = None) -> PilotDecision: ...
