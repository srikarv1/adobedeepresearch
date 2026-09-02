"""Stub for a small trained policy. Interface only — do not train here."""

from __future__ import annotations

from pathlib import Path

from adr.core.state import ResearchState
from adr.core.types import PilotDecision


class LearnedOrchestrator:
    name = "learned"

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.policy_path = cfg.get("policy_path")
        if self.policy_path and not Path(self.policy_path).exists():
            raise FileNotFoundError(
                f"LearnedOrchestrator policy not found: {self.policy_path}. "
                "Train later; until then use FixedOrchestrator or PromptedOrchestrator."
            )

    async def decide(self, state: ResearchState, ctx=None) -> PilotDecision:
        return PilotDecision(
            u=False,
            m=[ev.id for ev in state.retained()],
            w={},
            rationale="learned stub: identity keep-mask until a policy is loaded",
        )
