"""Constant breadth/depth and simple top-k retention. Baseline only."""

from __future__ import annotations

from adr.core.state import ResearchState
from adr.core.types import ActionType, OrchestratorAction


class FixedOrchestrator:
    name = "fixed"

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.max_keep = int(cfg.get("max_keep", 8))
        self.min_score = float(cfg.get("min_score", 0.0))

    async def decide(self, state: ResearchState, ctx=None) -> OrchestratorAction:
        ranked = sorted(state.retained(), key=lambda ev: (-ev.score, ev.added_step))
        if self.min_score:
            ranked = [ev for ev in ranked if ev.score >= self.min_score] or ranked
        keep = ranked[: self.max_keep]
        weights = {}
        for st in state.open_subtasks():
            if len(st.evidence_ids) < 2:
                weights[st.id] = 1.0
        terminate = (
            not state.open_subtasks()
            or state.budget.remaining_steps() <= 1
            or state.budget.remaining_tokens() <= 256
        )
        return OrchestratorAction(
            type=ActionType.TERMINATE if terminate else ActionType.PRUNE,
            evidence_ids=[ev.id for ev in keep],
            weights=weights,
            terminate=terminate,
            rationale=f"fixed top-{self.max_keep} keep={len(keep)}",
        )
