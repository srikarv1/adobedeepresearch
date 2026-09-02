"""Apply PILOT action (u, m, w) onto ResearchState."""

from __future__ import annotations

from adr.core.state import ResearchState
from adr.core.types import ActionType, OrchestratorAction, PilotDecision


def apply_decision(state: ResearchState, decision: PilotDecision) -> list[str]:
    """Keep-mask first, then branch weights, then optional stop. Returns dropped ids."""
    dropped: list[str] = []
    keep_ids = {eid.lower() for eid in decision.m}
    if keep_ids:
        drop = [ev.id for ev in state.retained() if ev.id.lower() not in keep_ids]
        if drop:
            dropped = state.prune(evidence_ids=drop, drop_duplicates=False)
    for sid, boost in decision.w.items():
        if sid in state.subtasks:
            state.allocate(sid, boost=float(boost))
    if decision.u:
        state.terminated = True
        state.termination_reason = state.termination_reason or "orchestrator terminate"
    return dropped


def apply_action(state: ResearchState, action: PilotDecision | OrchestratorAction) -> list[str]:
    """Accept paper decisions or older trajectory actions."""
    if isinstance(action, PilotDecision):
        return apply_decision(state, action)
    return apply_decision(
        state,
        PilotDecision(
            u=bool(action.terminate or action.type is ActionType.TERMINATE),
            m=list(action.evidence_ids or []),
            w=dict(action.weights or {}),
            rationale=action.rationale,
        ),
    )
