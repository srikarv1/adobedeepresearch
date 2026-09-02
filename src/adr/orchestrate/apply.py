"""Apply (u, m, w) onto ResearchState."""

from __future__ import annotations

from adr.core.state import ResearchState
from adr.core.types import ActionType, OrchestratorAction


def apply_action(state: ResearchState, action: OrchestratorAction) -> list[str]:
    """Apply keep-mask and branch weights. Returns dropped evidence ids."""
    dropped: list[str] = []
    keep_ids = {eid.lower() for eid in (action.evidence_ids or [])}
    if action.type is ActionType.PRUNE and keep_ids:
        drop = [ev.id for ev in state.retained() if ev.id.lower() not in keep_ids]
        if drop:
            dropped = state.prune(evidence_ids=drop, drop_duplicates=False)
    for sid, boost in (action.weights or {}).items():
        if sid in state.subtasks:
            state.allocate(sid, boost=float(boost))
    return dropped
