"""Eight PILOT-style features per evidence item (paper eq. 6 stand-in)."""

from __future__ import annotations

from adr.core.state import ResearchState
from adr.core.types import Evidence
from adr.features.encoder import EmbeddingCache, cosine

_SOURCE = {
    "gym": 0.8,
    "tavily": 0.7,
    "mock": 0.4,
    "unknown": 0.3,
}


def evidence_features(
    state: ResearchState,
    ev: Evidence,
    *,
    encoder: EmbeddingCache | None = None,
) -> dict[str, float]:
    """Return ρ, ν, δ, κ, ℓ, d, α, σ for one evidence item."""
    enc = encoder or EmbeddingCache()
    query_vec = enc.embed(state.query.text)
    body = f"{ev.title} {ev.body(800)}"
    vec = enc.embed(body)
    relevance = cosine(query_vec, vec)

    kept = [item for item in state.retained() if item.id != ev.id]
    novelty = 1.0
    redundancy = 0.0
    if kept:
        sims = [cosine(vec, enc.embed(f"{item.title} {item.body(400)}")) for item in kept]
        nearest = max(sims) if sims else 0.0
        novelty = max(0.0, 1.0 - nearest)
        redundancy = sum(1 for s in sims if s > 0.85) / max(1, len(sims))

    goal = ""
    depth = 0.0
    if ev.subtask_id and ev.subtask_id in state.subtasks:
        st = state.subtasks[ev.subtask_id]
        goal = st.goal
        depth = 1.0 if st.parent_id else 0.5
    coverage = cosine(enc.embed(goal or state.query.text), vec)

    length = min(1.0, len(ev.body()) / 4000)
    age = max(0, state.budget.used_steps - ev.added_step) / max(1, state.budget.max_steps)
    source = _SOURCE.get(ev.source_backend, 0.3)
    return {
        "rho": round(relevance, 4),
        "nu": round(novelty, 4),
        "delta": round(redundancy, 4),
        "kappa": round(coverage, 4),
        "ell": round(length, 4),
        "d": round(depth, 4),
        "alpha": round(age, 4),
        "sigma": round(source, 4),
    }


def pool_features(state: ResearchState, *, encoder: EmbeddingCache | None = None) -> dict:
    enc = encoder or EmbeddingCache()
    rows = []
    for ev in state.evidence.values():
        rows.append({"id": ev.id, "retained": ev.retained, **evidence_features(state, ev, encoder=enc)})
    return {
        "query_id": state.query.id,
        "n_evidence": len(state.evidence),
        "n_retained": len(state.retained()),
        "items": rows,
        "compact": state.compact_stats(),
    }
