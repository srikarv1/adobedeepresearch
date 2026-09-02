"""Thin FlashResearch-style prompted keep / allocate / terminate."""

from __future__ import annotations

import json
import re

from adr.agents.base import AgentContext
from adr.core.state import ResearchState
from adr.core.types import ActionType, OrchestratorAction

_DECISION = re.compile(r"^\s*DECISION\s*:\s*(CONTINUE|TERMINATE)\s*$", re.I | re.M)
_KEEP = re.compile(r"^\s*KEEP\s*:\s*(.+)$", re.I | re.M)
_ALLOC = re.compile(r"^\s*ALLOC\s*:\s*(.+)$", re.I | re.M)
_EV_ID = re.compile(r"ev_[a-f0-9]+", re.I)
_ALLOC_PAIR = re.compile(r"(st_[^\s=,:]+)\s*[=:]\s*([0-9]*\.?[0-9]+)", re.I)

ORCHESTRATOR_SYSTEM = """You orchestrate a deep research intern under a token budget.
You see compact evidence features and frontier stats, not full pages.
Reply with exactly this format and nothing else:

DECISION: CONTINUE
KEEP: ev_aaa, ev_bbb
ALLOC: st_1_xxx=2.0, st_2_yyy=1.0

Rules:
- DECISION is CONTINUE or TERMINATE.
- KEEP is the evidence ids to retain. Drop near-duplicates and off-topic items.
- ALLOC boosts unfinished branches that still have a coverage gap. Use the ids shown.
- TERMINATE when extra search is unlikely to help or the budget is nearly gone.
- If you TERMINATE, still list KEEP for the final retained set.
"""


def parse_orchestration(text: str) -> dict:
    decision_match = _DECISION.search(text or "")
    keep_match = _KEEP.search(text or "")
    alloc_match = _ALLOC.search(text or "")
    keep_ids = _EV_ID.findall(keep_match.group(1)) if keep_match else []
    weights: dict[str, float] = {}
    if alloc_match:
        for sid, value in _ALLOC_PAIR.findall(alloc_match.group(1)):
            try:
                weights[sid] = float(value)
            except ValueError:
                continue
    return {
        "terminate": bool(decision_match and decision_match.group(1).upper() == "TERMINATE"),
        "keep_ids": [eid.lower() for eid in keep_ids],
        "weights": weights,
        "raw": (text or "").strip(),
    }


def scorecard_text(state: ResearchState, *, snippet_chars: int = 280) -> str:
    stats = state.compact_stats()
    lines = [
        f"QUESTION: {state.query.text}",
        f"STATS: {json.dumps({k: stats[k] for k in stats if k != 'branches'})}",
        "BRANCHES:",
    ]
    for row in stats.get("branches") or []:
        lines.append(
            f"  {row['id']} status={row['status']} pri={row['priority']} "
            f"searches={row['searches']} retained={row['n_retained']} | {row['goal']}"
        )
    lines.append("EVIDENCE:")
    for ev in state.evidence.values():
        flag = "KEEP" if ev.retained and not ev.superseded_by else "DROP"
        snippet = (ev.snippet or ev.body(snippet_chars)).replace("\n", " ")[:snippet_chars]
        lines.append(
            f"  {ev.id} {flag} score={ev.score:.2f} chars={len(ev.body())} "
            f"url={ev.url} | {snippet}"
        )
    return "\n".join(lines)


class PromptedOrchestrator:
    name = "prompted"

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.snippet_chars = int(self.config.get("snippet_chars", 280))

    async def decide(self, state: ResearchState, ctx: AgentContext | None = None) -> OrchestratorAction:
        if ctx is None:
            raise RuntimeError("PromptedOrchestrator.decide needs AgentContext")
        if not state.evidence:
            return OrchestratorAction(type=ActionType.PRUNE, rationale="empty pool")
        reply = await ctx.llm.complete(
            [
                {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                {"role": "user", "content": scorecard_text(state, snippet_chars=self.snippet_chars)},
            ],
            max_tokens=256,
        )
        parsed = parse_orchestration(reply.text)
        terminate = parsed["terminate"]
        return OrchestratorAction(
            type=ActionType.TERMINATE if terminate else ActionType.PRUNE,
            evidence_ids=parsed["keep_ids"],
            weights=parsed["weights"],
            terminate=terminate,
            rationale=parsed["raw"][:400] or "prompted",
        )
