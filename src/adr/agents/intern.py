"""GPTR-shaped intern: plan, search, synthesize. Orchestrator is injected.

Inspired by gpt-researcher's planner / crawler / publisher loop. Every tool
call goes through ``ctx.llm`` and ``ctx.search`` so meters stay honest.
The intern does not prune; ``Orchestrator.decide`` does that after retrieve.
"""

from __future__ import annotations

import re
from adr.agents.base import AgentContext
from adr.core.instrument import BudgetExceeded
from adr.core.state import ResearchState
from adr.core.types import (
    ActionType,
    OrchestratorAction,
    Report,
    ResearchTask,
    TokenUsage,
    Trajectory,
)
from adr.features.extract import pool_features
from adr.llm.base import LLMResponse
from adr.logging.jsonl import append_jsonl
from adr.orchestrate.apply import apply_action
from adr.orchestrate.base import Orchestrator

_LINE = re.compile(r"^\s*(?:[-*]|\d+[.)])?\s*(.+?)\s*$")


def parse_subquestions(text: str, fallback: str, *, max_n: int = 4) -> list[str]:
    goals: list[str] = []
    seen: set[str] = set()
    for raw in (text or "").splitlines():
        match = _LINE.match(raw)
        if not match:
            continue
        goal = match.group(1).strip().strip("\"'")
        if len(goal) < 8 or goal.lower() in {"ok", "yes", "no"}:
            continue
        key = goal.lower()
        if key in seen:
            continue
        seen.add(key)
        goals.append(goal)
        if len(goals) >= max_n:
            break
    return goals or [fallback]


async def _complete(
    ctx: AgentContext,
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
) -> LLMResponse:
    return await ctx.llm.complete(messages, max_tokens=max_tokens)


async def decompose(
    ctx: AgentContext,
    question: str,
    *,
    max_n: int = 4,
) -> tuple[list[str], TokenUsage]:
    reply = await _complete(
        ctx,
        [
            {
                "role": "system",
                "content": (
                    "You plan deep research. Split the user question into "
                    f"{max_n} distinct subquestions that together cover it. "
                    "Return one subquestion per line and nothing else."
                ),
            },
            {"role": "user", "content": question},
        ],
        max_tokens=400,
    )
    return parse_subquestions(reply.text, question, max_n=max_n), reply.usage


def _evidence_block(state: ResearchState, *, max_chars: int = 700) -> str:
    lines: list[str] = []
    for i, ev in enumerate(state.retained(), start=1):
        title = ev.title or ev.url or ev.id
        lines.append(f"[{i}] {title}\nURL: {ev.url}\n{ev.body(max_chars)}")
    return "\n\n".join(lines) if lines else "(no retained evidence)"


async def synthesize(ctx: AgentContext, state: ResearchState) -> tuple[str, TokenUsage]:
    reply = await _complete(
        ctx,
        [
            {
                "role": "system",
                "content": (
                    "Write a structured research report that answers the question. "
                    "Cite sources inline as [1], [2] matching the evidence list. "
                    "Every factual claim needs a citation. End with a References "
                    "section listing those same numbered URLs. "
                    "If the evidence is thin, still write a sourced report from it."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {state.query.text}\n\nEvidence:\n{_evidence_block(state)}",
            },
        ],
    )
    article = reply.text.strip()
    if "http" not in article:
        article = _fallback_report(state, article)
    return article, reply.usage


def _fallback_report(state: ResearchState, synthesis: str) -> str:
    lines = ["# Research report", "", f"Question: {state.query.text}", ""]
    if synthesis:
        lines += [synthesis, ""]
    for i, ev in enumerate(state.retained(), start=1):
        lines += [f"## {ev.title or f'Source {i}'}", "", f"{ev.body(500)} [{i}]", ""]
    lines += ["## References", ""]
    for i, ev in enumerate(state.retained(), start=1):
        lines.append(f"[{i}] {ev.url}")
    return "\n".join(lines).strip() + "\n"


async def execute_branch(
    ctx: AgentContext,
    state: ResearchState,
    *,
    top_k: int,
    reads_per_round: int,
) -> list:
    branch = state.pick_branch()
    query = branch.goal if branch else state.query.text
    hits = await ctx.search.search(query, k=top_k)
    added = state.add_evidence(
        [
            hit.to_evidence(
                query=query,
                backend=ctx.search.name,
                subtask_id=branch.id if branch else None,
            )
            for hit in hits
        ]
    )
    if branch:
        branch.searches += 1
        state.mark_active(branch.id)

    reads = 0
    for ev in added:
        if reads >= reads_per_round:
            break
        if ev.text:
            continue
        if state.budget.remaining_reads() <= 0:
            break
        text = await ctx.search.fetch(ev.url)
        if text:
            state.attach_text(ev.id, text)
            reads += 1

    state.record_step(
        OrchestratorAction(
            type=ActionType.SEARCH,
            subtask_id=branch.id if branch else None,
            query=query,
            evidence_ids=[ev.id for ev in added],
            rationale="intern search",
        ),
        observation=f"retrieved {len(added)} passages" + (f", read {reads}" if reads else ""),
    )
    if branch and len(branch.evidence_ids) >= 2:
        state.mark_done(branch.id)
    return added


def _log_decision(
    ctx: AgentContext,
    state: ResearchState,
    action: OrchestratorAction,
    dropped: list[str],
    orch_name: str,
) -> None:
    path = ctx.extra.get("decision_log")
    if not path:
        return
    features = pool_features(state)
    append_jsonl(
        path,
        {
            "kind": "decision",
            "query_id": state.query.id,
            "orchestrator": orch_name,
            "action": action.type.value,
            "keep_ids": action.evidence_ids,
            "weights": action.weights,
            "terminate": action.terminate,
            "dropped": dropped,
            "features": features,
            "budget": {
                "remaining_tokens": state.budget.remaining_tokens(),
                "remaining_steps": state.budget.remaining_steps(),
                "remaining_latency_s": state.budget.remaining_latency_s(),
                "used_tokens": state.budget.used_tokens,
            },
            "rationale": action.rationale[:400],
        },
    )


async def run_intern(
    task: ResearchTask,
    ctx: AgentContext,
    *,
    max_subquestions: int = 4,
    top_k: int = 5,
    reads_per_round: int = 1,
    orchestrator: Orchestrator | None = None,
) -> Trajectory:
    """Algorithm 1. ``orchestrator`` is PILOT (fixed / prompted / learned)."""
    state = ResearchState(task.query, task.budget)
    try:
        goals, usage = await decompose(ctx, task.query.text, max_n=max_subquestions)
        created = state.add_subtasks(goals)
        state.record_step(
            OrchestratorAction(
                type=ActionType.PLAN,
                plan=[st.goal for st in created],
                rationale="decompose query",
            ),
            observation=f"planned {len(created)} subtasks",
            tokens=usage,
        )

        while not state.should_stop():
            await execute_branch(ctx, state, top_k=top_k, reads_per_round=reads_per_round)
            if orchestrator is not None:
                action = await orchestrator.decide(state, ctx)
                dropped = apply_action(state, action)
                _log_decision(ctx, state, action, dropped, orchestrator.name)
                state.record_step(
                    OrchestratorAction(
                        type=action.type,
                        evidence_ids=action.evidence_ids,
                        weights=action.weights,
                        terminate=action.terminate,
                        rationale=action.rationale,
                    ),
                    observation=(
                        f"orch={orchestrator.name} keep={len(action.evidence_ids)} "
                        f"dropped={len(dropped)} terminate={action.terminate}"
                    ),
                )
                if action.terminate or action.type is ActionType.TERMINATE:
                    break
            if not state.open_subtasks() and state.retained():
                break

        article, usage = await synthesize(ctx, state)
        state.report = Report(article=article, citations=state.citation_urls())
        state.record_step(
            OrchestratorAction(
                type=ActionType.WRITE,
                report_draft=article,
                rationale="synthesize retained evidence",
            ),
            observation="wrote report from retained evidence",
            tokens=usage,
        )
        if not state.terminated:
            state.record_step(
                OrchestratorAction(type=ActionType.TERMINATE, rationale="intern done"),
                observation="stop",
            )
    except BudgetExceeded as exc:
        if not state.report and state.retained():
            article, usage = await synthesize(ctx, state)
            state.report = Report(article=article, citations=state.citation_urls())
            state.record_step(
                OrchestratorAction(
                    type=ActionType.WRITE,
                    report_draft=article,
                    rationale="budget exceeded; synthesize what we have",
                ),
                tokens=usage,
            )
        state.terminated = True
        state.termination_reason = str(exc)
    return state.trajectory()
