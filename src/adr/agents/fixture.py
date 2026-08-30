"""Minimal agent used only to exercise the harness. Not a research baseline."""

from __future__ import annotations

from adr.agents.base import AgentContext, ResearchAgent
from adr.core.state import ResearchState
from adr.core.types import ActionType, OrchestratorAction, Report, ResearchTask, Trajectory


class FixtureAgent:
    name = "fixture"

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    async def run(self, task: ResearchTask, ctx: AgentContext) -> Trajectory:
        state = ResearchState(task.query, task.budget)
        state.add_subtasks([task.query.text])
        branch = state.pick_branch()
        query = self.config.get("search_query") or task.query.text
        hits = await ctx.search.search(query, k=int(self.config.get("max_passages", 4)))
        added = state.add_evidence(
            [hit.to_evidence(query=query, backend=ctx.search.name, subtask_id=branch.id if branch else None) for hit in hits]
        )
        if branch:
            branch.searches += 1
        state.record_step(
            OrchestratorAction(type=ActionType.SEARCH, query=query, rationale="fixture search"),
            observation=f"retrieved {len(added)} passages",
        )
        # One model call so a smoke run exercises the cost instrumentation too.
        synthesis = await ctx.llm.complete(
            [
                {"role": "system", "content": "Summarize the evidence in one sentence."},
                {
                    "role": "user",
                    "content": f"Question: {task.query.text}\n\n"
                    + "\n".join(ev.body(300) for ev in state.retained()),
                },
            ]
        )
        article = _render_report(task.query.text, state, synthesis.text)
        state.report = Report(article=article, citations=state.citation_urls())
        state.record_step(
            OrchestratorAction(type=ActionType.WRITE, report_draft=article, rationale="fixture write"),
            observation="wrote report from retrieved passages",
            tokens=synthesis.usage,
        )
        state.record_step(
            OrchestratorAction(type=ActionType.TERMINATE, rationale="fixture done"),
            observation="stop",
        )
        return state.trajectory()


def _render_report(question: str, state: ResearchState, synthesis: str = "") -> str:
    lines = [
        f"# Research report",
        "",
        f"Question: {question}",
        "",
        "This fixture report exists so the evaluation harness can be tested without a real agent.",
        "",
    ]
    if synthesis.strip():
        lines += [synthesis.strip(), ""]
    for i, ev in enumerate(state.retained(), start=1):
        body = ev.body(400)
        lines.append(f"## {ev.title or f'Source {i}'}")
        lines.append("")
        lines.append(f"{body} [{i}]")
        lines.append("")
    lines.append("## References")
    lines.append("")
    for i, ev in enumerate(state.retained(), start=1):
        lines.append(f"[{i}] {ev.url}")
    return "\n".join(lines).strip() + "\n"


def build(config: dict | None = None) -> ResearchAgent:
    return FixtureAgent(config)
