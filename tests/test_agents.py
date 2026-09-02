from __future__ import annotations

import pytest

from adr.agents.base import AgentContext
from adr.agents.deep_research import DeepResearchAgent
from adr.agents.intern import parse_subquestions
from adr.agents.learned import LearnedAgent
from adr.agents.pilot import PilotAgent
from adr.core.types import Query, ResearchTask
from adr.features.extract import evidence_features
from adr.llm.mock import MockLLM
from adr.orchestrate.fixed import FixedOrchestrator
from adr.orchestrate.prompted import parse_orchestration
from adr.tools.search import MockSearch


def test_parse_subquestions_keeps_distinct_lines():
    text = "- What caused the chip shortage?\n- How did automakers respond?\n- ok\n"
    assert parse_subquestions(text, "fallback", max_n=4)[0].startswith("What caused")


@pytest.mark.asyncio
async def test_deep_research_agent_writes_cited_report():
    agent = DeepResearchAgent(
        {"max_subquestions": 2, "top_k": 3, "reads_per_round": 0, "max_keep": 4}
    )
    task = ResearchTask(
        query=Query(id="923549", text="why is there a chip shortage", dataset="deep_research_gym")
    )
    ctx = AgentContext(llm=MockLLM(), search=MockSearch(), extra={})
    traj = await agent.run(task, ctx)
    assert traj.report and traj.report.article
    assert "http" in traj.report.article
    assert any(step.action.type.value == "search" for step in traj.steps)
    assert any(step.action.type.value in {"prune", "terminate"} for step in traj.steps)


@pytest.mark.asyncio
async def test_pilot_and_learned_agents_run():
    task = ResearchTask(
        query=Query(id="x", text="why is there a chip shortage", dataset="deep_research_gym")
    )
    ctx = AgentContext(llm=MockLLM(), search=MockSearch(), extra={})
    pilot = await PilotAgent({"max_subquestions": 2, "top_k": 2}).run(task, ctx)
    learned = await LearnedAgent({"max_subquestions": 2, "top_k": 2}).run(task, ctx)
    assert pilot.report and learned.report


def test_parse_orchestration_reads_keep_and_alloc():
    parsed = parse_orchestration(
        "DECISION: CONTINUE\nKEEP: ev_aaaaaaaaaaaa, ev_bbbbbbbbbbbb\nALLOC: st_1_deadbeef=2.0\n"
    )
    assert parsed["terminate"] is False
    assert parsed["keep_ids"] == ["ev_aaaaaaaaaaaa", "ev_bbbbbbbbbbbb"]
    assert parsed["weights"]["st_1_deadbeef"] == 2.0


@pytest.mark.asyncio
async def test_fixed_orchestrator_caps_keep():
    from adr.core.state import ResearchState
    from adr.core.types import Evidence

    state = ResearchState(Query(id="q", text="chip shortage", dataset="t"))
    state.add_subtasks(["why chips"])
    state.add_evidence(
        [
            Evidence(
                id=f"ev_{i:012x}",
                url=f"https://e/{i}",
                snippet="s" * 20,
                score=1.0 - i * 0.1,
            )
            for i in range(6)
        ]
    )
    action = await FixedOrchestrator({"max_keep": 2}).decide(state)
    assert len(action.evidence_ids) == 2


def test_evidence_features_have_eight_keys():
    from adr.core.state import ResearchState
    from adr.core.types import Evidence, Query

    state = ResearchState(Query(id="q", text="chip shortage causes", dataset="t"))
    ev = Evidence(
        id="ev_1", url="https://e/1", title="chips", snippet="foundry capacity", score=0.8
    )
    state.add_evidence([ev])
    feats = evidence_features(state, ev)
    assert set(feats) == {"rho", "nu", "delta", "kappa", "ell", "d", "alpha", "sigma"}
