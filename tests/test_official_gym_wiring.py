"""End-to-end check that the official DeepResearchGym judges are driven correctly.

Runs the real upstream scripts as subprocesses against a mock judge server, so a
failure here means the harness's invocation, file layout, or score aggregation
has drifted from upstream, not that a model changed its mind.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adr.eval.deep_research_gym import run_deep_research_gym
from adr.eval.importers import trajectory_from_pair
from adr.eval.repos import find_deep_research_gym, find_key_points
from adr.eval.scoring import headline_scores
from tests.mock_judge import EXPECTED_QUALITY, EXPECTED_SUPPORT_RATE, MockJudgeServer

REPORT = """# Do virtual training methods change HR's role?

Virtual platforms shifted HR from delivering training to curating it. Adoption of
VR and responsive online interfaces broadened the range of learning styles a
single program can serve, and remote delivery removed per-site instructor cost.

## Measurement

Completion telemetry lets HR report on skill acquisition rather than attendance.

## References

[1] https://example.com/hr-virtual-training
[2] https://example.com/vr-onboarding-study
"""

gym = find_deep_research_gym()
pytestmark = pytest.mark.skipif(
    not gym.ok, reason="DeepResearchGym checkout not available; set ADR_GYM_DIR"
)


def _query_with_key_points() -> str:
    key_points = find_key_points(gym.path)
    assert key_points is not None, "expected aggregated key points in the Gym checkout"
    return sorted(key_points.glob("*_aggregated.json"))[0].stem.replace("_aggregated", "")


def test_official_gym_judges_run_and_aggregate(tmp_path: Path, monkeypatch):
    query_id = _query_with_key_points()
    key_point_file = find_key_points(gym.path) / f"{query_id}_aggregated.json"
    question = json.loads(key_point_file.read_text(encoding="utf-8"))["question"]

    traj = trajectory_from_pair(
        question=question, report=REPORT, query_id=query_id, dataset="deep_research_gym"
    )

    with MockJudgeServer() as server:
        monkeypatch.setenv("OPENAI_API_KEY", "mock-key")
        monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
        result = run_deep_research_gym(
            [traj],
            run_dir=tmp_path,
            model_name="wiring-test",
            judge_model="gpt-4.1-mini",
            run_quality=True,
            run_kpr=True,
            timeout_s=300,
        )
        assert server.request_count > 0, "judge scripts never called the model"

    assert result["official"] is True, result

    quality = result["quality"]
    assert "error" not in quality, quality.get("error")
    assert quality["n"] == 1
    assert quality["average_normalized_score"] == pytest.approx(EXPECTED_QUALITY)
    assert set(quality["per_criterion"]) == {
        "Clarity",
        "Depth",
        "Balance",
        "Breadth",
        "Support",
        "Insightfulness",
    }

    kpr = result["kpr"]
    assert "error" not in kpr, kpr.get("error")
    assert kpr["n"] == 1
    assert kpr["average_support_rate"] == pytest.approx(EXPECTED_SUPPORT_RATE)
    assert kpr["per_query"][query_id]["n_key_points"] > 0

    flat = headline_scores({"deep_research_gym": result})
    assert flat["gym_quality"] == pytest.approx(EXPECTED_QUALITY)
    assert flat["gym_average_support_rate"] == pytest.approx(EXPECTED_SUPPORT_RATE)


def test_export_layout_matches_official_expectations(tmp_path: Path, monkeypatch):
    """The judges glob for <id>.q / <id>.a, so the export must land in one folder."""
    traj = trajectory_from_pair(
        question="why is there a chip shortage",
        report=REPORT,
        query_id="923549",
        dataset="deep_research_gym",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    result = run_deep_research_gym([traj], run_dir=tmp_path, model_name="layout")

    export = Path(result["export"])
    assert (export / "923549.q").read_text(encoding="utf-8").strip() == "why is there a chip shortage"
    assert "virtual platforms" in (export / "923549.a").read_text(encoding="utf-8").lower()
    assert result["official"] is False
    assert "OPENAI_API_KEY" in result["reason"]
