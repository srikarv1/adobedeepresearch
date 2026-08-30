import json

from adr.core.types import Query, Report, Trajectory
from adr.eval.exporters import export_deep_research_bench, export_deep_research_gym


def _traj(qid: str, text: str, article: str, dataset: str) -> Trajectory:
    return Trajectory(
        query=Query(id=qid, text=text, dataset=dataset, language="en"),
        report=Report(article=article, citations=["https://example.com"]),
    )


def test_drb_jsonl_matches_official_schema(tmp_path):
    dest = tmp_path / "model.jsonl"
    export_deep_research_bench(
        [_traj("51", "What are the investment philosophies of Duan Yongping, Warren Buffett, and Charlie Munger? ", "article body", "deep_research_bench")],
        dest,
    )
    row = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
    assert row == {
        "id": 51,
        "prompt": "What are the investment philosophies of Duan Yongping, Warren Buffett, and Charlie Munger? ",
        "article": "article body",
    }


def test_gym_qa_pair_files(tmp_path):
    dest = tmp_path / "gym"
    export_deep_research_gym(
        [_traj("923549", "why is there a chip shortage", "long report", "deep_research_gym")],
        dest,
    )
    assert (dest / "923549.q").read_text(encoding="utf-8").strip() == "why is there a chip shortage"
    assert (dest / "923549.a").read_text(encoding="utf-8").strip() == "long report"
