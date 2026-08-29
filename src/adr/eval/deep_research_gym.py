from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, create_model

from adr.core.types import Trajectory
from adr.eval.exporters import export_deep_research_gym

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_THIRD_PARTY = ROOT / "third_party" / "deepresearch_benchmarking"

# Official quality axes from cxcscmu/deepresearch_benchmarking/eval_quality_async.py
QUALITY_CRITERIA = [
    {
        "name": "Clarity",
        "description": "Assess how clearly, rigorously, and analytically distinct the answer is. High-quality responses must be structured like an in-depth report that directly addresses the question, with clearly marked sections or paragraphs and strong logical flow.",
    },
    {
        "name": "Depth",
        "description": "Assess the comprehensiveness and analytical depth of the report. Excellent reports demonstrate critical thinking, nuanced analysis, and/or synthesis of information.",
    },
    {
        "name": "Balance",
        "description": "Evaluate the fairness and objectivity of the answer. Excellent reports present multiple perspectives fairly and impartially.",
    },
    {
        "name": "Breadth",
        "description": "Evaluate how many distinct and relevant subtopics, perspectives, or contexts are covered.",
    },
    {
        "name": "Support",
        "description": "Evaluate the extent to which all key claims are substantiated by specific, identifiable, and credible evidence. Providing URLs in the report is the most basic requirement.",
    },
    {
        "name": "Insightfulness",
        "description": "Assess how insightful the answer is. Excellent reports go beyond summarizing common knowledge, offering original synthesis.",
    },
]


class KeyPointRecall(BaseModel):
    label: Literal["Supported", "Omitted", "Contradicted"]
    justification: str


def run_deep_research_gym(
    trajectories: list[Trajectory],
    *,
    run_dir: Path,
    model_name: str,
    third_party_dir: str | Path | None = None,
    key_point_dir: str | Path | None = None,
    judge_model: str = "gpt-4.1-mini",
    run_quality: bool = True,
    run_kpr: bool = True,
    run_citation: bool = False,
) -> dict[str, Any]:
    """Write Gym .q/.a files and optionally run official-style LLM judges."""
    export_dir = run_dir / "exports" / "deep_research_gym" / model_name
    export_deep_research_gym(trajectories, export_dir)
    bench_root = Path(third_party_dir) if third_party_dir else DEFAULT_THIRD_PARTY
    kp_dir = Path(key_point_dir) if key_point_dir else bench_root / "key_point"

    result: dict[str, Any] = {
        "bench": "deep_research_gym",
        "export": str(export_dir),
        "official": False,
        "reason": None,
        "quality": None,
        "kpr": None,
        "citation": None,
    }
    if not os.environ.get("OPENAI_API_KEY"):
        result["reason"] = "OPENAI_API_KEY missing; skipped Gym LLM judges"
        return result

    if run_quality:
        result["quality"] = _run_quality(export_dir, judge_model)
        result["official"] = True
    if run_kpr:
        if not kp_dir.exists():
            result["kpr"] = {"skipped": True, "reason": f"key points missing at {kp_dir}"}
        else:
            result["kpr"] = _run_kpr(export_dir, kp_dir, judge_model)
            result["official"] = True
    if run_citation:
        result["citation"] = {
            "skipped": True,
            "reason": "Citation crawl is expensive; run the official script in third_party/deepresearch_benchmarking",
        }
    result["reason"] = None if result["official"] else result["reason"]
    return result


def _client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _run_quality(report_dir: Path, model: str) -> dict[str, Any]:
    client = _client()
    possible = list(range(0, 11))
    schema = create_model("CriterionEvaluation", rating=(Literal[*possible], ...), justification=(str, ...))
    scores: dict[str, Any] = {}
    for q_path in sorted(report_dir.glob("*.q")):
        a_path = report_dir / f"{q_path.stem}.a"
        if not a_path.exists():
            continue
        question = q_path.read_text(encoding="utf-8").strip()
        answer = a_path.read_text(encoding="utf-8").strip()
        per_crit = {}
        for criterion in QUALITY_CRITERIA:
            prompt = (
                "You are a strict expert evaluator. Focus on a single criterion: "
                f"{criterion['name']}. {criterion['description']}\n\n"
                f"Question:\n{question}\n\nAnswer:\n{answer}\n\n"
                'Respond strictly in JSON: {"rating": <0-10 int>, "justification": <text>}'
            )
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
                temperature=0,
            )
            parsed = json.loads(response.choices[0].message.content)
            per_crit[criterion["name"]] = parsed
        ratings = [row["rating"] for row in per_crit.values()]
        scores[q_path.stem] = {
            "scores": per_crit,
            "normalized_score": (sum(ratings) / (len(ratings) * 10)) * 100 if ratings else 0.0,
        }
    out = report_dir / f"evaluation_results_detailed_{model}.json"
    out.write_text(json.dumps(scores, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    mean = sum(v["normalized_score"] for v in scores.values()) / len(scores) if scores else 0.0
    return {"path": str(out), "mean_normalized": round(mean, 3), "n": len(scores)}


def _run_kpr(report_dir: Path, key_point_dir: Path, model: str) -> dict[str, Any]:
    client = _client()
    labels: dict[str, Any] = {}
    for a_path in sorted(report_dir.glob("*.a")):
        kp_path = key_point_dir / f"{a_path.stem}_aggregated.json"
        if not kp_path.exists():
            continue
        answer = a_path.read_text(encoding="utf-8").strip()
        key_points = json.loads(kp_path.read_text(encoding="utf-8")).get("key_points") or []
        per_point = {}
        for point in key_points:
            prompt = (
                "Determine whether the report Supports, Omits, or Contradicts the key point.\n"
                f"Key Point: {point.get('point_content')}\nReport: {answer}\n"
                'Respond in JSON: {"label": "...", "justification": "..."}'
            )
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=KeyPointRecall,
                temperature=0,
            )
            parsed = json.loads(response.choices[0].message.content)
            per_point[str(point.get("point_number"))] = parsed
        supported = sum(1 for row in per_point.values() if row["label"] == "Supported")
        labels[a_path.stem] = {
            "labels": per_point,
            "recall": supported / len(per_point) if per_point else 0.0,
        }
    out = report_dir / f"evaluation_results_kpr_{model}.json"
    out.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    mean = sum(v["recall"] for v in labels.values()) / len(labels) if labels else 0.0
    return {"path": str(out), "mean_recall": round(mean, 4), "n": len(labels)}
