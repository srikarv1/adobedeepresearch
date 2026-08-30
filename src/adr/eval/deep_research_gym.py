"""DeepResearchGym evaluation via the upstream judge scripts.

We shell out to the real ``eval_quality_async.py`` / ``eval_kpr_async.py`` /
``eval_citation_async.py`` rather than reimplementing their prompts. The rubrics
carry hard scoring rules (for example, Support is zero when a report has no
source URLs) and dropping them inflates scores and makes results incomparable to
published numbers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from adr.core.types import Trajectory
from adr.eval.exporters import export_deep_research_gym
from adr.eval.procs import run_script, stage_relative_dir
from adr.eval.repos import find_deep_research_gym, find_key_points
from adr.eval.scoring import (
    aggregate_gym_citation,
    aggregate_gym_kpr,
    aggregate_gym_quality,
    load_json,
)

# Path eval_kpr_async.py expects key points at, relative to its working directory.
KPR_KEY_POINT_RELPATH = "deepresearch_benchmarking/key_point"


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
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Write Gym ``.q`` / ``.a`` files, then run the official judges on them."""
    export_dir = run_dir / "exports" / "deep_research_gym" / model_name
    export_deep_research_gym(trajectories, export_dir)

    result: dict[str, Any] = {
        "bench": "deep_research_gym",
        "export": str(export_dir),
        "judge_model": judge_model,
        "official": False,
        "reason": None,
        "quality": None,
        "kpr": None,
        "citation": None,
    }

    located = find_deep_research_gym(third_party_dir)
    if not located.ok:
        result["reason"] = located.reason
        return result
    gym_root = located.path
    result["repo"] = str(gym_root)

    if not os.environ.get("OPENAI_API_KEY"):
        result["reason"] = "OPENAI_API_KEY is not set; the Gym judges cannot run"
        return result

    out_dir = run_dir / "metrics" / "deep_research_gym"
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_quality:
        result["quality"] = _quality(gym_root, export_dir, out_dir, judge_model, timeout_s)
    if run_kpr:
        result["kpr"] = _kpr(
            gym_root, export_dir, out_dir, judge_model, key_point_dir, run_dir, timeout_s
        )
    if run_citation:
        result["citation"] = _citation(gym_root, export_dir, out_dir, judge_model, timeout_s)

    ran = [
        block
        for block in (result["quality"], result["kpr"], result["citation"])
        if isinstance(block, dict) and block.get("n")
    ]
    result["official"] = bool(ran)
    if not result["official"]:
        result["reason"] = "No Gym metric produced scores; see the per-metric log fields"
    return result


def _quality(
    gym_root: Path,
    export_dir: Path,
    out_dir: Path,
    judge_model: str,
    timeout_s: float | None,
) -> dict[str, Any]:
    log = run_script(
        [
            gym_root / "eval_quality_async.py",
            "--dir",
            export_dir.resolve(),
            "--output",
            out_dir.resolve(),
            "--open_ai_model",
            judge_model,
        ],
        cwd=gym_root,
        timeout_s=timeout_s,
    )
    path = out_dir / f"quality_{judge_model}.json"
    scores = aggregate_gym_quality(load_json(path))
    return {**scores, "path": str(path), "log": log}


def _kpr(
    gym_root: Path,
    export_dir: Path,
    out_dir: Path,
    judge_model: str,
    key_point_dir: str | Path | None,
    run_dir: Path,
    timeout_s: float | None,
) -> dict[str, Any]:
    key_points = find_key_points(gym_root, key_point_dir)
    if key_points is None:
        return {
            "n": 0,
            "skipped": True,
            "reason": (
                "No aggregated key points found. Expected *_aggregated.json under "
                f"{gym_root / 'key_point'} or set eval.deep_research_gym.key_point_dir"
            ),
        }

    # eval_kpr_async.py hardcodes a working-directory-relative key point path,
    # so give it a working directory where that path resolves.
    staging = run_dir / ".gym_kpr_cwd"
    staging.mkdir(parents=True, exist_ok=True)
    stage_relative_dir(staging, KPR_KEY_POINT_RELPATH, key_points)

    log = run_script(
        [
            gym_root / "eval_kpr_async.py",
            "--dir",
            export_dir.resolve(),
            "--output",
            out_dir.resolve(),
            "--open_ai_model",
            judge_model,
        ],
        cwd=staging,
        timeout_s=timeout_s,
    )
    path = out_dir / f"relevance_{judge_model}.json"
    scores = aggregate_gym_kpr(load_json(path))
    return {**scores, "key_point_dir": str(key_points), "path": str(path), "log": log}


def _citation(
    gym_root: Path,
    export_dir: Path,
    out_dir: Path,
    judge_model: str,
    timeout_s: float | None,
) -> dict[str, Any]:
    log = run_script(
        [
            gym_root / "eval_citation_async.py",
            "--dir",
            export_dir,
            "--output",
            out_dir,
            "--open_ai_model",
            judge_model,
        ],
        cwd=gym_root,
        timeout_s=timeout_s,
    )
    path = out_dir / f"faithfullness_{judge_model}.json"
    scores = aggregate_gym_citation(load_json(path))
    return {**scores, "path": str(path), "log": log}
