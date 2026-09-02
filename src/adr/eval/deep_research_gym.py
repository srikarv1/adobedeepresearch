"""DeepResearchGym evaluation via the upstream judge functions.

We dynamically import ``evaluate_folder_async`` from the upstream scripts
and call them directly, bypassing the ``__main__`` blocks which hardcode
cluster paths and accept different CLI flags than this harness needs.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

from adr.core.types import Trajectory
from adr.env import apply_azure_openai_compat_env
from adr.eval.exporters import export_deep_research_gym
from adr.eval.gpt5_compat import install_gpt5_openai_compat
from adr.eval.repos import find_deep_research_gym, find_key_points
from adr.eval.scoring import (
    aggregate_gym_citation,
    aggregate_gym_kpr,
    aggregate_gym_quality,
)


def _import_gym_module(gym_root: Path, name: str) -> Any:
    """Dynamically import a module from the Gym checkout by file path."""
    gym_root = Path(gym_root).resolve()
    root_str = str(gym_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    script = gym_root / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(script))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate_folder_call_args(
    fn: Any,
    export_dir: Path,
    judge_model: str,
    key_point_dir: str | Path | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build args for both Gym ``evaluate_folder_async`` signatures.

    Older checkouts take ``(subfolder, model, parent_dir[, key_point_dir])``.
    Current Flitternie checkouts take ``(path_to_reports, model[, key_point_dir])``.
    """
    params = list(inspect.signature(fn).parameters)
    names = set(params)
    if params and params[0] == "path_to_reports":
        args: list[Any] = [str(export_dir), judge_model]
        kwargs: dict[str, Any] = {}
        if key_point_dir is not None and "key_point_dir" in names:
            kwargs["key_point_dir"] = str(key_point_dir)
        elif key_point_dir is not None and len(params) >= 3 and params[2] != "num_workers":
            args.append(str(key_point_dir))
        return tuple(args), kwargs

    args = [export_dir.name, judge_model, str(export_dir.parent)]
    if key_point_dir is not None:
        args.append(str(key_point_dir))
    return tuple(args), {}


def _run_async(coro: Any, timeout_s: float | None = None) -> Any:
    if timeout_s is not None:
        async def _with_timeout() -> Any:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        return asyncio.run(_with_timeout())
    return asyncio.run(coro)


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

    apply_azure_openai_compat_env()
    install_gpt5_openai_compat()
    if not os.environ.get("OPENAI_API_KEY"):
        result["reason"] = (
            "OPENAI_API_KEY is not set; the Gym judges cannot run. "
            "Set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT or OPENAI_API_KEY."
        )
        return result

    out_dir = run_dir / "metrics" / "deep_research_gym"
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_quality:
        result["quality"] = _quality(gym_root, export_dir, out_dir, judge_model, timeout_s)
    if run_kpr:
        result["kpr"] = _kpr(
            gym_root, export_dir, out_dir, judge_model, key_point_dir, timeout_s
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
    try:
        mod = _import_gym_module(gym_root, "eval_quality_async")
        args, kwargs = evaluate_folder_call_args(mod.evaluate_folder_async, export_dir, judge_model)
        results = _run_async(mod.evaluate_folder_async(*args, **kwargs), timeout_s=timeout_s)
    except Exception as e:
        return {"n": 0, "error": str(e)}

    path = out_dir / f"quality_{judge_model}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**aggregate_gym_quality(results), "path": str(path)}


def _kpr(
    gym_root: Path,
    export_dir: Path,
    out_dir: Path,
    judge_model: str,
    key_point_dir: str | Path | None,
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

    try:
        mod = _import_gym_module(gym_root, "eval_kpr_async")
        args, kwargs = evaluate_folder_call_args(
            mod.evaluate_folder_async, export_dir, judge_model, key_points
        )
        results = _run_async(mod.evaluate_folder_async(*args, **kwargs), timeout_s=timeout_s)
    except Exception as e:
        return {"n": 0, "error": str(e)}

    path = out_dir / f"relevance_{judge_model}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**aggregate_gym_kpr(results), "key_point_dir": str(key_points), "path": str(path)}


def _citation(
    gym_root: Path,
    export_dir: Path,
    out_dir: Path,
    judge_model: str,
    timeout_s: float | None,
) -> dict[str, Any]:
    try:
        mod = _import_gym_module(gym_root, "eval_citation_async")
        args, kwargs = evaluate_folder_call_args(mod.evaluate_folder_async, export_dir, judge_model)
        results = _run_async(mod.evaluate_folder_async(*args, **kwargs), timeout_s=timeout_s)
    except Exception as e:
        return {"n": 0, "error": str(e)}

    path = out_dir / f"faithfullness_{judge_model}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**aggregate_gym_citation(results), "path": str(path)}
