from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from adr.core.types import Trajectory

_URL = re.compile(r"https?://[^\s\]\)>]+")

# Averaged across queries and diffed by `adr compare`.
_MEAN_KEYS = [
    "tokens",
    "prompt_tokens",
    "completion_tokens",
    "wall_s",
    "step_latency_s",
    "n_steps",
    "n_searches",
    "n_reads",
    "n_llm_calls",
    "n_retained",
    "n_pruned",
    "prune_rate",
    "n_citations",
    "article_chars",
]


def compute_local_metrics(trajectories: list[Trajectory]) -> dict[str, Any]:
    """Cost / structure metrics that do not need a judge model."""
    rows = [_one(traj) for traj in trajectories]
    n = len(rows) or 1
    averages = {f"mean_{k}": round(sum(row[k] for row in rows) / n, 4) for k in _MEAN_KEYS}
    return {
        "n_queries": len(rows),
        "n_with_report": sum(1 for row in rows if row["has_report"]),
        "n_errors": sum(1 for row in rows if row["error"]),
        "n_budget_violations": sum(1 for row in rows if row["budget_violations"]),
        "total_tokens": sum(row["tokens"] for row in rows),
        **averages,
        "per_query": rows,
    }


def _one(traj: Trajectory) -> dict[str, Any]:
    article = traj.report.article if traj.report else ""
    citations = traj.report.citations if traj.report else []
    if not citations:
        citations = _URL.findall(article)
    stats = traj.final_stats or {}

    n_searches = sum(1 for step in traj.steps if step.action.type.value == "search")
    n_reads = sum(1 for step in traj.steps if step.action.type.value == "read")
    n_retained = int(stats.get("n_retained") or 0)
    n_pruned = int(stats.get("n_pruned") or 0)
    seen = n_retained + n_pruned

    # Prefer harness-measured wall clock; fall back to summed step latency.
    step_latency = round(traj.total_latency_s(), 4)
    wall_s = stats.get("wall_s")
    wall_s = round(float(wall_s), 4) if isinstance(wall_s, (int, float)) else step_latency

    usage = stats.get("usage") or {}
    tokens = int(usage.get("total_tokens") or 0) or traj.total_tokens()

    return {
        "id": traj.query.id,
        "dataset": traj.query.dataset,
        "has_report": bool(article.strip()),
        "error": traj.error,
        "tokens": tokens,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "n_llm_calls": int(usage.get("n_calls") or 0),
        "wall_s": wall_s,
        "step_latency_s": step_latency,
        "n_steps": len(traj.steps),
        # Prefer the measured call count; step actions are only a fallback for
        # imported trajectories that carry no usage block.
        "n_searches": int(usage.get("n_search_calls") or 0) or n_searches,
        "n_reads": int(usage.get("n_fetch_calls") or 0) or n_reads,
        "n_retained": n_retained,
        "n_pruned": n_pruned,
        "prune_rate": round(n_pruned / seen, 4) if seen else 0.0,
        "n_citations": len(list(dict.fromkeys(citations))),
        "article_chars": len(article),
        "budget_violations": stats.get("budget_violations") or [],
    }


def write_local_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
