from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from adr.core.types import Trajectory

_URL = re.compile(r"https?://[^\s\]\)>]+")


def compute_local_metrics(trajectories: list[Trajectory]) -> dict[str, Any]:
    """Cost / structure metrics that do not need a judge model."""
    rows = [_one(traj) for traj in trajectories]
    n = len(rows) or 1
    keys = [
        "tokens",
        "latency_s",
        "n_steps",
        "n_searches",
        "n_reads",
        "n_retained",
        "n_pruned",
        "n_citations",
        "article_chars",
    ]
    averages = {f"mean_{k}": round(sum(row[k] for row in rows) / n, 4) for k in keys}
    return {
        "n_queries": len(rows),
        "n_with_report": sum(1 for row in rows if row["has_report"]),
        "n_errors": sum(1 for row in rows if row["error"]),
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
    return {
        "id": traj.query.id,
        "dataset": traj.query.dataset,
        "has_report": bool(article.strip()),
        "error": traj.error,
        "tokens": traj.total_tokens(),
        "latency_s": round(traj.total_latency_s(), 4),
        "n_steps": len(traj.steps),
        "n_searches": n_searches,
        "n_reads": n_reads,
        "n_retained": int(stats.get("n_retained") or 0),
        "n_pruned": int(stats.get("n_pruned") or 0),
        "n_citations": len(list(dict.fromkeys(citations))),
        "article_chars": len(article),
    }


def write_local_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps(metrics), encoding="utf-8")


def _dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
