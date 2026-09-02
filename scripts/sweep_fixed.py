#!/usr/bin/env python3
"""Sweep FixedOrchestrator breadth/depth on a small Gym sample.

Default: mock LLM/search (free). Writes a markdown quality-vs-cost table.
Live judges are optional and expensive; this sweep logs tokens/searches/retained
which is the cost axis. Official Q can be attached later with adr evaluate.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from adr.runner.config import load_config
from adr.runner.experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    base = load_config(ROOT / "configs" / "fixed_sweep.yaml")
    sweep = base.pop("sweep", {}) or {}
    grid = sweep.get("grid") or [
        {"name": "narrow", "max_subquestions": 2, "top_k": 3, "max_keep": 4},
        {"name": "default", "max_subquestions": 4, "top_k": 5, "max_keep": 8},
        {"name": "wide", "max_subquestions": 4, "top_k": 8, "max_keep": 12},
    ]
    n_queries = int(sweep.get("queries") or base.get("dataset", {}).get("limit") or 8)
    base.setdefault("dataset", {})["limit"] = n_queries

    rows: list[dict] = []
    out_root = Path(base.get("output_dir", "runs")) / "sweep_fixed"
    out_root.mkdir(parents=True, exist_ok=True)

    for cell in grid:
        cfg = yaml.safe_load(yaml.safe_dump(base))
        name = str(cell.get("name", "cell"))
        cfg["run_name"] = f"fixed-{name}"
        cfg["output_dir"] = str(out_root)
        agent_cfg = dict(cfg.get("agent") or {})
        raw_cfg = agent_cfg.get("config") or {}
        if isinstance(raw_cfg, str):
            loaded = yaml.safe_load((ROOT / raw_cfg).read_text(encoding="utf-8")) or {}
            inner = dict(loaded)
        else:
            inner = dict(raw_cfg)
        for key in ("max_subquestions", "top_k", "max_keep"):
            if key in cell:
                inner[key] = cell[key]
        agent_cfg["config"] = inner
        cfg["agent"] = agent_cfg
        manifest = run_experiment(cfg)
        summary = json.loads((manifest.run_dir / "metrics" / "summary.json").read_text())
        rows.append(
            {
                "name": name,
                "max_subquestions": inner.get("max_subquestions"),
                "top_k": inner.get("top_k"),
                "max_keep": inner.get("max_keep"),
                "mean_tokens": summary.get("mean_tokens"),
                "mean_n_searches": summary.get("mean_n_searches"),
                "mean_n_retained": summary.get("mean_n_retained"),
                "mean_n_pruned": summary.get("mean_n_pruned"),
                "mean_n_citations": summary.get("mean_n_citations"),
                "n_with_report": summary.get("n_with_report"),
                "run_dir": str(manifest.run_dir),
            }
        )

    table_path = out_root / "quality_vs_cost.md"
    lines = [
        "# FixedOrchestrator quality vs cost",
        "",
        "Cost axis is harness-measured tokens/searches. Official Gym Q is empty",
        "unless you re-run with `--official deep_research_gym`.",
        "",
        "| setting | B | k | keep | tokens | searches | retained | pruned | citations | reports |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['max_subquestions']} | {row['top_k']} | {row['max_keep']} | "
            f"{row['mean_tokens']} | {row['mean_n_searches']} | {row['mean_n_retained']} | "
            f"{row['mean_n_pruned']} | {row['mean_n_citations']} | {row['n_with_report']} |"
        )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_root / "quality_vs_cost.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(table_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
