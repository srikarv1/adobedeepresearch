from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from adr.core.types import Trajectory
from adr.eval.exporters import export_deep_research_bench

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_THIRD_PARTY = ROOT / "third_party" / "deep_research_bench"


def run_deep_research_bench(
    trajectories: list[Trajectory],
    *,
    run_dir: Path,
    model_name: str,
    third_party_dir: str | Path | None = None,
    language: str = "en",
    workers: int = 4,
    skip_cleaning: bool = False,
    run_race: bool = True,
    run_fact: bool = True,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write DRB jsonl and optionally invoke the official RACE/FACT scripts."""
    bench_root = Path(third_party_dir) if third_party_dir else DEFAULT_THIRD_PARTY
    export_path = run_dir / "exports" / "deep_research_bench" / f"{model_name}.jsonl"
    export_deep_research_bench(trajectories, export_path)

    result: dict[str, Any] = {
        "bench": "deep_research_bench",
        "export": str(export_path),
        "official": False,
        "reason": None,
        "race": None,
        "fact": None,
    }
    if not bench_root.exists():
        result["reason"] = f"Official repo missing at {bench_root}. Run scripts/bootstrap_third_party.sh"
        return result

    dest = bench_root / "data" / "test_data" / "raw_data" / f"{model_name}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(export_path.read_text(encoding="utf-8"), encoding="utf-8")

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    query_file = bench_root / "data" / "prompt_data" / "query.jsonl"
    if language == "en" and not (env.get("OPENROUTER_API_KEY") or env.get("OPENAI_API_KEY")):
        result["reason"] = "No OPENROUTER_API_KEY or OPENAI_API_KEY; skipped official RACE/FACT"
        return result

    if run_race:
        race_out = bench_root / "results" / "race" / model_name
        race_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-u",
            str(bench_root / "deepresearch_bench_race.py"),
            model_name,
            "--raw_data_dir",
            str(bench_root / "data" / "test_data" / "raw_data"),
            "--max_workers",
            str(workers),
            "--query_file",
            str(query_file),
            "--output_dir",
            str(race_out),
        ]
        if language == "en":
            cmd.append("--only_en")
        elif language == "zh":
            cmd.append("--only_zh")
        if skip_cleaning:
            cmd.append("--skip_cleaning")
        completed = subprocess.run(cmd, cwd=bench_root, env=env, capture_output=True, text=True)
        result["race"] = {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
            "output_dir": str(race_out),
        }
        if completed.returncode != 0:
            result["reason"] = "RACE script failed"
            return result

    if run_fact:
        if not env.get("JINA_API_KEY"):
            result["fact"] = {"skipped": True, "reason": "JINA_API_KEY missing"}
        else:
            fact_out = bench_root / "results" / "fact" / model_name
            fact_out.mkdir(parents=True, exist_ok=True)
            raw = dest
            steps = [
                ["-m", "utils.extract", "--raw_data_path", str(raw), "--output_path", str(fact_out / "extracted.jsonl"), "--query_data_path", str(query_file), "--n_total_process", str(workers)],
                ["-m", "utils.deduplicate", "--raw_data_path", str(fact_out / "extracted.jsonl"), "--output_path", str(fact_out / "deduplicated.jsonl"), "--query_data_path", str(query_file), "--n_total_process", str(workers)],
                ["-m", "utils.scrape", "--raw_data_path", str(fact_out / "deduplicated.jsonl"), "--output_path", str(fact_out / "scraped.jsonl"), "--n_total_process", str(workers)],
                ["-m", "utils.validate", "--raw_data_path", str(fact_out / "scraped.jsonl"), "--output_path", str(fact_out / "validated.jsonl"), "--query_data_path", str(query_file), "--n_total_process", str(workers)],
                ["-m", "utils.stat", "--input_path", str(fact_out / "validated.jsonl"), "--output_path", str(fact_out / "fact_result.txt")],
            ]
            fact_logs = []
            for args in steps:
                completed = subprocess.run(
                    [sys.executable, "-u", *args],
                    cwd=bench_root,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                fact_logs.append({"args": args, "returncode": completed.returncode})
                if completed.returncode != 0:
                    result["fact"] = {"steps": fact_logs, "stderr_tail": completed.stderr[-2000:]}
                    result["reason"] = "FACT script failed"
                    return result
            result["fact"] = {"steps": fact_logs, "output_dir": str(fact_out)}

    result["official"] = True
    result["reason"] = None
    return result
