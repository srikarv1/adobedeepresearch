from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from adr.datasets.loader import load_queries
from adr.eval.compare import compare_summaries
from adr.runner.config import load_config
from adr.runner.experiment import evaluate_run_dir, run_experiment

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command("queries")
def queries_cmd(
    dataset: str = typer.Option("deep_research_gym", "--dataset", "-d"),
    language: Optional[str] = typer.Option(None, "--language"),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    rows = load_queries(dataset, language=language, limit=limit)
    table = Table(title=f"{dataset} ({len(rows)} queries)")
    table.add_column("id")
    table.add_column("lang")
    table.add_column("text")
    for row in rows:
        text = row.text if len(row.text) < 80 else row.text[:77] + "..."
        table.add_row(row.id, row.language, text)
    console.print(table)


@app.command("run")
def run_cmd(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a"),
    llm_provider: Optional[str] = typer.Option(None, "--llm"),
    search_backend: Optional[str] = typer.Option(None, "--search"),
    language: Optional[str] = typer.Option(None, "--language"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    run_name: Optional[str] = typer.Option(None, "--run-name"),
    official: Optional[str] = typer.Option(None, "--official", help="Comma-separated: deep_research_bench,deep_research_gym"),
) -> None:
    overrides: dict = {}
    if dataset:
        overrides.setdefault("dataset", {})["name"] = dataset
    if language:
        overrides.setdefault("dataset", {})["language"] = language
    if limit is not None:
        overrides.setdefault("dataset", {})["limit"] = limit
    if agent:
        overrides.setdefault("agent", {})["name"] = agent
    if llm_provider:
        overrides.setdefault("llm", {})["provider"] = llm_provider
    if search_backend:
        overrides.setdefault("search", {})["backend"] = search_backend
    if run_name:
        overrides["run_name"] = run_name
    if official:
        overrides.setdefault("eval", {})["official_benches"] = [x.strip() for x in official.split(",") if x.strip()]
    cfg = load_config(config, overrides)
    manifest = run_experiment(cfg)
    console.print(f"[green]Run written to[/green] {manifest.run_dir}")


@app.command("evaluate")
def evaluate_cmd(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    official: str = typer.Option("", "--official", help="Comma-separated benches, or empty for local metrics only"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    cfg = load_config(config) if config else {}
    benches = [x.strip() for x in official.split(",") if x.strip()]
    summary = evaluate_run_dir(run_dir, official_benches=benches, config=cfg)
    console.print_json(json.dumps({k: v for k, v in summary.items() if k != "official"}))
    if summary.get("official"):
        console.print(summary["official"])


@app.command("compare")
def compare_cmd(
    left: Path = typer.Argument(..., exists=True),
    right: Path = typer.Argument(..., exists=True),
) -> None:
    left_summary = left / "metrics" / "summary.json" if left.is_dir() else left
    right_summary = right / "metrics" / "summary.json" if right.is_dir() else right
    console.print_json(json.dumps(compare_summaries(left_summary, right_summary)))


@app.command("bootstrap")
def bootstrap_cmd() -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_third_party.sh"
    raise typer.Exit(code=_run_script(script))


def _run_script(script: Path) -> int:
    import subprocess

    if not script.exists():
        console.print(f"[red]Missing {script}[/red]")
        return 1
    return subprocess.call(["bash", str(script)])


if __name__ == "__main__":
    app()
