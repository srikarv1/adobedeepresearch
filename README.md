# Adobe Deep Research harness

Evaluation harness for deep-research agents on **DeepResearch Bench** (RACE + FACT) and **DeepResearchGym** (quality, key-point recall, citation faithfulness).

```
query  ->  intern (plan / search / write)  ->  report
                 ^
                 |
           orchestrator.decide()
           keep mask, branch weights, stop
```

The intern is frozen and sequential. The orchestrator is the knob: **fixed** (top-k keep), **prompted** (thin LLM scorecard), or **learned** (stub). ParallelResearch's async tree is the published Adobe baseline, not this intern.

The harness owns query loading, budgets, cost meters, official export formats, and official judges. Agents must use `ctx.llm` and `ctx.search` so tokens and latency are measured, not self-reported.

## Design commitments

Two things drive most of the structure here:

**Judges are never reimplemented.** The official rubrics carry hard scoring rules, e.g. in DeepResearchGym a report with no source URLs scores zero on Support, and any rubric deficiency caps the score at 8. Paraphrasing those prompts inflates scores and makes results incomparable to published numbers. So the harness writes each benchmark's expected file layout and invokes the upstream judge functions directly, then aggregates raw results into structured scores.

**Cost is measured, not self-reported.** Token and latency numbers would be worthless if they depended on an agent remembering to log them. `MeteredLLM` and `MeteredSearch` wrap the model and search clients before the agent ever sees them, so every call is counted whether or not the agent cooperates. Budgets are charged from the same place.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Point the harness at the official judge repos. Existing clones anywhere on disk are symlinked rather than re-cloned:

```bash
bash scripts/bootstrap_third_party.sh
adr doctor          # shows which repos and keys resolved
```

`adr doctor` is the fastest way to see why a judge is being skipped.

### Which key does what

| Key | Needed for |
|---|---|
| `OPENAI_API_KEY` | DeepResearch Bench RACE + FACT (with `LLM_BACKEND=openai`), and all Gym judges. Honours `OPENAI_BASE_URL` for local vLLM/Ollama. |
| `OPENROUTER_API_KEY` | DeepResearch Bench RACE + FACT (with `LLM_BACKEND=openrouter`, the default). |
| `JINA_API_KEY` | FACT only; it scrapes every cited page. |
| `DEEPRESEARCHGYM_API_KEY` | The Gym retrieval sandbox. |
| `TAVILY_API_KEY` | Live web search, typical for DeepResearch Bench runs. |

## Score a question and report pair

The quickest way to exercise the whole evaluation path. Matching a benchmark query id matters: RACE pairs your report to its reference article by exact prompt string, and key-point recall needs an id that has key points.

```bash
adr score --report myreport.md --query-id 923549
```

The question is filled in from the benchmark. To score an off-benchmark query, pass `--question` instead; you then get Gym quality but not key-point recall.

```bash
adr score --report myreport.md --question "Does creatine help with cognition?"
adr score --report data/fixtures/sample_report_51.md --query-id 51 --dataset deep_research_bench
adr score --reports-dir path/to/folder-of-q-and-a-files
adr score --drb-jsonl path/to/raw.jsonl
adr score --report myreport.md --query-id 923549 --local-only   # no judge calls
```

To check the wiring without a judge key, run `pytest tests/test_official_gym_wiring.py`. It drives the real upstream Gym scripts against a stub judge and asserts an exactly predictable aggregate.

## Run an agent

```bash
adr run --config configs/default.yaml --limit 2          # fixture agent, mock LLM, mock corpus
adr run --config configs/intern_gym.yaml --limit 1       # intern + fixed orchestrator
adr run --config configs/prompted_gym.yaml --limit 1     # intern + prompted orchestrator (Azure)
python scripts/sweep_fixed.py                             # B/D quality-vs-cost table
```

### Where the report is

Each run writes a folder under `runs/` (gitignored). The article is the `.md` file:

```
runs/<timestamp>-<name>/
  reports/<query_id>.md                 <-- human-readable report
  exports/deep_research_gym/<agent>/<id>.a   <-- Gym judge input
  trajectories/<query_id>.json          <-- full step log
  trajectories/decisions.jsonl          <-- orchestrator (u, m, w) + features
  metrics/summary.json                  <-- tokens, searches, scores
```

Live Azure example (Gym query 923549, chip shortage):

`runs/20260901-205354-azure-intern/reports/923549.md`

Score that file without re-running the intern:

```bash
adr score --report runs/20260901-205354-azure-intern/reports/923549.md --query-id 923549
```

Then compare a baseline against a candidate. Quality, cost, and structure are reported separately, because "fewer tokens" and "higher score" should never be averaged into one number:

```bash
adr compare runs/<baseline> runs/<candidate>
```

## Implement an agent

There is one intern (`InternAgent` in `src/adr/agents/intern.py`). After each retrieve it calls `decide(state) -> PilotDecision(u, m, w)`: stop, keep-mask, branch weights. Register a name only to pick the orchestrator:

| `adr run` agent | Orchestrator | Meaning |
|---|---|---|
| `deep_research` | fixed | constant breadth, top-k keep |
| `pilot` | prompted | LLM keep / allocate / stop |
| `learned` | learned | policy stub; train later |

The contract is one method:

```python
async def run(self, task: ResearchTask, ctx: AgentContext) -> Trajectory:
    # ctx.llm.complete(...)       metered automatically
    # ctx.search.search / fetch   metered automatically
    # return a Trajectory with report.article set
```

Register it in `src/adr/agents/registry.py`. `ResearchState` is optional but gives you an evidence pool with retain/prune/superseded, a subtask frontier, budget accounting, and `compact_stats()` for a small observation instead of raw passages.

Budgets are charged as you spend. With `budget.enforce: true` the overrunning call raises `BudgetExceeded`; with `false` the overrun is recorded in `n_budget_violations` and the trajectory finishes.

## Layout

```
src/adr/
  agents/intern.py   plan / search / write  (one intern)
  orchestrate/       decide(state) -> keep, weights, stop
  runner/            writes runs/<id>/reports/<query>.md
  core/instrument.py MeteredLLM / MeteredSearch
  eval/              official Gym + Bench judges (never rewritten)
```

## Metrics

Local metrics need no judge and are always computed:

| Metric | Meaning |
|---|---|
| `mean_tokens`, `mean_prompt_tokens`, `mean_completion_tokens` | measured at the client, not self-reported |
| `mean_wall_s` | real end-to-end wall clock per query |
| `mean_n_llm_calls`, `mean_n_searches`, `mean_n_reads` | measured call counts |
| `mean_n_retained`, `mean_n_pruned`, `mean_prune_rate` | evidence pool churn |
| `n_budget_violations` | queries that overran their budget |

Judge metrics land in `summary.json` under `scores` (flattened, comparable) and `official` (full detail including subprocess logs):

| Metric | Source |
|---|---|
| `race_overall_score` and its four dimensions | `race_result.txt` |
| `fact_valid_rate`, `fact_total_citations` | `fact_result.txt` |
| `gym_quality` plus per-criterion ratings | `quality_<judge>.json` |
| `gym_average_support_rate` / `omitted` / `contradicted` | `relevance_<judge>.json` |
| `gym_citation_score` | `faithfullness_<judge>.json` |

## Report formats the judges expect

**DeepResearch Bench** — `data/test_data/raw_data/<model>.jsonl`, one row per query. The prompt must match the benchmark's query file exactly or RACE cannot find the reference article.

```json
{"id": 51, "prompt": "...", "article": "...markdown with citations..."}
```

**DeepResearchGym** — one folder per system containing `<id>.q` (query) and `<id>.a` (report).

## Tests

```bash
pytest
```

The suite runs offline. `tests/test_official_gym_wiring.py` drives the real upstream Gym scripts against the mock judge and asserts an exactly predictable aggregate, so it fails if the invocation contract or the scoring math drifts. `tests/test_official_drb_wiring.py` does the same for RACE with a stub, without touching your checkout. Tests that need a judge repo skip themselves when it is absent.
