# Adobe Deep Research harness

Evaluation-only scaffold for trying deep research agents on **DeepResearch Bench** and **DeepResearchGym**. The agent implementations are left for you; the harness already knows how to load queries, run an agent, record cost stats, and emit each bench’s official report format.

## What this is (and is not)

This repo does **not** implement PILOT or a full research agent. It implements the loop around them:

1. Load queries from a bench
2. Inject a model client and a search backend
3. Call `agent.run(task, ctx)`
4. Store the report + trajectory
5. Export to the official bench format
6. Optionally call the official judges

Swap models without touching eval code. `openai_compat` talks to OpenAI, OpenRouter, vLLM, Ollama, or any other OpenAI-style server.

## Layout

```
src/adr/
  agents/           # implement your agent here
    base.py         # ResearchAgent + AgentContext
    deep_research.py
    pilot.py
    fixture.py      # smoke-test double only
    registry.py     # add a one-line builder when you create a new agent
  core/             # shared types: Query, Budget, Evidence, Trajectory, ResearchState
  llm/              # mock | openai_compat
  tools/            # mock | gym (ClueWeb/FineWeb) | tavily
  datasets/         # DeepResearch Bench + Gym query loaders
  eval/             # exporters + official-bench runners + local metrics
  runner/           # experiment driver
  cli.py
data/benchmarks/    # official query files (copied)
configs/
third_party/        # gitignored clones of the official eval repos
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Clone the official judge repos only when you want RACE / FACT / Gym LLM judges:

```bash
bash scripts/bootstrap_third_party.sh
```

## Smoke test (no API keys)

```bash
adr run --config configs/default.yaml --limit 2
```

This uses the `fixture` agent, a local mock corpus, and a mock LLM. It writes:

```
runs/<timestamp>-local-smoke/
  reports/<id>.md
  trajectories/<id>.json
  exports/deep_research_gym/fixture/<id>.q
  exports/deep_research_gym/fixture/<id>.a
  metrics/local.json
  metrics/summary.json
```

## Implement an agent

Fill in `src/adr/agents/deep_research.py` (or `pilot.py`). The only required method:

```python
async def run(self, task: ResearchTask, ctx: AgentContext) -> Trajectory:
    # ctx.llm.complete(...)        any OpenAI-compatible model
    # ctx.search.search / fetch    mock, DeepResearchGym, or Tavily
    # return a Trajectory with report.article set
```

`ResearchState` is optional but useful: evidence pool, open branches, budget, `compact_stats()`, prune, trajectory logging.

Register a new class in `src/adr/agents/registry.py`, then:

```bash
adr run --agent deep_research --dataset deep_research_gym --limit 5 \
  --llm openai_compat --search gym
```

Point `llm.base_url` at a local vLLM / Ollama server if you are not using OpenAI.

## Evaluate

Local cost metrics (tokens, latency, steps, retained vs pruned evidence) never need a judge key:

```bash
adr evaluate runs/<run-id>
```

Official benches (needs keys + `third_party/`):

```bash
# DeepResearch Bench: RACE + FACT
export OPENROUTER_API_KEY=...
export JINA_API_KEY=...          # FACT only
adr run --dataset deep_research_bench --language en --search tavily \
  --official deep_research_bench

# DeepResearchGym: quality + key-point recall
export OPENAI_API_KEY=...
export DEEPRESEARCHGYM_API_KEY=...
adr run --dataset deep_research_gym --search gym \
  --official deep_research_gym
```

Compare two runs:

```bash
adr compare runs/<a> runs/<b>
```

### Report formats the judges expect

**DeepResearch Bench** (`data/test_data/raw_data/<name>.jsonl`):

```json
{"id": 51, "prompt": "...", "article": "...markdown with citations..."}
```

**DeepResearchGym** (one folder per system):

```
<id>.q    # query text
<id>.a    # report text
```

## Search backends

| Backend | Use when |
|---|---|
| `mock` | Tests and dry runs |
| `gym` | DeepResearchGym (ClueWeb22 / FineWeb `/search` + `/fetch`) |
| `tavily` | Live web, typical for DeepResearch Bench / FACT |

## Tests

```bash
pytest
```
