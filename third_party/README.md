# Third-party evaluation repos

The official judges are **not** vendored. Everything in this directory except
this file is gitignored, so nothing here is ever committed.

```bash
bash scripts/bootstrap_third_party.sh
adr doctor      # confirms what resolved
```

The script symlinks a checkout you already have rather than cloning a second
copy. Override with `ADR_DRB_DIR` / `ADR_GYM_DIR` / `ADR_GPTR_DIR`.

| Used for | Accepted directory names | Marker file |
|---|---|---|
| RACE + FACT | `deep_research_bench` | `deepresearch_bench_race.py` |
| quality / KPR / citation + key points | `deepresearchgym`, `deepresearch_benchmarking` | `eval_quality_async.py` |
| intern reference (not imported at runtime) | `gpt-researcher` | `gpt_researcher/__init__.py` |

Upstreams: [Ayanami0730/deep_research_bench](https://github.com/Ayanami0730/deep_research_bench),
[Flitternie/deepresearchgym](https://github.com/Flitternie/deepresearchgym),
[Flitternie/gpt-researcher](https://github.com/Flitternie/gpt-researcher).

The harness writes reports in each repo's official format, invokes their scripts
as subprocesses, and parses their result files back into scores.
