# Third-party evaluation repos

These are **not** vendored in git. Clone them with:

```bash
bash scripts/bootstrap_third_party.sh
```

| Directory | Upstream | Used for |
|---|---|---|
| `deep_research_bench/` | [Ayanami0730/deep_research_bench](https://github.com/Ayanami0730/deep_research_bench) | RACE + FACT |
| `deepresearch_benchmarking/` | [cxcscmu/deepresearch_benchmarking](https://github.com/cxcscmu/deepresearch_benchmarking) | quality / KPR / citation + key points |

The harness writes reports in each repo's official format, then optionally invokes their scripts.
