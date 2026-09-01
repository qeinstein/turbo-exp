# TurboQuant QJL measurement-budget study

This repository is an independent, falsification-oriented study of whether QJL
sketch dimension `m` beyond head dimension `d` improves TurboQuant-style
KV-cache attention enough to warrant more research. `../TurboQuant` is only an
implementation reference; no code here imports it at runtime.

Run a cheap mathematical sanity check:

```bash
python3 scripts/run_qjl_sanity.py --seeds 200 --trials 128
```

After installing the optional experiment dependencies, reproduce the empirical
study and generated report:

```bash
.venv/bin/python scripts/run_llm_sweep.py --models gpt2 --ratios 0.5 1 2 4 --qjl-seeds 11 23 37 53 71 --configs 4,2 3,2 --tokens 512 --stride 256
.venv/bin/python scripts/run_llm_sweep.py --models gpt2 --ratios 1 2 --qjl-seeds 11 23 37 53 71 --configs 4,2 --tokens 2048 --stride 512 --out results/raw/llm_length_check.jsonl
.venv/bin/python scripts/benchmark_cache_runtime.py
.venv/bin/python scripts/analyze.py
.venv/bin/python scripts/render_initial_study.py
```

Results are append-only JSONL under `results/raw/`; summaries and plots are
generated from those raw records. See `notes/research_log.md` before treating
any outcome as evidence.

The completed initial verdict and limitations are in `INITIAL_STUDY.md`.
