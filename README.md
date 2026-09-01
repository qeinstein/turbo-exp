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

## Phase 2 decision-grade validation

The Phase 2 configs are immutable experiment specifications. Reproduce the
allocation, fine-sweep, scalar-only, and projection-sensitivity runs with:

```bash
.venv/bin/python scripts/run_llm_sweep.py --models gpt2 --spec-file configs/phase2_equal_budget.json --qjl-seeds 11 23 37 53 71 --tokens 512 --stride 256 --study-id phase2-equal-budget-confirmatory --out results/raw/phase2_equal_budget.jsonl
.venv/bin/python scripts/run_llm_sweep.py --models gpt2 --spec-file configs/phase2_equal_budget_extension.json --qjl-seeds 11 23 37 53 71 --tokens 512 --stride 256 --study-id phase2-equal-budget-extension-confirmatory --out results/raw/phase2_equal_budget_extension.jsonl
.venv/bin/python scripts/run_llm_sweep.py --models gpt2 --spec-file configs/phase2_mse_only_extension.json --qjl-seeds 11 23 37 53 71 --tokens 512 --stride 256 --study-id phase2-mse-only-confirmatory --out results/raw/phase2_mse_only.jsonl
.venv/bin/python scripts/run_llm_sweep.py --models gpt2 --spec-file configs/phase2_high_scalar_controls.json --qjl-seeds 11 --tokens 512 --stride 256 --study-id phase2-high-scalar-confirmatory --out results/raw/phase2_high_scalar.jsonl
.venv/bin/python scripts/run_llm_sweep.py --models gpt2 --spec-file configs/phase2_fine_m.json --qjl-seeds 11 23 37 53 71 --tokens 512 --stride 256 --study-id phase2-fine-m-confirmatory --out results/raw/phase2_fine_m.jsonl
.venv/bin/python scripts/run_llm_sweep.py --models gpt2 --spec-file configs/phase2_block_orthogonal_control.json --qjl-seeds 11 23 37 53 71 --tokens 512 --stride 256 --projection-mode block_orthogonal --study-id phase2-block-orthogonal-confirmatory --out results/raw/phase2_block_orthogonal.jsonl
```

The runner refuses duplicate records in an existing output. To regenerate all
tables, plots, and the decision document solely from raw results:

```bash
.venv/bin/python scripts/analyze_phase2.py
.venv/bin/python scripts/render_phase2_validation.py
```

The result is [PHASE2_VALIDATION.md](PHASE2_VALIDATION.md), with the complete
primary-source audit in [notes/literature_audit.md](notes/literature_audit.md).
