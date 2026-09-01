# TurboQuant QJL measurement-budget study

This repository is an independent, falsification-oriented study of whether QJL
sketch dimension `m` beyond head dimension `d` improves TurboQuant-style
KV-cache attention enough to warrant more research. `../TurboQuant` is only an
implementation reference; no code here imports it at runtime.

Run a cheap mathematical sanity check:

```bash
python3 scripts/run_qjl_sanity.py --seeds 200 --trials 128
```

After installing the optional experiment dependencies, run the LLM sweep:

```bash
python3 scripts/run_llm_sweep.py --quick
python3 scripts/analyze.py
```

Results are append-only JSONL under `results/raw/`; summaries and plots are
generated from those raw records. See `notes/research_log.md` before treating
any outcome as evidence.
