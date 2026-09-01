# Initial study status — not yet complete

The requested small-model study has **not** reached a decision A/B/C. No LLM
perplexity result is reported in this file because no controlled LLM run has
completed and written a raw record.

## What has been verified

The reference at `../TurboQuant` commit `1ea420d` implements residual QJL with
`S` shaped `(m, d)` in its benchmark-only fast cache. It sets `m=d*sketch_mult`
and uses `sqrt(pi/2)/m` for the residual inner-product correction. Its reported
GPT-2/WikiText-2 `m=2d` number is a single deterministic projection run; it is
therefore insufficient to establish robustness to QJL seed, bit width, model,
or storage budget.

The dependency-light QJL check in `results/raw/qjl_sanity.json` confirms the
implementation's required normalization empirically: at `d=64`, its fitted
MSE scaling is approximately `1/m` (log slope `-0.999` over six ratios). This
does not imply downstream perplexity must improve monotonically.

## Designed controlled experiment

`scripts/run_llm_sweep.py` registers `m/d={0.5,1,2,4}` (optionally `0.25,8`),
five independent QJL seeds, fixed `4/2` and `3/2` key/value bit configurations,
GPT-2 plus DistilGPT-2, FP baseline, and per-run JSONL metadata. It measures
perplexity, QJL residual RMSE, attention-logit MAE/RMSE, attention KL, runtime,
and packed-storage estimates. `scripts/analyze.py` generates summaries and the
required plots only from that JSONL.

## Current limitation

The installed `datasets` resolver hung on WikiText metadata; the code was
changed to retrieve the official WikiText-2 test parquet shard directly. GPT-2
model loading did not complete during the short smoke-test window, so it was
stopped without a result. Until this is resolved and the multi-seed sweep has
run, the preliminary `m=2d` claim has neither replicated nor failed.
