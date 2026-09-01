# Research log

## 2026-09-01 — reference audit and harness design

- Hypothesis: the reported `m=2d` perplexity improvement is a robust downstream
  effect, not a single-seed or effective-storage artifact.
- Reference audited: `../TurboQuant` commit `1ea420d`. Its GPT-2/WikiText-2
  benchmark reports one deterministic projection per layer/head, with `m=d` and
  `m=2d` at key/value bits `4/2`; it does not provide multi-seed results,
  attention-logit diagnostics, or equal-storage controls.
- Implementation check: both the reference scalar estimator and fast cache use
  `sqrt(pi/2)/m`, which is the required normalization for arbitrary `m`.
- Confounders to control: QJL signs cost one bit each, so increasing `m` changes
  per-key storage; the shared dense projection matrix also grows with `m` but is
  normally regenerated from a seed rather than stored. Layer/head seeds in the
  reference are deterministic and do not form an independent experimental seed
  protocol.
- Next: run the dependency-light estimator scaling check; then execute the LLM
  sweep only after the environment can load a supported PyTorch build.

## 2026-09-01 — estimator sanity result and LLM-run blocker

- Exact setup: `d=64`, 64 independently sampled query/key pairs, 80 projection
  seeds for each `m/d in {0.25, 0.5, 1, 2, 4, 8}`; output is
  `results/raw/qjl_sanity.json`.
- Result: QJL MSE fell from 372.7 at `m/d=0.25` to 11.59 at `m/d=8`; the fitted
  log(MSE)-vs-log(m) slope was -0.999. Mean bias at all ratios was small relative
  to error. This is a normalization/scaling check, not LLM evidence.
- Failed attempt: the initial `datasets`-based WikiText loader hung during
  metadata resolution. The sweep now downloads the official test parquet shard
  directly and pins the source URL in code. A subsequent GPT-2 load did not
  complete within the smoke-test time window and was terminated before writing a
  result, so there are no downstream conclusions yet.
- Next: resolve/profile GPT-2 model loading, run a one-setting smoke test that
  produces JSONL, then start the registered five-seed two-bit-width sweep.

## 2026-09-01 — controlled GPT-2 sweeps complete

- Hypothesis: `m=2d` gives a seed-robust improvement over `m=d`, and the
  downstream response is not fully described by monotonic estimator scaling.
- Exact main setup: GPT-2 FP16 on MPS, WikiText-2 raw test first 512 tokens,
  context limit 1024, QJL seeds 11/23/37/53/71, `m/d={0.5,1,2,4}`. Rotations,
  scalar quantization, values, model weights, and text are fixed. Raw records:
  `results/raw/llm_sweep.jsonl`.
- 4/2-bit result: FP16 30.93 PPL; `m=d` 45.16 +/- 1.63; `m=2d` 37.84 +/- 0.53;
  `m=4d` 35.95 +/- 0.84. All paired seeds improved from `d` to `2d`.
- 3/2-bit confirmation: `m=d` 92.29 +/- 4.77; `m=2d` 54.43 +/- 1.94;
  `m=4d` 40.39 +/- 1.20. All paired seeds improved.
- Dataset-size check: at 2048 tokens, FP16 22.77; `m=d` 42.70 +/- 0.66;
  `m=2d` 30.57 +/- 0.24. This is close to the sibling repository's preliminary
  pattern and rules out a 512-token-only artifact for the main comparison.
- Interpretation: estimator/logit error follows the expected scaling, while
  4/2-bit downstream PPL saturates sharply after `2d`; more aggressive 3/2-bit
  quantization benefits materially through `4d`. This passes the initial gate
  but does not prove novelty or beat alternative uses of the same storage.
- Runtime confounder: synchronized MPS cache microbenchmarks are dominated by
  small-kernel overhead and are not monotonic. Compute and shared-projection
  storage scale linearly in theory; optimized incremental CUDA timing remains.
- Verdict: Outcome C, strong signal, with the larger-model study justified only
  as the next controlled gate.
