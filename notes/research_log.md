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
