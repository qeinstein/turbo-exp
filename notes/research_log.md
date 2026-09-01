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

## 2026-09-01 — Phase 2 equal-budget falsification

- Hypothesis: at a fixed total K+V byte budget, spending additional bytes on
  QJL signs can outperform spending those bytes on scalar key/value precision.
- Exact setup: GPT-2/WikiText-2 first 512 tokens, five paired QJL seeds, scalar
  key bits 2--5, value bits 1--4, `m/d in {0, 0.5, 1, 1.5, 2, 3, 4}` and exact
  packed storage including metadata. `m=0` is an explicit MSE-only control: the
  rotated scalar key approximation is used without a QJL residual correction.
  Raw records are `phase2_equal_budget*.jsonl` and `phase2_mse_only.jsonl`.
- Result: at the 60-byte exact match, 4/2-bit `m=d` scored 45.16 +/- 1.63 PPL
  versus 54.43 +/- 1.94 for 3/2-bit `m=2d`. At 68 bytes, 5/2-bit `m=d` scored
  35.43 +/- 0.79 versus 37.84 +/- 0.53 for 4/2-bit `m=2d`. More scalar key
  precision beat the oversized-QJL allocation in both planned comparisons.
- Stronger control: 4/2-bit MSE-only scored 39.91 PPL at 48 bytes, while
  4/2-bit `m=d` scored 45.16 at 60 bytes. At 3/2 bits, MSE-only scored 57.00 at
  40 bytes while `m=d` scored 92.29 at 52 bytes. Thus a noisy QJL correction
  can be worse than no residual correction at all.
- Confounder found: the original scalar sweep stopped at 5/4 bits, making
  5/4-bit `m=2d` and `m=3d` appear on the high-budget frontier only because no
  higher-scalar alternatives existed. A preregistered deterministic extension
  tested 13 MSE-only allocations up to 8-bit keys and 5-bit values. The entire
  tested PPL frontier through 88 bytes became MSE-only: 6/4-bit MSE-only scored
  31.29 at 80 bytes, beating 5/4-bit `m=2d` at 31.64 and 92 bytes; 6/5-bit
  MSE-only scored 30.90 at 88 bytes.
- Interpretation: increasing `m` does reduce the variance and harm of the QJL
  estimator conditional on using QJL, but it is not a competitive first use of
  the tested cache budget. This satisfies the preregistered Verdict A stop gate.
- Next: finish mechanism summaries and the primary-source novelty audit, then
  render the decision document. Cross-model/data compute is stopped unless a
  later question is explicitly reformulated around a non-QJL baseline.

## 2026-09-01 — Phase 2 fine measurement sweep

- Hypothesis: if the conditional QJL effect is structured rather than noise,
  the useful point should shift under more aggressive scalar key quantization.
- Exact setup: five paired seeds at `m/d={1,1.25,1.5,1.75,2,2.5,3,4}` for 4/2
  and 3/2 bits, otherwise identical to the 512-token confirmatory setup. Raw
  records are `results/raw/phase2_fine_m.jsonl`.
- Result: for 4/2 bits, the smallest tested `m` achieving 90% of the observed
  `m=d` to `4d` PPL gain was `2.5d`; marginal gain fell from 1.20 PPL/added byte
  over `1d -> 1.25d` to 0.026 over `2.5d -> 3d`. For 3/2 bits, the 90% point
  shifted to `3d`, and gains remained larger throughout the range.
- Interpretation: the severity-dependent crossover is real within GPT-2, but
  the MSE-only fixed-budget controls show it characterizes how many measurements
  are needed to make QJL non-harmful, not an allocation rule that improves the
  quality-memory frontier.

## 2026-09-01 — block-orthogonal projection sensitivity control

- Hypothesis: the negative fixed-budget result could be an artifact of the
  independent Gaussian projection used by TurboQuant rather than the
  block-orthogonal construction in the public QJL implementation.
- Exact setup: GPT-2/WikiText-2 first 512 tokens, five seeds, 3/2- and 4/2-bit
  quantization, `m/d in {1, 2}`, with independent scaled Haar-orthogonal blocks
  of `d` rows. All other settings match the confirmatory Gaussian runs. Raw
  records are `results/raw/phase2_block_orthogonal.jsonl`.
- Result: orthogonalization materially improved QJL. At 4/2 bits, mean PPL was
  36.95 +/- 0.59 at `m=d` and 34.87 +/- 0.62 at `m=2d`, versus Gaussian means
  45.16 and 37.84. At 3/2 bits, the corresponding orthogonal means were 49.16
  +/- 2.28 and 39.84 +/- 0.72, versus 92.29 and 54.43 for Gaussian QJL.
- Fixed-budget result: every orthogonal-QJL point remained dominated by an
  MSE-only allocation using no more storage. The closest cases were 4/2,
  `m=d` at 60 bytes/token (36.95 PPL) versus MSE-only 5/2 at 56 bytes (35.01),
  and 4/2, `m=2d` at 68 bytes (34.87) versus MSE-only 5/3 at 64 bytes (32.83).
- Interpretation: projection structure is a major quantitative confounder, but
  it does not reverse the preregistered fixed-budget stop decision. Claims in
  the final report must distinguish the Gaussian primary sweep from this
  stronger post-gate implementation control.

## 2026-09-01 — literature and public-code novelty audit

- Hypothesis audited: a downstream benefit from `m>d`, or a useful adaptive
  measurement-budget rule, is sufficiently unstudied to support a new paper.
- Direct prior: the 2024 QJL paper defines arbitrary `m` and normalization by
  `1/m`. Its public implementation (commit `648b364`) defaults to `m=2d` in
  general layers and `m=4d` in the first 15 layers, and its distortion script
  sweeps `m/d=0.5..4` with one repetition. Therefore the basic `m>d` idea is
  already public and cannot be claimed as novel.
- Strong post-TurboQuant overlap: arXiv:2605.08114 directly compares MSE-only
  with MSE+QJL under fair bit budgets in synthetic attention and proposes the
  same QJL-variance/softmax-amplification mechanism. UltraQuant and the current
  vLLM TurboQuant path explicitly remove QJL in deployment-oriented designs.
- Adjacent allocation prior: KVQuant, HIGGS, AQUA-KV, and JoLT already cover
  sensitivity-aware, layerwise, residual, or jointly optimized byte allocation.
- Interpretation: the only apparently open contribution here is a narrow
  multi-seed downstream boundary-of-use study asking when QJL ever beats the
  best scalar-only allocation. The current one-model negative evidence is not a
  defensible standalone paper and does not justify scaling the original positive
  hypothesis. Full source-by-source audit: `notes/literature_audit.md`.
