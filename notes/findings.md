# Replicated findings

- On GPT-2/WikiText-2 with 4/2-bit key/value quantization, `m=2d` beats `m=d`
  for every one of five QJL seeds. The result holds on both 512-token and
  2048-token evaluations. At 2048 tokens, mean PPL changes from 42.70 to 30.57
  against an FP16 PPL of 22.77.
- The direction survives a more aggressive 3/2-bit configuration: mean PPL
  changes from 92.29 at `m=d` to 54.43 at `m=2d`, with all five seeds improving.
- QJL residual/logit RMSE declines close to the expected `1/sqrt(m)` scaling,
  but downstream PPL is nonlinear: 4/2-bit PPL largely saturates after `2d`,
  whereas 3/2-bit PPL continues to improve materially from `2d` to `4d`.

These are replicated empirical findings for one model and one dataset slice,
not claims of novelty or broad model generality.

## Phase 2 findings that survived the fixed-budget controls

- The conditional `m` response is smooth and precision-dependent on GPT-2. In
  the fine sweep, the smallest tested point closing 90% of the observed
  `m=d -> 4d` PPL gain is `2.5d` at 4/2 bits and `3d` at 3/2 bits.
- The apparent benefit does not yield a useful fixed-storage allocation rule in
  the tested framework. Across scalar allocations up to 8-bit keys/5-bit
  values, every point on the measured PPL Pareto frontier through the best
  tested result is MSE-only (`m=0`).
- A finite-sample QJL residual correction can hurt. Relative to MSE-only,
  `m=d` raises PPL from 39.91 to 45.16 at 4/2 bits and from 57.00 to 92.29 at
  3/2 bits while also using 12 extra bytes/token. Raising `m` reduces this
  estimator-induced attention divergence, but spending the storage on scalar
  precision is more effective in these tests.

The fixed-budget finding is a falsification of the candidate paper hypothesis,
not evidence that QJL is universally inferior across models or implementations.
