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
