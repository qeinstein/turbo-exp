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
