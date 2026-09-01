# Future research direction: when is residual correction worth using?

## Status

The original positive research direction—choosing a QJL measurement dimension
`m>d` to improve TurboQuant-style KV-cache quantization—failed its Phase 2 gate.
Larger QJL sketches improved a QJL-based estimator, but no tested QJL allocation
entered the measured perplexity/storage Pareto frontier. Scalar-only
quantization achieved better quality with equal or fewer bytes.

This document preserves the narrower research question exposed by that failure.
It is a possible future direction, not a claim established by the current
experiments and not authorization to begin an expensive model sweep.

## One-sentence research question

> Under a fixed KV-cache budget, what measurable conditions make stochastic
> residual correction improve downstream softmax attention more than either
> omitting the residual or spending those bits on a lower-variance scalar
> representation?

## Why this is a better question

The completed experiments distinguish three correction regimes:

1. **Harmful correction:** estimator noise damages attention more than the
   omitted scalar residual would have.
2. **Locally useful correction:** correction improves quality relative to the
   same base scalar quantizer without correction.
3. **Globally worthwhile correction:** correction beats the best alternative
   use of the same total storage and enters the quality-memory Pareto frontier.

The GPT-2 study observed transitions from regime 1 to regime 2. It never
observed regime 3. This makes “how large should `m` be?” secondary. The more
fundamental question is whether correction should be used at all, and whether
that decision can be predicted before paying its storage and compute cost.

## Evidence that motivates the question

### Correction changed from harmful to locally useful

For Gaussian residual QJL, the first tested measurement counts that beat
MSE-only at the same base scalar precision were:

- 4/2-bit K/V: `m=1.75d`;
- 3/2-bit K/V: `m=2d`.

More aggressive quantization required more measurements before correction
became helpful. At `m=d`, the mean 3/2-bit residual norm was approximately 1.86
times the 4/2-bit value.

### Local usefulness did not imply efficient allocation

At 4/2 bits, Gaussian QJL improved from 39.91 PPL without correction to 37.84
at `m=2d`, but increased storage from 48 to 68 bytes/token. MSE-only 5/2
achieved 35.01 PPL at only 56 bytes/token.

At 3/2 bits, Gaussian `m=2d` improved from 57.00 to 54.43 PPL while increasing
storage from 40 to 60 bytes/token. MSE-only 5/2 again achieved 35.01 at 56
bytes/token.

Thus correction could help the underlying low-precision representation while
remaining a poor use of the global budget.

### Projection error structure mattered

Block-orthogonal projection blocks dramatically improved QJL at the same `m`:

| Configuration | Gaussian PPL | Block-orthogonal PPL |
|---|---:|---:|
| 3/2, `m=d` | 92.29 | 49.16 |
| 3/2, `m=2d` | 54.43 | 39.84 |
| 4/2, `m=d` | 45.16 | 36.95 |
| 4/2, `m=2d` | 37.84 | 34.87 |

All four block-orthogonal configurations became locally useful or much closer
to useful, but every point remained dominated by an MSE-only allocation using
four fewer bytes/token. Measurement count alone therefore does not determine
correction utility; covariance, tails, and projection construction matter.

### Aggregate RMSE did not predict downstream quality

At 3/2 bits:

| Method | Logit RMSE | Attention KL | PPL |
|---|---:|---:|---:|
| MSE-only | 1.604 | 0.339 | 57.00 |
| Gaussian QJL, `m=d` | 1.416 | 0.539 | 92.29 |

QJL improved global logit RMSE while substantially worsening attention KL and
PPL. An unbiased or lower-RMSE correction can still be harmful after softmax.
A useful predictor must describe where the error occurs and how attention
responds, not merely its global average magnitude.

### Sensitivity was concentrated

The three highest-error transformer layers accounted for approximately 62–66%
of summed layer attention KL. This suggests that residual correction value may
depend on layer/head sensitivity rather than a global measurement ratio.

## Candidate scientific formulation

Let `r` be the residual left by scalar key quantization and `q` a query. Without
correction, the omitted residual contributes logit error approximately
`q^T r`. A stochastic correction replaces that omission with an estimator error
whose variance, for Gaussian QJL, scales approximately as:

```text
Var(error_QJL) proportional to ||q||^2 ||r||^2 / m.
```

The downstream objective is not ordinary logit MSE. Softmax weights logit
directions non-uniformly, so correction should be worthwhile only when its
attention-sensitive error and storage cost are smaller than the
attention-sensitive cost of omitting the residual or increasing scalar
precision.

A conceptual decision criterion is:

```text
attention-weighted correction variance + storage/compute penalty
    < attention-weighted omitted-residual error.
```

This is a research hypothesis, not a derived or validated rule.

## Candidate predictors

The next study should test whether correction utility can be predicted from a
small set of quantities available before or during cache construction:

- scalar residual norm and variance;
- omitted-residual logit error or bias;
- query norm;
- correction-error variance, covariance, and tail quantiles;
- projection family and block orthogonality;
- attention entropy and concentration;
- top-logit margin or routing stability;
- layer/head sensitivity to controlled attention perturbations;
- key/value scalar precision;
- exact remaining packed byte budget;
- context length and attention pattern type.

Do not assume all of these are required. A useful result should identify a
small, interpretable subset that predicts correction value on held-out models
or data.

## Minimal next experiment

The next experiment should be a cheap, preregistered gate—not a larger-model
campaign.

### Model and data requirements

- Use one genuinely different small architecture, not another GPT-2 size.
- Retain GPT-2 as the development/reference model.
- Evaluate several non-overlapping text slices.
- Prefer a second accepted language-model corpus if it can be added without a
  large engineering or download burden.
- Use at least five isolated correction/projection seeds.

### Required experimental arms

At each exact packed KV-cache budget compare:

1. **Higher-precision scalar-only:** the best available use of the budget
   without residual correction.
2. **Lower-precision scalar-only:** establishes the error of omitting the
   residual at the same base scalar representation.
3. **Lower-precision plus block-orthogonal QJL:** isolates the value and cost of
   official-style stochastic residual correction.

Where cheap, include one deterministic residual-coding alternative to determine
whether any observed boundary is specific to QJL or applies to residual
correction more generally. Do not let this expand into a broad quantization
benchmark before the three core arms are complete.

### Required measurements

- downstream PPL and loss;
- individual seed results, mean, SD, and paired intervals;
- exact packed bytes/token and shared overhead;
- omitted-residual logit error;
- correction-error mean, variance, covariance summaries, and tails;
- attention-logit RMSE and attention KL;
- attention entropy/concentration and top-logit margin;
- per-layer and per-head error;
- residual norm/distribution;
- runtime and compute scaling where measurement is reliable.

### Controlled sensitivity intervention

Apply correction to one layer or head at a time while leaving the remaining
model state unchanged. Measure marginal loss and attention-distribution change.
This is preferable to inferring sensitivity only from observational
correlations.

Do not build an adaptive correction policy during this stage. First establish
whether a stable predictive boundary exists.

## Analysis plan

### Primary outcomes

For each configuration, classify residual correction as:

- harmful relative to the same scalar base;
- locally useful but globally dominated;
- Pareto-superior at the fixed total budget.

The main outcome is not the best `m`. It is whether pre-correction statistics
predict the correct regime.

### Predictor validation

1. Develop a simple interpretable predictor on GPT-2 and one subset of text
   slices.
2. Freeze its variables, thresholds, and coefficients.
3. Evaluate it on the second architecture and held-out slices.
4. Report classification errors and effect sizes, not only correlation.
5. Compare against simple baselines such as residual norm alone, scalar bit
   width alone, and always/never apply correction.

Avoid flexible models unless the dataset becomes large enough to support them.
A threshold or low-dimensional regression that generalizes is more valuable
than a high-capacity post-hoc fit.

### Budget analysis

Construct two frontiers:

```text
packed KV bytes/token <-> perplexity
packed KV bytes/token <-> attention divergence
```

Shared projection memory, prefill cost, and decode cost must be reported
separately. A correction method counts as globally useful only if it improves a
relevant frontier after all unavoidable storage is included.

## Success criteria

The direction should continue only if all of the following occur:

1. At least one block-orthogonal correction configuration enters the
   exact-budget quality-memory Pareto frontier on both architectures or on a
   clearly characterized reproducible subset.
2. Correction benefit is larger than seed and text-slice variability.
3. A small set of measurable statistics predicts whether correction helps on
   held-out architecture/data.
4. The predictor beats trivial always-correct, never-correct, residual-norm-only,
   and bit-width-only rules.
5. The result remains meaningful after storage and runtime costs are included.
6. The contribution remains differentiated after refreshing the literature
   audit.

## Stop criteria

Stop the reformulated direction if any of the following occurs:

- correction remains absent from the exact-budget Pareto frontier;
- any apparent benefit is isolated to one model, slice, layer, or seed;
- block-orthogonal improvement disappears in a second implementation or
  architecture;
- correction utility cannot be predicted better than always omitting it;
- residual norm or scalar bit width explains the effect completely, leaving no
  nontrivial attention-sensitive relationship;
- runtime or shared/per-token storage makes the quality gain operationally
  irrelevant;
- current literature already establishes the same real-LLM boundary or rule.

If a stop criterion is reached, preserve the negative result rather than
inventing an adaptive algorithm to rescue the project.

## Possible contribution if the gate succeeds

A defensible future contribution could be:

> An empirical and predictive bias–variance boundary for deciding when
> stochastic residual correction improves quantized softmax attention under a
> fixed memory budget.

Depending on the evidence, this might become:

- a cross-model empirical characterization of correction regimes;
- a simple correction-enable/disable rule based on residual and attention
  statistics;
- a layer/head-aware allocation rule, but only after controlled evidence shows
  that sensitivity is predictable and the rule improves the Pareto frontier;
- a comparison of stochastic and deterministic residual correction under equal
  storage.

It should not be presented as a new QJL measurement-scaling law. More one-bit
measurements reducing estimator error is already expected and `m>d` already
appears in original QJL work.

## Claims that are not currently supported

The existing results do not support claims that:

- QJL is universally harmful;
- residual correction never helps;
- block-orthogonal QJL is globally optimal;
- useful `m` generalizes beyond GPT-2;
- residual norm alone predicts correction utility;
- an adaptive layerwise QJL method is justified;
- the reformulated question is novel enough for publication;
- larger-model experiments should begin now.

## Recommended immediate action

Do not spend money on larger models. Preserve the completed study as a negative
result. Reopen experimental work only if there is willingness to run the single
small cross-architecture gate above and accept another stop decision if
correction does not enter the exact-budget Pareto frontier.

## Related repository evidence

- [README.md](README.md): full repository overview and failure summary.
- [PHASE2_VALIDATION.md](PHASE2_VALIDATION.md): final Phase 2 decision document.
- [INITIAL_STUDY.md](INITIAL_STUDY.md): preliminary study that established the
  conditional `m` effect.
- [notes/findings.md](notes/findings.md): replicated findings only.
- [notes/research_log.md](notes/research_log.md): chronological experimental
  record.
- [notes/literature_audit.md](notes/literature_audit.md): novelty and prior-work
  audit.
