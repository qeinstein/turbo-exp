#!/usr/bin/env python3
"""Render PHASE2_VALIDATION.md from generated summaries and raw metadata."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary"
RAW = ROOT / "results" / "raw"


def load_summary(name: str) -> list[dict]:
    return json.loads((SUMMARY / f"{name}.json").read_text())


budget = load_summary("phase2_budget_summary")
frontier = load_summary("phase2_ppl_frontier")
fine = load_summary("phase2_fine_m_summary")
comparisons = load_summary("phase2_comparisons")
mechanisms = load_summary("phase2_mechanism_summary")
projection = load_summary("phase2_projection_control")
length_check = json.loads((SUMMARY / "length_check_summary.json").read_text())
environment = json.loads((RAW / "environment.json").read_text())


def fine_row(key_bits: int, ratio: float) -> dict:
    return next(
        row
        for row in fine
        if row["key_bits"] == key_bits
        and row["value_bits"] == 2
        and row["m_over_d"] == ratio
    )


def comparison(name: str) -> dict:
    return next(row for row in comparisons if row["comparison"] == name)


def mechanism(key_bits: int) -> dict:
    return next(row for row in mechanisms if row["key_bits"] == key_bits)


def fmt_seed_values(serialized: str) -> str:
    return ", ".join(f"{value:.2f}" for value in json.loads(serialized))


def fine_table(key_bits: int) -> str:
    rows = sorted(
        [row for row in fine if row["key_bits"] == key_bits and row["value_bits"] == 2],
        key=lambda row: row["m_over_d"],
    )
    lines = [
        "| m/d | KV bytes/token | PPL mean +/- SD | 95% CI | Logit RMSE | Attention KL | PPL gain / added byte | Gain closed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        marginal = row["ppl_gain_per_added_byte"]
        marginal_text = "--" if marginal == "" else f"{marginal:.3f}"
        lines.append(
            f"| {row['m_over_d']:g} | {row['kv_bytes']} | "
            f"{row['perplexity_mean']:.2f} +/- {row['perplexity_sd']:.2f} | "
            f"[{row['perplexity_ci95_low']:.2f}, {row['perplexity_ci95_high']:.2f}] | "
            f"{row['attention_logit_rmse_mean']:.3f} | "
            f"{row['attention_kl_fp_to_quantized_mean']:.3f} | {marginal_text} | "
            f"{100 * row['fraction_observed_ppl_gain']:.1f}% |"
        )
    return "\n".join(lines)


def frontier_table() -> str:
    lines = [
        "| KV bytes/token | Allocation (K/V, m/d) | PPL | Logit RMSE | Attention KL |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in frontier:
        allocation = f"{row['key_bits']}/{row['value_bits']}, MSE-only"
        lines.append(
            f"| {row['kv_bytes']} | {allocation} | {row['perplexity_mean']:.2f} | "
            f"{row['attention_logit_rmse_mean']:.3f} | "
            f"{row['attention_kl_fp_to_quantized_mean']:.3f} |"
        )
    return "\n".join(lines)


def projection_table() -> str:
    lines = [
        "| Projection | K/V | m/d | Bytes | PPL mean +/- SD | Logit RMSE | Best MSE-only at <= bytes | PPL penalty |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        projection,
        key=lambda row: (row["key_bits"], row["m_over_d"], row["projection_mode"]),
    )
    for row in ordered:
        label = "block-orthogonal" if row["projection_mode"] == "block_orthogonal" else "Gaussian"
        comparator = (
            f"{row['best_mse_only_at_or_below_budget']} at "
            f"{row['best_mse_only_bytes']} B ({row['best_mse_only_ppl']:.2f})"
        )
        lines.append(
            f"| {label} | {row['key_bits']}/{row['value_bits']} | "
            f"{row['m_over_d']:g} | {row['kv_bytes']} | "
            f"{row['perplexity_mean']:.2f} +/- {row['perplexity_sd']:.2f} | "
            f"{row['attention_logit_rmse_mean']:.3f} | {comparator} | "
            f"+{row['ppl_above_best_mse_only']:.2f} |"
        )
    return "\n".join(lines)


def paired_table() -> str:
    selected = [
        comparison("4/2: m=d versus m=2d"),
        comparison("3/2: m=d versus m=2d"),
        comparison("60-byte allocation: 4/2,m=d versus 3/2,m=2d"),
        comparison("68-byte allocation: 5/2,m=d versus 4/2,m=2d"),
        comparison("4/2: MSE only versus m=d"),
        comparison("3/2: MSE only versus m=d"),
    ]
    lines = [
        "| Comparison (right - left PPL) | Mean difference | 95% paired CI | Paired Cohen dz | Seed-level differences |",
        "|---|---:|---:|---:|---|",
    ]
    for row in selected:
        diffs = ", ".join(
            f"{value:.2f}" for value in json.loads(row["individual_paired_differences"])
        )
        lines.append(
            f"| {row['left_config']} -> {row['right_config']} | "
            f"{row['right_minus_left_ppl_mean']:+.2f} | "
            f"[{row['right_minus_left_ci95_low']:+.2f}, "
            f"{row['right_minus_left_ci95_high']:+.2f}] | "
            f"{row['paired_cohens_dz']:+.2f} | {diffs} |"
        )
    return "\n".join(lines)


c60 = comparison("60-byte allocation: 4/2,m=d versus 3/2,m=2d")
c68 = comparison("68-byte allocation: 5/2,m=d versus 4/2,m=2d")
c4 = comparison("4/2: m=d versus m=2d")
c3 = comparison("3/2: m=d versus m=2d")
mech4 = mechanism(4)
mech3 = mechanism(3)
f41, f42 = fine_row(4, 1.0), fine_row(4, 2.0)
f31, f32 = fine_row(3, 1.0), fine_row(3, 2.0)
l1, l2 = length_check

text = f"""# Phase 2 validation: QJL measurement-budget allocation

This decision document is generated by `scripts/render_phase2_validation.py`
from immutable JSONL runs and programmatically generated summaries. The final
classification is **Verdict A — Stop**: larger QJL sketches improve a QJL-based
estimator, but oversized QJL is consistently dominated by scalar-only
allocations at equal or lower measured storage, and the basic `m>d` idea is
already present in the original QJL implementation. The evidence does not
justify larger-model spending on the proposed positive hypothesis.

## 1. Current hypothesis

The candidate hypothesis was that the useful QJL measurement budget is not
universally `m=d`, that it changes with quantization severity, and that spending
extra fixed-cache budget on QJL measurements can sometimes beat spending those
bytes on scalar key/value precision.

Phase 2 separates two claims:

1. **Conditional claim:** given a scalar K/V configuration that already uses
   residual QJL, increasing `m` improves attention and downstream PPL.
2. **Allocation claim:** given a fixed total KV-cache byte budget, extra QJL
   measurements are a competitive use of those bytes.

The first survives. The second—the paper-critical claim—does not.

## 2. Experimental controls

- Model: GPT-2 (`gpt2`, approximately 124M parameters), 12 layers, 12 heads,
  head dimension `d=64`; model revision `{environment['model_revision']}` and
  weight SHA-256 `{environment['model_weight_sha256']}`.
- Data: `Salesforce/wikitext:wikitext-2-raw-v1:test`, first 512 tokens,
  context limit 1024; shard SHA-256 `{environment['dataset_sha256']}`.
- Device/software: FP16 on Apple MPS; PyTorch
  `{environment['packages']['torch']}`, Transformers
  `{environment['packages']['transformers']}`.
- Randomness: QJL projection seeds `11, 23, 37, 53, 71`; model, tokens,
  rotations, scalar codebooks, quantization settings, and evaluation path are
  fixed. Seeds are paired for central QJL comparisons. MSE-only has no QJL
  randomness; duplicated seed-labelled runs are deterministic controls, not
  five independent estimates.
- Measurement normalization: `sqrt(pi/2)/m`; the independent sanity study fit
  QJL MSE slope `-0.999` versus `m`, rejecting accidental fixed-`d`
  normalization.
- Storage: packed scalar indices, packed QJL signs, one FP32 vector norm, an
  additional FP32 residual norm when QJL is active, and value metadata are
  included per token. Shared projections are reported separately. For example,
  4/2-bit Gaussian QJL uses {f41['kv_bytes']} bytes/token at `m=d` and
  {f42['kv_bytes']} at `m=2d`; its shared projections use
  {f41['shared_projection_bytes'] / 2**20:.2f} and
  {f42['shared_projection_bytes'] / 2**20:.2f} MiB total, respectively.
- Scope: 553 allocation runs covering 121 summarized configurations, 80
  fine-sweep runs covering 16 configurations, and 20 post-gate
  block-orthogonal runs covering four configurations.
- Analysis unit: the projection seed, never individual tokens. Central paired
  intervals are two-sided 95% t intervals over five seed-level differences.

The primary allocation and fine sweeps use independent Gaussian projection
rows, matching the local TurboQuant reference's residual QJL. Because the
original QJL paper orthogonalizes projection rows, a separately labelled
block-orthogonal sensitivity control was added after the stop gate.

## 3. Equal-budget results

At the two preregistered exact-byte comparisons, allocating the bytes to scalar
precision won:

- At 60 bytes/token, 4/2-bit `m=d` scored {f41['perplexity_mean']:.2f} +/-
  {f41['perplexity_sd']:.2f} PPL, while 3/2-bit `m=2d` scored
  {f32['perplexity_mean']:.2f} +/- {f32['perplexity_sd']:.2f}. The oversized
  allocation was worse by {c60['right_minus_left_ppl_mean']:.2f} PPL, with 95%
  paired CI [{c60['right_minus_left_ci95_low']:.2f},
  {c60['right_minus_left_ci95_high']:.2f}].
- At 68 bytes/token, 5/2-bit `m=d` scored 35.43 +/- 0.79, while 4/2-bit
  `m=2d` scored {f42['perplexity_mean']:.2f} +/- {f42['perplexity_sd']:.2f}.
  The oversized allocation was worse by {c68['right_minus_left_ppl_mean']:.2f}
  PPL, 95% paired CI [{c68['right_minus_left_ci95_low']:.2f},
  {c68['right_minus_left_ci95_high']:.2f}].

The stronger control removed residual QJL entirely (`m=0`) and used its bytes
for scalar precision. At 4/2 bits, MSE-only scored 39.91 PPL at 48 bytes versus
45.16 for `m=d` at 60 bytes. At 3/2 bits, it scored 57.00 at 40 bytes versus
92.29 for `m=d` at 52 bytes. A high-precision scalar extension closed an
initially truncated frontier: every measured PPL Pareto point through 88 bytes
is MSE-only.

{frontier_table()}

The 6/5-bit MSE-only point at 88 bytes is 0.03 PPL below this short slice's FP16
value (30.93). That tiny reversal is treated as numerical/slice-level variation,
not evidence that quantization improves the model.

![PPL-storage Pareto frontier](results/plots/phase2_storage_pareto_ppl.png)

![Attention-error storage frontier](results/plots/phase2_storage_pareto_logit.png)

### Projection-construction sensitivity

Block orthogonalization substantially improves QJL and was therefore a real
implementation confounder. It does not change the allocation verdict: all four
tested block-orthogonal QJL points remain worse than an MSE-only allocation
using four fewer bytes/token.

{projection_table()}

![Gaussian and block-orthogonal QJL control](results/plots/phase2_projection_control.png)

## 4. Fine-grained `m` results

### 4/2-bit Gaussian residual QJL

{fine_table(4)}

The marginal gain falls from 1.204 PPL per added byte over `1d -> 1.25d` to
0.026 over `2.5d -> 3d`, before a small 0.100 rebound over the wider
`3d -> 4d` interval. The smallest tested point achieving 90% of the observed
`m=d -> 4d` gain is `{mech4['smallest_m_over_d_for_90pct_observed_gain']:g}d`.
There is curvature, but no finite unconstrained optimum was located because PPL
still declines slightly through `4d`.

### 3/2-bit Gaussian residual QJL

{fine_table(3)}

The 90% operating point shifts to
`{mech3['smallest_m_over_d_for_90pct_observed_gain']:g}d`, and marginal gains
remain larger. Conditional on retaining QJL, more aggressive scalar
quantization therefore needs more measurements to suppress estimator harm.
Under an actual storage penalty, however, the observed optimum is `m=0`, not
either conditional knee.

![Fine m sweep and mechanism metrics](results/plots/phase2_fine_m_mechanism.png)

![Individual seed values](results/plots/phase2_fine_m_seed_variance.png)

## 5. Cross-model results

**Not run; generalization is inconclusive.** An OPT-125M setup/download was
attempted, but no valid run completed before the fixed-budget stop gate was
reached. It was terminated without writing a result. Continuing only to collect
a second conditional `m` curve would not rescue an allocation hypothesis that
is already dominated, and would violate the explicit stop criterion.

This is not evidence that replication failed on OPT. It is deliberately absent
evidence after a sequential stop decision.

## 6. Cross-data results

**Not run in Phase 2; robustness across text distributions is inconclusive.**
The initial study's 2048-token WikiText-2 check retained the 4/2-bit effect:
`m=d` was {l1['perplexity_mean']:.2f} +/- {l1['perplexity_std']:.2f} and `m=2d`
was {l2['perplexity_mean']:.2f} +/- {l2['perplexity_std']:.2f}, against FP16
{l1['fp16_perplexity']:.2f}. It extends the same prefix and is neither an
independent slice nor a second corpus. Further data sweeps were stopped for the
same reason as cross-model work.

## 7. Mechanism evidence

The causal chain is consistent with estimator variance reduction, followed by
a nonlinear softmax/downstream response:

- QJL residual RMSE slopes are {mech4['qjl_rmse_loglog_slope']:.3f} at 4/2 and
  {mech3['qjl_rmse_loglog_slope']:.3f} at 3/2 on log-log axes, essentially the
  expected `m^-1/2` behavior. Attention-logit RMSE has the same slopes.
- Across the eight aggregate `m` points, PPL correlates with logit RMSE at
  {mech4['pearson_ppl_vs_logit_rmse']:.3f} (4/2) and
  {mech3['pearson_ppl_vs_logit_rmse']:.3f} (3/2), and with attention KL at
  {mech4['pearson_ppl_vs_attention_kl']:.3f} and
  {mech3['pearson_ppl_vs_attention_kl']:.3f}. These are descriptive
  configuration-level correlations (`n=8`), not causal estimates or token-level
  significance tests.
- Mean residual norm at `m=d` is
  {mech3['m_equals_d_residual_norm_mean'] / mech4['m_equals_d_residual_norm_mean']:.2f}x
  larger at 3/2 than 4/2. More aggressive quantization therefore exposes QJL
  to a larger residual and makes finite-measurement variance more consequential.
- At `m=d`, layers {', '.join(str(x) for x in json.loads(mech4['top3_m_equals_d_kl_layers']))}
  account for {100 * mech4['top3_layers_fraction_of_layer_kl']:.1f}% of summed
  layer attention KL at 4/2; layers
  {', '.join(str(x) for x in json.loads(mech3['top3_m_equals_d_kl_layers']))}
  account for {100 * mech3['top3_layers_fraction_of_layer_kl']:.1f}% at 3/2.
  Error is concentrated, but this one-model observation does not justify an
  adaptive layer algorithm.
- Aggregate RMSE is not sufficient. At 3/2, `m=d` lowers logit RMSE versus
  MSE-only (1.416 versus 1.604) yet worsens attention KL (0.539 versus 0.339)
  and PPL (92.29 versus 57.00). The sign correction removes average bias while
  introducing projection-dependent noise that softmax can amplify. Increasing
  `m`, or orthogonalizing the rows, reduces that harm; scalar reconstruction
  avoids it more efficiently in the tested budget range.

![Layer attention KL](results/plots/phase2_layer_attention_kl.png)

## 8. Literature and novelty audit summary

The full source-by-source matrix is in
[`notes/literature_audit.md`](notes/literature_audit.md). The decisive findings
are:

1. The 2024 QJL paper defines arbitrary `m` and normalization by `1/m`; its
   public implementation already defaults to `m=2d` generally and `m=4d` in
   selected early layers, and sweeps `m/d=0.5..4` for distortion.
2. The expected one-bit measurement-error improvement is longstanding random
   hyperplane/compressive-sensing theory.
3. A 2026 fair-budget preprint already argues that QJL variance on K is
   amplified by softmax and compares MSE-only against MSE+QJL on synthetic
   attention metrics.
4. UltraQuant and current vLLM code explicitly remove QJL from practical
   TurboQuant-style paths.
5. KVQuant, HIGGS, AQUA-KV, and JoLT already occupy major parts of the
   sensitivity-aware and joint bit-allocation space.

The apparently open cell is a broad, end-to-end boundary-of-use study asking
when QJL beats the best scalar-only allocation under exact packed storage. This
repository supplies one small-model negative instance, not enough novelty or
generality for a paper.

## 9. Main threats to validity

- Only GPT-2 and one WikiText-2 prefix support the Phase 2 allocation result.
- The 512-token sample is intentionally cheap and can misestimate effects near
  FP quality; the tiny 6/5-bit improvement over FP is a warning.
- The harness emulates quantized full-sequence attention. It is not an optimized
  packed incremental decode kernel, so MPS wall time does not establish a
  deployment latency frontier. Arithmetic and shared projection storage scale
  linearly with `m`.
- Per-token accounting stores norms as FP32 and follows this harness's packing.
  Production layouts may use lower-precision metadata, padding, outlier tables,
  buffers, or amortization that move exact byte boundaries.
- The primary sweep is Gaussian because it tests TurboQuant-style residual QJL;
  official QJL uses orthogonalized blocks and outlier handling. The post-gate
  block-orthogonal control addresses projection structure only, not the full
  official QJL system.
- Paired projection seeds quantify QJL randomness, not dataset sampling,
  training variation, or model-family variation.
- Layer correlations and the proposed softmax mechanism are observational.

## 10. What failed

- Oversized Gaussian QJL never entered the measured PPL Pareto frontier after
  scalar-only allocations were extended to comparable high budgets.
- The exact 60- and 68-byte tests favored more scalar precision over more QJL
  measurements with paired intervals excluding zero.
- Removing QJL often improved PPL while also reducing storage.
- Stronger block-orthogonal QJL improved absolute quality markedly but still
  lost to MSE-only allocations using fewer bytes.
- The initial `m>d` framing failed the novelty audit: it is already theory- and
  implementation-supported in original QJL work.
- No general predictive rule `m*/d=f(...)` was established.
- Cross-model and cross-data evidence was not collected after the stop gate.

## 11. What survived

- The conditional phenomenon is real on GPT-2: Gaussian `m=2d` beats `m=d` for
  every seed at both 4/2 and 3/2 bits. The paired differences are
  {c4['right_minus_left_ppl_mean']:.2f} PPL (95% CI
  [{c4['right_minus_left_ci95_low']:.2f}, {c4['right_minus_left_ci95_high']:.2f}])
  and {c3['right_minus_left_ppl_mean']:.2f} (95% CI
  [{c3['right_minus_left_ci95_low']:.2f}, {c3['right_minus_left_ci95_high']:.2f}]);
  negative values mean `m=2d` is better.
- Estimator/logit RMSE follows the expected `m^-1/2` scaling with the correct
  normalization.
- Conditional PPL curves are nonlinear, and the 90%-of-observed-gain point
  shifts from `2.5d` at 4/2 to `3d` at 3/2.
- Projection row structure matters substantially; block orthogonalization is a
  real quantitative improvement.
- Attention error is layer-concentrated in this model, and residual magnitude
  helps explain why aggressive quantization remains measurement-limited longer.

### Central seed-level statistics

{paired_table()}

## 12. Final verdict

**Verdict A — Stop.** Two independent stop conditions are satisfied:

1. Equal-budget experiments show oversized QJL is consistently dominated by
   scalar-only allocations in the tested TurboQuant-style system, including
   after a stronger block-orthogonal projection control.
2. The basic `m>d` phenomenon is substantially established by original QJL
   theory and public code, while the fair-budget negative mechanism overlaps
   recent work.

This does not prove QJL is universally inferior. It does show that the proposed
positive allocation paper is not supported strongly enough to justify
escalating compute.

## 13. Exact recommended next step

Stop this project as a positive `m>d` paper and preserve it as a reproducible
negative repository result. Do not create an adaptive-QJL method or buy
larger-model compute. If the question is revisited later, reformulate it first as
“under what measurable conditions, if any, does residual QJL beat MSE-only at a
fixed packed byte budget?” The first—and only—new gate should be a
cross-architecture exact-budget comparison of official-style block-orthogonal
residual QJL against MSE-only. Proceed beyond that single gate only if QJL
actually enters the quality-memory Pareto frontier.

## Research Decision

- **Is there a real phenomenon?** Yes, conditionally: more QJL measurements
  reduce QJL/logit error and improve GPT-2 PPL when QJL is held in the pipeline.
- **Is there evidence of a useful measurement-budget rule?** No. The conditional
  knees do not improve the fixed-storage allocation frontier.
- **Does oversized QJL help at equal storage?** No in every tested Gaussian and
  block-orthogonal comparison; scalar-only allocations achieve lower PPL with
  equal or fewer bytes.
- **Does the useful `m` depend on quantization severity?** Yes within GPT-2's
  conditional QJL curves (`2.5d` versus `3d` for 90% of observed gain), but this
  is not a useful fixed-budget optimum.
- **Does the result generalize beyond GPT-2?** Inconclusive; the stop gate was
  reached before a valid second-model result.
- **Does the work appear novel after the literature audit?** No for the proposed
  positive claim; only the narrow multi-seed, equal-storage negative replication
  appears partially differentiated.
- **Is there a defensible paper here today?** No.
- **If yes, what exactly is the paper about?** Not applicable.
- **If not yet, what single missing experiment is most decisive?** For a
  reformulated boundary-of-use question, a cross-architecture exact-budget
  comparison of official-style block-orthogonal residual QJL versus MSE-only.
  It would not rescue the original novelty claim by itself.
- **Should we spend money on larger-model experiments now?** No.
"""

(ROOT / "PHASE2_VALIDATION.md").write_text(text)
print("wrote PHASE2_VALIDATION.md from generated summaries")
