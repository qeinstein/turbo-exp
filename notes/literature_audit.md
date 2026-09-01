# Phase 2 literature and public-code audit

Audit date: 2026-09-01. This is a claim audit, not a citation-count survey. I
prioritized primary papers and source repositories, separated peer-reviewed
work from recent preprints and implementation comments, and searched
specifically for variable QJL measurement count `m`, `m>d`, downstream quality
as `m` varies, fair-storage comparisons, bit-width interactions, and
layer-adaptive allocation.

Here `d` is the attention-head/key-vector dimension and `m` is the number of
one-bit QJL measurements. These must not be conflated with a paper's total
bits-per-coordinate notation.

## Directly relevant works

| Work / source | Exact contribution | Is `m` variable? | Is `m>d` considered? | Downstream LLM quality as `m` varies? | Equal total storage? | Bit-width-dependent optimum or layer allocation? | Overlap and remaining gap |
|---|---|---:|---:|---:|---:|---:|---|
| [QJL (Zandieh, Daliri, Han), arXiv:2406.03482v2, 2024](https://arxiv.org/html/2406.03482) | Defines QJL for arbitrary positive `d,m`, with the estimator normalized by `sqrt(pi/2)/m`, proves unbiasedness and a concentration bound in `m`, and evaluates 3-bit KV compression on LongBench. | Yes, in the definition, theory, and public distortion script. | Yes. The public Llama configuration uses head `d=128`, normally `m=256` and `m=512` for the first 15 layers. | No controlled PPL/LongBench sweep over `m`. The public distortion script sweeps `m=64,...,512` (`m/d=0.5,...,4`) for cached attention-score error with one repetition. | No scalar-bits-versus-QJL-measurements allocation study. | Yes, a hand-set layer split (`2d` generally, `4d` in early layers), but no learned or budget-optimal allocation. | The fact that `m>d`, including `2d` and `4d`, is useful is not novel. The open empirical cell is a replicated, downstream, equal-storage allocation test—not the existence of the effect. |
| [TurboQuant (Zandieh et al.), arXiv:2504.19874, 2025 / ICLR 2026](https://arxiv.org/html/2504.19874) | Combines an MSE-oriented rotated scalar quantizer at `b-1` bits with a one-bit QJL sketch of the residual to make the inner-product estimator unbiased; evaluates KV-cache and ANN tasks. | Not in the stated residual algorithm: its sign vector has length `d`. | No residual-`m` sweep. | No. | It compares algorithms at stated overall bit rates, but does not test reallocating the residual sign budget into higher scalar K/V precision. | No optimum in `m`; fixed construction. | This is the exact pipeline under test. It motivates the bias correction but does not establish that its variance is beneficial after softmax at a fixed byte budget. |
| [Official QJL source, commit `648b364`](https://github.com/amirzandieh/QJL/tree/648b3641f96b6e95e091217220b94e4739fd4d82) | Implements QJL with packed signs, outlier splitting, blockwise QR-orthogonalized projection chunks, and CUDA scoring. | Yes. | Yes, by concatenating `d`-wide orthogonal blocks. | Only a one-repetition attention-score distortion sweep in [`plot_distortion.py`](https://github.com/amirzandieh/QJL/blob/648b3641f96b6e95e091217220b94e4739fd4d82/plot_distortion.py#L30-L40); end-to-end defaults are fixed. | No. | Fixed early/general layer split in [`run_longbench.py`](https://github.com/amirzandieh/QJL/blob/648b3641f96b6e95e091217220b94e4739fd4d82/run_longbench.py#L40-L80). | Strong novelty threat and an implementation warning: the primary Gaussian sweep is not equivalent to official QJL experiments because the latter orthogonalize each projection block. Phase 2 therefore added a separate block-orthogonal control. |
| [Statistical Inference and Quality Measures of KV Cache Quantisations Inspired by TurboQuant (D'Alberto), arXiv:2605.08114, 2026 preprint](https://arxiv.org/html/2605.08114) | Compares MSE-only and MSE+QJL-style schemes at a fair bit budget on synthetic distributions/dimensions; argues that QJL on K inflates inner-product variance and softmax amplifies it, and reports lower attention KL for the non-QJL alternative. | No systematic measurement-count sweep. | No. | No downstream LLM PPL or task evaluation. | Yes—the central comparison spends the bit on scalar precision versus QJL. | Sweeps total bit budgets, not `m`, and finds metric-dependent crossovers; no learned layer allocation. | This is the closest conceptual prior to the fixed-budget negative result and substantially weakens novelty. Its limitation is that it is a recent single-author preprint with synthetic attention experiments rather than a multi-seed end-to-end LLM study. |
| [UltraQuant (Chakrabarti et al.), arXiv:2606.20474, 2026 preprint](https://arxiv.org/html/2606.20474) | Builds a serving-oriented 4-bit KV path around TurboQuant-style rotation/codebooks, asymmetric K/V choices, block scales, QJL removal, and native AMD matrix-core formats; measures task quality and serving performance. | No—QJL is removed. | No. | No. | It evaluates deployment tradeoffs, but not an `m` allocation curve. | Asymmetric K/V, not layer-adaptive `m`. | A strong practical novelty threat: QJL removal is already a documented design choice in a current end-to-end deployment study. It does not answer whether oversized QJL ever wins a fixed byte budget. |
| [vLLM TurboQuant configuration, commit `55aa766`](https://github.com/vllm-project/vllm/blob/55aa766dc8c1a7d739ef90c5580ca4b91050b35f/vllm/model_executor/layers/quantization/turboquant/config.py#L44-L92) | Current public implementation exposes MSE-only K/V presets and explicitly omits QJL, attributing the choice to variance amplification through softmax. | No. | No. | No controlled public `m` sweep in this configuration. | No formal allocation experiment in the source file. | K/V-asymmetric presets; no `m`. | Important engineering evidence, not a peer-reviewed result. The source comment's “community consensus” claim is not independently sufficient evidence and is treated only as a novelty/deployment signal. |
| [TurboESM (Hu, Wang, Liu), arXiv:2603.26110, 2026 preprint](https://arxiv.org/html/2603.26110) | Adapts rotation, calibrated LUTs, and a one-bit residual correction to ESM-2 650M; reports a no-correction cosine-similarity ablation, packed-memory accounting, and a fused kernel. | No. | No. | No PPL/task sweep over `m`; one model and fixed residual correction. | It reports component memory, but not scalar-bits-versus-QJL at equal storage. | Head-wise SVD calibration, not `m` allocation. | Shows that a residual correction can help a different modality/distribution, but does not establish the natural-language LLM allocation claim. Its metric and implementation are not directly comparable to this study. |
| [A JoLT for the KV cache (Krishnan, Schulz), arXiv:2607.12550v3, 2026 preprint](https://arxiv.org/abs/2607.12550) | Jointly allocates Tucker ranks and rotated-residual bit-widths per layer group and separately for K/V under one byte budget; evaluates PPL and tasks on Mistral-7B and Llama-2-13B. | No QJL `m`. | Not applicable. | Not applicable. | Yes. | Yes, explicit Lagrangian joint allocation. | A major novelty threat to any broad “adaptive measurement/bit allocation” paper. A QJL-specific result would need to beat or explain a distinct regime rather than rebrand joint budget allocation. |
| [HIGGS / Linearity Theorem (Malinovskii et al.), arXiv:2411.17525 / NAACL 2025](https://arxiv.org/abs/2411.17525) | Relates layer reconstruction error to PPL and solves nonuniform per-layer quantization-level selection by dynamic programming under a compression constraint. | No; weight quantization rather than QJL/KV cache. | Not applicable. | Not applicable. | Yes for its weight-compression setting. | Yes. | Establishes that principled layerwise bit allocation and PPL-linked error objectives are already developed ideas. It is adjacent rather than a direct answer to QJL measurement allocation. |
| [AQUA-KV (Shutova et al.), arXiv:2501.19392 / ICML 2025](https://arxiv.org/abs/2501.19392) | Uses compact predictors/adapters to exploit inter-layer K/V dependencies and quantizes prediction residuals, calibrated for near-lossless 2–2.5-bit KV compression. | No QJL `m`. | Not applicable. | Not applicable. | Reports compression/quality tradeoffs, not the same QJL allocation. | Adaptive to learned residual structure, not `m`; not primarily a layer-budget optimizer. | Occupies the broader “spend bits according to residual structure” space and supplies a strong larger-model baseline for any future allocation paper. |
| [KVQuant (Hooper et al.), arXiv:2401.18079v6 / NeurIPS 2024](https://arxiv.org/abs/2401.18079) | Introduces per-channel pre-RoPE key quantization, per-layer sensitivity-weighted nonuniform datatypes, and dense-and-sparse outlier handling; evaluates WikiText-2/C4 PPL and kernels. | No. | Not applicable. | Not applicable. | Quality/compression curves, not QJL reallocation. | Yes, per-layer sensitivity-weighted datatype design. | Another prior on layer-adaptive KV precision and a required comparison if work moves beyond the narrow QJL question. |
| [KIVI (Liu et al.), arXiv:2402.02750 / ICML 2024](https://arxiv.org/abs/2402.02750) | Establishes asymmetric per-channel K and per-token V 2-bit quantization across several LLM families with a hardware-friendly implementation. | No. | Not applicable. | Not applicable. | Reports effective memory and downstream quality, not QJL allocation. | K/V asymmetry, not layer-adaptive. | Makes “use different precision mechanisms for K and V” well established; useful baseline, not direct overlap with variable `m`. |
| [PolarQuant (Han et al.), arXiv:2502.02617, 2025](https://arxiv.org/abs/2502.02617) | Uses random preconditioning plus a polar representation to avoid normalization overhead and evaluates long-context KV compression. | No. | Not applicable. | Not applicable. | Compression/quality comparison, not QJL allocation. | No `m`; no central layer allocation claim. | Alternative no-QJL path demonstrating that QJL measurement allocation is only one of several cache-rate/distortion choices. |
| [Jacques et al., “Robust 1-Bit Compressive Sensing,” arXiv:1104.3160](https://arxiv.org/abs/1104.3160) and [Plan & Vershynin, “Dimension reduction by random hyperplane tessellations,” arXiv:1111.4452](https://arxiv.org/abs/1111.4452) | Develop sign-of-random-projection measurement/recovery and binary embedding theory, including error dependence on measurement count. | Yes. | Measurement counts can exceed ambient/intrinsic dimension. | No transformers or downstream language metrics. | Study sampling/bit-depth tradeoffs in signal models, not TurboQuant packed KV storage. | No LLM bit-width or layer allocation. | Broad theory makes “more one-bit random measurements reduce estimation error” unsurprising. Novelty must come from downstream allocation behavior, not the scaling law itself. |

## Public-code checks that change interpretation

- The QJL paper's formal estimator is explicitly
  `sqrt(pi/2) / m * ||k|| * <S q, sign(S k)>`; arbitrary `m` and the required
  normalization are already in [Definition 3.1](https://arxiv.org/html/2406.03482#S3).
  Its distortion guarantee also improves with measurement count.
- The official QJL source at commit `648b364` uses `d=128, m=256` normally and
  `m=512` for the initial 15 layers by default. It therefore operationalizes
  `2d` and `4d`, not merely `m=d`.
- Its [`QJLSketch`](https://github.com/amirzandieh/QJL/blob/648b3641f96b6e95e091217220b94e4739fd4d82/models/llama2_utils_qjl.py#L7-L31)
  QR-orthogonalizes each `d`-wide block and scales it by `sqrt(d)`. The paper
  says all experiments use orthogonalized rows. This directly motivated the
  separate block-orthogonal control in Phase 2.
- The CUDA score kernel divides by the actual sketch dimension, not `d`, in
  [`qjl_score_kernel.cu`](https://github.com/amirzandieh/QJL/blob/648b3641f96b6e95e091217220b94e4739fd4d82/qjl_kernel/csrc/qjl_score_kernel.cu#L153-L154).
- The official distortion script varies `m/d` from 0.5 to 4 but sets `rep=1`
  and measures relative attention-score error, not PPL. It is evidence that
  variable `m` was contemplated, not evidence of a stable downstream optimum.
- The local TurboQuant reference is pinned at commit
  `1ea420dc2d5184531023fd4a6c1356314c80a04b`; it uses a square Gaussian QJL
  residual projection by default. This repository was not modified.
- The current vLLM main branch was inspected at commit
  `55aa766dc8c1a7d739ef90c5580ca4b91050b35f`. Its omission of QJL is useful
  public implementation evidence, but its source-code assertion about five
  independent groups is not treated as a substitute for traceable experiments.

## Known

1. QJL is mathematically defined for arbitrary `m`, with estimator normalization
   proportional to `1/m` and concentration improving as measurements increase.
2. `m>d` is already present in the original QJL public implementation and its
   default end-to-end configuration: `2d` generally and `4d` in selected layers.
3. The public QJL code already varies `m/d` for attention-score distortion, and
   already uses a coarse layer-dependent measurement allocation.
4. One-bit random-measurement error improving with the number of measurements is
   longstanding one-bit embedding/compressive-sensing theory.
5. Recent post-TurboQuant work and the current vLLM implementation explicitly
   consider removing QJL because unbiased residual correction can add variance
   that softmax amplifies.
6. Equal-budget and layer-/component-adaptive quantization are crowded areas:
   HIGGS, KVQuant, AQUA-KV, and JoLT already cover important formulations.

## Apparently open

1. A well-powered, end-to-end natural-language LLM study that jointly sweeps QJL
   measurement count and scalar K/V precision at exactly accounted total cache
   storage, with multiple projection seeds and downstream PPL/tasks.
2. Conditions under which QJL residual correction beats the best MSE-only
   allocation at the same packed byte budget, if such conditions exist.
3. Whether that condition changes reliably with architecture, head dimension,
   layer, context distribution/length, and quantization severity.
4. A causal downstream account connecting residual norm and QJL estimator
   variance to attention KL and PPL, tested across models rather than inferred
   from a single model.

These cells are “apparently open” after this targeted audit, not proofs that no
unindexed concurrent work exists.

## Our possible contribution

The defensible contribution from the present repository is narrow: a
reproducible negative small-model result showing that conditional gains from
larger `m` can coexist with domination by scalar-only allocations at equal or
lower packed storage, with five projection seeds, attention/logit diagnostics,
and a projection-construction sensitivity control.

That is useful falsification evidence, but on GPT-2 and a short WikiText-2 slice
it is not a strong standalone paper. It neither introduces `m>d`, discovers the
expected measurement-error scaling, nor establishes a general allocation law.

## Novelty threats

1. **Fatal to the original framing:** original QJL code already uses `m=2d` and
   `m=4d`; “oversized QJL helps” cannot be claimed as new.
2. **Direct fair-budget overlap:** arXiv:2605.08114 already argues that spending
   the bit on MSE precision can beat QJL on K under a fair budget and proposes
   essentially the same variance/softmax mechanism, albeit synthetically.
3. **Deployment overlap:** UltraQuant and current vLLM remove QJL in practical
   TurboQuant-style paths.
4. **Allocation-space crowding:** KVQuant, HIGGS, AQUA-KV, and JoLT already make
   sensitivity-aware or jointly optimized allocation central contributions.
5. **Method sensitivity:** official QJL uses block-orthogonal projections;
   Gaussian-only findings would be too implementation-specific. The added
   control improves QJL markedly but does not reverse fixed-budget domination.
6. **Evidence scale:** the current end-to-end evidence is one 124M-parameter
   architecture, one dataset, and 512 tokens for the allocation study. It is
   decision-grade for stopping this hypothesis, not for a broad universal claim.

## Audit conclusion

The proposed positive paper claim is not novel and is empirically unsupported
at fixed storage in this study. The conditional curve—more QJL measurements
reduce estimator error and often improve downstream PPL—is expected from theory
and already reflected in original QJL code. The potentially publishable open
question is instead a boundary-of-use result: when, if ever, a QJL residual
sketch beats allocating those bytes to a lower-variance scalar reconstruction.
The present evidence says not to spend larger-model compute on the original
positive hypothesis.
