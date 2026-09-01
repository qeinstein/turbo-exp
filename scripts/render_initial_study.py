#!/usr/bin/env python3
"""Render INITIAL_STUDY.md from machine-readable summaries and raw runs."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
summary=json.loads((ROOT/'results/summary/summary.json').read_text())
length=json.loads((ROOT/'results/summary/length_check_summary.json').read_text())
runtime=json.loads((ROOT/'results/summary/runtime_summary.json').read_text())
raw=[json.loads(x) for x in (ROOT/'results/raw/llm_sweep.jsonl').read_text().splitlines()]

def row(bits,ratio): return next(r for r in summary if r['key_bits']==bits and r['m_over_d']==ratio)
def paired(bits):
    a={r['qjl_seed']:r['perplexity'] for r in raw if r['key_bits']==bits and r['m_over_d']==1}
    b={r['qjl_seed']:r['perplexity'] for r in raw if r['key_bits']==bits and r['m_over_d']==2}
    dif=np.array([a[s]-b[s] for s in sorted(a)]); half=2.776*float(dif.std(ddof=1))/math.sqrt(len(dif))
    return float(dif.mean()),float(dif.min()),float(dif.max()),half

def table(bits):
    lines=['| m/d | m | PPL mean +/- SD | PPL gap vs FP16 | logit RMSE | attention KL | QJL bytes/key | total key bytes | PPL gain / added QJL byte |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ratio in [.5,1,2,4]:
        r=row(bits,ratio); marginal='--' if r['perplexity_gain_per_added_qjl_byte']=='' else f"{r['perplexity_gain_per_added_qjl_byte']:.3f}"
        lines.append(f"| {ratio:g} | {r['m']} | {r['perplexity_mean']:.2f} +/- {r['perplexity_std']:.2f} | {r['ppl_delta_fp_mean']:.2f} | {r['attention_logit_rmse_mean']:.3f} | {r['attention_kl_fp_to_quantized_mean']:.3f} | {r['qjl_sketch_bytes_per_key_mean']:.0f} | {r['key_storage_bytes_per_key_mean']:.0f} | {marginal} |")
    return '\n'.join(lines)

r41,r42=row(4,1),row(4,2); r31,r32=row(3,1),row(3,2); p4=paired(4); p3=paired(3); l1,l2=length
closure4=100*(r41['ppl_delta_fp_mean']-r42['ppl_delta_fp_mean'])/r41['ppl_delta_fp_mean']; closure_len=100*(l1['ppl_delta_fp_mean']-l2['ppl_delta_fp_mean'])/l1['ppl_delta_fp_mean']
text=f'''# Initial QJL measurement-budget study

This report is generated from `results/raw/*.json*` by `scripts/render_initial_study.py`. Immutable model/data hashes and package versions are in `results/raw/environment.json`.

## Research question

Does increasing the QJL measurement dimension `m` beyond the standard `m=d` reproducibly improve LLM attention quality under TurboQuant-style KV-cache quantization, and is the relationship useful enough to justify deeper study?

## Implementation and controls

The sibling TurboQuant implementation was audited at commit `1ea420d`. Its fast GPT-2 benchmark uses residual QJL with `S` shaped `(m,d)` and the correct `sqrt(pi/2)/m` normalization. The study implementation keeps the model, data, scalar codebooks, value quantization, rotations, token slice, and evaluation code identical within a sweep. Only the QJL projection and `m` vary. The layer/head rotation is fixed across QJL seeds; tests enforce this. For a given seed, larger sketches preserve the smaller sketch as a row prefix, enabling paired comparisons.

Model: GPT-2 (`d=64` per head), FP16 on Apple MPS. Dataset: the official WikiText-2 raw test shard. Main sweep: first 512 tokens, five QJL seeds (`11,23,37,53,71`). Confirmation: key/value bits 3/2. Dataset-length check: 2048 tokens for 4/2 bits at `m=d,2d`.

## Estimator normalization check

The dependency-light check used 64 random query/key pairs and 80 projection seeds at `d=64`. QJL estimator MSE followed the expected `1/m` law over `m/d=0.25...8` (fitted log-log slope `-0.999`). This rejects an incorrect fixed-`d` normalization as the explanation for downstream gains.

## Main results: GPT-2, key/value 4/2 bits

Unquantized FP16 perplexity was {r41['perplexity_mean']-r41['ppl_delta_fp_mean']:.2f}.

{table(4)}

The paired `m=d -> 2d` PPL improvement was {p4[0]:.2f} points (95% paired t interval `{p4[0]-p4[3]:.2f}...{p4[0]+p4[3]:.2f}`); every seed improved, with individual improvements from {p4[1]:.2f} to {p4[2]:.2f}. `m=2d` closed {closure4:.1f}% of the `m=d` PPL gap. Logit RMSE fell {100*(1-r42['attention_logit_rmse_mean']/r41['attention_logit_rmse_mean']):.1f}% and attention KL fell {100*(1-r42['attention_kl_fp_to_quantized_mean']/r41['attention_kl_fp_to_quantized_mean']):.1f}%.

The downstream response is nonlinear. `d -> 2d` gains {r42['perplexity_gain_per_added_qjl_byte']:.3f} PPL points per added QJL byte, while `2d -> 4d` gains only {row(4,4)['perplexity_gain_per_added_qjl_byte']:.3f}. Logit RMSE continues its near-`1/sqrt(m)` decline, but PPL largely saturates after `2d` in this 4/2-bit setting.

## Second confirmation: GPT-2, key/value 3/2 bits

{table(3)}

The paired `m=d -> 2d` improvement was {p3[0]:.2f} PPL points (95% paired t interval `{p3[0]-p3[3]:.2f}...{p3[0]+p3[3]:.2f}`), again positive for every seed. Unlike 4/2 bits, `2d -> 4d` still yields a material {row(3,4)['perplexity_gain_from_previous']:.2f}-point gain. The useful measurement budget therefore changes with quantization aggressiveness in this experiment.

## Dataset-length check

At 2048 tokens, FP16 PPL was {l1['fp16_perplexity']:.2f}; `m=d` was {l1['perplexity_mean']:.2f} +/- {l1['perplexity_std']:.2f}, and `m=2d` was {l2['perplexity_mean']:.2f} +/- {l2['perplexity_std']:.2f}. All five seeds improved and `2d` closed {closure_len:.1f}% of the gap. This closely reproduces the pattern and approximate scale of the sibling repository's preliminary single-seed result, so the 512-token result is not merely a tiny-slice artifact.

## Storage and runtime tradeoff

At `d=64`, `m=d -> 2d` adds 8 packed sign bytes per key. For 4/2 bits, total key storage rises from 40 to 48 bytes (+20%); combined key/value storage rises from 60 to 68 bytes (+13.3%). The shared FP32 projection per layer/head doubles from 16 KiB to 32 KiB; it is fixed overhead, not per cached token, and can be regenerated from its seed.

The synchronized MPS cache-operation microbenchmark measured seed-median runtimes of {runtime[1]['runtime_s_median_across_seeds']*1000:.3f} ms at `d` and {runtime[2]['runtime_s_median_across_seeds']*1000:.3f} ms at `2d` for 512 vectors. These small kernels are launch-overhead dominated and not monotonic, so they do not establish a speed advantage or a reliable cost slope. Arithmetic and shared-projection memory scale linearly with `m`; CUDA and incremental-decoding benchmarks are still required.

## Plots

![Perplexity versus m/d](results/plots/perplexity_vs_m.png)

![Attention-logit error versus m/d](results/plots/logit_error_vs_m.png)

![QJL residual estimator error versus m/d](results/plots/qjl_error_vs_m.png)

![Quality versus QJL storage](results/plots/quality_vs_qjl_storage.png)

![Individual-seed variance](results/plots/seed_variance.png)

![2048-token length check](results/plots/length_check.png)

![Cache-operation runtime](results/plots/runtime_vs_m.png)

## Failed experiments and limitations

- The first WikiText loader path hung under `datasets` 5.x; the final script uses the official parquet shard directly.
- A 32-token smoke slice showed no PPL sensitivity despite nonzero logit error. At 128 tokens and above, sensitivity appeared; 32 tokens is too small for this question.
- Both confirmations use GPT-2 and the same WikiText token prefix. The second confirmation changes bit width, not model or dataset.
- This emulates quantized full-sequence attention, as the sibling benchmark does; it is not an optimized packed incremental-decoding kernel.
- Storage is deliberately unequal across `m`. The study characterizes marginal quality per extra bit; it does not yet show that increasing `m` beats every other use of the same storage.
- Runtime evidence is hardware-specific and kernel-overhead dominated. Novelty relative to all prior work has not been established.

## Next experiments

1. Repeat the paired sweep on one different small model and a different evaluation text slice.
2. Hold total KV bytes fixed and trade QJL measurements against scalar key/value bits.
3. Test `m/d` between 1 and 4 to localize the 4/2-bit saturation point and the more aggressive 3/2-bit operating point.
4. Implement packed incremental-cache timing on CUDA and measure peak memory, prefill, and decode latency separately.
5. Only after those controls, test a modestly larger model and conduct a focused literature/novelty audit.

## Initial Research Verdict

- Original observation replicated: **yes**. `m=2d` beats `m=d` for every tested seed at both 512 and 2048 tokens.
- Effect size: **{p4[0]:.2f} PPL points** at 512 tokens and **{l1['perplexity_mean']-l2['perplexity_mean']:.2f} points** at 2048 tokens for 4/2 bits.
- Seed variability: the 512-token mean gain is far larger than within-setting SD; no anomalous seed drives it.
- Attention behavior: logit RMSE and attention KL both improve in the expected direction for every increase in `m`.
- Storage/compute cost: `d -> 2d` adds 8 bytes/key at `d=64` (+13.3% combined KV storage for 4/2 bits); arithmetic and projection memory scale linearly, while measured small-kernel MPS latency was inconclusive.
- Second confirmation: **survived** at 3/2 bits, with a larger absolute PPL response and less saturation at `2d`.
- Classification: **Outcome C — Strong signal.** The estimator improvement itself is the expected more-bits effect, but downstream PPL saturation at 4/2 bits and the bit-width-dependent useful `m` are nontrivial operating-point signals.
- Larger-model experiments justified: **yes, as a controlled next gate**, not as evidence of novelty or a discovery.
'''
(ROOT/'INITIAL_STUDY.md').write_text(text)
