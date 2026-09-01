#!/usr/bin/env python3
"""Check QJL unbiasedness and empirical m scaling without model downloads."""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from qjlstudy.qjl_numpy import encode, estimate, projection, sketch_storage_bytes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4, 8])
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--trials", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--output", type=Path, default=Path("results/raw/qjl_sanity.json"))
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    q = rng.standard_normal((a.trials, a.d))
    k = rng.standard_normal((a.trials, a.d))
    truth = np.sum(q * k, axis=1)
    rows = []
    for ratio in a.ratios:
        m = max(1, round(a.d * ratio))
        errors = []
        biases = []
        for seed in range(a.seeds):
            S = projection(a.d, m, seed)
            signs, norms = encode(S, k)
            est = np.array([estimate(S, qi, si, ni) for qi, si, ni in zip(q, signs, norms)])
            errors.append(float(np.mean((est - truth) ** 2)))
            biases.append(float(np.mean(est - truth)))
        rows.append({"study": "qjl_sanity", "d": a.d, "m": m, "m_over_d": m/a.d,
                     "projection_seeds": a.seeds, "pair_trials": a.trials,
                     "mse_mean": float(np.mean(errors)), "mse_std": float(np.std(errors, ddof=1)),
                     "bias_mean": float(np.mean(biases)), "qjl_bytes_per_key": sketch_storage_bytes(m)})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print(f"m/d={r['m_over_d']:>5.2f} m={r['m']:>3} mse={r['mse_mean']:.4g} +/- {r['mse_std']:.2g} bias={r['bias_mean']:.3g}")
    loglog_slope = np.polyfit(np.log([r['m'] for r in rows]), np.log([r['mse_mean'] for r in rows]), 1)[0]
    print(f"log(MSE) vs log(m) slope: {loglog_slope:.3f} (ideal approximately -1)")


if __name__ == "__main__":
    main()
