#!/usr/bin/env python3
"""Generate Phase 2 decision tables and plots solely from immutable JSONL runs."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np


RAW = Path("results/raw")
SUMMARY = Path("results/summary")
PLOTS = Path("results/plots")
SEEDS = (11, 23, 37, 53, 71)
METRICS = (
    "perplexity",
    "ppl_delta_fp",
    "attention_logit_rmse",
    "qjl_residual_rmse",
    "attention_kl_fp_to_quantized",
    "residual_norm_mean",
    "residual_norm_std",
    "key_norm_mean",
    "runtime_s",
)


def read_jsonl(name: str) -> list[dict]:
    path = RAW / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sample_sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def ci95(values: list[float]) -> tuple[float, float]:
    """Two-sided t interval for five seed-level observations."""
    if len(values) < 2 or sample_sd(values) == 0:
        value = mean(values)
        return value, value
    try:
        from scipy.stats import t

        critical = float(t.ppf(0.975, len(values) - 1))
    except ImportError:
        critical = 2.776 if len(values) == 5 else 1.96
    half_width = critical * sample_sd(values) / math.sqrt(len(values))
    return mean(values) - half_width, mean(values) + half_width


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["model"],
            row["dataset"],
            row["tokens"],
            row["token_offset"],
            row["key_bits"],
            row["value_bits"],
            row["m_over_d"],
        )
        grouped[key].append(row)
    output = []
    for key, runs in sorted(grouped.items()):
        runs = sorted(runs, key=lambda run: run["qjl_seed"])
        observed_seeds = tuple(run["qjl_seed"] for run in runs)
        expected = (SEEDS[0],) if first_is_mse_only(runs) and len(runs) == 1 else SEEDS
        if observed_seeds != expected:
            raise ValueError(f"unexpected seeds for {key}: {[run['qjl_seed'] for run in runs]}")
        first = runs[0]
        record = {
            "model": key[0],
            "dataset": key[1],
            "tokens": key[2],
            "token_offset": key[3],
            "key_bits": key[4],
            "value_bits": key[5],
            "m_over_d": key[6],
            "m": first["m"],
            "n_seeds": len(runs),
            "kv_bytes": first["kv_storage_bytes_per_token"],
            "key_bytes": first["key_storage_bytes_per_key"],
            "value_bytes": first["value_storage_bytes_per_value"],
            "qjl_bytes": first["qjl_sketch_bytes_per_key"],
            "shared_projection_bytes": first["shared_projection_bytes_total"],
            "fp16_perplexity": first["fp16_perplexity"],
            "seed_perplexities": json.dumps([run["perplexity"] for run in runs]),
        }
        for metric in METRICS:
            values = [float(run[metric]) for run in runs]
            lo, hi = ci95(values)
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_sd"] = sample_sd(values)
            record[f"{metric}_ci95_low"] = lo
            record[f"{metric}_ci95_high"] = hi
        output.append(record)
    return output


def first_is_mse_only(runs: list[dict]) -> bool:
    return runs[0]["m_over_d"] == 0


def add_fine_metrics(summary: list[dict]) -> list[dict]:
    by_precision: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in summary:
        by_precision[(record["key_bits"], record["value_bits"])].append(record)
    output = []
    for precision, records in sorted(by_precision.items()):
        records.sort(key=lambda record: record["m_over_d"])
        initial = records[0]["perplexity_mean"]
        final = records[-1]["perplexity_mean"]
        possible_gain = initial - final
        previous = None
        for record in records:
            current = dict(record)
            current["fraction_observed_ppl_gain"] = (
                (initial - current["perplexity_mean"]) / possible_gain if possible_gain else 0.0
            )
            if previous is None:
                current["added_bytes_from_previous"] = ""
                current["ppl_gain_from_previous"] = ""
                current["ppl_gain_per_added_byte"] = ""
                current["logit_rmse_gain_per_added_byte"] = ""
            else:
                added = current["kv_bytes"] - previous["kv_bytes"]
                current["added_bytes_from_previous"] = added
                current["ppl_gain_from_previous"] = previous["perplexity_mean"] - current["perplexity_mean"]
                current["ppl_gain_per_added_byte"] = current["ppl_gain_from_previous"] / added
                current["logit_rmse_gain_per_added_byte"] = (
                    previous["attention_logit_rmse_mean"] - current["attention_logit_rmse_mean"]
                ) / added
            output.append(current)
            previous = current
    return output


def exact_budget_winners(summary: list[dict]) -> list[dict]:
    winners = []
    for budget in sorted({record["kv_bytes"] for record in summary}):
        candidates = [record for record in summary if record["kv_bytes"] == budget]
        winner = min(candidates, key=lambda record: record["perplexity_mean"])
        winners.append(
            {
                "kv_bytes": budget,
                "key_bits": winner["key_bits"],
                "value_bits": winner["value_bits"],
                "m_over_d": winner["m_over_d"],
                "perplexity_mean": winner["perplexity_mean"],
                "perplexity_sd": winner["perplexity_sd"],
                "attention_logit_rmse_mean": winner["attention_logit_rmse_mean"],
                "attention_kl_mean": winner["attention_kl_fp_to_quantized_mean"],
                "n_candidates": len(candidates),
            }
        )
    return winners


def pareto_frontier(summary: list[dict], metric: str) -> list[dict]:
    frontier = []
    best = math.inf
    for record in sorted(summary, key=lambda item: (item["kv_bytes"], item[metric])):
        if record[metric] < best:
            frontier.append(record)
            best = record[metric]
    return frontier


def paired_comparison(
    rows_by_config: dict[tuple[int, int, float], list[dict]],
    name: str,
    left: tuple[int, int, float],
    right: tuple[int, int, float],
) -> dict:
    left_runs = {run["qjl_seed"]: run for run in rows_by_config[left]}
    right_runs = {run["qjl_seed"]: run for run in rows_by_config[right]}
    if set(left_runs) != set(right_runs) or tuple(sorted(left_runs)) != SEEDS:
        raise ValueError(f"unpaired runs for {name}")
    differences = [right_runs[seed]["perplexity"] - left_runs[seed]["perplexity"] for seed in SEEDS]
    lo, hi = ci95(differences)
    difference_sd = sample_sd(differences)
    return {
        "comparison": name,
        "left_config": f"{left[0]}/{left[1]},m={left[2]}d",
        "right_config": f"{right[0]}/{right[1]},m={right[2]}d",
        "left_kv_bytes": left_runs[SEEDS[0]]["kv_storage_bytes_per_token"],
        "right_kv_bytes": right_runs[SEEDS[0]]["kv_storage_bytes_per_token"],
        "right_minus_left_ppl_mean": mean(differences),
        "right_minus_left_ppl_sd": difference_sd,
        "right_minus_left_ci95_low": lo,
        "right_minus_left_ci95_high": hi,
        "paired_cohens_dz": mean(differences) / difference_sd if difference_sd else "inf",
        "individual_paired_differences": json.dumps(differences),
    }


def layer_summary(rows_by_config: dict[tuple[int, int, float], list[dict]]) -> list[dict]:
    output = []
    for config in ((3, 2, 0.0), (3, 2, 1.0), (3, 2, 2.0), (4, 2, 0.0), (4, 2, 1.0), (4, 2, 2.0)):
        for layer in range(12):
            records = [run["layer_metrics"][layer] for run in rows_by_config[config]]
            output.append(
                {
                    "key_bits": config[0],
                    "value_bits": config[1],
                    "m_over_d": config[2],
                    "layer": layer,
                    "attention_logit_rmse_mean": mean(record["attention_logit_rmse"] for record in records),
                    "attention_logit_rmse_sd": sample_sd([record["attention_logit_rmse"] for record in records]),
                    "attention_kl_mean": mean(record["attention_kl"] for record in records),
                    "attention_kl_sd": sample_sd([record["attention_kl"] for record in records]),
                    "residual_norm_mean": mean(record["residual_norm_mean"] for record in records),
                    "key_norm_mean": mean(record["key_norm_mean"] for record in records),
                }
            )
    return output


def mechanism_summary(fine: list[dict], layers: list[dict], budget: list[dict]) -> list[dict]:
    output = []
    for key_bits in (4, 3):
        records = sorted(
            [record for record in fine if record["key_bits"] == key_bits and record["value_bits"] == 2],
            key=lambda record: record["m_over_d"],
        )
        ratios = np.array([record["m_over_d"] for record in records])
        qjl_error = np.array([record["qjl_residual_rmse_mean"] for record in records])
        logit_error = np.array([record["attention_logit_rmse_mean"] for record in records])
        perplexity = np.array([record["perplexity_mean"] for record in records])
        attention_kl = np.array([record["attention_kl_fp_to_quantized_mean"] for record in records])
        operating_point = min(record["m_over_d"] for record in records if record["fraction_observed_ppl_gain"] >= 0.9)
        layer_records = [record for record in layers if record["key_bits"] == key_bits and record["m_over_d"] == 1.0]
        layer_records.sort(key=lambda record: record["attention_kl_mean"], reverse=True)
        total_layer_kl = sum(record["attention_kl_mean"] for record in layer_records)
        mse_only = next(record for record in budget if record["key_bits"] == key_bits and record["value_bits"] == 2 and record["m_over_d"] == 0)
        m_one = records[0]
        m_two = next(record for record in records if record["m_over_d"] == 2)
        output.append(
            {
                "key_bits": key_bits,
                "value_bits": 2,
                "qjl_rmse_loglog_slope": float(np.polyfit(np.log(ratios), np.log(qjl_error), 1)[0]),
                "logit_rmse_loglog_slope": float(np.polyfit(np.log(ratios), np.log(logit_error), 1)[0]),
                "pearson_ppl_vs_logit_rmse": float(np.corrcoef(perplexity, logit_error)[0, 1]),
                "pearson_ppl_vs_attention_kl": float(np.corrcoef(perplexity, attention_kl)[0, 1]),
                "smallest_m_over_d_for_90pct_observed_gain": operating_point,
                "mse_only_ppl": mse_only["perplexity_mean"],
                "m_equals_d_ppl": m_one["perplexity_mean"],
                "m_equals_2d_ppl": m_two["perplexity_mean"],
                "mse_only_attention_kl": mse_only["attention_kl_fp_to_quantized_mean"],
                "m_equals_d_attention_kl": m_one["attention_kl_fp_to_quantized_mean"],
                "m_equals_2d_attention_kl": m_two["attention_kl_fp_to_quantized_mean"],
                "m_equals_d_residual_norm_mean": m_one["residual_norm_mean_mean"],
                "top3_m_equals_d_kl_layers": json.dumps([record["layer"] for record in layer_records[:3]]),
                "top3_layers_fraction_of_layer_kl": sum(record["attention_kl_mean"] for record in layer_records[:3]) / total_layer_kl,
            }
        )
    return output


def plot_results(
    budget: list[dict],
    fine: list[dict],
    ppl_frontier: list[dict],
    error_frontier: list[dict],
    layers: list[dict],
) -> None:
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    colors = {0.0: "#111827", 0.5: "#9ca3af", 1.0: "#2563eb", 1.5: "#60a5fa", 2.0: "#dc2626", 3.0: "#f59e0b", 4.0: "#7c3aed"}

    for metric, ylabel, filename, frontier in (
        ("perplexity_mean", "Perplexity", "phase2_storage_pareto_ppl.png", ppl_frontier),
        ("attention_logit_rmse_mean", "Attention-logit RMSE", "phase2_storage_pareto_logit.png", error_frontier),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for ratio in sorted({record["m_over_d"] for record in budget}):
            records = [record for record in budget if record["m_over_d"] == ratio]
            ax.scatter(
                [record["kv_bytes"] for record in records],
                [record[metric] for record in records],
                color=colors.get(ratio, "#6b7280"),
                marker="X" if ratio == 0 else "o",
                alpha=0.72,
                label="MSE only" if ratio == 0 else f"m={ratio:g}d",
            )
        ordered = sorted(frontier, key=lambda record: record["kv_bytes"])
        ax.plot([record["kv_bytes"] for record in ordered], [record[metric] for record in ordered], color="#111827", linewidth=1.5, label="Pareto frontier")
        ax.set_xlabel("Total K+V bytes per token")
        ax.set_ylabel(ylabel)
        if metric == "perplexity_mean":
            ax.set_yscale("log")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(PLOTS / filename, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    for (key_bits, value_bits), color in (((4, 2), "#2563eb"), ((3, 2), "#dc2626")):
        records = [record for record in fine if (record["key_bits"], record["value_bits"]) == (key_bits, value_bits)]
        records.sort(key=lambda record: record["m_over_d"])
        label = f"{key_bits}/{value_bits}-bit"
        axes[0].errorbar([record["m_over_d"] for record in records], [record["perplexity_mean"] for record in records], yerr=[record["perplexity_sd"] for record in records], marker="o", color=color, label=label)
        axes[1].plot([record["m_over_d"] for record in records], [record["attention_logit_rmse_mean"] for record in records], marker="o", color=color, label=label)
        axes[2].plot([record["m_over_d"] for record in records], [record["attention_kl_fp_to_quantized_mean"] for record in records], marker="o", color=color, label=label)
    for ax, ylabel in zip(axes, ("Perplexity", "Attention-logit RMSE", "Attention KL(FP || quantized)")):
        ax.set_xlabel("m/d")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "phase2_fine_m_mechanism.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)
    for axis, key_bits in zip(axes, (4, 3)):
        for ratio, color, label in ((0.0, "#111827", "MSE only"), (1.0, "#2563eb", "m=d"), (2.0, "#dc2626", "m=2d")):
            records = [record for record in layers if record["key_bits"] == key_bits and record["m_over_d"] == ratio]
            axis.plot([record["layer"] for record in records], [record["attention_kl_mean"] for record in records], marker="o", color=color, label=label)
        axis.set_title(f"{key_bits}/2-bit")
        axis.set_xlabel("Transformer layer")
        axis.set_ylabel("Attention KL(FP || quantized)")
        axis.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "phase2_layer_attention_kl.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
    for axis, key_bits in zip(axes, (4, 3)):
        for ratio, color in ((1.0, "#2563eb"), (1.25, "#60a5fa"), (1.5, "#22c55e"), (1.75, "#f59e0b"), (2.0, "#dc2626"), (2.5, "#a855f7"), (3.0, "#7c3aed"), (4.0, "#111827")):
            records = [record for record in fine if record["key_bits"] == key_bits and record["m_over_d"] == ratio]
            seed_values = json.loads(records[0]["seed_perplexities"])
            axis.scatter([ratio] * len(seed_values), seed_values, color=color, alpha=0.8)
        axis.set_title(f"{key_bits}/2-bit")
        axis.set_xlabel("m/d")
        axis.set_ylabel("Individual-seed perplexity")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS / "phase2_fine_m_seed_variance.png", dpi=180)
    plt.close(fig)


def main() -> None:
    equal_rows = (
        read_jsonl("phase2_equal_budget.jsonl")
        + read_jsonl("phase2_equal_budget_extension.jsonl")
        + read_jsonl("phase2_mse_only.jsonl")
        + read_jsonl("phase2_high_scalar.jsonl")
    )
    fine_rows = read_jsonl("phase2_fine_m.jsonl")
    budget_summary = summarize(equal_rows)
    fine_summary = add_fine_metrics(summarize(fine_rows))

    rows_by_config: dict[tuple[int, int, float], list[dict]] = defaultdict(list)
    for row in equal_rows:
        config = (row["key_bits"], row["value_bits"], row["m_over_d"])
        rows_by_config[config].append(row)
    for row in fine_rows:
        config = (row["key_bits"], row["value_bits"], row["m_over_d"])
        if config not in rows_by_config:
            rows_by_config[config].append(row)
    # Prefer confirmatory fine-sweep records for its overlapping 3/2 and 4/2 configurations.
    for config in ((3, 2, 1.0), (3, 2, 2.0), (4, 2, 1.0), (4, 2, 2.0)):
        rows_by_config[config] = [row for row in fine_rows if (row["key_bits"], row["value_bits"], row["m_over_d"]) == config]

    winners = exact_budget_winners(budget_summary)
    ppl_frontier = pareto_frontier(budget_summary, "perplexity_mean")
    error_frontier = pareto_frontier(budget_summary, "attention_logit_rmse_mean")
    comparisons = [
        paired_comparison(rows_by_config, "4/2: MSE only versus m=d", (4, 2, 0.0), (4, 2, 1.0)),
        paired_comparison(rows_by_config, "4/2: m=d versus m=2d", (4, 2, 1.0), (4, 2, 2.0)),
        paired_comparison(rows_by_config, "3/2: MSE only versus m=d", (3, 2, 0.0), (3, 2, 1.0)),
        paired_comparison(rows_by_config, "3/2: m=d versus m=2d", (3, 2, 1.0), (3, 2, 2.0)),
        paired_comparison(rows_by_config, "60-byte allocation: 4/2,m=d versus 3/2,m=2d", (4, 2, 1.0), (3, 2, 2.0)),
        paired_comparison(rows_by_config, "68-byte allocation: 5/2,m=d versus 4/2,m=2d", (5, 2, 1.0), (4, 2, 2.0)),
        paired_comparison(rows_by_config, "MSE-only 5/2 versus 4/2,m=2d", (5, 2, 0.0), (4, 2, 2.0)),
        paired_comparison(rows_by_config, "5/4: MSE only versus m=2d", (5, 4, 0.0), (5, 4, 2.0)),
        paired_comparison(rows_by_config, "5/4: m=2d versus m=3d", (5, 4, 2.0), (5, 4, 3.0)),
    ]
    layers = layer_summary(rows_by_config)
    mechanisms = mechanism_summary(fine_summary, layers, budget_summary)

    SUMMARY.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "phase2_budget_summary": budget_summary,
        "phase2_fine_m_summary": fine_summary,
        "phase2_budget_winners": winners,
        "phase2_ppl_frontier": ppl_frontier,
        "phase2_logit_frontier": error_frontier,
        "phase2_comparisons": comparisons,
        "phase2_layer_summary": layers,
        "phase2_mechanism_summary": mechanisms,
    }
    for name, records in artifacts.items():
        (SUMMARY / f"{name}.json").write_text(json.dumps(records, indent=2) + "\n")
        write_csv(SUMMARY / f"{name}.csv", records)

    plot_results(budget_summary, fine_summary, ppl_frontier, error_frontier, layers)
    print(
        f"summarized {len(equal_rows)} equal-budget runs ({len(budget_summary)} configs), "
        f"{len(fine_rows)} fine-sweep runs ({len(fine_summary)} configs)"
    )


if __name__ == "__main__":
    main()
