"""Exact packed-storage accounting and equal-budget configuration enumeration."""
from __future__ import annotations

import math
from collections import defaultdict


def storage_bytes(d: int, key_bits: int, value_bits: int, m: int) -> dict[str, int]:
    key_scalar = math.ceil(d * (key_bits - 1) / 8)
    qjl_signs = math.ceil(m / 8)
    key_metadata = 8  # original-key norm + residual norm, fp32 each
    value_scalar = math.ceil(d * value_bits / 8)
    value_metadata = 4  # value norm, fp32
    return {
        "key_scalar_bytes": key_scalar,
        "qjl_sign_bytes": qjl_signs,
        "key_metadata_bytes": key_metadata,
        "value_scalar_bytes": value_scalar,
        "value_metadata_bytes": value_metadata,
        "key_bytes": key_scalar + qjl_signs + key_metadata,
        "value_bytes": value_scalar + value_metadata,
        "kv_bytes": key_scalar + qjl_signs + key_metadata + value_scalar + value_metadata,
    }


def enumerate_matched_configs(
    d: int,
    key_bits=(2, 3, 4),
    value_bits=(2, 3, 4),
    ratios=(0.5, 1, 1.5, 2, 3, 4),
) -> list[dict]:
    candidates = []
    for kb in key_bits:
        for vb in value_bits:
            for ratio in ratios:
                m = round(d * ratio)
                candidates.append({"key_bits": kb, "value_bits": vb, "m": m, "m_over_d": m / d, **storage_bytes(d, kb, vb, m)})
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate["kv_bytes"]].append(candidate)
    selected = []
    for budget, group in groups.items():
        has_allocation_trade = any(
            a["m"] > b["m"] and (a["key_bits"] + a["value_bits"]) < (b["key_bits"] + b["value_bits"])
            for a in group for b in group
        )
        if has_allocation_trade:
            for candidate in group:
                candidate["budget_group"] = budget
                selected.append(candidate)
    return sorted(selected, key=lambda r: (r["kv_bytes"], r["key_bits"], r["value_bits"], r["m"]))
