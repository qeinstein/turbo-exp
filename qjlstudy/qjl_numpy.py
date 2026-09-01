"""Dependency-light QJL estimator used for normalization and scaling checks."""
from __future__ import annotations

import math
import numpy as np


def projection(d: int, m: int, seed: int) -> np.ndarray:
    if d <= 0 or m <= 0:
        raise ValueError("d and m must be positive")
    return np.random.default_rng(seed).standard_normal((m, d))


def encode(S: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(keys, dtype=np.float64)
    signs = np.where(keys @ S.T >= 0, 1.0, -1.0)
    return signs, np.linalg.norm(keys, axis=-1)


def estimate(S: np.ndarray, query: np.ndarray, signs: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """Unbiased asymmetric QJL estimator: sqrt(pi/2) * ||k|| * <Sq, sign(Sk)> / m."""
    return math.sqrt(math.pi / 2.0) * np.asarray(norms) * (signs @ (S @ query)) / S.shape[0]


def sketch_storage_bytes(m: int, per_key_metadata_bytes: int = 4) -> int:
    """Packed one-bit signs plus an fp32 residual norm; excludes shared projection matrix."""
    return math.ceil(m / 8) + per_key_metadata_bytes
