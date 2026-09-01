"""TurboQuant-style vectorized residual-QJL cache with arbitrary m.

Adapted independently from the public algorithms used by the sibling reference.
The important experimental control is that m changes only residual sign-sketch
length and the 1/m estimator normalization; all scalar quantization is fixed.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch


def _rotation(d: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device); g.manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g, device=device))
    return q * torch.sign(torch.diag(r)).unsqueeze(0)


@lru_cache(maxsize=32)
def _codebook_cpu(d: int, bits: int) -> tuple[float, ...]:
    # Same coordinate distribution as TurboQuant; scipy computes Lloyd-Max points.
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.special import gamma
    import numpy as np
    levels = 2 ** bits
    c = gamma(d / 2) / (math.sqrt(math.pi) * gamma((d - 1) / 2))
    def pdf(x): return c * max(0.0, 1 - x*x) ** ((d - 3) / 2)
    def cdf(x): return quad(pdf, -1, x, limit=100)[0]
    centers = np.array([brentq(lambda x: cdf(x) - p, -1 + 1e-9, 1 - 1e-9)
                        for p in np.linspace(1/(2*levels), 1-1/(2*levels), levels)])
    for _ in range(200):
        edges = np.r_[-1., (centers[:-1] + centers[1:]) / 2, 1.]
        new = np.array([quad(lambda x: x*pdf(x), edges[i], edges[i+1], limit=100)[0] /
                        quad(pdf, edges[i], edges[i+1], limit=100)[0] for i in range(levels)])
        if np.max(np.abs(new - centers)) < 1e-7: break
        centers = new
    return tuple(float(x) for x in centers)


def _rotation_seed(layer: int, head: int) -> int:
    """Match the reference's collision-free layer/head rotation seed."""
    return (layer + head) * (layer + head + 1) // 2 + head


def _projection_seed(layer: int, head: int, qjl_seed: int) -> int:
    """Vary only QJL randomness; qjl_seed=0 matches the reference projection."""
    return _rotation_seed(layer, head) + 1 + 1000003 * qjl_seed


class FastResidualQJLCache:
    def __init__(self, d: int, key_bits: int, val_bits: int, layer: int, head: int,
                 m: int, qjl_seed: int, device: torch.device):
        if m < 0: raise ValueError("m must be non-negative")
        self.rotation = _rotation(d, _rotation_seed(layer, head), device)
        g = torch.Generator(device=device)
        g.manual_seed(_projection_seed(layer, head, qjl_seed))
        self.S = torch.randn(m, d, generator=g, device=device) if m else None
        self.key_cb = torch.tensor(_codebook_cpu(d, key_bits - 1), device=device)
        self.val_cb = torch.tensor(_codebook_cpu(d, val_bits), device=device)
        self.m, self.d, self.key_bits = m, d, key_bits

    def encode(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        kn = torch.linalg.norm(keys.float(), dim=1, keepdim=True)
        ku = keys.float() / kn.clamp_min(1e-12)
        kr = ku @ self.rotation.T
        self.K = self.key_cb[(kr.unsqueeze(-1) - self.key_cb).square().argmin(-1)]
        khat = kn * (self.K @ self.rotation)
        residual = keys.float() - khat
        self.Rsign = torch.where(residual @ self.S.T >= 0, 1., -1.) if self.m else None
        self.Rnorm, self.Knorm = torch.linalg.norm(residual, dim=1), kn.squeeze(1)
        vn = torch.linalg.norm(values.float(), dim=1, keepdim=True)
        vr = values.float() / vn.clamp_min(1e-12) @ self.rotation.T
        self.V = self.val_cb[(vr.unsqueeze(-1) - self.val_cb).square().argmin(-1)]
        self.Vnorm = vn.squeeze(1)

    def scores(self, queries: torch.Tensor) -> torch.Tensor:
        mse = (queries.float() @ self.rotation.T @ self.K.T) * self.Knorm
        if not self.m:
            return mse
        qjl = (math.sqrt(math.pi/2) / self.m) * (queries.float() @ self.S.T @ self.Rsign.T) * self.Rnorm
        return mse + qjl

    def values(self) -> torch.Tensor:
        return self.Vnorm.unsqueeze(1) * (self.V @ self.rotation)

    def key_storage_bytes(self) -> float:
        return math.ceil(self.d * (self.key_bits - 1) / 8) + math.ceil(self.m / 8) + 4 + (4 if self.m else 0)
