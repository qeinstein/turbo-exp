import numpy as np
from qjlstudy.qjl_numpy import encode, estimate, projection, sketch_storage_bytes


def test_shapes_and_storage():
    s = projection(8, 16, 3)
    signs, norms = encode(s, np.ones((2, 8)))
    assert signs.shape == (2, 16)
    assert norms.shape == (2,)
    assert sketch_storage_bytes(16) == 6


def test_estimator_is_approximately_unbiased_across_projections():
    rng = np.random.default_rng(4)
    q, k = rng.standard_normal(32), rng.standard_normal(32)
    truth = q @ k
    estimates = []
    for seed in range(2000):
        s = projection(32, 32, seed)
        sign, norm = encode(s, k)
        estimates.append(estimate(s, q, sign, norm))
    assert abs(np.mean(estimates) - truth) < 0.2
