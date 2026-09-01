import pytest

torch = pytest.importorskip("torch")

from qjlstudy.torch_quant import FastResidualQJLCache


def test_qjl_seed_does_not_change_rotation_but_changes_projection():
    a = FastResidualQJLCache(8, 4, 2, layer=1, head=2, m=8, qjl_seed=11, device=torch.device("cpu"))
    b = FastResidualQJLCache(8, 4, 2, layer=1, head=2, m=8, qjl_seed=23, device=torch.device("cpu"))
    assert torch.equal(a.rotation, b.rotation)
    assert not torch.equal(a.S, b.S)


def test_m_changes_only_projection_shape_and_preserves_rotation():
    a = FastResidualQJLCache(8, 4, 2, layer=0, head=0, m=4, qjl_seed=11, device=torch.device("cpu"))
    b = FastResidualQJLCache(8, 4, 2, layer=0, head=0, m=16, qjl_seed=11, device=torch.device("cpu"))
    assert torch.equal(a.rotation, b.rotation)
    assert a.S.shape == (4, 8)
    assert b.S.shape == (16, 8)


def test_m_zero_is_mse_only_and_has_no_projection():
    cache = FastResidualQJLCache(8, 4, 2, layer=0, head=0, m=0, qjl_seed=11, device=torch.device("cpu"))
    keys, values, queries = torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 8)
    cache.encode(keys, values)
    expected = (queries @ cache.rotation.T @ cache.K.T) * cache.Knorm
    assert cache.S is None
    assert cache.Rsign is None
    assert torch.allclose(cache.scores(queries), expected)
