from qjlstudy.budget import enumerate_matched_configs, storage_bytes


def test_known_exact_budget_match():
    standard = storage_bytes(64, key_bits=4, value_bits=2, m=64)
    oversized = storage_bytes(64, key_bits=3, value_bits=2, m=128)
    assert standard["kv_bytes"] == oversized["kv_bytes"] == 60
    assert standard["key_bytes"] == 40
    assert standard["value_bytes"] == 20
    more_key = storage_bytes(64, key_bits=5, value_bits=2, m=64)
    oversized_at_68 = storage_bytes(64, key_bits=4, value_bits=2, m=128)
    assert more_key["kv_bytes"] == oversized_at_68["kv_bytes"] == 68
    mse_only = storage_bytes(64, key_bits=5, value_bits=2, m=0)
    assert mse_only["kv_bytes"] == 56
    assert mse_only["qjl_sign_bytes"] == 0
    assert mse_only["key_metadata_bytes"] == 4


def test_enumerator_only_keeps_groups_with_real_allocation_tradeoffs():
    configs = enumerate_matched_configs(64)
    assert any(r["key_bits"] == 4 and r["value_bits"] == 2 and r["m_over_d"] == 1 for r in configs)
    assert any(r["key_bits"] == 3 and r["value_bits"] == 2 and r["m_over_d"] == 2 for r in configs)
    for budget in {r["kv_bytes"] for r in configs}:
        group = [r for r in configs if r["kv_bytes"] == budget]
        assert any(a["m"] > b["m"] and a["key_bits"] + a["value_bits"] < b["key_bits"] + b["value_bits"] for a in group for b in group)
