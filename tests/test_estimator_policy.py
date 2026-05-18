from ai_modules.estimator_policy import (
    BHK_LAYOUT_DEFAULTS,
    clamp_layout_inputs,
    feasibility_caps,
    resolve_layout_from_bhk,
)


def test_feasibility_caps_one_marla_band():
    mx_b, mx_bt, mx_k = feasibility_caps(225)
    assert mx_b == 1 and mx_bt == 1


def test_clamp_layout_inputs():
    b, w, k, clamped, msg = clamp_layout_inputs(225, 4, 2, 2)
    assert clamped is True
    assert b == 1 and w == 1 and k == 1
    assert "Adjusted" in msg


def test_resolve_layout_from_bhk_typical_counts():
    b, w, k, note = resolve_layout_from_bhk(3)
    assert (b, w, k) == (3, 3, 1)
    assert "3 BHK" in note
    assert 3 in BHK_LAYOUT_DEFAULTS


def test_resolve_layout_from_bhk_clamps_extremes():
    assert resolve_layout_from_bhk(0)[0] >= 1
    assert resolve_layout_from_bhk(99)[0] <= 6
