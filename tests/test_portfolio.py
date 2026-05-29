"""Tests for src.services.portfolio — pure logic, no network."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.services.portfolio import (
    _active_day_std,
    _single_linkage_leaf_order,
    _turnover_l1,
    allocate,
    equal_weight,
    hrp,
    inverse_vol,
)


# ---------------------------------------------------------------------------
# equal_weight

def test_equal_weight_sums_to_one() -> None:
    w = equal_weight(["a", "b", "c", "d"])
    assert math.isclose(sum(w.values()), 1.0)
    assert all(math.isclose(v, 0.25) for v in w.values())


def test_equal_weight_empty() -> None:
    assert equal_weight([]) == {}


# ---------------------------------------------------------------------------
# inverse_vol

def test_inverse_vol_high_vol_gets_low_weight() -> None:
    rng = np.random.default_rng(0)
    returns = {
        "calm":  rng.normal(0.0, 0.5, 250),
        "wild":  rng.normal(0.0, 5.0, 250),
    }
    w = inverse_vol(returns)
    assert math.isclose(sum(w.values()), 1.0)
    assert w["calm"] > w["wild"]
    # The 10× vol gap should give roughly 10× weight ratio.
    assert w["calm"] / w["wild"] > 5.0


def test_inverse_vol_constant_series_capped_by_min_vol() -> None:
    # A flat-zero strategy has zero std; clamp prevents inf-weight.
    returns = {"flat": np.zeros(200), "noisy": np.random.default_rng(1).normal(0, 1, 200)}
    w = inverse_vol(returns)
    assert math.isclose(sum(w.values()), 1.0)
    # `flat` gets a finite but very large weight; `noisy` gets nearly zero.
    assert 0.0 < w["noisy"] < 1.0
    assert w["flat"] > w["noisy"]


def test_inverse_vol_empty() -> None:
    assert inverse_vol({}) == {}


# ---------------------------------------------------------------------------
# Single-linkage clustering

def test_single_linkage_groups_close_items_first() -> None:
    # 4 items: items 0/1 are very close, items 2/3 are very close.
    # Single-linkage should produce a leaf-order that pairs them.
    d = np.array([
        [0.0, 0.1, 1.0, 1.0],
        [0.1, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.1],
        [1.0, 1.0, 0.1, 0.0],
    ])
    order = _single_linkage_leaf_order(d)
    assert len(order) == 4
    assert set(order) == {0, 1, 2, 3}
    # 0 and 1 should be adjacent; 2 and 3 should be adjacent.
    pos = {v: i for i, v in enumerate(order)}
    assert abs(pos[0] - pos[1]) == 1
    assert abs(pos[2] - pos[3]) == 1


def test_single_linkage_handles_singletons() -> None:
    assert _single_linkage_leaf_order(np.zeros((1, 1))) == [0]
    assert _single_linkage_leaf_order(np.zeros((0, 0))) == []


# ---------------------------------------------------------------------------
# HRP

def test_hrp_correlated_pair_plus_outlier_concentrates_on_outlier() -> None:
    """Two strategies are perfectly correlated; a third is uncorrelated.
    HRP's bisection first separates the outlier (50% weight), then splits
    the correlated pair (~25% each). This is the canonical sanity test."""
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 500)
    returns = {
        "A": base,                                  # twin of B
        "B": base + rng.normal(0, 0.01, 500),       # almost-perfect corr with A
        "C": rng.normal(0, 1, 500),                 # uncorrelated outlier
    }
    w = hrp(returns)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)
    # Outlier gets ~50%; correlated pair splits the rest ~25/25.
    assert w["C"] > 0.4
    assert abs(w["A"] - w["B"]) < 0.1
    assert w["A"] + w["B"] < 0.6


def test_hrp_uncorrelated_equal_vol_yields_equal_weight() -> None:
    """When all strategies are mutually uncorrelated and have equal vol,
    HRP should converge to ~equal-weight (the recursive bisection has no
    differential signal to allocate against)."""
    rng = np.random.default_rng(7)
    returns = {f"S{i}": rng.normal(0, 1, 400) for i in range(4)}
    w = hrp(returns)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)
    # All weights should be roughly 0.25 — accept ±0.10 of slop given
    # finite-sample correlation estimates from 400 draws.
    for v in w.values():
        assert abs(v - 0.25) < 0.10


def test_hrp_single_strategy_takes_all() -> None:
    assert hrp({"only": np.array([0.1, 0.2, -0.1])}) == {"only": 1.0}


def test_hrp_short_series_fallback_to_equal_weight() -> None:
    # Series of length < 2 can't yield covariance → equal-weight fallback.
    w = hrp({"a": np.array([0.1]), "b": np.array([0.2])})
    assert w == {"a": 0.5, "b": 0.5}


def test_hrp_constant_series_fallback_to_equal_weight() -> None:
    # A zero-variance strategy makes the correlation matrix undefined.
    w = hrp({"flat": np.ones(100), "noisy": np.random.default_rng(0).normal(0, 1, 100)})
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(w["flat"], 0.5)
    assert math.isclose(w["noisy"], 0.5)


# ---------------------------------------------------------------------------
# Turnover helper

def test_turnover_l1_zero_when_identical() -> None:
    w = {"a": 0.5, "b": 0.5}
    assert _turnover_l1(w, w) == 0.0


def test_turnover_l1_one_for_full_rotation() -> None:
    prev = {"a": 1.0, "b": 0.0}
    new = {"a": 0.0, "b": 1.0}
    # 0.5 * (|0-1| + |1-0|) = 1.0
    assert math.isclose(_turnover_l1(new, prev), 1.0)


def test_turnover_l1_handles_disjoint_keys() -> None:
    prev = {"a": 1.0}
    new = {"b": 1.0}
    assert math.isclose(_turnover_l1(new, prev), 1.0)


# ---------------------------------------------------------------------------
# Allocator dispatcher

def test_allocate_equal_method() -> None:
    rng = np.random.default_rng(0)
    returns = {f"S{i}": rng.normal(0, 1, 100) for i in range(4)}
    r = allocate(returns, method="equal")
    assert r.method_used == "equal"
    assert math.isclose(sum(r.weights.values()), 1.0)
    assert all(math.isclose(v, 0.25) for v in r.weights.values())


def test_allocate_inverse_vol_method() -> None:
    rng = np.random.default_rng(0)
    returns = {"calm": rng.normal(0, 0.5, 250), "wild": rng.normal(0, 5.0, 250)}
    r = allocate(returns, method="inverse_vol")
    assert r.method_used == "inverse_vol"
    assert r.weights["calm"] > r.weights["wild"]


def test_allocate_hrp_method() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 300)
    returns = {
        "A": base, "B": base + rng.normal(0, 0.01, 300),
        "C": rng.normal(0, 1, 300),
    }
    r = allocate(returns, method="hrp")
    assert r.method_used == "hrp"
    assert r.weights["C"] > 0.4


def test_allocate_hrp_falls_back_on_high_turnover() -> None:
    """If HRP would move significantly from the previous weights, the
    dispatcher should fall back to inverse_vol."""
    rng = np.random.default_rng(99)
    base = rng.normal(0, 1, 300)
    returns = {"A": base, "B": base + rng.normal(0, 0.01, 300),
               "C": rng.normal(0, 1, 300)}
    # Stage a prev_weights that's far from any reasonable HRP output.
    prev = {"A": 0.9, "B": 0.05, "C": 0.05}
    r = allocate(returns, method="hrp", fallback="inverse_vol",
                  turnover_threshold=0.2, prev_weights=prev)
    assert r.method_used == "inverse_vol"
    assert "fell back" in r.reason


def test_allocate_hrp_no_fallback_when_turnover_below_threshold() -> None:
    rng = np.random.default_rng(99)
    base = rng.normal(0, 1, 300)
    returns = {"A": base, "B": base + rng.normal(0, 0.01, 300),
               "C": rng.normal(0, 1, 300)}
    # Stage a prev close to what HRP would produce.
    prev = {"A": 0.25, "B": 0.25, "C": 0.50}
    r = allocate(returns, method="hrp", turnover_threshold=0.5, prev_weights=prev)
    assert r.method_used == "hrp"


def test_allocate_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        allocate({"a": np.array([0.1, 0.2])}, method="momentum")


def test_allocate_no_prev_weights_means_zero_turnover() -> None:
    rng = np.random.default_rng(0)
    returns = {"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)}
    r = allocate(returns, method="hrp")
    assert r.turnover == 0.0


# ---------------------------------------------------------------------------
# Active-day volatility — the fix for zero-padding deflating thin-sleeve risk

def test_active_day_std_ignores_idle_days() -> None:
    # 10 active days; padding to 100 days must not change the estimate.
    active = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    padded = np.concatenate([active, np.zeros(90)])
    assert math.isclose(_active_day_std(padded), float(active.std(ddof=1)),
                        rel_tol=1e-9)


def test_active_day_std_too_few_active_returns_floor() -> None:
    assert _active_day_std(np.zeros(50)) == 1e-12          # never traded
    assert _active_day_std(np.array([0.0, 0.0, 3.0, 0.0])) == 1e-12  # 1 active


def test_inverse_vol_ignores_idle_padding() -> None:
    """Two sleeves with an identical active-day return distribution — one
    trading daily, the other only 1 day in 10 (zero-padded) — must receive
    ~equal weight. The old std-of-padded-series logic handed the thin sleeve
    several times the weight purely for being idle."""
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1.0, 40)
    daily = base.copy()
    thin = np.zeros(400)
    thin[::10] = base                                   # same 40 values, scattered
    w = inverse_vol({"daily": daily, "thin": thin})
    assert math.isclose(sum(w.values()), 1.0)
    assert abs(w["daily"] - w["thin"]) < 0.05


def test_hrp_does_not_concentrate_on_idle_volatile_sleeve() -> None:
    """A sleeve that trades rarely but with HIGH active-day volatility should
    get the SMALLER weight. The zero-padding bug inverted this — its deflated
    padded variance won it the lion's share (the live 71%-on-a-20-day-sleeve
    pathology)."""
    rng = np.random.default_rng(5)
    daily = rng.normal(0, 1.0, 300)                     # active daily, vol 1
    thin = np.zeros(300)
    idx = np.arange(0, 300, 15)                         # ~20 active days
    thin[idx] = rng.normal(0, 3.0, len(idx))           # higher active-day vol
    w = hrp({"daily": daily, "thin": thin})
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)
    assert w["thin"] < w["daily"]


# ---------------------------------------------------------------------------
# Evidence floor + weight cap

def test_allocate_evidence_floor_zeros_thin_sleeve() -> None:
    rng = np.random.default_rng(0)
    daily = rng.normal(0, 1, 100)
    thin = np.zeros(100)
    thin[:5] = rng.normal(0, 1, 5)                      # 5 active days
    r = allocate({"daily": daily, "thin": thin}, method="equal",
                  min_active_days=30)
    assert r.weights["thin"] == 0.0
    assert math.isclose(r.weights["daily"], 1.0)
    assert math.isclose(sum(r.weights.values()), 1.0)
    assert "evidence floor" in r.reason


def test_allocate_evidence_floor_keeps_book_when_all_thin() -> None:
    """If no sleeve clears the floor, leave weights untouched rather than
    blank the whole book."""
    rng = np.random.default_rng(0)
    a = np.zeros(100); a[:5] = rng.normal(0, 1, 5)
    b = np.zeros(100); b[:4] = rng.normal(0, 1, 4)
    r = allocate({"a": a, "b": b}, method="equal", min_active_days=30)
    assert math.isclose(sum(r.weights.values()), 1.0)
    assert r.weights["a"] > 0.0 and r.weights["b"] > 0.0


def test_allocate_weight_cap_redistributes() -> None:
    rng = np.random.default_rng(0)
    returns = {
        "big": rng.normal(0, 0.1, 200),                 # low vol → huge ivol weight
        "m1":  rng.normal(0, 1.0, 200),
        "m2":  rng.normal(0, 1.0, 200),
    }
    r = allocate(returns, method="inverse_vol", max_weight=0.5)
    assert all(v <= 0.5 + 1e-9 for v in r.weights.values())
    assert math.isclose(sum(r.weights.values()), 1.0)
    assert "capped" in r.reason


def test_allocate_cap_infeasible_single_survivor_holds_all() -> None:
    """When the floor leaves a single survivor, it holds 100% — the cap is
    raised to its feasibility floor rather than producing weights that can't
    sum to 1."""
    rng = np.random.default_rng(0)
    daily = rng.normal(0, 1, 100)
    thin = np.zeros(100); thin[:3] = rng.normal(0, 1, 3)
    r = allocate({"daily": daily, "thin": thin}, method="equal",
                  min_active_days=30, max_weight=0.35)
    assert math.isclose(r.weights["daily"], 1.0)
    assert r.weights["thin"] == 0.0


def test_allocate_defaults_are_noops() -> None:
    """Default min_active_days=0 / max_weight=1.0 preserve raw weights."""
    rng = np.random.default_rng(1)
    returns = {f"S{i}": rng.normal(0, 1, 80) for i in range(3)}
    r = allocate(returns, method="equal")
    assert all(math.isclose(v, 1.0 / 3) for v in r.weights.values())
    assert r.reason == ""
