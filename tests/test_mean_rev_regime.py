"""Tests for Hurst / VR / OU regime gate."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.services.mean_rev_regime import (
    RegimeGateParams,
    hurst_exponent,
    ou_half_life_bars,
    passes_regime_gate,
    passes_variance_ratio,
    variance_ratio_test,
)


def test_hurst_trend_is_persistent():
    # Deterministic geometric growth → H ≥ ~0.9 (very persistent)
    prices = [100.0 * (1.005 ** i) for i in range(300)]
    h = hurst_exponent(prices, max_lag=20)
    assert h is not None
    assert h > 0.85


def test_hurst_random_walk_near_half():
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 0.01, 500)
    prices = np.cumsum(rets) + 100
    h = hurst_exponent(prices.tolist(), max_lag=20)
    assert h is not None
    assert 0.40 <= h <= 0.60  # random-walk should land near 0.5


def test_hurst_mean_reverting_is_low():
    # Strong AR(1) with negative coefficient → H well below 0.5
    rng = np.random.default_rng(7)
    n = 500
    x = [0.0]
    for _ in range(n):
        x.append(-0.7 * x[-1] + rng.normal(0, 1))
    h = hurst_exponent(x, max_lag=20)
    assert h is not None
    assert h < 0.40


def test_variance_ratio_mean_reverting():
    # AR(1) with rho=-0.6 → VR(2) < 1
    rng = np.random.default_rng(1)
    n = 1000
    x = [0.0]
    for _ in range(n):
        x.append(-0.6 * x[-1] + rng.normal(0, 1))
    rets = list(np.diff(np.asarray(x)))
    vr2 = variance_ratio_test(rets, lag=2)
    assert vr2 is not None
    assert vr2 < 1.0


def test_variance_ratio_random_walk_is_one():
    rng = np.random.default_rng(2)
    rets = rng.normal(0, 0.01, 2000)
    vr2 = variance_ratio_test(list(rets), lag=2)
    assert vr2 is not None
    assert 0.85 <= vr2 <= 1.15


def test_passes_variance_ratio_strict_rejects_trend():
    rng = np.random.default_rng(3)
    rets = rng.normal(0.001, 0.01, 500)  # positive drift, no mean reversion
    assert passes_variance_ratio(list(rets), lags=(2, 4, 8)) is False


def test_ou_half_life_returns_none_for_trend():
    prices = [100.0 * (1.001 ** i) for i in range(300)]
    hl = ou_half_life_bars(prices)
    assert hl is None


def test_ou_half_life_estimates_reasonable_for_mean_reverter():
    # Classic OU: dx = -theta x dt + sigma dW with known half-life
    rng = np.random.default_rng(5)
    theta = 0.10
    sigma = 1.0
    n = 1000
    x = [0.0]
    for _ in range(n):
        x.append(x[-1] + (-theta * x[-1]) + sigma * rng.normal(0, 1))
    hl = ou_half_life_bars(x)
    assert hl is not None
    # Theoretical half-life = ln(2) / theta = 6.93
    assert 4.0 <= hl <= 12.0


def test_passes_regime_gate_rejects_pure_trend():
    prices = [100.0 * (1.005 ** i) for i in range(500)]
    assert passes_regime_gate(prices) is False


def test_passes_regime_gate_accepts_strong_mean_reverter():
    # Strong AR(1) with negative coefficient → all three tests should clear
    rng = np.random.default_rng(11)
    n = 500
    x = [0.0]
    for _ in range(n):
        x.append(-0.6 * x[-1] + rng.normal(0, 1))
    p = RegimeGateParams(
        hurst_max=0.50, ou_min_half_life_bars=0.1, ou_max_half_life_bars=200.0,
    )
    assert passes_regime_gate(x, p) is True
