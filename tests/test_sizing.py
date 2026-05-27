"""Tests for vol-target sizing."""
from __future__ import annotations

import math

import pytest

from src.services.sizing import (
    VolTargetConfig,
    annualize_vol,
    ewma_volatility,
    qty_from_notional,
    realized_vol_annual_from_klines,
    vol_target_notional,
)


def test_ewma_vol_empty_returns_zero():
    assert ewma_volatility([]) == 0.0
    assert ewma_volatility([0.01]) == 0.0


def test_ewma_vol_constant_returns():
    # Constant non-zero returns → vol == abs(r)
    rets = [0.02] * 50
    vol = ewma_volatility(rets, lam=0.94)
    assert vol == pytest.approx(0.02, rel=0.01)


def test_ewma_vol_increases_with_dispersion():
    low_vol = ewma_volatility([0.001] * 100, lam=0.94)
    high_vol = ewma_volatility([0.05] * 100, lam=0.94)
    assert high_vol > low_vol * 10


def test_annualize_vol():
    # 5m bars: 12 per hour × 24 × 365 = 105,120
    period = 0.001
    annual = annualize_vol(period, 105_120)
    assert annual == pytest.approx(0.001 * math.sqrt(105_120))


def test_realized_vol_from_klines_basic():
    # Flat price → zero vol
    closes = [100.0] * 100
    vol = realized_vol_annual_from_klines(closes, bars_per_year=8760)
    assert vol == 0.0


def test_realized_vol_from_klines_geometric():
    # +1% per bar deterministic — log return = ln(1.01) ≈ 0.00995 constant
    closes = [100.0 * (1.01 ** i) for i in range(50)]
    vol = realized_vol_annual_from_klines(closes, bars_per_year=8760)
    # Constant log-returns → period_vol = abs(r) ≈ 0.00995
    expected_annual = 0.00995 * math.sqrt(8760)
    assert vol == pytest.approx(expected_annual, rel=0.05)


def test_vol_target_notional_scales_inverse_vol():
    cfg = VolTargetConfig()
    # 25% vol asset, $10k equity → 0.20 * 10000 * 0.25 / 0.25 = $2000
    n_25 = vol_target_notional(10_000.0, 0.25, cfg)
    # 50% vol asset → halve
    n_50 = vol_target_notional(10_000.0, 0.50, cfg)
    assert n_25 == pytest.approx(2000.0, rel=0.01)
    assert n_50 == pytest.approx(1000.0, rel=0.01)


def test_vol_target_notional_caps_leverage():
    # Floor low so we can isolate cap behavior. Raw = 0.20 * 1000 * 0.25 / 0.005
    # = $10_000, capped at 2x equity = $2000.
    cfg = VolTargetConfig(notional_cap_x_equity=2.0, vol_floor=0.001)
    n = vol_target_notional(1000.0, 0.005, cfg)
    assert n == pytest.approx(2000.0)


def test_vol_target_notional_floors_vol():
    # Realized vol below floor uses floor value
    cfg = VolTargetConfig(vol_floor=0.10)
    n_below = vol_target_notional(10_000.0, 0.05, cfg)
    n_at = vol_target_notional(10_000.0, 0.10, cfg)
    assert n_below == n_at


def test_vol_target_notional_minimum():
    cfg = VolTargetConfig(notional_floor_usd=10.0)
    # Tiny equity, big vol → notional below floor → returns 0
    n = vol_target_notional(1.0, 10.0, cfg)
    assert n == 0.0


def test_qty_from_notional():
    assert qty_from_notional(1000.0, 50.0) == pytest.approx(20.0)
    assert qty_from_notional(0.0, 50.0) == 0.0
    assert qty_from_notional(1000.0, 0.0) == 0.0
