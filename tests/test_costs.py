"""Tests for the realistic cost model."""
from __future__ import annotations

import math

import pytest

from src.services.costs import (
    Costs,
    IMPACT_K_MAJOR,
    IMPACT_K_MID,
    IMPACT_K_SMALL,
    adjust_entry_price,
    adjust_exit_price,
    funding_accrual_usd,
    impact_k_for_symbol,
    round_trip_cost_bps,
    slippage_bps,
    taker_fee_usd,
)


def test_impact_k_classification():
    assert impact_k_for_symbol("BTCUSDT") == IMPACT_K_MAJOR
    assert impact_k_for_symbol("ETHUSDT") == IMPACT_K_MAJOR
    assert impact_k_for_symbol("DOGEUSDT") == IMPACT_K_SMALL
    assert impact_k_for_symbol("DOGEUSDT", mid_caps={"DOGEUSDT"}) == IMPACT_K_MID
    assert impact_k_for_symbol("LITTLECOINUSDT") == IMPACT_K_SMALL


def test_slippage_bps_zero_notional():
    assert slippage_bps(0.0, 1e7, IMPACT_K_MAJOR, 1.0) == 0.0


def test_slippage_bps_no_adv_falls_back_to_half_spread():
    assert slippage_bps(1_000.0, 0.0, IMPACT_K_MAJOR, 3.0) == 3.0


def test_slippage_bps_grows_with_participation():
    # 0.01% participation: impact = 0.05 * sqrt(0.0001) * 10000 = 5 bps
    s_small = slippage_bps(100.0, 1_000_000.0, IMPACT_K_MAJOR, 1.0)
    # 1% participation: impact = 0.05 * sqrt(0.01) * 10000 = 50 bps
    s_big = slippage_bps(10_000.0, 1_000_000.0, IMPACT_K_MAJOR, 1.0)
    assert s_big > s_small
    assert s_small == pytest.approx(1.0 + 5.0, rel=0.01)
    assert s_big == pytest.approx(1.0 + 50.0, rel=0.01)


def test_slippage_alt_costs_3x_major():
    # Same participation, alt k vs major k → roughly 3x more impact
    s_major = slippage_bps(1_000.0, 100_000.0, IMPACT_K_MAJOR, 0.0)
    s_alt = slippage_bps(1_000.0, 100_000.0, IMPACT_K_SMALL, 0.0)
    assert s_alt > s_major * 5  # 0.30 / 0.05 = 6x


def test_adjust_entry_price_longs_pay_higher():
    p = adjust_entry_price(100.0, "long", 100.0, 1_000_000.0, IMPACT_K_MAJOR, 1.0)
    assert p > 100.0
    p_short = adjust_entry_price(100.0, "short", 100.0, 1_000_000.0, IMPACT_K_MAJOR, 1.0)
    assert p_short < 100.0


def test_adjust_exit_price_longs_sell_lower():
    p = adjust_exit_price(100.0, "long", 100.0, 1_000_000.0, IMPACT_K_MAJOR, 1.0)
    assert p < 100.0
    p_short = adjust_exit_price(100.0, "short", 100.0, 1_000_000.0, IMPACT_K_MAJOR, 1.0)
    assert p_short > 100.0


def test_taker_fee_matches_bps():
    costs = Costs()
    assert taker_fee_usd(10_000.0, "perp", costs) == pytest.approx(5.0)
    assert taker_fee_usd(10_000.0, "spot", costs) == pytest.approx(10.0)


def test_funding_accrual_long_pays_positive():
    # Long, $1000 notional, one event with +0.01 funding rate during hold
    events = [(2000, 0.01)]
    cost = funding_accrual_usd("long", 1000.0, events, 1000, 3000)
    assert cost == pytest.approx(10.0)  # 1000 * 0.01


def test_funding_accrual_short_receives_positive():
    events = [(2000, 0.01)]
    cost = funding_accrual_usd("short", 1000.0, events, 1000, 3000)
    assert cost == pytest.approx(-10.0)  # short paid funding → negative cost


def test_funding_accrual_skips_events_outside_window():
    events = [(500, 0.01), (2000, 0.01), (4000, 0.01)]
    cost = funding_accrual_usd("long", 1000.0, events, 1000, 3000)
    assert cost == pytest.approx(10.0)  # only ts=2000 in (1000, 3000]


def test_round_trip_cost_bps_perp():
    # Tiny participation: 100 notional vs 1M ADV = 0.01% → impact ~5 bps
    cost_bps = round_trip_cost_bps(
        notional_usd=100.0,
        venue="perp",
        adv_5m_usd=1_000_000.0,
        impact_k=IMPACT_K_MAJOR,
        half_spread_bps=1.0,
        costs=Costs(),
        n_funding_cycles=0,
    )
    # 2 * (5 fee + 1 half_spread + ~5 impact) ≈ 22 bps
    assert 18.0 <= cost_bps <= 28.0


def test_round_trip_cost_bps_with_funding():
    # 3 cycles, +10bps each, long → adds 30bps of funding cost on top of execution
    base = round_trip_cost_bps(
        notional_usd=100.0,
        venue="perp",
        adv_5m_usd=1_000_000.0,
        impact_k=IMPACT_K_MAJOR,
        half_spread_bps=1.0,
        costs=Costs(),
        n_funding_cycles=0,
    )
    with_fund = round_trip_cost_bps(
        notional_usd=100.0,
        venue="perp",
        adv_5m_usd=1_000_000.0,
        impact_k=IMPACT_K_MAJOR,
        half_spread_bps=1.0,
        costs=Costs(),
        n_funding_cycles=3,
        avg_funding_rate=0.001,  # 10bps each
        funding_sign=+1.0,
    )
    assert with_fund - base == pytest.approx(30.0, abs=0.5)
