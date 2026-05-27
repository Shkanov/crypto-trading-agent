"""Tests for src.services.risk_circuits — three independent circuits + composition."""
from __future__ import annotations

import math

import pytest

from src.services.risk_circuits import (
    DAY_MS,
    AccountTimeSeries,
    CircuitConfig,
    evaluate_circuits,
)


T_NOW = 1_700_000_000_000  # arbitrary anchor


def _flat(equity: float, n_days: int, pnl_pct: float = 0.0) -> AccountTimeSeries:
    return AccountTimeSeries(
        equity_curve=tuple([equity] * n_days),
        daily_pnl_pct=tuple([pnl_pct] * n_days),
        last_day_ms=T_NOW,
    )


# ---------------------------------------------------------------------------
# All-clear baseline

def test_no_history_returns_neutral() -> None:
    ts = AccountTimeSeries(equity_curve=(), daily_pnl_pct=(), last_day_ms=T_NOW)
    state = evaluate_circuits(ts)
    assert state.size_multiplier == 1.0
    assert not state.flatten and not state.no_new_entries
    assert state.cooloff_until_ms == 0
    assert state.triggered == ()


def test_flat_equity_clean() -> None:
    state = evaluate_circuits(_flat(1000.0, 30))
    assert state.size_multiplier == 1.0
    assert not state.flatten and not state.no_new_entries
    assert state.reason == "all clear"


# ---------------------------------------------------------------------------
# Circuit 1: trailing DD

def test_dd_below_halve_threshold_no_action() -> None:
    # Peak 1000 → cur 950 = 5% DD, under threshold
    curve = tuple([1000.0] * 10 + [950.0])
    ts = AccountTimeSeries(curve, (0.0,) * 11, T_NOW)
    state = evaluate_circuits(ts)
    assert state.size_multiplier == 1.0
    assert "dd_halve" not in state.triggered


def test_dd_at_halve_threshold_halves_sizing() -> None:
    # Peak 1000 → cur 900 = 10% DD, exact threshold
    curve = tuple([1000.0] * 10 + [900.0])
    ts = AccountTimeSeries(curve, (0.0,) * 11, T_NOW)
    state = evaluate_circuits(ts)
    assert state.size_multiplier == 0.5
    assert "dd_halve" in state.triggered
    assert not state.flatten
    assert state.cooloff_until_ms == 0


def test_dd_above_flatten_triggers_flatten_and_cooloff() -> None:
    # Peak 1000 → cur 750 = 25% DD
    curve = tuple([1000.0] * 10 + [750.0])
    ts = AccountTimeSeries(curve, (0.0,) * 11, T_NOW)
    cfg = CircuitConfig()
    state = evaluate_circuits(ts, cfg)
    assert state.flatten is True
    assert state.no_new_entries is True
    assert state.size_multiplier == 0.0
    assert state.cooloff_until_ms == T_NOW + cfg.dd_cooloff_days * DAY_MS
    assert "dd_flatten" in state.triggered


def test_cooloff_persists_after_recovery() -> None:
    """Once flatten + cooloff fires, recovery doesn't release the brake
    until the cooloff window expires."""
    cfg = CircuitConfig()
    # Equity recovered — no current DD — but we're inside an active cooloff.
    curve = tuple([1000.0] * 11)
    ts = AccountTimeSeries(curve, (0.0,) * 11, T_NOW)
    active_cooloff = T_NOW + 5 * DAY_MS  # 5d remaining
    state = evaluate_circuits(ts, cfg, now_ms=T_NOW,
                              active_cooloff_until_ms=active_cooloff)
    assert state.size_multiplier == 0.0
    assert state.no_new_entries is True
    assert state.cooloff_until_ms == active_cooloff
    assert "dd_cooloff_active" in state.triggered


def test_cooloff_expires_when_now_past_window() -> None:
    cfg = CircuitConfig()
    curve = tuple([1000.0] * 11)
    ts = AccountTimeSeries(curve, (0.0,) * 11, T_NOW)
    # Cooloff already lapsed
    expired = T_NOW - 1 * DAY_MS
    state = evaluate_circuits(ts, cfg, now_ms=T_NOW,
                              active_cooloff_until_ms=expired)
    assert state.size_multiplier == 1.0
    assert "dd_cooloff_active" not in state.triggered


# ---------------------------------------------------------------------------
# Circuit 2: vol regime

def test_low_vol_no_trigger() -> None:
    # 30 days of tiny PnL (well under target vol)
    ts = AccountTimeSeries(
        equity_curve=tuple([1000.0] * 30),
        daily_pnl_pct=tuple([0.1 if i % 2 == 0 else -0.1 for i in range(30)]),
        last_day_ms=T_NOW,
    )
    state = evaluate_circuits(ts)
    assert "vol_regime" not in state.triggered


def test_vol_breach_5_consecutive_days_halves() -> None:
    # 30 days of pnl pct that produces annualised vol well above 2×15% = 30%
    # σ_daily ≈ 30% / sqrt(365) ≈ 1.57% — need bigger swings to clear 2× target.
    # Use ±2.5% alternating returns: σ ≈ 2.5%, annualised ≈ 47% > 30%.
    pnl = tuple([2.5 if i % 2 == 0 else -2.5 for i in range(40)])
    ts = AccountTimeSeries(
        equity_curve=tuple([1000.0] * 40),
        daily_pnl_pct=pnl,
        last_day_ms=T_NOW,
    )
    state = evaluate_circuits(ts)
    assert "vol_regime" in state.triggered
    assert state.size_multiplier == 0.5
    assert state.consecutive_high_vol_days >= 5


def test_vol_breach_only_1_day_no_trigger() -> None:
    # Quiet for 30 days, then one wild day — consec must be < 5.
    pnl = [0.05] * 30 + [10.0]
    ts = AccountTimeSeries(
        equity_curve=tuple([1000.0] * len(pnl)),
        daily_pnl_pct=tuple(pnl),
        last_day_ms=T_NOW,
    )
    state = evaluate_circuits(ts)
    # Even though the most-recent rolling-window has 1 wild sample, the
    # CONSECUTIVE-days counter needs ≥5 to trip.
    assert state.consecutive_high_vol_days < 5
    assert "vol_regime" not in state.triggered


# ---------------------------------------------------------------------------
# Circuit 3: daily loss block

def test_daily_loss_blocks_new_entries() -> None:
    ts = AccountTimeSeries(
        equity_curve=tuple([1000.0] * 5 + [965.0]),
        daily_pnl_pct=tuple([0.0] * 5 + [-3.5]),
        last_day_ms=T_NOW,
    )
    state = evaluate_circuits(ts)
    assert state.no_new_entries is True
    assert "daily_loss_block" in state.triggered
    # Doesn't flatten — just blocks new entries
    assert state.flatten is False


def test_daily_gain_no_block() -> None:
    ts = AccountTimeSeries(
        equity_curve=tuple([1000.0] * 5 + [1035.0]),
        daily_pnl_pct=tuple([0.0] * 5 + [3.5]),
        last_day_ms=T_NOW,
    )
    state = evaluate_circuits(ts)
    assert "daily_loss_block" not in state.triggered


def test_daily_loss_below_threshold_no_block() -> None:
    ts = AccountTimeSeries(
        equity_curve=tuple([1000.0] * 5 + [980.0]),
        daily_pnl_pct=tuple([0.0] * 5 + [-2.0]),
        last_day_ms=T_NOW,
    )
    state = evaluate_circuits(ts)
    assert "daily_loss_block" not in state.triggered


# ---------------------------------------------------------------------------
# Composition

def test_dd_flatten_overrides_lesser_circuits() -> None:
    """When -20% DD fires, the daily-loss circuit's no-new is also implied —
    we shouldn't see size_multiplier > 0 even if vol is low."""
    curve = tuple([1000.0] * 10 + [780.0])  # 22% DD
    # Same day is also a 3.5% daily loss
    pnl = tuple([0.0] * 10 + [-3.5])
    ts = AccountTimeSeries(curve, pnl, T_NOW)
    state = evaluate_circuits(ts)
    assert state.flatten is True
    assert state.size_multiplier == 0.0
    assert state.no_new_entries is True
    assert "dd_flatten" in state.triggered
    assert "daily_loss_block" in state.triggered


def test_dd_halve_plus_vol_breach_takes_minimum() -> None:
    """A 12% DD halves sizing; if vol regime ALSO fires, the result is still
    0.5 — both circuits set multiplier to 0.5, min() is 0.5."""
    # 12% DD with high vol returns
    n = 40
    pnl_list = [2.5 if i % 2 == 0 else -2.5 for i in range(n - 1)] + [-12.0]
    # equity curve: start 1000, peak around mid, end at 12% DD
    equity = [1000.0]
    for r in pnl_list:
        equity.append(equity[-1] * (1.0 + r / 100.0))
    # Force exactly 12% trailing DD: scale final equity
    peak = max(equity)
    equity[-1] = peak * 0.88
    ts = AccountTimeSeries(
        equity_curve=tuple(equity),
        daily_pnl_pct=tuple([0.0] + pnl_list),
        last_day_ms=T_NOW,
    )
    state = evaluate_circuits(ts)
    assert state.size_multiplier == 0.5
    assert "dd_halve" in state.triggered or "vol_regime" in state.triggered


def test_invalid_series_lengths_rejected() -> None:
    with pytest.raises(ValueError):
        AccountTimeSeries(equity_curve=(1.0, 2.0), daily_pnl_pct=(0.0,), last_day_ms=T_NOW)


def test_config_custom_thresholds_respected() -> None:
    # Tighten the halve threshold to 5% — a 5% DD should now trip it.
    cfg = CircuitConfig(dd_halve_pct=5.0, dd_flatten_pct=15.0)
    curve = tuple([1000.0] * 10 + [950.0])
    ts = AccountTimeSeries(curve, (0.0,) * 11, T_NOW)
    state = evaluate_circuits(ts, cfg)
    assert state.size_multiplier == 0.5
    assert "dd_halve" in state.triggered


def test_state_records_metrics() -> None:
    """CircuitState should expose enough metadata for caller logging."""
    curve = tuple([1000.0] * 10 + [850.0])  # 15% DD
    ts = AccountTimeSeries(curve, (0.0,) * 11, T_NOW)
    state = evaluate_circuits(ts)
    assert abs(state.dd_from_peak_pct - 15.0) < 1e-9
    assert state.size_multiplier == 0.5
    assert "dd_halve" in state.triggered
