"""Tests for the basis funding-gate logic (pure functions)."""
from __future__ import annotations

from src.strategies.basis_carry import (
    BasisParams,
    compute_signal,
    included_names,
    update_regime,
)

DAY_MS = 86_400_000


def test_hurdle_and_on_thresholds_from_economics():
    p = BasisParams(usd_yield_pct=4.5, borrow_exec_pct=2.5, on_margin_pct=2.0)
    assert p.hurdle_pct == 7.0        # OFF line = USD + borrow
    assert p.on_pct == 9.0            # ON line = hurdle + hysteresis


def test_hysteresis_state_machine():
    # below OFF → stays/turns OFF
    assert update_regime(False, 5.0, on_pct=9, off_pct=7) is False
    # between OFF and ON, currently OFF → stays OFF (needs to clear ON)
    assert update_regime(False, 8.0, on_pct=9, off_pct=7) is False
    # clears ON → turns ON
    assert update_regime(False, 9.5, on_pct=9, off_pct=7) is True
    # between thresholds, currently ON → stays ON (hysteresis holds)
    assert update_regime(True, 8.0, on_pct=9, off_pct=7) is True
    # drops below OFF → turns OFF
    assert update_regime(True, 6.9, on_pct=9, off_pct=7) is False


def test_compute_signal_is_pit_and_annualizes():
    now = 100 * DAY_MS
    # BTC: constant 0.01% (0.0001) per 8h → 0.0001*3*365*100 = 10.95%/yr
    ev = [(now - k * (DAY_MS // 3), 0.0001) for k in range(1, 40)]
    # an event AT/after `now` must be excluded (PIT)
    ev_future = ev + [(now, 0.05), (now + DAY_MS, 0.05)]
    funding = {"BTCUSDT": ev_future}
    basket, per_name = compute_signal(funding, now, lookback_days=21)
    assert abs(per_name["BTCUSDT"] - 10.95) < 0.2      # ~10.95%/yr, future excluded
    assert basket == per_name["BTCUSDT"]                # single name


def test_compute_signal_excludes_pre_window_events():
    now = 100 * DAY_MS
    old = [(now - 50 * DAY_MS, 0.5)]                    # outside 21d window
    recent = [(now - k * (DAY_MS // 3), 0.0002) for k in range(1, 30)]
    basket, per_name = compute_signal({"X": old + recent}, now, 21)
    # old huge event must NOT inflate the signal
    assert per_name["X"] < 30.0


def test_included_names_drops_below_hurdle():
    per_name = {"BTCUSDT": 9.0, "ETHUSDT": 5.0, "LINKUSDT": 7.1, "AAVEUSDT": -2.0}
    incl = included_names(per_name, hurdle_pct=7.0)
    assert incl == ["BTCUSDT", "LINKUSDT"]             # ETH(5) & AAVE(-2) dropped
    assert included_names({}, 7.0) == []


def test_current_regime_off_at_todays_lean_funding():
    # ~5%/yr basket (like now) with OFF state stays OFF; must clear 9% to flip on
    assert update_regime(False, 4.9, on_pct=9, off_pct=7) is False
