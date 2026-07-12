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


# ── Execution reconcile logic (Phase 2) ──
import pytest  # noqa: E402

from src.strategies.basis_carry import BasisStrategy  # noqa: E402


class _FakeCtx:
    def __init__(self):
        self.opened: list[str] = []
        self.closed: list[tuple[str, str]] = []
        self.pair_executor = self  # so _close_name uses close_pair path
        self.last_price = {s: 100.0 for s in
                           ("BTCUSDT", "ETHUSDT", "LINKUSDT", "UNIUSDT")}

        class _Ind:
            def latest(self, sym, tf):
                return None
        self.indicators = _Ind()

        class _S:
            approval_timeout_sec = 60
        self.settings = _S()

    def equity_available_usd(self, name=None):
        return 1000.0

    async def propose_pair(self, pair):
        self.opened.append(pair.legs[0].symbol)

    async def close_pair(self, legs, reason=None, price_map=None):
        # mimic PairExecutor.close_pair; record the symbol closed
        self.closed.append((legs[0].symbol, reason))
        return legs


def _mk(execute=True):
    from src.strategies.basis_carry import BasisParams
    s = BasisStrategy(BasisParams(execute_legs=execute))
    s.ctx = _FakeCtx()
    return s


@pytest.mark.asyncio
async def test_reconcile_opens_missing_target_names():
    s = _mk()
    await s._reconcile_book({"BTCUSDT", "ETHUSDT"})
    assert sorted(s.ctx.opened) == ["BTCUSDT", "ETHUSDT"]
    assert s.ctx.closed == []


@pytest.mark.asyncio
async def test_reconcile_closes_dropped_name():
    s = _mk()
    # pretend BTC + ETH are held
    from src.models.types import Trade
    for sym in ("BTCUSDT", "ETHUSDT"):
        s.active[sym] = [Trade(proposal_id="p", symbol=sym, market="spot",
                               side="long", qty=1.0, entry_price=100.0)]
    # target drops ETH → ETH closed, BTC untouched, nothing new opened
    await s._reconcile_book({"BTCUSDT"})
    assert s.ctx.opened == []
    assert s.ctx.closed == [("ETHUSDT", "basis_dropped")]
    assert "ETHUSDT" not in s.active


@pytest.mark.asyncio
async def test_reconcile_off_closes_everything():
    s = _mk()
    from src.models.types import Trade
    for sym in ("BTCUSDT", "ETHUSDT"):
        s.active[sym] = [Trade(proposal_id="p", symbol=sym, market="spot",
                               side="long", qty=1.0, entry_price=100.0)]
    await s._reconcile_book(set())          # regime OFF → empty target
    assert sorted(sym for sym, _ in s.ctx.closed) == ["BTCUSDT", "ETHUSDT"]
    assert all(r == "basis_regime_off" for _, r in s.ctx.closed)
    assert s.active == {}


@pytest.mark.asyncio
async def test_explicit_per_leg_notional_is_primary():
    s = _mk()  # default notional_per_leg_usd=20
    assert s._per_name_notional(5) == pytest.approx(20.0)   # fixed, ignores n
    assert s._per_name_notional(0) == 0.0


@pytest.mark.asyncio
async def test_equity_derived_notional_when_explicit_zero():
    from src.strategies.basis_carry import BasisParams
    s = BasisStrategy(BasisParams(execute_legs=True, notional_per_leg_usd=0.0))
    s.ctx = _FakeCtx()
    # equity 1000 * 0.5 / 2 / 2 names = 125
    assert s._per_name_notional(2) == pytest.approx(125.0)


@pytest.mark.asyncio
async def test_reconcile_holds_when_leg_below_min_notional():
    from src.strategies.basis_carry import BasisParams
    # explicit $3/leg < min $6 → must NOT open (would reject/strand)
    s = BasisStrategy(BasisParams(execute_legs=True, notional_per_leg_usd=3.0,
                                  min_leg_notional_usd=6.0))
    s.ctx = _FakeCtx()
    await s._reconcile_book({"BTCUSDT", "ETHUSDT"})
    assert s.ctx.opened == []          # held, nothing opened below min


@pytest.mark.asyncio
async def test_reconcile_caps_target_to_max_names():
    from src.strategies.basis_carry import BasisParams
    s = BasisStrategy(BasisParams(execute_legs=True, notional_per_leg_usd=20.0,
                                  max_names=2))
    s.ctx = _FakeCtx()
    await s._reconcile_book({"BTCUSDT", "ETHUSDT", "LINKUSDT", "UNIUSDT"})
    assert len(s.ctx.opened) == 2      # capped to max_names
