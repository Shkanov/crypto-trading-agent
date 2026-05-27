"""Tests for the sprint #17 allocator wire-up.

Pure-logic `allocate()` and the three weighting methods (equal/inverse_vol/HRP)
are covered exhaustively in `test_portfolio.py`. This file covers the glue
between Storage + portfolio + Orchestrator:

  - `realized_pnl_by_day_per_strategy` bucketing
  - allocator state round-trip in Storage
  - `build_strategy_returns` shape + zero-fill + normalisation
  - Orchestrator.equity_available_usd honours stored weights
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.models.types import TradeStatus
from src.services.portfolio import build_strategy_returns
from src.services.storage import Storage, TradeRow


DAY_MS = 86_400_000


# ---------------------------------------------------------------------------
# Per-strategy daily P&L bucketing

@pytest.mark.asyncio
async def test_realized_pnl_by_day_per_strategy_partitions_by_strategy() -> None:
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        s = Storage(database_url=url)
        await s.init()
        anchor = (1_700_000_000_000 // DAY_MS) * DAY_MS
        # Two strategies, three closes:
        #   indicator day0 +5, day0 +2  ⇒ {day0: 7}
        #   funding   day0 −1, day1 +3  ⇒ {day0: −1, day1: 3}
        rows = [
            ("indicator", anchor + 100, 5.0),
            ("indicator", anchor + 200, 2.0),
            ("funding_harvest", anchor + 300, -1.0),
            ("funding_harvest", anchor + DAY_MS + 100, 3.0),
        ]
        async with s.session() as sess, sess.begin():
            for i, (strat, ts_ms, pnl) in enumerate(rows):
                sess.add(TradeRow(
                    id=f"t{i}", strategy=strat, proposal_id=f"p{i}",
                    symbol="BTCUSDT", market="spot", side="long",
                    qty=1.0, leverage=1, entry_price=100.0,
                    entry_ts_ms=ts_ms - 1_000, exit_ts_ms=ts_ms,
                    exit_price=100.0, exit_reason="tp",
                    realized_pnl_usd=pnl, status=TradeStatus.CLOSED.value,
                ))
        out = await s.realized_pnl_by_day_per_strategy(anchor, anchor + 2 * DAY_MS)
        assert out == {
            "indicator": {anchor: 7.0},
            "funding_harvest": {anchor: -1.0, anchor + DAY_MS: 3.0},
        }


# ---------------------------------------------------------------------------
# Storage round-trip of allocator state

@pytest.mark.asyncio
async def test_allocator_state_save_load_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        s = Storage(database_url=url)
        await s.init()
        assert await s.load_allocator_state() is None
        weights = {"indicator": 0.4, "funding_harvest": 0.6}
        prev = {"indicator": 0.5, "funding_harvest": 0.5}
        await s.save_allocator_state(
            last_rebalance_ms=1_700_000_000_000,
            method_used="hrp", weights=weights, prev_weights=prev,
        )
        loaded = await s.load_allocator_state()
        assert loaded is not None
        last_ms, method, w, p = loaded
        assert last_ms == 1_700_000_000_000
        assert method == "hrp"
        assert w == weights
        assert p == prev
        # Update path: overwrites in place.
        await s.save_allocator_state(
            last_rebalance_ms=1_700_000_001_000,
            method_used="equal", weights={"indicator": 1.0}, prev_weights=weights,
        )
        loaded2 = await s.load_allocator_state()
        assert loaded2 is not None
        assert loaded2[1] == "equal" and loaded2[2] == {"indicator": 1.0}


# ---------------------------------------------------------------------------
# build_strategy_returns

@pytest.mark.asyncio
async def test_build_strategy_returns_shape_and_zero_fill() -> None:
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        s = Storage(database_url=url)
        await s.init()
        anchor_today = (1_700_000_000_000 // DAY_MS) * DAY_MS
        # Three-day lookback: days [-3, -2, -1] (today is excluded by the
        # half-open range — the returns matrix covers PAST days only).
        # Put a +10 close on day -2 for "indicator", nothing on day -1
        # or day -3, nothing at all for "funding_harvest".
        day_minus_2 = anchor_today - 2 * DAY_MS
        async with s.session() as sess, sess.begin():
            sess.add(TradeRow(
                id="t0", strategy="indicator", proposal_id="p0",
                symbol="BTCUSDT", market="spot", side="long",
                qty=1.0, leverage=1, entry_price=100.0,
                entry_ts_ms=day_minus_2, exit_ts_ms=day_minus_2 + 1_000,
                exit_price=110.0, exit_reason="tp",
                realized_pnl_usd=10.0, status=TradeStatus.CLOSED.value,
            ))
        out = await build_strategy_returns(
            s, strategy_names=["indicator", "funding_harvest"],
            reference_equity_usd=1000.0, now_ms=anchor_today + 3_600_000,
            lookback_days=3,
        )
        assert set(out.keys()) == {"indicator", "funding_harvest"}
        # Days are [today-3, today-2, today-1] in order.
        assert out["indicator"].shape == (3,)
        assert out["funding_harvest"].shape == (3,)
        # Indicator: 0 at day -3, +10/1000*100 = 1.0% at day -2, 0 at day -1.
        np.testing.assert_allclose(out["indicator"], [0.0, 1.0, 0.0])
        # Funding: never traded ⇒ all zeros.
        np.testing.assert_allclose(out["funding_harvest"], [0.0, 0.0, 0.0])


@pytest.mark.asyncio
async def test_build_strategy_returns_zero_equity_safe() -> None:
    """`reference_equity_usd=0` must not divide-by-zero; series ends up all-0."""
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        s = Storage(database_url=url)
        await s.init()
        out = await build_strategy_returns(
            s, strategy_names=["x"], reference_equity_usd=0.0,
            now_ms=1_700_000_000_000, lookback_days=5,
        )
        assert out["x"].shape == (5,)
        np.testing.assert_allclose(out["x"], np.zeros(5))


# ---------------------------------------------------------------------------
# Orchestrator.equity_available_usd

def _make_orchestrator_with_weights(
    weights: dict[str, float],
    *,
    enabled: bool = True,
    equity: float = 1_000.0,
    pnl_today: float = 0.0,
    strategies: int = 2,
):
    from src.orchestrator import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)
    orch.equity = equity
    orch.pnl_today = pnl_today
    orch.allocator_weights = weights
    orch.notional_ramp = None
    orch.s = type("S", (), {"allocator_enabled": enabled})()
    # Lightweight strategy stubs: `Strategy.name` is read but nothing else.
    orch.strategies = [type("ST", (), {"name": f"s{i}"})() for i in range(strategies)]
    return orch


def test_equity_available_usd_uses_weights_when_enabled() -> None:
    orch = _make_orchestrator_with_weights(
        {"s0": 0.7, "s1": 0.3}, equity=1_000.0, strategies=2,
    )
    assert orch.equity_available_usd("s0") == pytest.approx(700.0)
    assert orch.equity_available_usd("s1") == pytest.approx(300.0)


def test_equity_available_usd_falls_back_to_even_when_disabled() -> None:
    orch = _make_orchestrator_with_weights(
        {"s0": 0.7, "s1": 0.3}, enabled=False, equity=1_000.0, strategies=2,
    )
    # Disabled ⇒ ignore weights, even split.
    assert orch.equity_available_usd("s0") == pytest.approx(500.0)


def test_equity_available_usd_unknown_strategy_uses_even() -> None:
    orch = _make_orchestrator_with_weights(
        {"s0": 0.7, "s1": 0.3}, equity=1_000.0, strategies=2,
    )
    # New strategy registered after last rebalance ⇒ even split until next.
    assert orch.equity_available_usd("s_new") == pytest.approx(500.0)


def test_equity_available_usd_no_weights_uses_even() -> None:
    orch = _make_orchestrator_with_weights({}, equity=1_000.0, strategies=2)
    # No weights yet (e.g. first boot, before housekeeping fires) ⇒ even.
    assert orch.equity_available_usd("s0") == pytest.approx(500.0)


def test_equity_available_usd_none_strategy_arg_is_even() -> None:
    """Backwards-compat: callers that pass `strategy_name=None` get even split."""
    orch = _make_orchestrator_with_weights(
        {"s0": 0.7, "s1": 0.3}, equity=1_000.0, strategies=2,
    )
    assert orch.equity_available_usd(None) == pytest.approx(500.0)


def test_equity_available_usd_includes_pnl_today() -> None:
    orch = _make_orchestrator_with_weights(
        {"s0": 0.6, "s1": 0.4}, equity=1_000.0, pnl_today=200.0, strategies=2,
    )
    # live_equity = 1200; 60% slice → 720
    assert orch.equity_available_usd("s0") == pytest.approx(720.0)


def test_equity_available_usd_clamps_negative_equity_to_zero() -> None:
    orch = _make_orchestrator_with_weights(
        {"s0": 1.0}, equity=100.0, pnl_today=-500.0, strategies=1,
    )
    # live_equity = max(0, -400) = 0 ⇒ 0 slice
    assert orch.equity_available_usd("s0") == pytest.approx(0.0)
