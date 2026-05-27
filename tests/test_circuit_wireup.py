"""Tests for the sprint #12 orchestrator wire-up.

The circuit primitives themselves are exhaustively covered in
`test_risk_circuits.py`. This file covers the wire-up surface:

  - Storage round-trip of the cooloff timestamp.
  - `realized_pnl_by_day` bucketing.
  - `build_equity_series` correctness against a controlled storage.
  - `_propose` blocks on `no_new_entries` (skipping ramp / executor).
  - `_propose` halves `decision.qty/notional_usd` on `size_multiplier=0.5`.
  - `propose_pair` blocks on `no_new_entries` but does NOT rescale (would
    unbalance the dollar-hedge).
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta

import pytest
from sqlalchemy import insert

from src.models.types import FeatureVector, Signal, TradeStatus
from src.services.performance import build_equity_series
from src.services.risk_circuits import (
    DAY_MS,
    AccountTimeSeries,
    CircuitConfig,
    CircuitState,
)
from src.services.storage import Storage, TradeRow


# ---------------------------------------------------------------------------
# Storage round-trip

@pytest.mark.asyncio
async def test_circuit_state_save_load_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        s = Storage(database_url=url)
        await s.init()
        # Never-set: returns 0, not None.
        assert await s.load_circuit_state() == 0
        await s.save_circuit_state(1_700_000_000_000)
        assert await s.load_circuit_state() == 1_700_000_000_000
        # Overwrites on second save.
        await s.save_circuit_state(0)
        assert await s.load_circuit_state() == 0


# ---------------------------------------------------------------------------
# realized_pnl_by_day

@pytest.mark.asyncio
async def test_realized_pnl_by_day_buckets_closes_by_utc_day() -> None:
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        s = Storage(database_url=url)
        await s.init()
        # Anchor on a UTC midnight to avoid fencepost confusion.
        anchor = (1_700_000_000_000 // DAY_MS) * DAY_MS
        rows = [
            # day 0: two closes, +10 and −3 ⇒ +7
            (anchor + 100, 10.0),
            (anchor + 80_000_000, -3.0),
            # day 1: one close, +5
            (anchor + DAY_MS + 1_000, 5.0),
            # day 2: open trade (status OPEN, excluded)
            # day 3: closed but `exit_ts_ms` None ⇒ excluded
            # day 4: close +1
            (anchor + 4 * DAY_MS + 1, 1.0),
        ]
        async with s.session() as sess, sess.begin():
            for i, (ts_ms, pnl) in enumerate(rows):
                sess.add(TradeRow(
                    id=f"t{i}", strategy="x", proposal_id=f"p{i}",
                    symbol="BTCUSDT", market="spot", side="long",
                    qty=1.0, leverage=1, entry_price=100.0,
                    entry_ts_ms=ts_ms - 1_000, exit_ts_ms=ts_ms,
                    exit_price=100.0, exit_reason="tp",
                    realized_pnl_usd=pnl, status=TradeStatus.CLOSED.value,
                ))
            # OPEN trade — excluded.
            sess.add(TradeRow(
                id="open1", strategy="x", proposal_id="po",
                symbol="ETHUSDT", market="spot", side="long",
                qty=1.0, leverage=1, entry_price=100.0,
                entry_ts_ms=anchor + 2 * DAY_MS, exit_ts_ms=None,
                exit_price=None, realized_pnl_usd=None,
                status=TradeStatus.OPEN.value,
            ))
            # CLOSED but exit_ts_ms None — excluded by SQL filter.
            sess.add(TradeRow(
                id="missing_exit", strategy="x", proposal_id="pm",
                symbol="ETHUSDT", market="spot", side="long",
                qty=1.0, leverage=1, entry_price=100.0,
                entry_ts_ms=anchor + 3 * DAY_MS, exit_ts_ms=None,
                exit_price=99.0, realized_pnl_usd=-1.0,
                status=TradeStatus.CLOSED.value,
            ))
        out = await s.realized_pnl_by_day(anchor, anchor + 5 * DAY_MS)
        assert out == {
            anchor: 7.0,
            anchor + DAY_MS: 5.0,
            anchor + 4 * DAY_MS: 1.0,
        }


# ---------------------------------------------------------------------------
# build_equity_series

@pytest.mark.asyncio
async def test_build_equity_series_threads_today_pnl_and_history() -> None:
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        s = Storage(database_url=url)
        await s.init()
        # Today's UTC midnight; populate yesterday with a +50 close.
        anchor_today = (1_700_000_000_000 // DAY_MS) * DAY_MS
        yesterday = anchor_today - DAY_MS
        async with s.session() as sess, sess.begin():
            sess.add(TradeRow(
                id="y1", strategy="x", proposal_id="p1",
                symbol="BTCUSDT", market="spot", side="long",
                qty=1.0, leverage=1, entry_price=100.0,
                entry_ts_ms=yesterday - 1_000, exit_ts_ms=yesterday + 1_000,
                exit_price=150.0, exit_reason="tp",
                realized_pnl_usd=50.0, status=TradeStatus.CLOSED.value,
            ))
        ts = await build_equity_series(
            s, start_equity_usd=1000.0,
            today_pnl_usd=-10.0,
            now_ms=anchor_today + 3_600_000,
            lookback_days=3,
        )
        # 4 rows (lookback_days + 1): days [-3, -2, -1, 0] from today.
        assert len(ts.equity_curve) == 4
        # day -3, -2 are flat (no trades): equity = start = 1000
        assert ts.equity_curve[0] == 1000.0
        assert ts.equity_curve[1] == 1000.0
        # day -1 (yesterday) closed +50 ⇒ 1050
        assert ts.equity_curve[2] == 1050.0
        # day 0 (today) uses today_pnl_usd = -10 ⇒ 1040
        assert ts.equity_curve[3] == 1040.0
        # daily_pnl_pct: first row 0; yesterday +50/1000 = 5%; today -10/1050
        assert ts.daily_pnl_pct[0] == 0.0
        assert ts.daily_pnl_pct[1] == 0.0
        assert ts.daily_pnl_pct[2] == pytest.approx(5.0)
        assert ts.daily_pnl_pct[3] == pytest.approx(-10.0 / 1050.0 * 100.0)
        assert ts.last_day_ms == anchor_today


@pytest.mark.asyncio
async def test_build_equity_series_empty_storage_flat_curve() -> None:
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        s = Storage(database_url=url)
        await s.init()
        ts = await build_equity_series(
            s, start_equity_usd=500.0, today_pnl_usd=0.0,
            now_ms=1_700_000_000_000, lookback_days=5,
        )
        assert len(ts.equity_curve) == 6
        assert all(e == 500.0 for e in ts.equity_curve)
        assert all(p == 0.0 for p in ts.daily_pnl_pct)


# ---------------------------------------------------------------------------
# Orchestrator wire-up via direct method exercise.
# We don't spin up the full Orchestrator (it would require Binance + Telegram
# wiring); instead we patch the methods we exercise onto a minimal stub.

class _StubTelegram:
    async def send_info(self, *_a, **_k): pass
    async def send_critical(self, *_a, **_k): pass
    async def send_proposal(self, *_a, **_k): pass


class _StubStorage:
    async def save_proposal(self, *_a, **_k): pass
    async def audit(self, *_a, **_k): pass


class _StubRiskGate:
    """Stub that mirrors RiskGate.check() — always passes with a fixed-size
    decision. Lets us inspect what the orchestrator does to qty/notional
    AFTER the gate returns ok."""

    def check(self, signal, acct, now_ms, market, leverage,
              max_notional_override=None):
        from src.tools.risk_gate import RiskDecision
        return RiskDecision(ok=True, qty=10.0, notional_usd=1000.0,
                             leverage=leverage)


@pytest.mark.asyncio
async def test_propose_blocks_when_circuit_says_no_new_entries() -> None:
    from src.orchestrator import Orchestrator
    # Sidestep the full ctor — instantiate without running __init__.
    orch = Orchestrator.__new__(Orchestrator)
    # Minimum attributes touched by _propose.
    orch.s = type("S", (), {"symbol_list": ["BTCUSDT"]})()
    orch.correlation = None
    orch.notional_ramp = None
    orch.equity = 1000.0
    orch.pnl_today = 0.0
    orch.consecutive_losses = 0
    orch.open_positions = []
    orch.last_trade_ms_by_symbol = {}
    orch.halted_until_ms = 0
    orch.risk = _StubRiskGate()
    orch.pending = {}
    orch.storage = _StubStorage()
    orch.telegram = None
    orch.executor = None
    orch.position_manager = None
    # The block path
    orch.circuit_state = CircuitState(
        size_multiplier=0.0, flatten=False, no_new_entries=True,
        cooloff_until_ms=0, triggered=("daily_loss_block",),
        reason="today's PnL −3.5% ≤ −3.0% block threshold",
    )
    sig = Signal(symbol="BTCUSDT", side="long", confidence=0.6, score=0.5,
                 entry=100, stop=98, take_profit=104, edge_bps=20,
                 features=FeatureVector())
    p, reject = await orch._propose(sig, market="spot", leverage=1)
    assert p is None
    assert reject is not None and "risk circuit" in reject
    assert "daily_loss_block" not in reject  # we surface the human reason


@pytest.mark.asyncio
async def test_propose_halves_qty_when_size_multiplier_is_half() -> None:
    from src.orchestrator import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)
    orch.s = type("S", (), {
        "symbol_list": ["BTCUSDT"], "approval_timeout_sec": 60,
        "auto_approve_max_notional_usd": 100_000.0,  # auto-approve path
        "twofa_threshold_notional_usd": 1_000_000.0,
    })()
    orch.correlation = None
    orch.notional_ramp = None
    orch.equity = 1000.0
    orch.pnl_today = 0.0
    orch.consecutive_losses = 0
    orch.open_positions = []
    orch.last_trade_ms_by_symbol = {}
    orch.halted_until_ms = 0
    orch.risk = _StubRiskGate()
    orch.pending = {}
    orch.storage = _StubStorage()
    orch.telegram = None
    orch.executor = None  # _submit() short-circuits when executor is None
    orch.position_manager = None
    orch.circuit_state = CircuitState(
        size_multiplier=0.5, flatten=False, no_new_entries=False,
        cooloff_until_ms=0, triggered=("dd_halve",),
        reason="trailing DD 10.5% ≥ halve threshold 10.0%",
    )
    sig = Signal(symbol="BTCUSDT", side="long", confidence=0.6, score=0.5,
                 entry=100, stop=98, take_profit=104, edge_bps=20,
                 features=FeatureVector())
    p, reject = await orch._propose(sig, market="spot", leverage=1)
    assert reject is None
    assert p is not None
    # Stub gate returned qty=10, notional=1000; circuit halves both.
    assert p.qty == pytest.approx(5.0)
    assert p.notional_usd == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_propose_passes_through_when_circuit_clear() -> None:
    from src.orchestrator import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)
    orch.s = type("S", (), {
        "symbol_list": ["BTCUSDT"], "approval_timeout_sec": 60,
        "auto_approve_max_notional_usd": 100_000.0,
        "twofa_threshold_notional_usd": 1_000_000.0,
    })()
    orch.correlation = None
    orch.notional_ramp = None
    orch.equity = 1000.0
    orch.pnl_today = 0.0
    orch.consecutive_losses = 0
    orch.open_positions = []
    orch.last_trade_ms_by_symbol = {}
    orch.halted_until_ms = 0
    orch.risk = _StubRiskGate()
    orch.pending = {}
    orch.storage = _StubStorage()
    orch.telegram = None
    orch.executor = None
    orch.position_manager = None
    # circuit_state None ⇒ subsystem disabled / never evaluated.
    orch.circuit_state = None
    sig = Signal(symbol="BTCUSDT", side="long", confidence=0.6, score=0.5,
                 entry=100, stop=98, take_profit=104, edge_bps=20,
                 features=FeatureVector())
    p, reject = await orch._propose(sig, market="spot", leverage=1)
    assert reject is None
    assert p is not None and p.qty == pytest.approx(10.0)
    assert p.notional_usd == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_propose_pair_blocks_but_does_not_rescale() -> None:
    """Pair trades are dollar-hedged at the strategy level; rescaling one
    side would unbalance the hedge. The block path applies but qty/notional
    must not be touched in propose_pair (we verify via the block path here
    since the non-block path isn't reachable without PairExecutor)."""
    from src.orchestrator import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)
    orch.s = type("S", (), {
        "max_daily_loss_pct": 3.0, "symbol_list": ["BTCUSDT", "ETHUSDT"],
    })()
    orch.pair_executor = "stub"
    orch.position_manager = "stub"
    orch.equity = 1000.0
    orch.pnl_today = 0.0
    orch.halted_until_ms = 0
    orch.strategies = []
    orch.notional_ramp = None
    orch.circuit_state = CircuitState(
        size_multiplier=0.5, flatten=False, no_new_entries=True,
        cooloff_until_ms=0, triggered=("dd_cooloff_active",),
        reason="cooloff active",
    )
    pair = type("Pair", (), {
        "strategy": "test", "notional_usd": 100.0,
        "legs": [type("L", (), {"symbol": "BTCUSDT"})],
        "id": "x", "rationale": "",
    })()
    # propose_pair returns None (void) — we just want it to early-out without
    # raising and without touching pair_executor (a string).
    out = await orch.propose_pair(pair)
    assert out is None
