from __future__ import annotations

import os
import tempfile

import pytest

from src.models.types import Trade
from src.services.storage import Storage
from src.tools.position_manager import PositionManager, TickRange, _exit_with_slippage


def test_exit_slippage_directions():
    # Long stop: adverse = lower price.
    assert _exit_with_slippage(100.0, "long", "stop", 25.0, 5.0) == pytest.approx(99.75)
    # Short stop: adverse = higher price.
    assert _exit_with_slippage(100.0, "short", "stop", 25.0, 5.0) == pytest.approx(100.25)
    # TP uses tp_slip_bps.
    assert _exit_with_slippage(110.0, "long", "tp", 25.0, 5.0) == pytest.approx(109.945)


@pytest.mark.asyncio
async def test_stop_hit_closes_long_with_loss():
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        st = Storage(database_url=url)
        await st.init()
        closed: list[Trade] = []

        async def on_close(t: Trade) -> None:
            closed.append(t)

        pm = PositionManager(storage=st, on_close=on_close, paper=True)
        trade = Trade(
            proposal_id="p1", symbol="BTCUSDT", market="spot", side="long",
            qty=0.1, entry_price=100.0, intended_stop=98.0, intended_tp=104.0,
        )
        await pm.register(trade)

        # Bar low pierces the stop.
        await pm.on_bar(TickRange(symbol="BTCUSDT", high=99.5, low=97.5, close=98.5, ts_ms=1))

        assert len(closed) == 1
        c = closed[0]
        assert c.exit_reason == "stop"
        # PnL should be negative (close near 98 with 25 bps adverse → ~97.755).
        assert (c.realized_pnl_usd or 0) < 0


@pytest.mark.asyncio
async def test_tp_hit_closes_long_with_profit():
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        st = Storage(database_url=url)
        await st.init()
        closed: list[Trade] = []

        async def on_close(t: Trade) -> None:
            closed.append(t)

        pm = PositionManager(storage=st, on_close=on_close, paper=True)
        trade = Trade(
            proposal_id="p2", symbol="BTCUSDT", market="spot", side="long",
            qty=0.1, entry_price=100.0, intended_stop=98.0, intended_tp=104.0,
        )
        await pm.register(trade)

        await pm.on_bar(TickRange(symbol="BTCUSDT", high=104.5, low=100.2, close=104.1, ts_ms=1))

        assert len(closed) == 1
        c = closed[0]
        assert c.exit_reason == "tp"
        assert (c.realized_pnl_usd or 0) > 0


@pytest.mark.asyncio
async def test_stop_and_tp_in_same_bar_resolves_stop():
    """Conservative assumption: without intra-bar data, resolve the worse outcome."""
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        st = Storage(database_url=url)
        await st.init()
        closed: list[Trade] = []

        async def on_close(t: Trade) -> None:
            closed.append(t)

        pm = PositionManager(storage=st, on_close=on_close, paper=True)
        trade = Trade(
            proposal_id="p3", symbol="BTCUSDT", market="spot", side="long",
            qty=0.1, entry_price=100.0, intended_stop=98.0, intended_tp=104.0,
        )
        await pm.register(trade)

        # Bar spans 96..106 (both stop and TP touched).
        await pm.on_bar(TickRange(symbol="BTCUSDT", high=106.0, low=96.0, close=101.0, ts_ms=1))

        assert len(closed) == 1
        assert closed[0].exit_reason == "stop"


@pytest.mark.asyncio
async def test_daily_pnl_and_consecutive_losses_track_closed_trades():
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        st = Storage(database_url=url)
        await st.init()

        async def noop(t: Trade) -> None:
            pass

        pm = PositionManager(storage=st, on_close=noop, paper=True)

        # Loser: long, stop hit.
        t1 = Trade(proposal_id="p1", symbol="BTCUSDT", market="spot", side="long",
                   qty=1.0, entry_price=100.0, intended_stop=98.0, intended_tp=104.0)
        await pm.register(t1)
        await pm.on_bar(TickRange("BTCUSDT", high=99.0, low=97.0, close=98.0, ts_ms=1))

        # Loser: long, stop hit again.
        t2 = Trade(proposal_id="p2", symbol="BTCUSDT", market="spot", side="long",
                   qty=1.0, entry_price=100.0, intended_stop=98.0, intended_tp=104.0)
        await pm.register(t2)
        await pm.on_bar(TickRange("BTCUSDT", high=99.0, low=97.0, close=98.0, ts_ms=2))

        pnl = await st.realized_pnl_today_usd()
        assert pnl < 0, f"two losers should produce negative pnl, got {pnl}"
        cl = await st.consecutive_losses()
        assert cl == 2

        # Winner: long, TP hit — should RESET consecutive losses.
        t3 = Trade(proposal_id="p3", symbol="BTCUSDT", market="spot", side="long",
                   qty=1.0, entry_price=100.0, intended_stop=98.0, intended_tp=104.0)
        await pm.register(t3)
        await pm.on_bar(TickRange("BTCUSDT", high=105.0, low=100.5, close=104.5, ts_ms=3))

        cl2 = await st.consecutive_losses()
        assert cl2 == 0


# ── Live-close path (Bug 2 fix): force_close must place a real exchange order
#    and must NOT phantom-close (DB-close a still-open position) on failure. ──

def _mk_storage(td):
    url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
    return Storage(database_url=url)


@pytest.mark.asyncio
async def test_live_force_close_uses_real_fill_and_closes():
    with tempfile.TemporaryDirectory() as td:
        st = _mk_storage(td)
        await st.init()
        closed: list[Trade] = []

        async def on_close(t: Trade) -> None:
            closed.append(t)

        calls: list[str] = []

        async def close_exec(t: Trade):
            calls.append(t.id)
            return 95.0                       # real exchange fill price

        pm = PositionManager(storage=st, on_close=on_close, paper=False,
                             close_executor=close_exec)
        trade = Trade(proposal_id="p", symbol="HOMEUSDT", market="perps",
                      side="short", qty=100.0, entry_price=100.0)
        await pm.register(trade)

        res = await pm.force_close(trade.id, 99.0, "dfunding_rebalance")
        assert res is not None
        assert calls == [trade.id]                    # exchange order placed
        assert trade.id not in pm.open                # removed from open set
        assert res.exit_price == pytest.approx(95.0)  # used REAL fill, not 99.0
        assert (await st.list_open_trades()) == []    # DB agrees: flat
        # PnL correctness (item 2): short 100 units from 100.0 → fill 95.0 is a
        # +$500 gross price gain. With the real fill recorded, that price move
        # lands in realized_pnl (not lost to exit==entry / mislabeled to
        # funding, which was the phantom-close symptom). Allow the exit fee.
        assert res.realized_pnl_usd > 400.0
        assert res.funding_accrued_usd == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_live_force_close_failure_keeps_position_open_no_phantom():
    with tempfile.TemporaryDirectory() as td:
        st = _mk_storage(td)
        await st.init()

        async def on_close(t: Trade) -> None:
            pass

        async def close_exec_fail(t: Trade):
            return None                       # exchange close failed

        pm = PositionManager(storage=st, on_close=on_close, paper=False,
                             close_executor=close_exec_fail)
        trade = Trade(proposal_id="p", symbol="HOMEUSDT", market="perps",
                      side="short", qty=100.0, entry_price=100.0)
        await pm.register(trade)

        res = await pm.force_close(trade.id, 99.0, "dfunding_rebalance")
        assert res is None                            # not closed
        assert trade.id in pm.open                    # STILL tracked open
        still_open = await st.list_open_trades()      # DB still shows it OPEN
        assert any(t.id == trade.id for t in still_open)


@pytest.mark.asyncio
async def test_paper_force_close_ignores_close_executor():
    with tempfile.TemporaryDirectory() as td:
        st = _mk_storage(td)
        await st.init()

        async def on_close(t: Trade) -> None:
            pass

        called = []

        async def close_exec(t: Trade):
            called.append(t.id)
            return 1.0

        # Paper mode: close_executor must NOT be invoked (no real exchange).
        pm = PositionManager(storage=st, on_close=on_close, paper=True,
                             close_executor=close_exec)
        trade = Trade(proposal_id="p", symbol="HOMEUSDT", market="perps",
                      side="long", qty=1.0, entry_price=100.0)
        await pm.register(trade)
        res = await pm.force_close(trade.id, 101.0, "manual")
        assert res is not None
        assert called == []                           # executor untouched
        assert res.exit_price == pytest.approx(101.0)
