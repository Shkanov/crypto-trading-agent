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
