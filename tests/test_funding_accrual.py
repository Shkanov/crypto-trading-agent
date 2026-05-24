from __future__ import annotations

import os
import tempfile

import pytest

from src.models.types import PairLeg, PairProposal, Trade, TradeStatus
from src.services.storage import Storage


@pytest.mark.asyncio
async def test_funding_accrued_in_realized_pnl():
    """The headline F1 bug: a short-perp leg that collected $1 of funding
    should close with realized_pnl = gross - fees + funding."""
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        st = Storage(database_url=url)
        await st.init()

        t = Trade(
            strategy="funding_harvest", proposal_id="p1",
            symbol="BTCUSDT", market="perps", side="short",
            qty=0.01, leverage=2, entry_price=100_000.0, fee_total_usd=0.5,
        )
        ok = await st.open_trade(t)
        assert ok

        # Two funding boundaries credit $0.50 each → $1 total.
        await st.credit_funding(t.id, 0.5)
        await st.credit_funding(t.id, 0.5)

        # Close perp at $98k (favorable for short — gross $20).
        closed = await st.close_trade(t.id, exit_price=98_000.0,
                                       exit_reason="funding_flip",
                                       fee_total_usd=0.5)
        assert closed is not None
        # gross = (98000 - 100000) * 0.01 = -20; short flips → +20.
        # PnL = 20 - (0.5 entry + 0.5 exit) + 1 funding = 20.0
        assert closed.realized_pnl_usd == pytest.approx(20.0, abs=0.01)
        assert closed.funding_accrued_usd == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_open_trade_duplicate_id_returns_false():
    """F12: open_trade must NOT silently no-op on duplicate. It must signal
    the bug so a retry-then-second-leg path doesn't leave you single-legged."""
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        st = Storage(database_url=url)
        await st.init()
        t = Trade(strategy="x", proposal_id="p", symbol="BTCUSDT",
                  market="spot", side="long", qty=1.0, entry_price=100.0)
        assert await st.open_trade(t) is True
        # Second call with same ID must return False.
        assert await st.open_trade(t) is False


@pytest.mark.asyncio
async def test_credit_funding_skips_closed_trade():
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        st = Storage(database_url=url)
        await st.init()
        t = Trade(strategy="x", proposal_id="p", symbol="BTCUSDT",
                  market="perps", side="short", qty=0.01, entry_price=100_000.0)
        await st.open_trade(t)
        await st.close_trade(t.id, exit_price=99_000.0, exit_reason="manual",
                              fee_total_usd=0.0)
        # Crediting after close must be rejected.
        new_total = await st.credit_funding(t.id, 5.0)
        assert new_total is None
