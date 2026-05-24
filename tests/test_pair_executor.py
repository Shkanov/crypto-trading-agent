from __future__ import annotations

import os
import tempfile

import pytest

from src.config.settings import Settings
from src.models.types import PairLeg, PairProposal
from src.services.storage import Storage
from src.tools.pair_executor import PairExecutor


def _binance_stub():
    class _S: pass
    return _S()


def _settings() -> Settings:
    s = Settings()
    s.spot_taker_fee_bps = 10.0
    s.perps_taker_fee_bps = 5.0
    s.slippage_bps = 5.0
    return s


@pytest.mark.asyncio
async def test_close_paper_uses_per_market_price_map():
    """F2: perp leg must close at perp mark; spot leg must close at spot mid."""
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        st = Storage(database_url=url)
        await st.init()
        s = _settings()
        pe = PairExecutor(_binance_stub(), st, paper=True, settings=s)

        pair = PairProposal(
            id="pair1", strategy="funding_harvest",
            legs=[
                PairLeg(symbol="BTCUSDT", market="spot", side="BUY",
                        qty=0.001, expected_price=100_000.0, leverage=1),
                PairLeg(symbol="BTCUSDT", market="perps", side="SELL",
                        qty=0.001, expected_price=100_500.0, leverage=2),
            ],
            notional_usd=200.0,
            expires_at_ms=0,
        )
        result = await pe.open_pair(pair)
        assert result.ok
        assert len(result.legs) == 2

        # Close: spot leg should close at spot mid=99_900, perp leg at perp mark=100_400.
        price_map = {
            ("BTCUSDT", "spot"): 99_900.0,
            ("BTCUSDT", "perps"): 100_400.0,
        }
        closed = await pe.close_pair(result.legs, reason="funding_flip",
                                      price_map=price_map)
        assert len(closed) == 2
        spot_leg = next(t for t in closed if t.market == "spot")
        perp_leg = next(t for t in closed if t.market == "perps")
        # Spot exit (long close = SELL) fills LOWER than ref by 5bps → 99_900 * (1 - 5/10_000) = 99_850.05
        assert spot_leg.exit_price == pytest.approx(99_900.0 * 0.9995, rel=1e-5)
        # Perp exit (short close = BUY) fills HIGHER than ref by 5bps → 100_400 * 1.0005
        assert perp_leg.exit_price == pytest.approx(100_400.0 * 1.0005, rel=1e-5)


@pytest.mark.asyncio
async def test_open_paper_sell_fills_below_expected():
    """F5: SELL fill should be ADVERSE (lower than expected), not favorable.
    The comment said 'favorable to seller' but the code is actually correct;
    this test pins that down so the comment fix in 2026 doesn't drift the code."""
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        st = Storage(database_url=url)
        await st.init()
        s = _settings()
        pe = PairExecutor(_binance_stub(), st, paper=True, settings=s)

        pair = PairProposal(
            id="pair2", strategy="funding_harvest",
            legs=[
                PairLeg(symbol="BTCUSDT", market="spot", side="BUY",
                        qty=0.001, expected_price=100_000.0, leverage=1),
                PairLeg(symbol="BTCUSDT", market="perps", side="SELL",
                        qty=0.001, expected_price=100_000.0, leverage=2),
            ],
            notional_usd=200.0,
            expires_at_ms=0,
        )
        result = await pe.open_pair(pair)
        assert result.ok
        spot_leg = next(t for t in result.legs if t.market == "spot")
        perp_leg = next(t for t in result.legs if t.market == "perps")
        # BUY fills HIGHER (paid more); SELL fills LOWER (received less). Both adverse.
        assert spot_leg.entry_price == pytest.approx(100_000.0 * 1.0005, rel=1e-5)
        assert perp_leg.entry_price == pytest.approx(100_000.0 * 0.9995, rel=1e-5)
