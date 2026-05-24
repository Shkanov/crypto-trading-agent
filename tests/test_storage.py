from __future__ import annotations

import os
import tempfile

import pytest

from src.models.types import FeatureVector, Proposal, ProposalStatus, Signal, StrategyConfig
from src.services.storage import Storage


@pytest.mark.asyncio
async def test_save_and_load_proposal_roundtrips():
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        s = Storage(database_url=url)
        await s.init()
        sig = Signal(symbol="BTCUSDT", side="long", confidence=0.6, score=0.5,
                     entry=100, stop=98, take_profit=104, edge_bps=20,
                     features=FeatureVector())
        p = Proposal(id="abc123", signal=sig, market="spot", qty=0.01, notional_usd=1.0,
                     status=ProposalStatus.AWAITING_USER, expires_at_ms=1)
        await s.save_proposal(p)
        loaded = await s.get_proposal("abc123")
        assert loaded is not None and loaded.signal.symbol == "BTCUSDT"
        assert loaded.status == ProposalStatus.AWAITING_USER


@pytest.mark.asyncio
async def test_strategy_config_latest():
    with tempfile.TemporaryDirectory() as td:
        url = f"sqlite+aiosqlite:///{os.path.join(td, 'agent.db')}"
        s = Storage(database_url=url)
        await s.init()
        c1 = StrategyConfig(version=1, allowed_symbols=["BTCUSDT"])
        c2 = StrategyConfig(version=2, allowed_symbols=["BTCUSDT", "ETHUSDT"], notes="add ETH")
        await s.save_strategy_config(c1)
        await s.save_strategy_config(c2)
        latest = await s.latest_strategy_config()
        assert latest is not None and latest.version == 2
        assert "ETHUSDT" in latest.allowed_symbols
