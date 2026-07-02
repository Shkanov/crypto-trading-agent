"""Bug 1 fix: dfunding must HOLD (not close-then-fail-to-open) when the
effective per-leg notional — after the ramp cap + risk-circuit multiplier —
falls below the exchange minimum. Otherwise a drawdown-halve strands the book
flat and paralysed (which is exactly what happened live on 2026-06/07)."""
from __future__ import annotations

import pytest

from src.strategies.dfunding_carry import DFundingCarryParams, DFundingCarryStrategy


class _FakeTelegram:
    def __init__(self):
        self.infos: list[str] = []

    async def send_info(self, msg: str) -> None:
        self.infos.append(msg)


class _FakeCtx:
    """Minimal StrategyContext for the feasibility guard. effective_notional
    returns a fixed value to simulate the ramp/circuit shrink."""
    def __init__(self, effective_per_leg: float, equity: float = 40.0):
        self._eff = effective_per_leg
        self._equity = equity
        self.telegram = _FakeTelegram()

    def equity_available_usd(self, name=None) -> float:
        return self._equity

    def effective_notional(self, nominal_usd: float) -> float:
        return self._eff

    def open_trades(self, name=None):
        return []


@pytest.mark.asyncio
async def test_rebalance_holds_when_effective_leg_below_min_notional():
    strat = DFundingCarryStrategy(DFundingCarryParams(min_leg_notional_usd=5.5))
    strat.ctx = _FakeCtx(effective_per_leg=3.3)   # circuit 0.5× of $6.67 → below $5.5

    compute_called = []

    async def _boom():
        compute_called.append(1)
        raise AssertionError("_compute_signal must NOT run when held")

    strat._compute_signal = _boom  # type: ignore[assignment]

    await strat._rebalance()

    assert compute_called == []                # guard returned before the API build
    assert strat.last_rebalance_ms == 0        # timer NOT advanced → retries next cycle
    assert strat._held_alerted is True         # alert latched
    assert len(strat.ctx.telegram.infos) == 1  # alerted once

    # Second held cycle: must NOT re-alert (throttled).
    await strat._rebalance()
    assert len(strat.ctx.telegram.infos) == 1


@pytest.mark.asyncio
async def test_rebalance_proceeds_when_effective_leg_above_min_notional():
    strat = DFundingCarryStrategy(DFundingCarryParams(min_leg_notional_usd=5.5))
    strat.ctx = _FakeCtx(effective_per_leg=6.67)  # healthy sizing

    proceeded = []

    async def _stub_compute():
        proceeded.append(1)
        return {}          # empty signal → _rebalance returns after coverage check

    strat._compute_signal = _stub_compute  # type: ignore[assignment]

    await strat._rebalance()

    assert proceeded == [1]                 # guard passed → signal build ran
