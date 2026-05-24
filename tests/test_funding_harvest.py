from __future__ import annotations

import pytest

from src.services.funding_monitor import FundingMonitor, FundingPoint
from src.strategies.funding_harvest import FundingHarvestStrategy, HarvestParams


def _seed(fm: FundingMonitor, symbol: str, rates_bps: list[float], mark: float = 100.0) -> None:
    """Inject a fake history at the desired bps levels."""
    st = fm.state[symbol]
    for i, bps in enumerate(rates_bps):
        st.history.append(FundingPoint(
            symbol=symbol, rate=bps / 10_000, mark_price=mark, index_price=mark,
            next_funding_ms=i, ts_ms=i,
        ))
    st.last = st.history[-1]


def test_funding_monitor_avg_and_extreme():
    fm = FundingMonitor(symbols=["BTCUSDT"])
    _seed(fm, "BTCUSDT", [5.0] * 20 + [15.0])
    assert fm.current_bps("BTCUSDT") == pytest.approx(15.0)
    # avg over last 21 = (20*5 + 15) / 21 ≈ 5.476
    assert fm.avg_bps("BTCUSDT", n=21) == pytest.approx(5.476, abs=1e-2)
    assert fm.is_extreme("BTCUSDT", threshold_bps=10.0) == 1
    assert fm.is_extreme("BTCUSDT", threshold_bps=20.0) == 0


def test_annualized_pct():
    fm = FundingMonitor(symbols=["BTCUSDT"])
    _seed(fm, "BTCUSDT", [10.0])  # 10 bps per 8h
    # 10 bps * 3 * 365 = 10950 bps = 109.5%
    assert fm.annualized_pct("BTCUSDT") == pytest.approx(109.5, abs=0.1)


def test_no_extreme_when_below_threshold():
    fm = FundingMonitor(symbols=["BTCUSDT"])
    _seed(fm, "BTCUSDT", [3.0])
    assert fm.is_extreme("BTCUSDT", threshold_bps=10.0) == 0


def test_short_side_signaled_on_negative_funding():
    fm = FundingMonitor(symbols=["BTCUSDT"])
    _seed(fm, "BTCUSDT", [-15.0])
    assert fm.is_extreme("BTCUSDT", threshold_bps=10.0) == -1
