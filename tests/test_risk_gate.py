from __future__ import annotations

from src.config.settings import Settings
from src.models.types import FeatureVector, Signal
from src.tools.risk_gate import AccountState, RiskGate


def _settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _signal(entry=100.0, stop=98.0, tp=104.0, side="long", edge_bps=30.0, conf=0.6) -> Signal:
    return Signal(
        symbol="BTCUSDT", side=side, confidence=conf, score=conf if side == "long" else -conf,
        entry=entry, stop=stop, take_profit=tp, edge_bps=edge_bps,
        features=FeatureVector(),
    )


def _acct(equity=1000.0, pnl=0.0, losses=0):
    return AccountState(
        equity_usd=equity, pnl_today_usd=pnl, consecutive_losses=losses,
        open_positions=[], last_trade_ms_by_symbol={},
    )


def test_basic_long_passes():
    # Per-coin cap defaults to 20% of equity ($200 on a $1k account); the
    # signal here sizes to ~$500 (1% of $1k risk, $2 stop distance, hits
    # max_notional cap at $500). Lift both caps so this test exercises only
    # the basic-pass path, not per-coin enforcement.
    g = RiskGate(_settings(risk_per_trade_pct=1.0, max_notional_usd=500,
                            max_exposure_per_coin_pct=100.0,
                            max_correlated_exposure_pct=100.0))
    d = g.check(_signal(), _acct(), now_ms=1, market="spot", leverage=1)
    assert d.ok, d.reason
    assert d.qty > 0 and d.notional_usd > 0


def test_daily_loss_cap_blocks():
    g = RiskGate(_settings(max_daily_loss_pct=3.0))
    d = g.check(_signal(), _acct(pnl=-31.0), now_ms=1, market="spot", leverage=1)
    assert not d.ok and "daily" in d.reason


def test_leverage_cap_blocks():
    g = RiskGate(_settings(max_leverage=3))
    d = g.check(_signal(), _acct(), now_ms=1, market="perps", leverage=10)
    assert not d.ok and "leverage" in d.reason


def test_negative_edge_blocks():
    g = RiskGate()
    d = g.check(_signal(edge_bps=-5.0), _acct(), now_ms=1, market="spot", leverage=1)
    assert not d.ok and "edge" in d.reason


def test_min_gap_blocks():
    g = RiskGate(_settings(min_gap_between_trades_sec=300))
    a = _acct()
    a.last_trade_ms_by_symbol["BTCUSDT"] = 1_000_000
    d = g.check(_signal(), a, now_ms=1_100_000, market="spot", leverage=1)  # 100s later
    assert not d.ok and "gap" in d.reason
