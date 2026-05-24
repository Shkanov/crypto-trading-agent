from __future__ import annotations

from src.models.types import IndicatorSnapshot, StrategyConfig
from src.tools.signal_generator import generate_signal, score_features
from src.tools.signal_generator import _build_feature_vector


def _snap(close=100.0, ema21=101.0, ema55=99.0, macd_hist=0.5, rsi=65.0,
          atr=1.0, st_dir=1, vwap=99.5, vol_z=1.5, cvd_slope=200.0,
          bb_upper=103.0, bb_lower=97.0, bb_mid=100.0, bb_rank=0.1) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        symbol="BTCUSDT", timeframe="5m", close=close, ema21=ema21, ema55=ema55,
        macd_hist=macd_hist, rsi14=rsi, atr14=atr, supertrend_dir=st_dir, vwap=vwap,
        volume_z=vol_z, cvd_slope=cvd_slope,
        bb_upper=bb_upper, bb_middle=bb_mid, bb_lower=bb_lower,
        bb_width_pct_rank=bb_rank,
    )


def test_generates_long_in_uptrend():
    snap = _snap()
    cfg = StrategyConfig(allowed_symbols=["BTCUSDT"])
    sig = generate_signal("BTCUSDT", snap, snap, cfg)
    assert sig is not None
    assert sig.side == "long"
    assert sig.entry == 100.0
    assert sig.stop < sig.entry < sig.take_profit


def test_blocks_against_htf_regime():
    trig = _snap()  # bullish trigger
    htf = _snap(ema21=98.0, ema55=101.0, st_dir=-1)  # bearish 1h regime
    cfg = StrategyConfig(allowed_symbols=["BTCUSDT"])
    sig = generate_signal("BTCUSDT", trig, htf, cfg)
    assert sig is None


def test_score_is_signed():
    bullish = _snap()
    bearish = _snap(ema21=99.0, ema55=101.0, macd_hist=-0.5, rsi=35.0, st_dir=-1, vwap=100.5,
                    cvd_slope=-200.0, vol_z=1.5)
    cfg = StrategyConfig(allowed_symbols=["BTCUSDT"])
    fv_b = _build_feature_vector(bullish)
    fv_d = _build_feature_vector(bearish)
    assert score_features(fv_b, cfg) > 0
    assert score_features(fv_d, cfg) < 0
