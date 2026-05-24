"""Confluence-based signal generator.

Reads `IndicatorSnapshot`s across timeframes (e.g., 1h regime, 15m bias, 5m
trigger, 1m exec), folds them into a `FeatureVector` in [-1, 1], applies the
weighted-vote scorer from `StrategyConfig`, and emits a `Signal` when the
score breaches the threshold AND the higher-timeframe regime filter agrees.

This module is also pure — no I/O, no LLM.
"""
from __future__ import annotations

import math
from typing import Optional

import structlog

from src.config.settings import get_settings
from src.models.types import FeatureVector, IndicatorSnapshot, Signal, StrategyConfig

log = structlog.get_logger(__name__)


def _sign(x: float) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _normalize_macd_hist(hist: float, atr: float) -> float:
    if atr <= 0:
        return 0.0
    return _clip(hist / (atr * 0.5))


def _normalize_rsi(rsi: float) -> float:
    # 30 -> -1, 50 -> 0, 70 -> +1 (saturating beyond)
    return _clip((rsi - 50.0) / 20.0)


def _normalize_volume_z(z: float) -> float:
    return _clip(z / 3.0)


def _trend_score(snap: IndicatorSnapshot) -> float:
    contribs: list[float] = []
    if snap.ema21 is not None and snap.ema55 is not None:
        diff = snap.ema21 - snap.ema55
        scale = max(abs(snap.ema55) * 0.005, 1e-9)
        contribs.append(_clip(diff / scale))
    if snap.supertrend_dir is not None:
        contribs.append(float(snap.supertrend_dir))
    if snap.vwap is not None:
        diff = snap.close - snap.vwap
        scale = max(snap.atr14 or (snap.close * 0.005), 1e-9)
        contribs.append(_clip(diff / scale))
    # Donchian breakout direction: close near upper = bullish, near lower = bearish.
    if (snap.donchian_upper is not None and snap.donchian_lower is not None
            and snap.donchian_upper > snap.donchian_lower):
        rng = snap.donchian_upper - snap.donchian_lower
        # Map [lower, upper] to [-1, +1].
        contribs.append(_clip(((snap.close - snap.donchian_lower) / rng) * 2.0 - 1.0))
    base = sum(contribs) / len(contribs) if contribs else 0.0
    # ADX (trend strength) damps the trend contribution when there's no
    # actual trend. <15 = chop (×0.25); 15-25 = weak (×0.6); 25-40 = decent
    # (×1.0); >40 = very strong (clamped to ×1.0). This is a multiplicative
    # gate — sign of the trend stays the same, magnitude shrinks in chop.
    if snap.adx14 is not None:
        if snap.adx14 < 15:
            base *= 0.25
        elif snap.adx14 < 25:
            base *= 0.6
    return base


def _momentum_score(snap: IndicatorSnapshot) -> float:
    contribs: list[float] = []
    if snap.macd_hist is not None and snap.atr14 is not None:
        contribs.append(_normalize_macd_hist(snap.macd_hist, snap.atr14))
    if snap.rsi14 is not None:
        contribs.append(_normalize_rsi(snap.rsi14))
    # StochRSI K maps 20→-1 (oversold), 50→0, 80→+1 (overbought). Same
    # sign convention as RSI for trend-follow: higher = stronger momentum.
    if snap.stoch_rsi_k is not None:
        contribs.append(_clip((snap.stoch_rsi_k - 50.0) / 30.0))
    return sum(contribs) / len(contribs) if contribs else 0.0


def _volume_score(snap: IndicatorSnapshot) -> float:
    contribs: list[float] = []
    if snap.volume_z is not None:
        contribs.append(_normalize_volume_z(snap.volume_z))
    if snap.cvd_slope is not None:
        # heuristic: positive CVD slope = buyers in control
        contribs.append(_clip(snap.cvd_slope / (abs(snap.cvd_slope) + 1e-9)))
    if snap.obv_slope is not None:
        # OBV is in raw volume units; sign-of-slope is the durable signal,
        # not magnitude (which scales with the symbol's typical volume).
        contribs.append(_clip(snap.obv_slope / (abs(snap.obv_slope) + 1e-9)))
    return sum(contribs) / len(contribs) if contribs else 0.0


def _volatility_score(snap: IndicatorSnapshot) -> float:
    # Squeeze (low BB width rank) preferred for breakouts on the trigger TF.
    if snap.bb_width_pct_rank is None:
        return 0.0
    # 0.0 = tightest 100 bars; map to +1 (favorable to breakout direction)
    return _clip(1.0 - 2.0 * snap.bb_width_pct_rank)


def _build_feature_vector(snap: IndicatorSnapshot) -> FeatureVector:
    return FeatureVector(
        trend=_trend_score(snap),
        momentum=_momentum_score(snap),
        volume=_volume_score(snap),
        volatility=_volatility_score(snap),
        pattern=0.0,  # phase-2: candlestick patterns at S/R
    )


ADX_TREND_MIN = 20.0  # below this on the regime TF, treat as no-trend


def htf_regime(snap: Optional[IndicatorSnapshot]) -> int:
    """Higher-TF regime filter. +1 bullish, -1 bearish, 0 unclear/chop.

    Requires ADX above ADX_TREND_MIN on the regime timeframe — otherwise we
    return 0 even if EMA/Supertrend agree. This prevents trend-follow from
    firing in choppy regimes where the mean-reversion strategy should own
    the bar instead.
    """
    if snap is None or snap.ema21 is None or snap.ema55 is None:
        return 0
    # ADX is optional — if missing (early bars), fall back to old behavior.
    if snap.adx14 is not None and snap.adx14 < ADX_TREND_MIN:
        return 0
    if snap.ema21 > snap.ema55 and (snap.supertrend_dir or 1) > 0:
        return 1
    if snap.ema21 < snap.ema55 and (snap.supertrend_dir or -1) < 0:
        return -1
    return 0


def score_features(fv: FeatureVector, cfg: StrategyConfig) -> float:
    w = cfg.feature_weights
    return (
        w["trend"] * fv.trend
        + w["momentum"] * fv.momentum
        + w["volume"] * fv.volume
        + w["volatility"] * fv.volatility
        + w["pattern"] * fv.pattern
    )


def generate_signal(
    symbol: str,
    trigger_snap: IndicatorSnapshot,
    htf_snap: Optional[IndicatorSnapshot],
    cfg: StrategyConfig,
) -> Optional[Signal]:
    """Generate a Signal from indicator snapshots, or None.

    `trigger_snap` is on the entry-trigger timeframe (e.g., 5m).
    `htf_snap` is on the regime timeframe (e.g., 1h).
    """
    if symbol not in cfg.allowed_symbols:
        return None
    if trigger_snap.atr14 is None or trigger_snap.atr14 <= 0:
        return None

    fv = _build_feature_vector(trigger_snap)
    score = score_features(fv, cfg)
    conf = abs(score)
    if conf < cfg.min_confidence:
        return None

    side: str
    if score >= cfg.long_score_threshold:
        side = "long"
    elif score <= cfg.short_score_threshold:
        side = "short"
    else:
        return None

    if side not in cfg.enabled_sides:
        return None

    if cfg.htf_regime_filter:
        regime = htf_regime(htf_snap)
        if regime != 0 and ((side == "long" and regime < 0) or (side == "short" and regime > 0)):
            return None

    entry = trigger_snap.close
    stop_dist = cfg.atr_stop_mult * trigger_snap.atr14
    if side == "long":
        stop = entry - stop_dist
        tp = entry + cfg.rr_target * stop_dist
    else:
        stop = entry + stop_dist
        tp = entry - cfg.rr_target * stop_dist

    # Edge model: confidence + RR vs round-trip cost (in bps).
    s = get_settings()
    fee_bps = s.perps_taker_fee_bps * 2 + s.slippage_bps
    expected_move_bps = (abs(tp - entry) / entry) * 10_000
    edge_bps = expected_move_bps * conf - fee_bps

    rationale = (
        f"score={score:+.2f} conf={conf:.0%} "
        f"trend={fv.trend:+.2f} mom={fv.momentum:+.2f} "
        f"vol={fv.volume:+.2f} vlt={fv.volatility:+.2f}"
    )

    return Signal(
        symbol=symbol, side=side, confidence=conf, score=score,
        entry=entry, stop=stop, take_profit=tp,
        edge_bps=edge_bps, features=fv, rationale=rationale,
    )
