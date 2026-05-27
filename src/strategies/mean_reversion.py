"""Mean-reversion strategy — counterpart to IndicatorConfluence.

Uses the SAME indicators (RSI, StochRSI, Bollinger, ADX, ATR) but in the
opposite direction. Premise: in low-trend regimes (ADX < ~20), liquid majors
oscillate around a mean (VWAP / BB middle), and extreme deviations revert.
This strategy waits for stretched + oversold/overbought conditions and
fades them back toward the mean.

Entry (long):
  - close < bb_lower                  (stretched below mean)
  - stoch_rsi_k < `stoch_oversold`    (sensitive OB/OS confirms)
  - rsi14 < `rsi_oversold`            (slower OB/OS confirms; reduces false-fades)
  - adx14 < `adx_max_for_meanrev`     (no strong trend — mean is meaningful)
  - bb_middle defined and > entry     (TP target above us)

Entry (short): mirror of the above with `stoch_overbought` / `rsi_overbought`
and close > bb_upper.

Exit:
  - take_profit = bb_middle (the mean we're reverting to)
  - stop_loss   = entry ± atr14 * `atr_stop_mult` (1.5x by default — tight,
    since mean-rev edge dies when momentum keeps going)

This module is pure: no I/O, no LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from src.config.settings import get_settings
from src.models.types import FeatureVector, IndicatorSnapshot, Kline, Signal, StrategyConfig
from src.strategies.base import Strategy, StrategyContext

log = structlog.get_logger(__name__)


@dataclass
class ScaledMeanReversionConfig:
    """Scaled mean-rev parameters (v2). Fixes the R/R inversion of v1
    by widening the stop, scaling into deeper stretches, and taking
    partial profits before the full revert.

    Geometry (long entry below BB lower, target BB middle):

      entry_1 ────┐
                  │  (price drops further)
      entry_2 ───>│
        avg ────  │            <─ TP1 close 50%
                  │            <─ TP2 close 50%
                  │
        stop ─── (3 ATR below avg, much wider than v1)

    With wider stops, the strategy can survive a deeper drawdown before
    being stopped out, giving the mean-revert thesis time. Partial TP
    captures gains on partial-reverts that don't reach the BB middle.
    """
    allowed_symbols: list[str]
    enabled_sides: tuple[str, ...] = ("long", "short")
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    adx_max_for_meanrev: float = 20.0
    # Scale-in: how many entries allowed and step between them in ATR.
    max_entries: int = 2
    scale_in_atr_step: float = 1.0
    # Stop is `atr_stop_mult` ATR from AVERAGE entry (using ATR at first entry).
    atr_stop_mult: float = 3.0
    # TP1 takes `tp1_close_fraction` of position off at this fraction of the
    # way from avg entry to TP2 anchor (BB middle). 0.5 = halfway.
    tp1_distance_frac: float = 0.5
    tp1_close_fraction: float = 0.5
    min_target_atr: float = 0.5
    time_stop_bars: int = 48
    htf_timeframe: str = "1h"
    trigger_timeframe_pref: tuple[str, ...] = ("5m", "15m", "3m", "1m")


@dataclass
class MeanReversionConfig:
    allowed_symbols: list[str]
    enabled_sides: tuple[str, ...] = ("long", "short")
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    adx_max_for_meanrev: float = 20.0
    atr_stop_mult: float = 1.5
    # If mean (bb_middle) is closer than min_target_atr * ATR, skip — too
    # little room to make the trade worth costs.
    min_target_atr: float = 0.5
    # Mean-rev positions that don't revert within `time_stop_bars` trigger-TF
    # bars are closed at market. Critical for honest backtest accounting: a
    # mean-rev trade that's still open after this many bars is a failed
    # revert thesis, not a winner-in-progress. 24 bars ≈ 2h on 5m, 6h on 15m,
    # 1 day on 1h — long enough for the BB midline to come to the price,
    # short enough that we cut losing fades before the market trends through.
    time_stop_bars: int = 24
    htf_timeframe: str = "1h"
    trigger_timeframe_pref: tuple[str, ...] = ("5m", "15m", "3m", "1m")

    # Strict Hurst+VR+OU regime gate (Chan ch.2; Macrosynergy 2023). When True,
    # replaces ADX<20 as the regime check; called from the simulator with the
    # last N closing prices. The 3-test stack typically kills 60-80% of false
    # positives in directional crypto regimes.
    use_strict_regime_gate: bool = False

    # Triple-barrier exits (López de Prado, AFML ch.3). When True, replaces
    # fixed ATR stops/TP with σ-scaled barriers + a hard time stop derived
    # from the OU half-life estimate.
    use_triple_barrier: bool = False
    tp_sigma: float = 0.7        # take profit at 0.7 σ of entry deviation
    sl_sigma: float = 2.0        # stop loss at 2.0 σ of entry deviation
    time_stop_mult_of_half_life: float = 3.0  # hard time stop = 3 × OU HL bars


def generate_mean_reversion_signal(
    symbol: str,
    snap: IndicatorSnapshot,
    htf_snap: Optional[IndicatorSnapshot],
    cfg: MeanReversionConfig,
) -> Optional[Signal]:
    """Pure scorer. Returns a Signal if conditions hit, else None.

    `htf_snap` is read only for an ADX sanity check on the regime TF — if
    the higher TF shows ADX > 30, we skip even if the trigger TF says
    oversold (a strong trend is in progress, fading it is dangerous).
    """
    if symbol not in cfg.allowed_symbols:
        return None
    if snap.atr14 is None or snap.atr14 <= 0:
        return None
    if snap.bb_upper is None or snap.bb_lower is None or snap.bb_middle is None:
        return None
    if snap.rsi14 is None or snap.stoch_rsi_k is None:
        return None
    # ADX gate — both on trigger TF and (if available) on regime TF.
    if snap.adx14 is None or snap.adx14 > cfg.adx_max_for_meanrev:
        return None
    if htf_snap is not None and htf_snap.adx14 is not None and htf_snap.adx14 > 30.0:
        return None

    entry = snap.close
    atr = snap.atr14
    target = snap.bb_middle

    side: Optional[str] = None
    if (entry < snap.bb_lower
            and snap.stoch_rsi_k < cfg.stoch_oversold
            and snap.rsi14 < cfg.rsi_oversold
            and target > entry):
        side = "long"
    elif (entry > snap.bb_upper
            and snap.stoch_rsi_k > cfg.stoch_overbought
            and snap.rsi14 > cfg.rsi_overbought
            and target < entry):
        side = "short"
    if side is None or side not in cfg.enabled_sides:
        return None

    # Require minimum room to the mean (else costs eat the trade).
    if abs(target - entry) < cfg.min_target_atr * atr:
        return None

    stop_dist = cfg.atr_stop_mult * atr
    if side == "long":
        stop = entry - stop_dist
        tp = target
    else:
        stop = entry + stop_dist
        tp = target

    s = get_settings()
    fee_bps = s.perps_taker_fee_bps * 2 + s.slippage_bps
    expected_move_bps = (abs(tp - entry) / entry) * 10_000
    # Confidence proxy: how stretched we are beyond the band, normalized
    # by ATR. Capped at 1.0.
    if side == "long":
        stretch = max(0.0, (snap.bb_lower - entry) / atr)
    else:
        stretch = max(0.0, (entry - snap.bb_upper) / atr)
    conf = min(1.0, 0.5 + stretch)  # baseline 0.5, +stretch
    edge_bps = expected_move_bps * conf - fee_bps

    fv = FeatureVector(
        trend=0.0, momentum=-1.0 if side == "long" else 1.0,
        volume=0.0, volatility=1.0, pattern=0.0,
    )
    rationale = (
        f"mean-rev {side} stretch={stretch:.2f}atr rsi={snap.rsi14:.1f} "
        f"stoch_k={snap.stoch_rsi_k:.1f} adx={snap.adx14:.1f} target={target:.4f}"
    )
    return Signal(
        symbol=symbol, side=side, confidence=conf, score=stretch,
        entry=entry, stop=stop, take_profit=tp,
        edge_bps=edge_bps, features=fv, rationale=rationale,
    )


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def __init__(self, cfg: MeanReversionConfig) -> None:
        self.cfg = cfg
        self.ctx: StrategyContext | None = None

    async def start(self, ctx: StrategyContext) -> None:
        self.ctx = ctx
        log.info("strategy.start", name=self.name, symbols=self.cfg.allowed_symbols)

    def _trigger_timeframe(self) -> str:
        if not self.ctx:
            return "5m"
        tfs = self.ctx.settings.timeframe_list
        for cand in self.cfg.trigger_timeframe_pref:
            if cand in tfs:
                return cand
        return tfs[0]

    async def on_bar(self, k: Kline, snap: IndicatorSnapshot) -> None:
        if not self.ctx:
            return
        if k.timeframe != self._trigger_timeframe():
            return
        htf_snap = self.ctx.indicators.latest(k.symbol, self.cfg.htf_timeframe)
        sig = generate_mean_reversion_signal(k.symbol, snap, htf_snap, self.cfg)
        if not sig:
            return
        market = "perps" if self.ctx.settings.market_type in ("perps", "both") else "spot"
        leverage = self.ctx.settings.max_leverage if market == "perps" else 1
        await self.ctx.propose(sig, market=market, leverage=leverage,
                                strategy_name=self.name)
