"""Level-breakout strategy — multi-timeframe break of prior-HTF level,
with optional rising/falling trendline (наклонка) break variant.

The pattern (distilled from a scalping signal channel — NOT a copy of the
channel; the channel is a hint about which mechanical setup to encode):

  LONG  := trigger-TF close strictly above the prior closed HTF candle's
           high, with prior trigger close at-or-below that level (real
           break, not a gap-through), HTF regime up, volume_z > threshold,
           and RSI > threshold. Stop = level − k*ATR. TP = 2R.

  SHORT := symmetric on the prior HTF low.

  TRENDLINE long  := trendline TF close strictly above a recently-fitted
                     falling-resistance line (two confirmed pivot highs,
                     slope < 0). HTF regime up still required.

  TRENDLINE short := close strictly below a rising-support line (two
                     confirmed pivot lows, slope > 0). HTF regime down.

Honest caveats:

- This is *one* class of momentum setup. It will perform badly in chop
  (low ADX, mean-reverting). The HTF regime filter is the main defense
  against that, but no filter is perfect.
- The cost-of-edge filter is NOT pre-baked here (unlike `generate_signal`).
  The risk gate downstream applies its own slippage/fee math via
  `edge_bps` on the Signal. We compute `edge_bps` from the configured
  ATR distance — if the stop is too tight for fees, the gate rejects.
- Trendline pivots are confirmed with a `pivot_window` lookback AND
  lookforward; the line is only usable AFTER the second pivot has been
  fully confirmed. This intentionally lags real-time discretionary chart-
  reading (a human draws the line as soon as they "see" it). Don't try
  to fix that by removing the confirmation — it would introduce
  look-ahead bias.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import structlog

from src.models.types import (
    FeatureVector,
    IndicatorSnapshot,
    Kline,
    Signal,
    now_ms,
)
from src.strategies.base import Strategy, StrategyContext

log = structlog.get_logger(__name__)


# ───────────────────────────────── params ──────────────────────────────────


@dataclass
class LevelBreakoutParams:
    htf: str = "1d"
    trigger_tf: str = "5m"
    atr_stop_mult: float = 1.0
    rr_target: float = 2.0
    cooldown_min: int = 240
    vol_z_min: float = 1.0
    rsi_long_min: float = 50.0
    rsi_short_max: float = 50.0
    max_atr_pct: float = 5.0
    # Trendline (наклонка) variant
    trendline_enabled: bool = True
    trendline_tf: str = "15m"
    pivot_window: int = 3
    trendline_max_age_bars: int = 100
    # Confidence / score baked into the emitted Signal. Two different
    # signal qualities — straight HTF-level break gets the higher score
    # because the reference is exact and unambiguous (yesterday's high is
    # a single number); a trendline depends on which two pivots you chose
    # to fit, so it's softer.
    htf_break_confidence: float = 0.65
    htf_break_score: float = 0.55
    trendline_confidence: float = 0.55
    trendline_score: float = 0.40


# ───────────────────────────────── pivots ──────────────────────────────────


@dataclass
class _Pivot:
    """A confirmed swing point on a given TF.

    `bar_ix` is a monotonically-increasing index of the bar on that TF
    (used for max-age checks and slope-time arithmetic so we don't have
    to deal with calendar-time math)."""
    bar_ix: int
    price: float
    kind: str  # "high" | "low"


class _PivotBuffer:
    """Confirmed-pivot detector with symmetric lookback/lookforward window.

    A bar at index `i` is a pivot HIGH when its high is strictly greater
    than all `window` bars before AND after. Confirmation therefore lags
    real-time by `window` bars — this is the price of avoiding lookahead.
    """

    def __init__(self, window: int, max_keep: int = 40) -> None:
        self.window = max(1, window)
        # Sliding buffer of recent bars (need 2*window+1 to evaluate a pivot
        # at the center). Stored as (bar_ix, high, low).
        self._recent: Deque[tuple[int, float, float]] = deque(
            maxlen=2 * self.window + 1
        )
        self.highs: Deque[_Pivot] = deque(maxlen=max_keep)
        self.lows: Deque[_Pivot] = deque(maxlen=max_keep)
        self._next_ix = 0

    def push(self, high: float, low: float) -> None:
        self._recent.append((self._next_ix, high, low))
        self._next_ix += 1
        # Evaluate the center bar of the window if we have a full window.
        if len(self._recent) == 2 * self.window + 1:
            c_ix, c_high, c_low = self._recent[self.window]
            left = list(self._recent)[: self.window]
            right = list(self._recent)[self.window + 1 :]
            if all(c_high > h for _, h, _ in left) and all(c_high > h for _, h, _ in right):
                self.highs.append(_Pivot(bar_ix=c_ix, price=c_high, kind="high"))
            if all(c_low < ll for _, _, ll in left) and all(c_low < ll for _, _, ll in right):
                self.lows.append(_Pivot(bar_ix=c_ix, price=c_low, kind="low"))

    @property
    def current_bar_ix(self) -> int:
        """Index of the most recent bar pushed (the one we're currently
        evaluating). Always == bars_pushed - 1."""
        return self._next_ix - 1


def _trendline_value_at(p1: _Pivot, p2: _Pivot, at_bar_ix: int) -> float:
    """Project the line through two pivots to a given bar index. Bars are
    equally spaced (this is *bar time*, not calendar time — gaps in a 24/7
    crypto feed are ~nonexistent, and even if they happen, a bar-time line
    is what a human chart-reader actually fits)."""
    if p2.bar_ix == p1.bar_ix:
        return p2.price
    slope = (p2.price - p1.price) / (p2.bar_ix - p1.bar_ix)
    return p2.price + slope * (at_bar_ix - p2.bar_ix)


def _trendline_slope(p1: _Pivot, p2: _Pivot) -> float:
    if p2.bar_ix == p1.bar_ix:
        return 0.0
    return (p2.price - p1.price) / (p2.bar_ix - p1.bar_ix)


# ────────────────────────────── per-symbol state ────────────────────────────


@dataclass
class _SymbolState:
    # HTF-level break path
    prior_htf_high: Optional[float] = None
    prior_htf_low: Optional[float] = None
    last_trigger_close: Optional[float] = None  # to require "real break"
    # Cooldown so a failed break doesn't re-fire on the same level
    cooldown_until_ms: int = 0
    # Trendline path — pivot buffer lives on the trendline TF
    pivots: Optional[_PivotBuffer] = None
    # Dedup: per-bar lockout so the two paths don't both fire on the same
    # closing bar of the same TF.
    last_signal_bar_ts_ms: int = 0


# ───────────────────────────── pure decision logic ──────────────────────────


def _htf_regime_long_ok(htf_snap: Optional[IndicatorSnapshot]) -> bool:
    """Long-side HTF agreement: close above ema55 OR supertrend up. Either
    is sufficient — they tend to disagree at the very turn, and we'd
    rather catch the second leg of a turn than miss it entirely waiting
    for both."""
    if htf_snap is None:
        return False
    if htf_snap.ema55 is not None and htf_snap.close > htf_snap.ema55:
        return True
    if htf_snap.supertrend_dir is not None and htf_snap.supertrend_dir > 0:
        return True
    return False


def _htf_regime_short_ok(htf_snap: Optional[IndicatorSnapshot]) -> bool:
    if htf_snap is None:
        return False
    if htf_snap.ema55 is not None and htf_snap.close < htf_snap.ema55:
        return True
    if htf_snap.supertrend_dir is not None and htf_snap.supertrend_dir < 0:
        return True
    return False


def _trigger_filters_ok(
    side: str, snap: IndicatorSnapshot, params: LevelBreakoutParams
) -> tuple[bool, str]:
    """Momentum + volatility filters on the trigger-TF snapshot. Returns
    (ok, reason_if_not)."""
    if snap.atr14 is None or snap.atr14 <= 0:
        return False, "no_atr"
    atr_pct = (snap.atr14 / snap.close) * 100.0
    if atr_pct > params.max_atr_pct:
        return False, f"atr_pct_{atr_pct:.2f}>{params.max_atr_pct}"
    if snap.volume_z is None or snap.volume_z < params.vol_z_min:
        return False, "volume_z_too_low"
    if side == "long":
        if snap.rsi14 is None or snap.rsi14 < params.rsi_long_min:
            return False, "rsi_below_long_floor"
    else:
        if snap.rsi14 is None or snap.rsi14 > params.rsi_short_max:
            return False, "rsi_above_short_ceiling"
    return True, ""


def _build_signal(
    *,
    symbol: str,
    side: str,
    entry: float,
    atr: float,
    params: LevelBreakoutParams,
    confidence: float,
    score: float,
    rationale: str,
) -> Signal:
    """Construct a Signal with ATR-distance stop and RR-multiple TP.

    `edge_bps` is computed from the *stop distance*, which is the right
    floor the risk gate's cost-of-edge check needs (you can't credibly
    say your edge is larger than the move that proves you wrong)."""
    stop_dist = params.atr_stop_mult * atr
    if side == "long":
        stop = entry - stop_dist
        tp = entry + params.rr_target * stop_dist
    else:
        stop = entry + stop_dist
        tp = entry - params.rr_target * stop_dist
    edge_bps = (stop_dist / entry) * 10_000.0
    return Signal(
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        confidence=confidence,
        score=score if side == "long" else -score,
        entry=entry,
        stop=stop,
        take_profit=tp,
        edge_bps=edge_bps,
        features=FeatureVector(trend=0.6 if side == "long" else -0.6,
                               momentum=0.5 if side == "long" else -0.5,
                               volume=0.5),
        rationale=rationale,
    )


# ─────────────────────────────── the strategy ──────────────────────────────


class LevelBreakoutStrategy(Strategy):
    name = "levelbreak"

    def __init__(self, params: Optional[LevelBreakoutParams] = None,
                 symbols: Optional[list[str]] = None) -> None:
        self.params = params or LevelBreakoutParams()
        self.symbols = symbols  # None = trade all symbols the engine sees
        self.ctx: Optional[StrategyContext] = None
        self._state: dict[str, _SymbolState] = {}
        self._htf_warning_logged = False
        self._tf_warning_logged = False

    # public for tests / backtest
    def state(self, symbol: str) -> _SymbolState:
        st = self._state.get(symbol)
        if st is None:
            st = _SymbolState()
            if self.params.trendline_enabled:
                st.pivots = _PivotBuffer(window=self.params.pivot_window)
            self._state[symbol] = st
        return st

    async def start(self, ctx: StrategyContext) -> None:
        self.ctx = ctx
        tfs = ctx.settings.timeframe_list
        missing = [tf for tf in {self.params.htf, self.params.trigger_tf,
                                 self.params.trendline_tf if self.params.trendline_enabled else None}
                   if tf and tf not in tfs]
        if missing:
            log.warning(
                "levelbreak.missing_timeframes",
                missing=missing,
                hint=(
                    "Add the missing timeframe(s) to settings.timeframes "
                    "(e.g. TIMEFRAMES=1m,5m,15m,1h,1d). The strategy will "
                    "no-op until they are present."
                ),
            )
        log.info(
            "strategy.start",
            name=self.name,
            symbols=self.symbols or "ALL",
            htf=self.params.htf,
            trigger_tf=self.params.trigger_tf,
            trendline=self.params.trendline_enabled,
        )

    def _symbol_allowed(self, symbol: str) -> bool:
        return self.symbols is None or symbol in self.symbols

    async def on_bar(self, k: Kline, snap: IndicatorSnapshot) -> None:
        if not self.ctx or not self._symbol_allowed(k.symbol):
            return
        st = self.state(k.symbol)

        # HTF closed bar → update the level the trigger will reference.
        # Note: snap is the trigger-TF snapshot for the on_bar call where
        # k.timeframe == HTF — but the *engine* updates that snap from this
        # closed HTF bar's high/low (we read them off the Kline directly,
        # not the snapshot, which only carries close).
        if k.timeframe == self.params.htf:
            st.prior_htf_high = k.high
            st.prior_htf_low = k.low
            return

        # Trendline-TF closed bar → push to pivot buffer.
        if self.params.trendline_enabled and k.timeframe == self.params.trendline_tf:
            if st.pivots is None:
                st.pivots = _PivotBuffer(window=self.params.pivot_window)
            st.pivots.push(k.high, k.low)
            # If trendline_tf == trigger_tf we fall through to the trigger
            # path below; otherwise stop here.
            if k.timeframe != self.params.trigger_tf:
                # But still attempt trendline signal on this trendline TF.
                await self._maybe_emit_trendline(k, snap, st)
                return

        if k.timeframe != self.params.trigger_tf:
            return

        # Dedup: don't double-fire on the same bar
        if st.last_signal_bar_ts_ms == k.close_time:
            self._remember_close(st, k.close)
            return

        # Cooldown
        if now_ms() < st.cooldown_until_ms:
            self._remember_close(st, k.close)
            return

        emitted = await self._maybe_emit_htf_break(k, snap, st)
        if not emitted and self.params.trendline_enabled and self.params.trendline_tf == self.params.trigger_tf:
            emitted = await self._maybe_emit_trendline(k, snap, st)

        self._remember_close(st, k.close)
        if emitted:
            st.last_signal_bar_ts_ms = k.close_time
            st.cooldown_until_ms = now_ms() + self.params.cooldown_min * 60_000

    def _remember_close(self, st: _SymbolState, close: float) -> None:
        st.last_trigger_close = close

    async def _maybe_emit_htf_break(
        self, k: Kline, snap: IndicatorSnapshot, st: _SymbolState
    ) -> bool:
        """Returns True if a signal was proposed."""
        assert self.ctx is not None
        if st.prior_htf_high is None or st.prior_htf_low is None:
            return False
        if st.last_trigger_close is None:
            # First trigger bar — we need a previous close to know whether
            # this is a real break or a gap. Conservative: skip.
            return False

        htf_snap = self.ctx.indicators.latest(k.symbol, self.params.htf)

        # LONG: current close strictly above prior HTF high, prior close at-or-below.
        if (
            k.close > st.prior_htf_high
            and st.last_trigger_close <= st.prior_htf_high
            and _htf_regime_long_ok(htf_snap)
        ):
            ok, why = _trigger_filters_ok("long", snap, self.params)
            if ok:
                sig = _build_signal(
                    symbol=k.symbol,
                    side="long",
                    entry=k.close,
                    atr=snap.atr14 or 0.0,
                    params=self.params,
                    confidence=self.params.htf_break_confidence,
                    score=self.params.htf_break_score,
                    rationale=f"htf_break_long level={st.prior_htf_high:.6f} "
                              f"htf={self.params.htf}",
                )
                await self._propose(sig)
                return True
            log.debug("levelbreak.filter_reject", side="long", reason=why,
                      symbol=k.symbol)

        # SHORT: current close strictly below prior HTF low, prior close at-or-above.
        if (
            k.close < st.prior_htf_low
            and st.last_trigger_close >= st.prior_htf_low
            and _htf_regime_short_ok(htf_snap)
        ):
            ok, why = _trigger_filters_ok("short", snap, self.params)
            if ok:
                sig = _build_signal(
                    symbol=k.symbol,
                    side="short",
                    entry=k.close,
                    atr=snap.atr14 or 0.0,
                    params=self.params,
                    confidence=self.params.htf_break_confidence,
                    score=self.params.htf_break_score,
                    rationale=f"htf_break_short level={st.prior_htf_low:.6f} "
                              f"htf={self.params.htf}",
                )
                await self._propose(sig)
                return True
            log.debug("levelbreak.filter_reject", side="short", reason=why,
                      symbol=k.symbol)

        return False

    async def _maybe_emit_trendline(
        self, k: Kline, snap: IndicatorSnapshot, st: _SymbolState
    ) -> bool:
        """Trendline-break path. Evaluated on the trendline TF only; uses
        the two most-recent confirmed pivots."""
        assert self.ctx is not None
        if st.pivots is None or len(st.pivots.highs) < 2 and len(st.pivots.lows) < 2:
            return False

        # Dedup against HTF path firing in the same bar.
        if st.last_signal_bar_ts_ms == k.close_time:
            return False
        if now_ms() < st.cooldown_until_ms:
            return False

        cur_ix = st.pivots.current_bar_ix
        max_age = self.params.trendline_max_age_bars
        htf_snap = self.ctx.indicators.latest(k.symbol, self.params.htf)

        # Falling-resistance long: last two pivot highs, slope < 0
        if len(st.pivots.highs) >= 2 and _htf_regime_long_ok(htf_snap):
            p2 = st.pivots.highs[-1]
            p1 = st.pivots.highs[-2]
            if (
                cur_ix - p2.bar_ix <= max_age
                and _trendline_slope(p1, p2) < 0
            ):
                line_val = _trendline_value_at(p1, p2, cur_ix)
                if k.close > line_val:
                    ok, why = _trigger_filters_ok("long", snap, self.params)
                    if ok:
                        sig = _build_signal(
                            symbol=k.symbol, side="long",
                            entry=k.close, atr=snap.atr14 or 0.0,
                            params=self.params,
                            confidence=self.params.trendline_confidence,
                            score=self.params.trendline_score,
                            rationale=f"trendline_long break={line_val:.6f}",
                        )
                        await self._propose(sig)
                        st.last_signal_bar_ts_ms = k.close_time
                        st.cooldown_until_ms = now_ms() + self.params.cooldown_min * 60_000
                        return True
                    log.debug("levelbreak.tl_filter_reject", side="long",
                              reason=why, symbol=k.symbol)

        # Rising-support short: last two pivot lows, slope > 0
        if len(st.pivots.lows) >= 2 and _htf_regime_short_ok(htf_snap):
            p2 = st.pivots.lows[-1]
            p1 = st.pivots.lows[-2]
            if (
                cur_ix - p2.bar_ix <= max_age
                and _trendline_slope(p1, p2) > 0
            ):
                line_val = _trendline_value_at(p1, p2, cur_ix)
                if k.close < line_val:
                    ok, why = _trigger_filters_ok("short", snap, self.params)
                    if ok:
                        sig = _build_signal(
                            symbol=k.symbol, side="short",
                            entry=k.close, atr=snap.atr14 or 0.0,
                            params=self.params,
                            confidence=self.params.trendline_confidence,
                            score=self.params.trendline_score,
                            rationale=f"trendline_short break={line_val:.6f}",
                        )
                        await self._propose(sig)
                        st.last_signal_bar_ts_ms = k.close_time
                        st.cooldown_until_ms = now_ms() + self.params.cooldown_min * 60_000
                        return True
                    log.debug("levelbreak.tl_filter_reject", side="short",
                              reason=why, symbol=k.symbol)

        return False

    async def _propose(self, sig: Signal) -> None:
        assert self.ctx is not None
        market = "perps" if self.ctx.settings.market_type in ("perps", "both") else "spot"
        leverage = self.ctx.settings.max_leverage if market == "perps" else 1
        await self.ctx.propose(sig, market=market, leverage=leverage,
                                strategy_name=self.name)
