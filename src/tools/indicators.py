"""Rolling indicator engine.

One IndicatorState per (symbol, timeframe). Updates from closed klines only —
we never read in-progress candles into signal logic. Uses numpy/pandas-classic
for batch warmup; per-tick updates are O(1) recurrence formulas.

This module is pure: no I/O, no LLM calls.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

from src.models.types import IndicatorSnapshot, Kline


def _ema(prev: Optional[float], value: float, period: int) -> float:
    k = 2.0 / (period + 1)
    return value if prev is None else (value - prev) * k + prev


def _wilder(prev: Optional[float], value: float, period: int) -> float:
    return value if prev is None else (prev * (period - 1) + value) / period


@dataclass
class IndicatorState:
    symbol: str
    timeframe: str

    closes: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    highs: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    lows: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    volumes: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    quote_volumes: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    taker_buy_volumes: Deque[float] = field(default_factory=lambda: deque(maxlen=500))

    ema21: Optional[float] = None
    ema55: Optional[float] = None
    macd_fast: Optional[float] = None  # EMA12
    macd_slow: Optional[float] = None  # EMA26
    macd_signal: Optional[float] = None  # EMA9 of macd_line
    rsi_avg_gain: Optional[float] = None
    rsi_avg_loss: Optional[float] = None
    atr: Optional[float] = None
    prev_close: Optional[float] = None

    # Supertrend (10, 3)
    st_period: int = 10
    st_mult: float = 3.0
    st_upper: Optional[float] = None
    st_lower: Optional[float] = None
    st_value: Optional[float] = None
    st_dir: int = 1

    # VWAP — session-anchored (we use UTC daily rollover via reset())
    vwap_pv: float = 0.0
    vwap_v: float = 0.0
    session_day: Optional[str] = None

    # CVD running sum (signed taker-buy minus taker-sell, derived from kline)
    cvd: float = 0.0
    cvd_history: Deque[float] = field(default_factory=lambda: deque(maxlen=50))

    # ADX (Wilder, 14) — Wilder-smoothed DM/TR and DX
    adx_period: int = 14
    adx_plus_dm: Optional[float] = None
    adx_minus_dm: Optional[float] = None
    adx_tr: Optional[float] = None
    adx_value: Optional[float] = None
    prev_high: Optional[float] = None
    prev_low: Optional[float] = None

    # Stochastic RSI — needs rolling history of raw RSI values
    stoch_rsi_period: int = 14
    rsi_history: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    stoch_rsi_raw_history: Deque[float] = field(default_factory=lambda: deque(maxlen=16))
    stoch_rsi_k_history: Deque[float] = field(default_factory=lambda: deque(maxlen=16))

    # OBV running sum
    obv_value: float = 0.0
    obv_history: Deque[float] = field(default_factory=lambda: deque(maxlen=50))

    # Choppiness Index (Dreiss) — true-range history for CHOP(14)
    chop_period: int = 14
    true_ranges: Deque[float] = field(default_factory=lambda: deque(maxlen=64))

    last_snapshot: Optional[IndicatorSnapshot] = None

    def reset_session(self) -> None:
        self.vwap_pv = 0.0
        self.vwap_v = 0.0

    def on_closed_kline(self, k: Kline) -> IndicatorSnapshot:
        c, h, l, v = k.close, k.high, k.low, k.volume

        self.closes.append(c)
        self.highs.append(h)
        self.lows.append(l)
        self.volumes.append(v)
        self.quote_volumes.append(k.quote_volume)
        self.taker_buy_volumes.append(k.taker_buy_volume)

        # EMAs
        self.ema21 = _ema(self.ema21, c, 21)
        self.ema55 = _ema(self.ema55, c, 55)

        # MACD
        self.macd_fast = _ema(self.macd_fast, c, 12)
        self.macd_slow = _ema(self.macd_slow, c, 26)
        macd_line: Optional[float] = None
        macd_hist: Optional[float] = None
        if self.macd_fast is not None and self.macd_slow is not None:
            macd_line = self.macd_fast - self.macd_slow
            self.macd_signal = _ema(self.macd_signal, macd_line, 9)
            if self.macd_signal is not None:
                macd_hist = macd_line - self.macd_signal

        # RSI (Wilder, 14)
        rsi: Optional[float] = None
        if self.prev_close is not None:
            change = c - self.prev_close
            gain = max(change, 0.0)
            loss = max(-change, 0.0)
            self.rsi_avg_gain = _wilder(self.rsi_avg_gain, gain, 14)
            self.rsi_avg_loss = _wilder(self.rsi_avg_loss, loss, 14)
            # Canonical Wilder: rsi=100 when avg_loss==0 (pure uptrend),
            # rsi=0 when avg_gain==0 (pure downtrend). Previous version
            # silently returned None for pure uptrends — caught by tests.
            if self.rsi_avg_gain is not None and self.rsi_avg_loss is not None:
                if self.rsi_avg_loss == 0:
                    rsi = 100.0 if self.rsi_avg_gain > 0 else 50.0
                else:
                    rs = self.rsi_avg_gain / self.rsi_avg_loss
                    rsi = 100 - (100 / (1 + rs))

        # ATR (Wilder, 14) from True Range
        if self.prev_close is not None:
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
            self.atr = _wilder(self.atr, tr, 14)
            self.true_ranges.append(tr)

        # Bollinger (20, 2)
        bb_upper = bb_middle = bb_lower = None
        bb_width_rank: Optional[float] = None
        if len(self.closes) >= 20:
            window = np.array(list(self.closes)[-20:])
            mean = window.mean()
            std = window.std(ddof=0)
            bb_middle = float(mean)
            bb_upper = float(mean + 2 * std)
            bb_lower = float(mean - 2 * std)
            # rolling percentile rank of width — useful for "squeeze" detection
            if len(self.closes) >= 100:
                widths = []
                arr = np.array(list(self.closes)[-100:])
                for i in range(20, 100):
                    w = arr[i - 20:i]
                    widths.append(w.std(ddof=0) * 4)
                cur_w = std * 4
                widths = np.array(widths)
                bb_width_rank = float((widths <= cur_w).mean())

        # Supertrend (10, 3)
        if self.atr is not None and len(self.closes) >= self.st_period:
            mid = (h + l) / 2.0
            upper = mid + self.st_mult * self.atr
            lower = mid - self.st_mult * self.atr
            if self.st_upper is None:
                self.st_upper, self.st_lower = upper, lower
                self.st_value = lower
                self.st_dir = 1
            else:
                self.st_upper = min(upper, self.st_upper) if c <= self.st_upper else upper
                self.st_lower = max(lower, self.st_lower) if c >= self.st_lower else lower
                if self.st_dir == 1 and c < self.st_lower:
                    self.st_dir = -1
                elif self.st_dir == -1 and c > self.st_upper:
                    self.st_dir = 1
                self.st_value = self.st_lower if self.st_dir == 1 else self.st_upper

        # VWAP (session)
        typical = (h + l + c) / 3.0
        self.vwap_pv += typical * v
        self.vwap_v += v
        vwap = self.vwap_pv / self.vwap_v if self.vwap_v > 0 else None

        # CVD proxy from kline taker buy vs sell
        taker_buy = k.taker_buy_volume
        taker_sell = max(v - taker_buy, 0.0)
        self.cvd += (taker_buy - taker_sell)
        self.cvd_history.append(self.cvd)
        cvd_slope: Optional[float] = None
        if len(self.cvd_history) >= 10:
            hist = np.array(list(self.cvd_history)[-10:])
            xs = np.arange(len(hist))
            cvd_slope = float(np.polyfit(xs, hist, 1)[0])

        # Volume z-score (20-bar)
        volume_z: Optional[float] = None
        if len(self.volumes) >= 20:
            window = np.array(list(self.volumes)[-20:])
            mu = window.mean()
            sd = window.std(ddof=0)
            if sd > 0:
                volume_z = float((v - mu) / sd)

        # ADX / DI+ / DI- (Wilder, 14). Standard formulation:
        #   +DM = max(high-prev_high, 0) if high-prev_high > prev_low-low else 0
        #   -DM = max(prev_low-low, 0)  if prev_low-low > high-prev_high else 0
        #   TR  = max(h-l, |h-prev_close|, |l-prev_close|)
        # All three Wilder-smoothed; DX = 100*|+DI - -DI|/(+DI + -DI);
        # ADX = Wilder-smoothed DX.
        adx14: Optional[float] = None
        di_plus_val: Optional[float] = None
        di_minus_val: Optional[float] = None
        if self.prev_high is not None and self.prev_low is not None \
                and self.prev_close is not None:
            up_move = h - self.prev_high
            dn_move = self.prev_low - l
            plus_dm = up_move if up_move > dn_move and up_move > 0 else 0.0
            minus_dm = dn_move if dn_move > up_move and dn_move > 0 else 0.0
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
            self.adx_plus_dm = _wilder(self.adx_plus_dm, plus_dm, self.adx_period)
            self.adx_minus_dm = _wilder(self.adx_minus_dm, minus_dm, self.adx_period)
            self.adx_tr = _wilder(self.adx_tr, tr, self.adx_period)
            if self.adx_tr and self.adx_tr > 0:
                di_plus_val = 100.0 * self.adx_plus_dm / self.adx_tr
                di_minus_val = 100.0 * self.adx_minus_dm / self.adx_tr
                denom = di_plus_val + di_minus_val
                if denom > 0:
                    dx = 100.0 * abs(di_plus_val - di_minus_val) / denom
                    self.adx_value = _wilder(self.adx_value, dx, self.adx_period)
                    adx14 = self.adx_value

        # Stochastic RSI (RSI lookback 14, stoch period 14, K=3 SMA, D=3 SMA).
        # Needs at least `stoch_rsi_period` raw RSI values in history.
        stoch_k: Optional[float] = None
        stoch_d: Optional[float] = None
        if rsi is not None:
            self.rsi_history.append(rsi)
            if len(self.rsi_history) >= self.stoch_rsi_period:
                window_rsi = list(self.rsi_history)[-self.stoch_rsi_period:]
                lo = min(window_rsi)
                hi = max(window_rsi)
                if hi > lo:
                    raw = (rsi - lo) / (hi - lo) * 100.0
                else:
                    raw = 50.0
                self.stoch_rsi_raw_history.append(raw)
                if len(self.stoch_rsi_raw_history) >= 3:
                    k = float(np.mean(list(self.stoch_rsi_raw_history)[-3:]))
                    self.stoch_rsi_k_history.append(k)
                    stoch_k = k
                    if len(self.stoch_rsi_k_history) >= 3:
                        stoch_d = float(np.mean(list(self.stoch_rsi_k_history)[-3:]))

        # Donchian channels (20-bar high/low)
        donchian_upper = donchian_lower = donchian_mid = None
        if len(self.highs) >= 20 and len(self.lows) >= 20:
            donchian_upper = float(max(list(self.highs)[-20:]))
            donchian_lower = float(min(list(self.lows)[-20:]))
            donchian_mid = (donchian_upper + donchian_lower) / 2.0

        # Choppiness Index (Dreiss, 14): 100 * log10(sum(TR_n) / range_n) / log10(n).
        # Bounded 0..100; CHOP < 38.2 = trending, > 61.8 = chop, in-between = mixed.
        choppiness14: Optional[float] = None
        n = self.chop_period
        if (len(self.true_ranges) >= n and len(self.highs) >= n
                and len(self.lows) >= n):
            tr_sum = float(sum(list(self.true_ranges)[-n:]))
            high_n = float(max(list(self.highs)[-n:]))
            low_n = float(min(list(self.lows)[-n:]))
            channel = high_n - low_n
            if channel > 0 and tr_sum > 0:
                choppiness14 = 100.0 * math.log10(tr_sum / channel) / math.log10(n)

        # OBV — cumulative volume weighted by close-direction.
        if self.prev_close is not None:
            if c > self.prev_close:
                self.obv_value += v
            elif c < self.prev_close:
                self.obv_value -= v
        self.obv_history.append(self.obv_value)
        obv_slope: Optional[float] = None
        if len(self.obv_history) >= 10:
            hist = np.array(list(self.obv_history)[-10:])
            xs = np.arange(len(hist))
            obv_slope = float(np.polyfit(xs, hist, 1)[0])

        # Record prev_* AFTER all computations that need them above.
        self.prev_close = c
        self.prev_high = h
        self.prev_low = l

        snap = IndicatorSnapshot(
            symbol=self.symbol, timeframe=self.timeframe, close=c,
            ema21=self.ema21, ema55=self.ema55,
            macd=macd_line, macd_signal=self.macd_signal, macd_hist=macd_hist,
            rsi14=rsi, atr14=self.atr,
            bb_upper=bb_upper, bb_middle=bb_middle, bb_lower=bb_lower,
            bb_width_pct_rank=bb_width_rank,
            vwap=vwap, supertrend=self.st_value, supertrend_dir=self.st_dir,
            cvd=self.cvd, cvd_slope=cvd_slope, volume_z=volume_z,
            adx14=adx14, di_plus=di_plus_val, di_minus=di_minus_val,
            stoch_rsi_k=stoch_k, stoch_rsi_d=stoch_d,
            donchian_upper=donchian_upper, donchian_lower=donchian_lower,
            donchian_mid=donchian_mid,
            obv=self.obv_value, obv_slope=obv_slope,
            choppiness14=choppiness14,
        )
        self.last_snapshot = snap
        return snap


class IndicatorEngine:
    """Per (symbol, timeframe) bucket of IndicatorState."""

    def __init__(self) -> None:
        self.states: dict[tuple[str, str], IndicatorState] = {}

    def get(self, symbol: str, timeframe: str) -> IndicatorState:
        key = (symbol, timeframe)
        if key not in self.states:
            self.states[key] = IndicatorState(symbol=symbol, timeframe=timeframe)
        return self.states[key]

    def latest(self, symbol: str, timeframe: str) -> Optional[IndicatorSnapshot]:
        st = self.states.get((symbol, timeframe))
        return st.last_snapshot if st else None

    def warmup(self, symbol: str, timeframe: str, klines: list[Kline]) -> None:
        st = self.get(symbol, timeframe)
        for k in klines:
            if k.is_closed:
                st.on_closed_kline(k)
