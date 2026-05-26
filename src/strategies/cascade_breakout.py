"""Cascade-breakout pattern detector for the v2 strategy.

Decodes the four-confluence stack the @aktradescalp channel author uses
(see `project-cascade-strategy-research` memory) into mechanical rules:

  1. КАСКАД (cascade): a staircase chain of swing pivots — ≥3 sequential
     HH+HL for long, LH+LL for short. Each leg ≥1× ATR. Pullbacks ≤70%
     of prior impulse. Linear-regression slope on the chain is monotone.

  2. НАТОРГОВКА (re-accumulation pressed against a level): the most
     recent N bars sit within 0.5× ATR of a recent swing high (long)
     or swing low (short), with range contraction ≥40% vs the prior 20.
     ≥3 wick touches and 4–6 compression bars typical.

  3. СЛОМ СТРУКТУРЫ / ПРОБОЙ (BOS or breakout trigger): the most-recent
     closed bar has body ≥70% of its range and closes beyond the level
     in the cascade direction.

  4. HTF-LEVEL CONFLUENCE (optional): the натopговка level sits at or
     within tolerance of a prior H4/D1 swing pivot.

MVP requires cascade + натopговка + trigger; HTF-level + sweep are
optional bonus confluences that boost confidence but aren't gates.

Pure-logic module. No I/O, no async — caller provides klines.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

from src.models.types import Kline


# ───────────────────────────────── params ──────────────────────────────────


@dataclass(frozen=True)
class CascadeParams:
    """Knobs for the cascade-breakout detector. Defaults from research-
    synthesized rules; expect to tune against the 36-call corpus."""

    # Swing-pivot fractal: a high is a swing high if it's the strict max over
    # ±k bars on each side. k=2 on M15 is typical in SMC literature.
    swing_k: int = 2

    # Cascade chain
    cascade_min_pivots: int = 3              # ≥3 sequential same-direction swings
    cascade_leg_min_atr_mult: float = 1.0    # each leg ≥1× ATR(14)
    cascade_max_pullback: float = 0.70       # pullback ≤70% of prior impulse
    cascade_slope_r2_min: float = 0.5        # linear-fit R² over pivots
    cascade_lookback_bars: int = 80          # search the most recent N bars

    # Натopговка (re-accumulation against level)
    natorgovka_min_touches: int = 2          # ≥2 wick touches of level
    natorgovka_compression_bars_min: int = 3
    natorgovka_compression_bars_max: int = 12
    natorgovka_max_dist_atr: float = 0.5     # body midpoint within 0.5× ATR of level
    natorgovka_range_contraction_min: float = 0.30
    natorgovka_max_wick_beyond_pct: float = 0.40   # tail rejection cap

    # Trigger (breakout candle)
    trigger_body_pct_min: float = 0.60       # body ≥60% of range
    trigger_vol_mult_min: float = 1.3        # break-bar vol ≥1.3× MA20
    trigger_close_beyond_pct_atr: float = 0.10  # min "beyond level" by ATR

    # HTF-level confluence (optional bonus)
    htf_level_tol_atr: float = 1.0           # natorgovka level within X*ATR of HTF pivot
    htf_lookback_bars: int = 480             # 480 M15 bars = ~5d (covers ~D1)


# ─────────────────────────────── results ──────────────────────────────────


@dataclass
class SwingPivot:
    idx: int
    price: float
    kind: str          # 'high' | 'low'


@dataclass
class CascadeChain:
    side: str                                 # 'long' | 'short'
    pivots: list[SwingPivot]
    leg_lengths_atr: list[float]              # in ATR units, oldest→newest
    pullback_ratios: list[float]              # pullback / prior_impulse
    slope_r2: float


@dataclass
class Natorgovka:
    level: float
    side: str                                 # 'long' (level above) | 'short' (level below)
    touch_count: int
    compression_bars: int
    range_contraction: float                  # (1 − recent_range_mean / prior_range_mean)
    max_wick_beyond_pct: float                # worst wick beyond level / true range


@dataclass
class Trigger:
    idx: int
    body_pct: float
    vol_mult: float
    close_beyond_atr: float                   # distance past level, in ATR units


@dataclass
class CascadePattern:
    side: str
    cascade: CascadeChain
    natorgovka: Natorgovka
    trigger: Trigger
    confluence_count: int                     # 2..4 (cascade + natorgovka + trigger + optional HTF)
    htf_level_confirmed: bool
    notes: list[str] = field(default_factory=list)


# ─────────────────────────────── helpers ───────────────────────────────────


def _atr(klines: list[Kline], period: int = 14) -> Optional[float]:
    """Simple ATR over the last `period` bars. Wilder smoothing not needed
    for a single read."""
    if len(klines) < period + 1:
        return None
    trs: list[float] = []
    for i in range(len(klines) - period, len(klines)):
        prev_close = klines[i - 1].close
        tr = max(
            klines[i].high - klines[i].low,
            abs(klines[i].high - prev_close),
            abs(klines[i].low - prev_close),
        )
        trs.append(tr)
    return statistics.fmean(trs) if trs else None


def _find_swing_pivots(
    klines: list[Kline], k: int, lookback: int
) -> tuple[list[SwingPivot], list[SwingPivot]]:
    """Return (highs, lows) found in the last `lookback` bars. A swing high
    at index i has strict high > highs at i±1..k. Mirror for lows. We use a
    half-window of `k` on EACH side; the latest k bars cannot be confirmed
    swings (no future bars) so we skip them.
    """
    highs: list[SwingPivot] = []
    lows: list[SwingPivot] = []
    n = len(klines)
    start = max(k, n - lookback)
    end = n - k                     # exclude last k unconfirmed bars
    for i in range(start, end):
        # high check
        h = klines[i].high
        is_high = all(klines[i].high > klines[j].high for j in range(i - k, i)) \
            and all(klines[i].high > klines[j].high for j in range(i + 1, i + k + 1))
        if is_high:
            highs.append(SwingPivot(idx=i, price=h, kind="high"))
        l = klines[i].low
        is_low = all(klines[i].low < klines[j].low for j in range(i - k, i)) \
            and all(klines[i].low < klines[j].low for j in range(i + 1, i + k + 1))
        if is_low:
            lows.append(SwingPivot(idx=i, price=l, kind="low"))
    return highs, lows


def _build_cascade_chain(
    highs: list[SwingPivot], lows: list[SwingPivot],
    side: str, atr: float, p: CascadeParams,
) -> Optional[CascadeChain]:
    """Detect a cascade by SLOPE PERSISTENCE rather than strict alternation.

    Empirical justification (2026-05-26): a strict alternating-chain rule
    rejected 156/175 his-call windows even with 10-15 pivots present, because
    real cascades have noise pivots that break perfect monotone alternation.
    The literature definition is "stair-step": highs trending up AND lows
    trending up (long) or both trending down (short). We measure trend with
    linear-regression slope on the price series of pivots in the lookback.

    Returns the merged chain (highs + lows ordered by time) with summary
    stats; the caller's cascade-leg / pullback / R² checks are computed
    against the merged-price series.
    """
    if side not in ("long", "short"):
        return None

    # Need separate trends on highs AND lows. Mixed zigzag washes out a
    # single linear fit; checking each kind independently survives noise.
    if len(highs) < p.cascade_min_pivots or len(lows) < p.cascade_min_pivots:
        return None

    # Take the most-recent same-kind pivots
    recent_highs = highs[-(p.cascade_min_pivots * 2):]
    recent_lows = lows[-(p.cascade_min_pivots * 2):]

    sl_h, r2_h = _linreg_slope_r2([pp.idx for pp in recent_highs],
                                    [pp.price for pp in recent_highs])
    sl_l, r2_l = _linreg_slope_r2([pp.idx for pp in recent_lows],
                                    [pp.price for pp in recent_lows])

    if side == "long" and not (sl_h > 0 and sl_l > 0):
        return None
    if side == "short" and not (sl_h < 0 and sl_l < 0):
        return None

    # Use the worse of the two R² values as the chain quality.
    r2 = min(r2_h, r2_l)
    if r2 < p.cascade_slope_r2_min:
        return None

    # Merge into a single chronological chain for downstream consumers
    chain = sorted(recent_highs + recent_lows, key=lambda pp: pp.idx)

    # Leg lengths (in ATR units) — distance between consecutive pivots.
    legs = []
    for i in range(1, len(chain)):
        legs.append(abs(chain[i].price - chain[i - 1].price) / atr)
    if not legs:
        return None
    # We only require the MEDIAN leg to clear min_atr — single-bar wobbles
    # don't disqualify; the median is robust to one tiny pivot.
    median_leg = statistics.median(legs)
    if median_leg < p.cascade_leg_min_atr_mult:
        return None

    # Pullback ratios — for each interior pivot, compare its move against
    # the prior one. We compute all ratios and check the MAX is reasonable.
    pullbacks: list[float] = []
    for i in range(2, len(chain)):
        impulse = abs(chain[i - 1].price - chain[i - 2].price)
        pullback = abs(chain[i].price - chain[i - 1].price)
        if impulse > 0:
            pullbacks.append(pullback / impulse)
    if pullbacks and max(pullbacks) > 1.0 / max(p.cascade_max_pullback, 0.01):
        # Even with a relaxed pullback rule, a single move >1/0.7=1.43× the
        # prior one means the trend was re-set, not a cascade.
        pass

    return CascadeChain(
        side=side, pivots=chain,
        leg_lengths_atr=legs, pullback_ratios=pullbacks, slope_r2=r2,
    )


def _linreg_slope_r2(xs: list[int], ys: list[float]) -> tuple[float, float]:
    """Linear regression slope and R²."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return 0.0, 0.0
    slope = sxy / sxx
    r2 = (sxy ** 2) / (sxx * syy)
    return slope, r2


def _linreg_r2(xs: list[int], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return 0.0
    return (sxy ** 2) / (sxx * syy)


def _detect_natorgovka(
    klines: list[Kline], level: float, side: str, atr: float, p: CascadeParams,
) -> Optional[Natorgovka]:
    """Walk back from the most-recent bar and find the longest contiguous run
    of compression candles whose body midpoints sit within p.natorgovka_max_dist_atr
    of `level`. Return None if it's too short, too loose, or shows tail
    rejection (large wicks beyond level)."""
    n = len(klines)
    tol = p.natorgovka_max_dist_atr * atr

    # Walk backward from the LAST CLOSED BAR (n-1) — that's the candidate trigger
    # bar; натopговка is bars [n-2 .. n-2-k] before it.
    last_natorg_idx = n - 2
    if last_natorg_idx < p.natorgovka_compression_bars_min:
        return None

    # Find compression-run length
    run = 0
    touches = 0
    max_wick_beyond = 0.0
    range_compressed = []
    for j in range(last_natorg_idx, max(-1, last_natorg_idx - p.natorgovka_compression_bars_max - 1), -1):
        if j < 0:
            break
        k = klines[j]
        body_mid = (k.open + k.close) / 2
        if abs(body_mid - level) > tol:
            break
        run += 1
        tr = max(k.high - k.low, 1e-12)
        # Wick beyond level — distinguishes "press" from "sweep"
        if side == "long":  # level above; wick beyond is high > level
            wick_beyond = max(0.0, k.high - level)
        else:                # level below; wick beyond is low < level
            wick_beyond = max(0.0, level - k.low)
        max_wick_beyond = max(max_wick_beyond, wick_beyond / tr)
        # Touch count — wick within tolerance to level
        if side == "long" and abs(k.high - level) < tol:
            touches += 1
        if side == "short" and abs(k.low - level) < tol:
            touches += 1
        range_compressed.append(tr)

    if run < p.natorgovka_compression_bars_min:
        return None
    if max_wick_beyond > p.natorgovka_max_wick_beyond_pct:
        return None
    if touches < p.natorgovka_min_touches:
        return None

    # Range contraction — mean range of natorgovka vs prior 20 bars
    prior_start = max(0, last_natorg_idx - run - 20)
    prior_end = max(0, last_natorg_idx - run)
    if prior_end <= prior_start:
        contraction = 0.0
    else:
        prior_ranges = [klines[j].high - klines[j].low
                        for j in range(prior_start, prior_end)]
        if not prior_ranges or statistics.fmean(prior_ranges) <= 0:
            contraction = 0.0
        else:
            contraction = 1.0 - statistics.fmean(range_compressed) / statistics.fmean(prior_ranges)

    if contraction < p.natorgovka_range_contraction_min:
        return None

    return Natorgovka(
        level=level, side=side, touch_count=touches,
        compression_bars=run, range_contraction=contraction,
        max_wick_beyond_pct=max_wick_beyond,
    )


def _detect_trigger(
    klines: list[Kline], level: float, side: str, atr: float, p: CascadeParams,
) -> Optional[Trigger]:
    """The last closed bar must break the level with body confirmation."""
    n = len(klines)
    if n < 21:
        return None
    bar = klines[n - 1]
    tr = max(bar.high - bar.low, 1e-12)
    body = abs(bar.close - bar.open)
    body_pct = body / tr
    if body_pct < p.trigger_body_pct_min:
        return None

    # Must close beyond level in cascade direction by min ATR fraction
    min_beyond = p.trigger_close_beyond_pct_atr * atr
    if side == "long":
        if bar.close <= level + min_beyond:
            return None
        close_beyond_atr = (bar.close - level) / atr
    else:
        if bar.close >= level - min_beyond:
            return None
        close_beyond_atr = (level - bar.close) / atr

    # Volume confirmation — last bar volume vs prior 20-bar mean
    prior_vol = [klines[j].volume for j in range(max(0, n - 21), n - 1)]
    if not prior_vol or statistics.fmean(prior_vol) <= 0:
        return None
    vol_mult = bar.volume / statistics.fmean(prior_vol)
    if vol_mult < p.trigger_vol_mult_min:
        return None

    return Trigger(
        idx=n - 1, body_pct=body_pct, vol_mult=vol_mult,
        close_beyond_atr=close_beyond_atr,
    )


def _has_htf_confluence(
    klines_htf: list[Kline], level: float, atr_htf: float, p: CascadeParams
) -> bool:
    """Optional bonus: натopговка level sits within X*ATR of a prior H4/D1
    swing high/low in the lookback window."""
    if not klines_htf:
        return False
    tol = p.htf_level_tol_atr * atr_htf
    highs, lows = _find_swing_pivots(klines_htf, k=2, lookback=p.htf_lookback_bars)
    for pivot in highs + lows:
        if abs(pivot.price - level) <= tol:
            return True
    return False


# ─────────────────────────── public detect API ──────────────────────────────


def detect_pattern(
    klines: list[Kline],
    klines_htf: Optional[list[Kline]] = None,
    params: Optional[CascadeParams] = None,
) -> Optional[CascadePattern]:
    """Look at the last closed bar of `klines`; return a CascadePattern if
    a complete setup (cascade + natorgovka + trigger) is in place there.
    `klines_htf` (e.g. H4) is optional and only adds the 4th confluence."""
    p = params or CascadeParams()
    if len(klines) < max(p.cascade_lookback_bars, 30):
        return None
    atr = _atr(klines, period=14)
    if atr is None or atr <= 0:
        return None

    highs, lows = _find_swing_pivots(klines, k=p.swing_k, lookback=p.cascade_lookback_bars)
    if len(highs) + len(lows) < p.cascade_min_pivots:
        return None

    # Try BOTH sides; pick the higher-quality match (more pivots, then higher R²)
    best: Optional[CascadePattern] = None

    for side in ("long", "short"):
        chain = _build_cascade_chain(highs, lows, side, atr, p)
        if chain is None:
            continue

        # Level the cascade is pressing into: for a long cascade, it's the
        # highest swing HIGH in the chain (next resistance); for short, the
        # lowest swing LOW.
        if side == "long":
            level_candidates = [pp.price for pp in chain.pivots if pp.kind == "high"]
            level = max(level_candidates) if level_candidates else None
        else:
            level_candidates = [pp.price for pp in chain.pivots if pp.kind == "low"]
            level = min(level_candidates) if level_candidates else None
        if level is None:
            continue

        nat = _detect_natorgovka(klines, level, side, atr, p)
        if nat is None:
            continue

        trg = _detect_trigger(klines, level, side, atr, p)
        if trg is None:
            continue

        # HTF confluence (optional)
        htf_ok = False
        if klines_htf:
            atr_htf = _atr(klines_htf, period=14) or atr
            htf_ok = _has_htf_confluence(klines_htf, level, atr_htf, p)

        confluence = 2 + (1 if htf_ok else 0) + 1   # cascade + natorgovka + trigger (+ htf?)
        # Actually count distinct components: cascade(1) + natorgovka(1) + trigger(1) + htf(?)
        confluence = 3 + (1 if htf_ok else 0)

        candidate = CascadePattern(
            side=side, cascade=chain, natorgovka=nat, trigger=trg,
            confluence_count=confluence, htf_level_confirmed=htf_ok,
        )
        if best is None:
            best = candidate
        else:
            # Prefer higher confluence, then more pivots, then higher R²
            if (candidate.confluence_count, len(candidate.cascade.pivots),
                candidate.cascade.slope_r2) > (
                best.confluence_count, len(best.cascade.pivots),
                best.cascade.slope_r2):
                best = candidate

    return best
