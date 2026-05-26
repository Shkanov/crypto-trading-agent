"""Candidate-universe scanner for the cascade-breakout strategy (v2).

Encodes the selection edge of the @aktradescalp channel author (see
project memory `project_cascade_strategy_research`) as a cross-sectional
ranking on Binance perp futures.

Edge thesis (3 confirmed legs + 1 weak leg):
  - deep-vol session window (07-12 UTC peaks at 11:00, Amberdata depth study)
  - fresh-listing perp MM mispricing (Kaiko, Coin Bureau)
  - cross-sectional momentum in alt long-tail (Drogen/Hoffstein SSRN 4322637)
  - Friday concentration (weak/behavioral — kept as a 1.2× weight only)

The scanner is a PURE-LOGIC module. Feature computation is no-lookahead
(uses only bars closed <= at_ts_ms) so the same code can run live and in
backtest. Data fetching lives in the validation harness, not here.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.models.types import Kline


# ───────────────────────────────── params ──────────────────────────────────


@dataclass(frozen=True)
class UniverseParams:
    """Membership filter for the candidate universe.

    The original research thesis ("fresh-listing perps in 2-60d window") was
    FALSIFIED against the 36-call corpus on 2026-05-26: his picks span
    listing-age 54d to 2400d+, with no concentration in the fresh window.
    Recalibrated to his actual profile — mid-cap alts ($50M-$2B 24h vol)
    of any age, excluding top-10 majors. See `project-cascade-strategy-research`
    memory for the original (falsified) thesis.
    """
    min_vol_24h_usd: float = 50_000_000
    max_vol_24h_usd: float = 2_000_000_000
    min_listing_age_days: float = 2.0
    max_listing_age_days: float = 9999.0     # effectively no upper bound
    excluded_symbols: frozenset[str] = frozenset({
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT",
    })


@dataclass(frozen=True)
class ScannerParams:
    """Per-hour scoring thresholds. Defaults from the v2-research synthesis."""
    vol_z_min: float = 2.0
    oi_z_min: float = 2.0
    funding_short_bias: float = 0.001     # > +0.10%/8h → short bias
    funding_long_bias: float = -0.0005    # < −0.05%/8h → long bias
    rank_topn: int = 10
    # 2.0 fits the empirical reality that OI history is unavailable >30d back
    # on Binance, so historical validation effectively has 3 components
    # {vol_z, funding, rank}. Live trading will have all 4 and naturally
    # produces higher scores. Friday × 1.2 means Friday score=2 → 2.4.
    score_min: float = 2.0
    friday_multiplier: float = 1.2
    # Session gate — only fire candidates in this UTC hour window.
    session_start_utc: int = 7
    session_end_utc: int = 12             # inclusive


# ────────────────────────────── data shapes ────────────────────────────────


@dataclass
class SymbolHistory:
    """All historical data the scanner needs to feature a symbol at any ts."""
    symbol: str
    klines_1h: list[Kline]                            # ascending close_time
    funding_rates: list[tuple[int, float]]            # (ts_ms, rate per 8h)
    oi_history: list[tuple[int, float]]               # (ts_ms, OI in base units)
    listing_date_ms: Optional[int] = None


@dataclass
class SymbolFeatures:
    symbol: str
    ts_ms: int
    quote_vol_24h_usd: Optional[float] = None
    days_since_listing: Optional[float] = None
    vol_z_1h_sameHour_30d: Optional[float] = None
    oi_z_24h_30d: Optional[float] = None
    ret_24h_bps: Optional[float] = None
    funding_rate_8h: Optional[float] = None
    history_ok: bool = True


@dataclass
class Candidate:
    """A symbol that crossed the attention gate at ts_ms.

    Side is NOT determined here — the downstream pattern detector (M2)
    inspects каскад + наторговка structure on the symbol and chooses
    long/short based on the chart. `side_hint` is a soft suggestion
    derived from rank + funding alignment; it can be overridden.

    Empirical justification (2026-05-26): in the 36-call corpus his calls
    are split across extreme-top-of-rank (fade) and extreme-bot-of-rank
    (continuation), with no consistent rank→side mapping. Tying side to
    rank here would falsely reject ~50% of his actual picks.
    """
    symbol: str
    score: float
    ts_ms: int
    rank_in_returns: int                              # 1..N (1 = lowest ret)
    universe_n: int
    side_hint: Optional[str] = None                   # 'long' | 'short' | None
    components: dict[str, bool] = field(default_factory=dict)


# ──────────────────────────── feature compute ──────────────────────────────


def _zscore(x: float, series: list[float], min_n: int = 10) -> Optional[float]:
    if len(series) < min_n:
        return None
    mu = statistics.fmean(series)
    sd = statistics.pstdev(series)
    if sd == 0:
        return None
    return (x - mu) / sd


def _interp_at(series: list[tuple[int, float]], ts_ms: int) -> Optional[float]:
    """Last sample with timestamp <= ts_ms. No interpolation — point-in-time."""
    val: Optional[float] = None
    for ts, v in series:
        if ts <= ts_ms:
            val = v
        else:
            break
    return val


def compute_features(hist: SymbolHistory, at_ts_ms: int) -> SymbolFeatures:
    """Compute features as-of `at_ts_ms`. No look-ahead: only uses bars/OI/
    funding entries with timestamp <= at_ts_ms."""
    sym = hist.symbol
    valid = [k for k in hist.klines_1h if k.close_time <= at_ts_ms]
    # Need at least 30 days of 1h bars for the same-hour-of-day z-baseline
    if len(valid) < 30 * 24:
        return SymbolFeatures(symbol=sym, ts_ms=at_ts_ms, history_ok=False)

    cur = valid[-1]
    px = cur.close

    last_24 = valid[-24:]
    quote_vol_24h_usd = sum(k.quote_volume for k in last_24)

    days_since_listing: Optional[float] = None
    if hist.listing_date_ms and hist.listing_date_ms > 0:
        days_since_listing = (at_ts_ms - hist.listing_date_ms) / (1000 * 86400)

    cur_hour = datetime.fromtimestamp(cur.close_time / 1000, tz=timezone.utc).hour
    # 30d of same-hour-of-day samples, excluding the current bar
    same_hour_history = [
        k.quote_volume
        for k in valid[-30 * 24:-1]
        if datetime.fromtimestamp(k.close_time / 1000, tz=timezone.utc).hour == cur_hour
    ]
    vol_z = _zscore(cur.quote_volume, same_hour_history)

    oi_now = _interp_at(hist.oi_history, at_ts_ms)
    oi_24h_ago = _interp_at(hist.oi_history, at_ts_ms - 86_400_000)
    oi_z: Optional[float] = None
    if oi_now is not None and oi_24h_ago is not None and oi_24h_ago > 0:
        delta_now = (oi_now - oi_24h_ago) / oi_24h_ago
        baseline = []
        for d in range(1, 31):
            t_end = at_ts_ms - d * 86_400_000
            t_start = t_end - 86_400_000
            a = _interp_at(hist.oi_history, t_end)
            b = _interp_at(hist.oi_history, t_start)
            if a is not None and b is not None and b > 0:
                baseline.append((a - b) / b)
        oi_z = _zscore(delta_now, baseline, min_n=10)

    ret_24h_bps: Optional[float] = None
    if len(valid) >= 24:
        ref = valid[-24].open
        if ref > 0:
            ret_24h_bps = (px - ref) / ref * 10_000

    funding = _interp_at(hist.funding_rates, at_ts_ms)

    return SymbolFeatures(
        symbol=sym,
        ts_ms=at_ts_ms,
        quote_vol_24h_usd=quote_vol_24h_usd,
        days_since_listing=days_since_listing,
        vol_z_1h_sameHour_30d=vol_z,
        oi_z_24h_30d=oi_z,
        ret_24h_bps=ret_24h_bps,
        funding_rate_8h=funding,
        history_ok=True,
    )


# ─────────────────────────── universe filter ───────────────────────────────


def passes_universe(f: SymbolFeatures, p: UniverseParams) -> bool:
    if f.symbol in p.excluded_symbols:
        return False
    if not f.history_ok:
        return False
    if f.quote_vol_24h_usd is None:
        return False
    if not (p.min_vol_24h_usd <= f.quote_vol_24h_usd <= p.max_vol_24h_usd):
        return False
    if f.days_since_listing is None:
        return False
    if not (p.min_listing_age_days <= f.days_since_listing <= p.max_listing_age_days):
        return False
    return True


# ──────────────────────────── scoring + rank ───────────────────────────────


def score_universe(
    features_by_symbol: dict[str, SymbolFeatures],
    ts_ms: int,
    u: UniverseParams,
    s: ScannerParams,
) -> list[Candidate]:
    """Score every eligible symbol at ts_ms and return Candidates whose
    side-neutral attention score crosses s.score_min, sorted desc.

    Attention components (all side-neutral):
      - vol_z hit:         unusual 1h volume vs 30d same-hour-of-day baseline
      - oi_z hit:          unusual 24h ΔOI vs 30d baseline (None when OI absent)
      - extreme rank:      symbol's 24h return is in top-N or bot-N of universe
      - funding extreme:   |funding| > min(short_bias, |long_bias|) — any direction

    `side_hint` is set only when rank-side and funding-side agree:
      - top-rank + crowded-long funding → fade short
      - bot-rank + crowded-short funding → squeeze long
    Else None — pattern detector decides.

    Session gating: returns [] if ts_ms's UTC hour is outside the window.
    """
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    hour = dt.hour
    if not (s.session_start_utc <= hour <= s.session_end_utc):
        return []

    eligible = {
        sym: f for sym, f in features_by_symbol.items()
        if passes_universe(f, u)
    }
    if not eligible:
        return []

    ret_pairs = sorted(
        [(sym, f.ret_24h_bps) for sym, f in eligible.items()
         if f.ret_24h_bps is not None],
        key=lambda x: x[1],
    )
    rank_lookup = {sym: i + 1 for i, (sym, _) in enumerate(ret_pairs)}
    n = len(ret_pairs)
    bot_set = {sym for sym, _ in ret_pairs[: s.rank_topn]}
    top_set = {sym for sym, _ in ret_pairs[max(0, n - s.rank_topn):]}

    is_friday = dt.weekday() == 4
    mult = s.friday_multiplier if is_friday else 1.0

    out: list[Candidate] = []
    for sym, f in eligible.items():
        vol_hit = (f.vol_z_1h_sameHour_30d is not None
                   and f.vol_z_1h_sameHour_30d >= s.vol_z_min)
        oi_hit = (f.oi_z_24h_30d is not None
                  and f.oi_z_24h_30d >= s.oi_z_min)
        in_top = sym in top_set
        in_bot = sym in bot_set
        rank_hit = in_top or in_bot
        fund = f.funding_rate_8h
        fund_long_hit = fund is not None and fund < s.funding_long_bias
        fund_short_hit = fund is not None and fund > s.funding_short_bias
        fund_hit = fund_long_hit or fund_short_hit

        components = {
            "vol_z": vol_hit,
            "oi_z": oi_hit,
            "rank_extreme": rank_hit,
            "funding_extreme": fund_hit,
        }
        attention = sum(components.values()) * mult

        side_hint: Optional[str] = None
        if in_top and fund_short_hit:
            side_hint = "short"
        elif in_bot and fund_long_hit:
            side_hint = "long"

        if attention >= s.score_min:
            out.append(Candidate(
                symbol=sym,
                score=attention,
                ts_ms=ts_ms,
                rank_in_returns=rank_lookup.get(sym, 0),
                universe_n=n,
                side_hint=side_hint,
                components=components,
            ))

    out.sort(key=lambda c: c.score, reverse=True)
    return out


# ─────────────────────────────── helpers ───────────────────────────────────


def in_session(ts_ms: int, p: ScannerParams = ScannerParams()) -> bool:
    """True iff ts_ms's UTC hour falls in the configured session window."""
    h = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    return p.session_start_utc <= h <= p.session_end_utc
