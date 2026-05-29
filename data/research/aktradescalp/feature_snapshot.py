"""Feature snapshot at @aktradescalp's call timestamps vs a same-symbol /
same-side random baseline in his trading window.

Purpose: see whether any measurable market feature at the moment of his entry
discriminates his picks from a random eligible bar on the same symbol — i.e.
whether his discretion is reverse-engineerable from kline data alone.

Outputs:
  - data/research/aktradescalp/feature_snapshot.jsonl  (per-row feature rows)
  - stdout summary table comparing his-call distribution vs random baseline

NOTE on scope: this is a *per-symbol-history* snapshot — we ask "at his entry
time, is the symbol unusual against its own 7-day history?". A true cross-
sectional rank ("is this symbol the top vol_z perp in the universe right
now?") would need klines for all ~300 perps and is a v1 follow-up if any
single-symbol feature shows separation.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.models.types import Kline
from src.tools.binance_client import BinanceClient
from src.tools.indicators import IndicatorEngine

HERE = Path(__file__).resolve().parent
CALLS_PATH = HERE / "aktradescalp_calls.json"
OUT_PATH = HERE / "feature_snapshot.jsonl"

TF = "5m"
TF_MS = 300_000
BARS_PER_HOUR = 12
BARS_PER_DAY = 288
BARS_PER_WEEK = 2016

# Random-baseline knobs
RANDOM_SAMPLES_PER_CALL = 4
RANDOM_EXCLUSION_BARS = 24       # ±2h around his actual call: don't pick from here
HOUR_WINDOW_UTC = (7, 12)        # 10-15 MSK = 07-12 UTC


@dataclass
class Features:
    symbol: str
    side: str
    kind: str                    # "his" | "random"
    dt_iso: str
    msg_id: int

    # Volume / liquidity
    quote_vol_5m_usd: Optional[float] = None
    vol_z_5m_7d: Optional[float] = None          # z-score of last M5 qv vs 7d
    vol_z_1h_7d: Optional[float] = None          # z-score of last hour sum vs 7d hours
    taker_buy_ratio_1h: Optional[float] = None   # taker-buy / total over last 12 bars

    # Volatility / range
    atr14_bps: Optional[float] = None
    atr_expansion_7d: Optional[float] = None     # atr_now / mean(atr last 7d)
    range_pct_5m: Optional[float] = None         # (h-l)/c of last bar

    # Trend / momentum
    ret_1h_bps: Optional[float] = None
    ret_4h_bps: Optional[float] = None
    ret_24h_bps: Optional[float] = None

    # Structure
    dist_24h_high_bps: Optional[float] = None    # positive if below high
    dist_24h_low_bps: Optional[float] = None     # positive if above low
    dist_7d_high_bps: Optional[float] = None
    dist_7d_low_bps: Optional[float] = None
    # Side-aligned: if short → distance to recent high (small = near resistance,
    #               which is where shorts get set up). If long → distance to low.
    dist_aligned_24h_bps: Optional[float] = None
    dist_aligned_7d_bps: Optional[float] = None

    skipped: Optional[str] = None


def _rows_to_klines(rows: list[list], symbol: str, tf: str) -> list[Kline]:
    return [Kline(
        symbol=symbol, timeframe=tf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in rows]


def _bps(num: float, denom: float) -> Optional[float]:
    if denom <= 0:
        return None
    return num / denom * 10_000


def _zscore(x: float, series: list[float]) -> Optional[float]:
    if len(series) < 30:
        return None
    mu = statistics.fmean(series)
    sd = statistics.pstdev(series)
    if sd == 0:
        return None
    return (x - mu) / sd


def _snapshot(ks: list[Kline], entry_ix: int, side: str) -> Features:
    """Compute features as of bar `entry_ix` (the bar the trader would enter on,
    i.e. the first closed bar after the call). Uses ONLY bars 0..entry_ix-1 for
    history and bar entry_ix as the 'current' state — no look-ahead."""
    f = Features(symbol=ks[0].symbol, side=side, kind="", dt_iso="", msg_id=0)
    if entry_ix < BARS_PER_WEEK:
        f.skipped = "insufficient_history"
        return f

    entry_bar = ks[entry_ix]
    px = entry_bar.close

    # Volume
    f.quote_vol_5m_usd = entry_bar.quote_volume

    qv_hist_5m = [k.quote_volume for k in ks[entry_ix - BARS_PER_WEEK:entry_ix]]
    f.vol_z_5m_7d = _zscore(entry_bar.quote_volume, qv_hist_5m)

    # Last-hour aggregated volume, vs prior 7d of rolling-hour sums
    last_hour_qv = sum(k.quote_volume for k in ks[entry_ix - BARS_PER_HOUR + 1:entry_ix + 1])
    rolling_hours = []
    # Stride by 1 bar — 2016 - 12 ≈ 2004 overlapping samples; fine for z
    for j in range(entry_ix - BARS_PER_WEEK, entry_ix - BARS_PER_HOUR + 1):
        rolling_hours.append(sum(k.quote_volume for k in ks[j:j + BARS_PER_HOUR]))
    f.vol_z_1h_7d = _zscore(last_hour_qv, rolling_hours)

    # Taker buy ratio last hour
    last_hour = ks[entry_ix - BARS_PER_HOUR + 1:entry_ix + 1]
    tot_v = sum(k.volume for k in last_hour)
    tb_v = sum(k.taker_buy_volume for k in last_hour)
    if tot_v > 0:
        f.taker_buy_ratio_1h = tb_v / tot_v

    # ATR(14) — incremental Wilder over bars 0..entry_ix. Capture history
    # over the last 7d to compute the expansion ratio in one pass.
    state = IndicatorEngine().get(ks[0].symbol, TF)
    atr_hist: list[float] = []
    for j, k in enumerate(ks[:entry_ix + 1]):
        state.on_closed_kline(k)
        if j >= entry_ix - BARS_PER_WEEK and state.atr:
            atr_hist.append(state.atr)
    atr_now = state.atr
    if atr_now and atr_now > 0:
        f.atr14_bps = atr_now / px * 10_000
        if atr_hist:
            mean_atr = statistics.fmean(atr_hist)
            if mean_atr > 0:
                f.atr_expansion_7d = atr_now / mean_atr

    f.range_pct_5m = (entry_bar.high - entry_bar.low) / px * 100 if px > 0 else None

    # Returns
    def _ret_bps(lookback: int) -> Optional[float]:
        if entry_ix - lookback < 0:
            return None
        prev = ks[entry_ix - lookback].close
        return _bps(px - prev, prev)
    f.ret_1h_bps = _ret_bps(BARS_PER_HOUR)
    f.ret_4h_bps = _ret_bps(4 * BARS_PER_HOUR)
    f.ret_24h_bps = _ret_bps(BARS_PER_DAY)

    # Structure: 24h high/low using highs/lows over last day (exclusive of entry)
    win_24h = ks[entry_ix - BARS_PER_DAY:entry_ix]
    h24 = max(k.high for k in win_24h)
    l24 = min(k.low for k in win_24h)
    f.dist_24h_high_bps = _bps(h24 - px, px)
    f.dist_24h_low_bps = _bps(px - l24, px)

    win_7d = ks[entry_ix - BARS_PER_WEEK:entry_ix]
    h7d = max(k.high for k in win_7d)
    l7d = min(k.low for k in win_7d)
    f.dist_7d_high_bps = _bps(h7d - px, px)
    f.dist_7d_low_bps = _bps(px - l7d, px)

    if side == "short":
        f.dist_aligned_24h_bps = f.dist_24h_high_bps
        f.dist_aligned_7d_bps = f.dist_7d_high_bps
    elif side == "long":
        f.dist_aligned_24h_bps = f.dist_24h_low_bps
        f.dist_aligned_7d_bps = f.dist_7d_low_bps

    return f


def _entry_ix_for_ts(ks: list[Kline], ts_ms: int) -> Optional[int]:
    """First bar whose close_time > ts_ms — i.e. the first bar a trader could
    actually enter on after the message was published."""
    for i, k in enumerate(ks):
        if k.close_time > ts_ms:
            return i
    return None


def _random_eligible_indices(ks: list[Kline], avoid_ts_ms: int,
                             n: int) -> list[int]:
    """Pick n random bar indices that:
      - have enough history (>= BARS_PER_WEEK before)
      - have enough room after (>= 1 bar)
      - fall in HOUR_WINDOW_UTC (7..12)
      - are not within RANDOM_EXCLUSION_BARS of avoid_ts_ms
    Without-replacement sampling.
    """
    eligible = []
    avoid_lo = avoid_ts_ms - RANDOM_EXCLUSION_BARS * TF_MS
    avoid_hi = avoid_ts_ms + RANDOM_EXCLUSION_BARS * TF_MS
    for i in range(BARS_PER_WEEK + 5, len(ks) - 1):
        k = ks[i]
        if not (avoid_lo <= k.close_time <= avoid_hi):
            hour = datetime.fromtimestamp(k.close_time / 1000,
                                          tz=timezone.utc).hour
            if HOUR_WINDOW_UTC[0] <= hour <= HOUR_WINDOW_UTC[1]:
                eligible.append(i)
    if not eligible:
        return []
    k_sample = min(n, len(eligible))
    return random.sample(eligible, k_sample)


# ── Stats helpers (no scipy) ────────────────────────────────────────────────
def _mann_whitney_u_p(a: list[float], b: list[float]) -> Optional[float]:
    """Two-sided Mann-Whitney U p-value via normal approximation (no tie
    correction). Good enough at n>=20 per side for triage; we're not making
    inference claims, just ranking features by separation."""
    n1, n2 = len(a), len(b)
    if n1 < 5 or n2 < 5:
        return None
    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort(key=lambda x: x[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed midrank
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, combined) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu_u = n1 * n2 / 2
    sd_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sd_u == 0:
        return None
    z = (u1 - mu_u) / sd_u
    # 2-sided p
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def _summary(name: str, his: list[float], rnd: list[float]) -> dict:
    if not his or not rnd:
        return {"feature": name, "his_n": len(his), "rnd_n": len(rnd)}
    pool_sd = math.sqrt((statistics.pvariance(his) + statistics.pvariance(rnd)) / 2)
    mean_diff = statistics.fmean(his) - statistics.fmean(rnd)
    sep = mean_diff / pool_sd if pool_sd > 0 else 0.0
    return {
        "feature": name,
        "his_n": len(his),
        "rnd_n": len(rnd),
        "his_med": statistics.median(his),
        "rnd_med": statistics.median(rnd),
        "his_mean": statistics.fmean(his),
        "rnd_mean": statistics.fmean(rnd),
        "sep_cohen_d": sep,
        "mw_p": _mann_whitney_u_p(his, rnd),
    }


FEATURE_FIELDS = [
    "quote_vol_5m_usd", "vol_z_5m_7d", "vol_z_1h_7d", "taker_buy_ratio_1h",
    "atr14_bps", "atr_expansion_7d", "range_pct_5m",
    "ret_1h_bps", "ret_4h_bps", "ret_24h_bps",
    "dist_24h_high_bps", "dist_24h_low_bps", "dist_7d_high_bps", "dist_7d_low_bps",
    "dist_aligned_24h_bps", "dist_aligned_7d_bps",
]


async def amain() -> None:
    load_dotenv()
    calls = json.load(open(CALLS_PATH))
    # Keep only single-ticker long/short calls (same filter as analyzer)
    calls = [c for c in calls if c["side"] in ("long", "short")]
    print(f"loaded {len(calls)} calls")

    b = BinanceClient()
    await b.start()

    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for c in calls:
        by_symbol[c["symbol"]].append(c)

    rows: list[Features] = []
    skipped_symbols: list[str] = []

    try:
        for symbol, group in sorted(by_symbol.items()):
            try:
                raw = await b.fetch_klines_paginated(
                    symbol, TF, total=20_000, market="perps")
            except Exception as e:
                skipped_symbols.append(f"{symbol}({str(e).splitlines()[0][:50]})")
                for c in group:
                    rows.append(Features(symbol=symbol, side=c["side"],
                                         kind="his", dt_iso=c["dt_iso"],
                                         msg_id=c["msg_id"],
                                         skipped="fetch_err"))
                continue
            ks = _rows_to_klines(raw, symbol, TF)
            if len(ks) < BARS_PER_WEEK + 100:
                skipped_symbols.append(f"{symbol}({len(ks)}bars)")
                for c in group:
                    rows.append(Features(symbol=symbol, side=c["side"],
                                         kind="his", dt_iso=c["dt_iso"],
                                         msg_id=c["msg_id"],
                                         skipped="symbol_unavailable"))
                continue

            for c in group:
                ts_ms = int(datetime.fromisoformat(c["dt_iso"]).timestamp() * 1000)
                ix = _entry_ix_for_ts(ks, ts_ms)
                if ix is None:
                    rows.append(Features(symbol=symbol, side=c["side"],
                                         kind="his", dt_iso=c["dt_iso"],
                                         msg_id=c["msg_id"],
                                         skipped="no_entry_bar"))
                    continue
                his_feat = _snapshot(ks, ix, c["side"])
                his_feat.kind = "his"
                his_feat.dt_iso = c["dt_iso"]
                his_feat.msg_id = c["msg_id"]
                rows.append(his_feat)

                # Random baseline: same side, same symbol, eligible bar
                for ix_rnd in _random_eligible_indices(
                        ks, ts_ms, RANDOM_SAMPLES_PER_CALL):
                    rnd_feat = _snapshot(ks, ix_rnd, c["side"])
                    rnd_feat.kind = "random"
                    rnd_feat.dt_iso = datetime.fromtimestamp(
                        ks[ix_rnd].close_time / 1000, tz=timezone.utc).isoformat()
                    rnd_feat.msg_id = -1
                    rows.append(rnd_feat)

    finally:
        await b.close()

    if skipped_symbols:
        print(f"skipped symbols: {len(skipped_symbols)} — "
              f"{', '.join(skipped_symbols[:6])}"
              + (f" (+{len(skipped_symbols) - 6} more)"
                 if len(skipped_symbols) > 6 else ""))

    # Persist
    with OUT_PATH.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT_PATH}")

    # Summary
    his_rows = [r for r in rows if r.kind == "his" and r.skipped is None]
    rnd_rows = [r for r in rows if r.kind == "random" and r.skipped is None]
    print(f"\nValid rows: his={len(his_rows)}  random={len(rnd_rows)}")

    print("\n=== Per-feature separation (his vs random) ===")
    fmt = ("{feature:24s}  n={his_n:>3d}/{rnd_n:<3d}  "
           "his_med={his_med:>+10.2f}  rnd_med={rnd_med:>+10.2f}  "
           "his_mean={his_mean:>+10.2f}  rnd_mean={rnd_mean:>+10.2f}  "
           "d={sep_cohen_d:>+5.2f}  p={mw_p_str}")
    summaries = []
    for field_name in FEATURE_FIELDS:
        h = [getattr(r, field_name) for r in his_rows
             if getattr(r, field_name) is not None]
        rn = [getattr(r, field_name) for r in rnd_rows
              if getattr(r, field_name) is not None]
        s = _summary(field_name, h, rn)
        summaries.append(s)
        if s.get("his_med") is None:
            print(f"{field_name:24s}  insufficient data")
            continue
        s["mw_p_str"] = f"{s['mw_p']:.4f}" if s.get("mw_p") is not None else "  n/a "
        print(fmt.format(**s))

    # Rank by |d|
    print("\n=== Top features by |Cohen's d| ===")
    ranked = sorted(
        [s for s in summaries if s.get("sep_cohen_d") is not None],
        key=lambda x: abs(x["sep_cohen_d"]),
        reverse=True,
    )
    for s in ranked[:8]:
        direction = "HIGHER in his" if s["sep_cohen_d"] > 0 else "LOWER in his"
        p_str = f"p={s['mw_p']:.4f}" if s.get("mw_p") is not None else ""
        print(f"  {s['feature']:24s}  d={s['sep_cohen_d']:+.2f}  "
              f"{direction:14s}  {p_str}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(amain())
