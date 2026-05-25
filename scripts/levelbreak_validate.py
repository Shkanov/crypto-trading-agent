"""Validation sweep for the LevelBreakoutStrategy.

This is the honest "does the pattern have edge" probe — not a tuning loop.
It runs three reads, side-by-side, with shared formatting:

  1. CROSS-SYMBOL on perps: BTC/ETH/SOL as control + the channel-mentioned
     alts that exist on the configured Binance endpoint. Same window length,
     same params. If the pattern is a low-cap-alt momentum thing, the alts
     should look better than majors. If it doesn't, that's a real finding.

  2. WALK-FORWARD on BTCUSDT/perps: ~4 non-overlapping windows of equal
     length. If results swing wildly window-to-window, the strategy is
     a regime gamble, not an edge.

  3. STRATEGY-COMPARE on BTCUSDT/perps: levelbreak vs the existing
     indicator-confluence and mean-reversion backtests on the SAME recent
     window. Levelbreak is the new entrant; we want to see how it ranks
     against what's already there before flipping any flags.

Usage:
  .venv/bin/python -m scripts.levelbreak_validate
  .venv/bin/python -m scripts.levelbreak_validate --skip-walk    # faster
  .venv/bin/python -m scripts.levelbreak_validate --bars 4000    # smaller windows
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Optional

from dotenv import load_dotenv

from src.models.types import Kline
from src.services.backtest import (
    BacktestStats,
    backtest_indicator,
    backtest_mean_reversion,
    simulate_level_breakout,
)
from src.strategies.level_breakout import LevelBreakoutParams
from src.tools.binance_client import BinanceClient

# Channel-mentioned alts that exist on the configured endpoint (perps).
# Probed via scripts._probe_symbols — re-run that if symbol availability
# changes upstream.
CHANNEL_ALTS_PERPS = ["FIDAUSDT", "PROVEUSDT", "BANANAS31USDT", "GRASSUSDT"]
CONTROL_MAJORS_PERPS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _fmt_row(label: str, stats: BacktestStats) -> str:
    """Compact one-line summary; aligned for grep-ability."""
    return (
        f"{label:28s}  "
        f"n={stats.trades:3d}  "
        f"wr={stats.win_rate * 100:5.1f}%  "
        f"pnl=${stats.total_pnl_usd:+7.2f}  "
        f"avg=${stats.avg_pnl_usd:+5.2f}  "
        f"sharpe={stats.sharpe:+5.2f}  "
        f"defl={stats.deflated_sharpe:+5.2f}  "
        f"mdd={stats.max_drawdown_pct:4.1f}%  "
        f"ann={stats.annualized_pct:+6.1f}%"
    )


def _rows_to_klines(rows: list[list], symbol: str, tf: str) -> list[Kline]:
    return [Kline(
        symbol=symbol, timeframe=tf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in rows]


async def _fetch_ks_pair(
    b: BinanceClient, symbol: str, tf: str, htf: str, bars: int, market: str,
) -> tuple[list[Kline], list[Kline]]:
    """Fetch trigger + HTF klines from one endpoint, in one shot."""
    htf_total = max(60, bars // _bars_per_day(tf) + 60)
    raw = await b.fetch_klines_paginated(symbol, tf, total=bars, market=market)
    htf_raw = await b.fetch_klines_paginated(symbol, htf, total=htf_total, market=market)
    return _rows_to_klines(raw, symbol, tf), _rows_to_klines(htf_raw, symbol, htf)


def _bars_per_day(tf: str) -> int:
    n = "".join(c for c in tf if c.isdigit()) or "1"
    unit = tf[-1]
    mult_min = {"m": 1, "h": 60, "d": 1440}.get(unit, 1)
    return max(1, 1440 // (int(n) * mult_min))


# ───────────────────────────── 1. cross-symbol ─────────────────────────────


async def cross_symbol(b: BinanceClient, *, tf: str, htf: str, bars: int) -> None:
    print()
    print("══ 1. CROSS-SYMBOL  ({} bars × {})  perps  ".format(bars, tf)
          + "═" * 30)
    print()
    params = LevelBreakoutParams(htf=htf, trigger_tf=tf, trendline_tf=tf)
    for symbol in CONTROL_MAJORS_PERPS + CHANNEL_ALTS_PERPS:
        tag = "alt   " if symbol in CHANNEL_ALTS_PERPS else "major "
        try:
            ks, htf_ks = await _fetch_ks_pair(b, symbol, tf, htf, bars, "perps")
            if len(ks) < 500:
                print(f"  {tag}{symbol:14s}  insufficient history "
                      f"({len(ks)} bars, want ≥500) — skipping")
                continue
            stats, _ = simulate_level_breakout(
                symbol, ks, htf_ks, params=params, market="perps",
            )
            print("  " + tag + _fmt_row(symbol, stats))
        except Exception as e:
            msg = str(e).split('\n')[0][:80]
            print(f"  {tag}{symbol:14s}  ERROR: {msg}")


# ───────────────────────────── 2. walk-forward ─────────────────────────────


async def walk_forward(b: BinanceClient, *, tf: str, htf: str,
                       window_bars: int, n_windows: int = 4) -> None:
    print()
    print("══ 2. WALK-FORWARD  BTCUSDT perps  "
          f"{n_windows} × {window_bars} bars × {tf}  "
          + "═" * 22)
    print()
    symbol = "BTCUSDT"
    total_bars = window_bars * n_windows
    # Fetch a single long history and slice. The HTF series spans the whole
    # range, so each window can do its own prior-level lookups correctly.
    ks_all, htf_ks = await _fetch_ks_pair(b, symbol, tf, htf, total_bars, "perps")
    if len(ks_all) < total_bars * 0.9:
        print(f"  insufficient history: got {len(ks_all)} bars, "
              f"wanted ~{total_bars} — running fewer windows")
        n_windows = max(1, len(ks_all) // window_bars)
    params = LevelBreakoutParams(htf=htf, trigger_tf=tf, trendline_tf=tf)
    for i in range(n_windows):
        slice_ks = ks_all[i * window_bars:(i + 1) * window_bars]
        if len(slice_ks) < 500:
            break
        # Slice HTF to "what would be knowable" at the end of this window.
        slice_htf = [hk for hk in htf_ks if hk.close_time <= slice_ks[-1].close_time]
        stats, _ = simulate_level_breakout(
            symbol, slice_ks, slice_htf, params=params, market="perps",
        )
        first_ms = slice_ks[0].close_time
        last_ms = slice_ks[-1].close_time
        span_d = (last_ms - first_ms) / 1000 / 86400
        label = f"win{i + 1} ({span_d:.0f}d)"
        print("  " + _fmt_row(label, stats))


# ───────────────────────────── 3. strategy-compare ─────────────────────────


async def strategy_compare(b: BinanceClient, *, tf: str, htf: str, bars: int) -> None:
    """Same symbol + same window, three strategies. Uses the existing
    backtest_indicator/backtest_mean_reversion (spot path) for parity with
    how they're normally run; uses levelbreak on perps because that's the
    honest market for this strategy. Numbers are NOT cost-normalized across
    venues — read them as relative orderings, not absolute alpha."""
    print()
    print("══ 3. STRATEGY-COMPARE  BTCUSDT  {} bars × {}  ═".format(bars, tf)
          + "═" * 26)
    print()
    symbol = "BTCUSDT"
    # indicator + meanrev: spot (their default; HTF defaults to 1h)
    try:
        stats_ind, _ = await backtest_indicator(b, symbol=symbol, tf=tf,
                                                htf="1h", bars=bars)
        print("  spot  " + _fmt_row("indicator-confluence", stats_ind))
    except Exception as e:
        print(f"  indicator   ERROR: {str(e)[:80]}")
    try:
        stats_mr, _ = await backtest_mean_reversion(b, symbol=symbol, tf=tf,
                                                    htf="1h", bars=bars)
        print("  spot  " + _fmt_row("mean-reversion", stats_mr))
    except Exception as e:
        print(f"  meanrev     ERROR: {str(e)[:80]}")
    # levelbreak: perps (its honest market) + 1d HTF
    try:
        params = LevelBreakoutParams(htf=htf, trigger_tf=tf, trendline_tf=tf)
        ks, htf_ks = await _fetch_ks_pair(b, symbol, tf, htf, bars, "perps")
        stats_lb, _ = simulate_level_breakout(
            symbol, ks, htf_ks, params=params, market="perps",
        )
        print("  perps " + _fmt_row("level-breakout", stats_lb))
    except Exception as e:
        print(f"  levelbreak  ERROR: {str(e)[:80]}")


# ───────────────────────────────── main ────────────────────────────────────


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="15m")
    ap.add_argument("--htf", default="1d")
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--n-windows", type=int, default=4)
    ap.add_argument("--skip-cross", action="store_true")
    ap.add_argument("--skip-walk", action="store_true")
    ap.add_argument("--skip-compare", action="store_true")
    args = ap.parse_args()

    b = BinanceClient()
    await b.start()
    try:
        if not args.skip_cross:
            await cross_symbol(b, tf=args.tf, htf=args.htf, bars=args.bars)
        if not args.skip_walk:
            await walk_forward(b, tf=args.tf, htf=args.htf,
                               window_bars=args.bars, n_windows=args.n_windows)
        if not args.skip_compare:
            await strategy_compare(b, tf=args.tf, htf=args.htf, bars=args.bars)
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
