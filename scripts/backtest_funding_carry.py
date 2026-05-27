"""Backtest driver for the cross-sectional funding carry overlay (sprint #16).

Fan-Jiao-Lu-Tong SSRN 4666425 reports 43.4% p.a. / Sharpe 0.74 for the
long-top minus short-bottom decile funding spread on crypto perps,
concentrated in lower-cap higher-OI alts. This driver implements the
mechanic:

  1. Build a universe of top-N currently-active USDT perps by 24h volume,
     excluding majors (Ethena dominates BTC/ETH) and stablecoins.
  2. Fetch the full funding-rate history per symbol (8h cadence).
  3. Fetch 8h klines per symbol for price-PnL reference.
  4. Walk forward in weekly rebalance steps. At each rebalance ts:
     - Snapshot the most-recent funding rate per symbol.
     - `build_rebalance` → top-N longs + bottom-N shorts (equal-weighted).
     - Hold for one week, accrue funding cycle-by-cycle + price PnL.
     - Aggregate the 2*N per-position results into a single weekly PnL.
  5. Compute Sharpe / DSR / drawdown across the weekly PnL series.

Notes on honesty:
  - This run uses today's universe (no PIT correction). Survivorship bias
    is real; the right path is to layer #9's `universe_pit` filter once we
    have enough delisted-symbol entries. For now, top-50 today is the
    most-similar feasible approximation.
  - All notionals are paper-money. The driver assumes maker-or-taker
    perp fees from `Costs()`; slippage is not modelled (Fan et al. use
    end-of-cycle midpoint fills, same convention).

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.backtest_funding_carry
  BINANCE_TESTNET=false .venv/bin/python -m scripts.backtest_funding_carry \\
      --days 365 --top-n 3 --book-pct-per-side 0.25 --top-n-universe 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.services.backtest import (
    _deflated_sharpe,
    _equity_curve,
    _max_drawdown,
    _sharpe_from_pnls_and_span,
)
from src.services.costs import Costs
from src.strategies.funding_carry import (
    CarryParams,
    CarryPosition,
    build_rebalance,
    cycle_pnl,
)
from src.tools.binance_client import BinanceClient


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data/research/strategy_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)


STABLE_BASE_TOKENS = ("FDUSD", "USDC", "EUR", "USD1", "TUSD", "BUSD", "DAI", "PYUSD")
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BEARUSDT", "BULLUSDT")
MAJORS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"})


# ---------------------------------------------------------------------------
# Data shapes for the driver

@dataclass
class SymbolHistory:
    funding: list[tuple[int, float]] = field(default_factory=list)   # (ts_ms, rate)
    closes_8h: dict[int, float] = field(default_factory=dict)         # close_time → close


@dataclass
class WeeklyResult:
    rebalance_ts_ms: int
    universe_n: int
    longs: list[str]
    shorts: list[str]
    long_price_pnl: float
    short_price_pnl: float
    long_funding_pnl: float
    short_funding_pnl: float
    fee_pnl: float
    total_pnl_usd: float


# ---------------------------------------------------------------------------
# Universe + fetch

async def build_universe(b: BinanceClient, top_n_universe: int) -> list[str]:
    """Top-N USDT perps by 24h quote volume, majors and stablecoins out."""
    assert b.client is not None
    async with b.rest_limiter:
        tickers = await b.client.futures_ticker()
    rows = [
        r for r in tickers
        if r["symbol"].endswith("USDT")
        and r["symbol"] not in MAJORS
        and not any(tok in r["symbol"][:-4] for tok in STABLE_BASE_TOKENS)
        and not any(r["symbol"].endswith(suf) for suf in LEVERAGED_SUFFIXES)
    ]
    rows.sort(key=lambda r: float(r["quoteVolume"]), reverse=True)
    return [r["symbol"] for r in rows[:top_n_universe]]


async def fetch_funding_history(b: BinanceClient, symbol: str,
                                start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    """Page through `futures_funding_rate` to cover [start_ms, end_ms]."""
    assert b.client is not None
    out: list[tuple[int, float]] = []
    cursor = start_ms
    while cursor < end_ms:
        async with b.rest_limiter:
            try:
                page = await b.client.futures_funding_rate(
                    symbol=symbol, startTime=cursor, endTime=end_ms, limit=1000,
                )
            except Exception as e:  # noqa: BLE001
                print(f"   {symbol}: funding page failed: {type(e).__name__}: {e}")
                return out
        if not page:
            break
        for row in page:
            out.append((int(row["fundingTime"]), float(row["fundingRate"])))
        last_t = int(page[-1]["fundingTime"])
        if len(page) < 1000 or last_t <= cursor:
            break
        cursor = last_t + 1
    out.sort(key=lambda x: x[0])
    return out


async def fetch_perp_closes_8h(b: BinanceClient, symbol: str,
                                bars: int) -> dict[int, float]:
    """8h close prices for the symbol, indexed by close_time."""
    try:
        raw = await b.fetch_klines_paginated(symbol, "8h", total=bars,
                                              market="perps")
    except Exception as e:  # noqa: BLE001
        print(f"   {symbol}: 8h kline fetch failed: {type(e).__name__}: {e}")
        return {}
    return {int(r[6]): float(r[4]) for r in raw}


# ---------------------------------------------------------------------------
# Simulation

def _funding_rate_at(history: list[tuple[int, float]], ts_ms: int) -> float | None:
    """Most recent funding rate observed strictly before `ts_ms`."""
    val: float | None = None
    for t, r in history:
        if t < ts_ms:
            val = r
        else:
            break
    return val


def _nearest_price(closes_8h: dict[int, float], ts_ms: int) -> float | None:
    """Look up the most recent 8h close at or before `ts_ms`. Returns None
    when no kline is available."""
    if not closes_8h:
        return None
    cts = [t for t in closes_8h.keys() if t <= ts_ms]
    if not cts:
        return None
    return closes_8h[max(cts)]


def simulate_carry(
    histories: dict[str, SymbolHistory],
    start_ms: int,
    end_ms: int,
    p: CarryParams,
    start_equity: float,
    costs: Costs,
) -> list[WeeklyResult]:
    """Walk weekly rebalances over [start_ms, end_ms]."""
    rebalance_step = p.rebalance_period_hours * 3_600_000
    results: list[WeeklyResult] = []
    equity = start_equity
    ts = start_ms

    while ts + rebalance_step <= end_ms:
        next_ts = ts + rebalance_step

        # Build funding snapshot at `ts` from each symbol's history.
        snapshot: dict[str, float] = {}
        for sym, hist in histories.items():
            r = _funding_rate_at(hist.funding, ts)
            if r is not None:
                snapshot[sym] = r

        rb = build_rebalance(snapshot, equity_usd=equity, ts_ms=ts, p=p)
        if not rb.is_active:
            # Skip this week — keep equity flat, advance cursor.
            ts = next_ts
            continue

        weekly_pnl = 0.0
        long_px_pnl = short_px_pnl = 0.0
        long_fnd_pnl = short_fnd_pnl = 0.0
        fee_pnl = 0.0
        for pos in rb.longs + rb.shorts:
            hist = histories[pos.symbol]
            entry_px = _nearest_price(hist.closes_8h, ts)
            exit_px = _nearest_price(hist.closes_8h, next_ts)
            if entry_px is None or exit_px is None:
                # Skip leg if we don't have prices on either end.
                continue
            # Funding events strictly within (ts, next_ts] are accrued by
            # `funding_accrual_usd` inside `cycle_pnl`.
            r = cycle_pnl(pos, entry_price=entry_px, exit_price=exit_px,
                           funding_events=hist.funding,
                           entry_ts_ms=ts, exit_ts_ms=next_ts,
                           costs=costs)
            weekly_pnl += r.total_pnl_usd
            fee_pnl += r.fee_pnl_usd
            if pos.side == "long":
                long_px_pnl += r.price_pnl_usd
                long_fnd_pnl += r.funding_pnl_usd
            else:
                short_px_pnl += r.price_pnl_usd
                short_fnd_pnl += r.funding_pnl_usd

        equity += weekly_pnl
        results.append(WeeklyResult(
            rebalance_ts_ms=ts, universe_n=rb.universe_n,
            longs=[p.symbol for p in rb.longs],
            shorts=[p.symbol for p in rb.shorts],
            long_price_pnl=long_px_pnl, short_price_pnl=short_px_pnl,
            long_funding_pnl=long_fnd_pnl, short_funding_pnl=short_fnd_pnl,
            fee_pnl=fee_pnl, total_pnl_usd=weekly_pnl,
        ))
        ts = next_ts

    return results


def summarise(results: list[WeeklyResult], start_equity: float,
              span_days: float) -> dict:
    pnls = [r.total_pnl_usd for r in results]
    if not pnls:
        return {"weeks": 0, "total_pnl_usd": 0.0, "sharpe": 0.0,
                "deflated_sharpe": 0.0, "max_drawdown_pct": 0.0,
                "annualized_pct": 0.0, "ending_equity_usd": start_equity}
    sharpe = _sharpe_from_pnls_and_span(pnls, span_days)
    dsr = _deflated_sharpe(sharpe, len(pnls), pnls=pnls)
    curve = _equity_curve(pnls, start_equity)
    dd_usd, dd_pct = _max_drawdown(curve)
    total = float(sum(pnls))
    return {
        "weeks": len(pnls),
        "total_pnl_usd": total,
        "long_price_pnl": float(sum(r.long_price_pnl for r in results)),
        "short_price_pnl": float(sum(r.short_price_pnl for r in results)),
        "long_funding_pnl": float(sum(r.long_funding_pnl for r in results)),
        "short_funding_pnl": float(sum(r.short_funding_pnl for r in results)),
        "fee_pnl": float(sum(r.fee_pnl for r in results)),
        "win_weeks": sum(1 for p in pnls if p > 0),
        "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
        "sharpe": sharpe,
        "deflated_sharpe": dsr,
        "max_drawdown_pct": dd_pct,
        "annualized_pct": (total / start_equity) * (365.0 / span_days) * 100.0
            if span_days > 0 and start_equity > 0 else 0.0,
        "ending_equity_usd": float(curve[-1]),
    }


# ---------------------------------------------------------------------------
# CLI

async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--top-n", type=int, default=3,
                    help="positions per leg (longs + shorts)")
    ap.add_argument("--top-n-universe", type=int, default=30,
                    help="universe size (top-N USDT perps by 24h volume)")
    ap.add_argument("--book-pct-per-side", type=float, default=0.25)
    ap.add_argument("--rebalance-hours", type=int, default=168)
    ap.add_argument("--equity-usd", type=float, default=1_000.0)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    p = CarryParams(top_n=args.top_n,
                    rebalance_period_hours=args.rebalance_hours,
                    book_pct_per_side=args.book_pct_per_side)
    costs = Costs()

    b = BinanceClient()
    await b.start()
    try:
        universe = await build_universe(b, top_n_universe=args.top_n_universe)
        print(f"universe ({len(universe)} symbols):\n  {', '.join(universe[:10])}"
              f"{'...' if len(universe) > 10 else ''}")

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - args.days * 86_400_000
        # Fetch per-symbol history.
        histories: dict[str, SymbolHistory] = {}
        bars_8h = math.ceil(args.days * 3) + 10
        t0 = time.time()
        for i, sym in enumerate(universe, 1):
            funding, closes = await asyncio.gather(
                fetch_funding_history(b, sym, start_ms, now_ms),
                fetch_perp_closes_8h(b, sym, bars_8h),
            )
            histories[sym] = SymbolHistory(funding=funding, closes_8h=closes)
            if i % 5 == 0 or i == len(universe):
                print(f"   [{i}/{len(universe)}] {sym}  "
                      f"funding rows={len(funding)}  closes={len(closes)}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

        results = simulate_carry(
            histories, start_ms=start_ms, end_ms=now_ms,
            p=p, start_equity=args.equity_usd, costs=costs,
        )
        span_days = (now_ms - start_ms) / 86_400_000
        stats = summarise(results, start_equity=args.equity_usd, span_days=span_days)

        # Report
        print("\n" + "=" * 90)
        print(f"FUNDING CARRY BACKTEST  ·  {args.days}d  ·  "
              f"top {p.top_n} L/S  ·  {p.book_pct_per_side*100:.0f}%/side")
        print("=" * 90)
        print(f"  weeks:           {stats['weeks']}")
        print(f"  total PnL:       ${stats['total_pnl_usd']:+.2f}")
        print(f"  win rate:        {stats.get('win_rate', 0)*100:.1f}%")
        print(f"  Sharpe:          {stats['sharpe']:+.3f}")
        print(f"  deflated SR:     {stats['deflated_sharpe']:+.3f}")
        print(f"  dd %:            {stats['max_drawdown_pct']:.2f}")
        print(f"  annualized:      {stats['annualized_pct']:+.1f}%")
        print(f"  long  price PnL: ${stats.get('long_price_pnl', 0):+.2f}")
        print(f"  short price PnL: ${stats.get('short_price_pnl', 0):+.2f}")
        print(f"  long  fund PnL:  ${stats.get('long_funding_pnl', 0):+.2f}")
        print(f"  short fund PnL:  ${stats.get('short_funding_pnl', 0):+.2f}")
        print(f"  fees:            ${stats.get('fee_pnl', 0):+.2f}")

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.out_tag}" if args.out_tag else ""
        out_path = OUT_DIR / f"funding_carry_{ts}{tag}.json"
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "days": args.days,
            "params": asdict(p),
            "universe": universe,
            "stats": stats,
            "weekly": [asdict(r) for r in results],
        }, indent=2, default=str))
        print(f"\nwrote {out_path}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
