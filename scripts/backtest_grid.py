"""Run trend + mean-reversion backtests across many symbols, then print
a comparison grid. Funding-harvest is run separately (different cadence).

Usage:
  python -m scripts.backtest_grid --symbols BTCUSDT,ETHUSDT,... --bars 10000 --tf 5m

Reads SPOT klines via the existing BinanceClient. For long-tail symbols
that don't exist on spot, the call will raise and the symbol is skipped
(reported in the final summary). Mainnet recommended (testnet klines are
sparse for alts):

  BINANCE_TESTNET=false python -m scripts.backtest_grid ...
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Optional

from dotenv import load_dotenv

from src.services.backtest import (
    BacktestStats,
    backtest_indicator,
    backtest_mean_reversion,
)
from src.tools.binance_client import BinanceClient


def _row(stats: BacktestStats) -> str:
    return (
        f"{stats.trades:>5d}  "
        f"{stats.win_rate*100:>5.1f}%  "
        f"${stats.total_pnl_usd:>+8.2f}  "
        f"{stats.sharpe:>+6.2f}  "
        f"{stats.deflated_sharpe:>+5.2f}  "
        f"{stats.max_drawdown_pct:>5.1f}%  "
        f"{stats.annualized_pct:>+7.1f}%"
    )


HEADER = (
    f"{'symbol':<12s} {'strategy':<14s} "
    f"{'trades':>5s}  {'win%':>5s}   {'P&L':>8s}  "
    f"{'Sharpe':>6s}  {'dSh':>5s}  {'mDD%':>5s}  {'ann%':>7s}"
)


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--symbols",
        default=("BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,"
                 "APTUSDT,SUIUSDT,ARBUSDT,WIFUSDT,INJUSDT,SEIUSDT"),
        help="comma-separated symbols",
    )
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--htf", default="1h")
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--strategies", default="indicator,meanrev",
                    help="comma list: indicator,meanrev")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    b = BinanceClient()
    await b.start()
    results: dict[tuple[str, str], Optional[BacktestStats]] = {}
    errors: dict[str, str] = {}
    try:
        for sym in symbols:
            for strat in strategies:
                try:
                    if strat == "indicator":
                        stats, _ = await backtest_indicator(
                            b, symbol=sym, tf=args.tf, htf=args.htf, bars=args.bars,
                        )
                    elif strat == "meanrev":
                        stats, _ = await backtest_mean_reversion(
                            b, symbol=sym, tf=args.tf, htf=args.htf, bars=args.bars,
                        )
                    else:
                        continue
                    results[(sym, strat)] = stats
                except Exception as e:
                    errors[f"{sym}/{strat}"] = str(e)
                    results[(sym, strat)] = None
    finally:
        await b.close()

    print()
    print(HEADER)
    print("-" * len(HEADER))
    for sym in symbols:
        for strat in strategies:
            stats = results.get((sym, strat))
            if stats is None:
                err = errors.get(f"{sym}/{strat}", "no data")
                print(f"{sym:<12s} {strat:<14s} ERROR: {err[:80]}")
                continue
            print(f"{sym:<12s} {strat:<14s} {_row(stats)}")

    # Aggregate per-strategy totals
    print()
    print("=== aggregate ===")
    for strat in strategies:
        rows = [s for (sy, st), s in results.items() if st == strat and s is not None]
        if not rows:
            print(f"  {strat}: no successful runs")
            continue
        trades = sum(r.trades for r in rows)
        pnl = sum(r.total_pnl_usd for r in rows)
        wins = sum(r.wins for r in rows)
        win_rate = wins / trades if trades else 0.0
        symbols_pos = sum(1 for r in rows if r.total_pnl_usd > 0)
        print(f"  {strat:<14s}  symbols={len(rows)}  trades={trades}  "
              f"win%={win_rate*100:.1f}  total_pnl=${pnl:+.2f}  "
              f"profitable_symbols={symbols_pos}/{len(rows)}")


if __name__ == "__main__":
    asyncio.run(amain())
