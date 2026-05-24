"""Walk-forward backtest for indicator, mean-reversion, and funding strategies.

Examples:
  python -m scripts.backtest --strategy indicator --symbol BTCUSDT --bars 5000
  python -m scripts.backtest --strategy meanrev   --symbol BTCUSDT --bars 5000
  python -m scripts.backtest --strategy funding   --symbol BTCUSDT --days 60
  python -m scripts.backtest --strategy all       --symbol ETHUSDT --bars 5000

The honest workflow: run all three, on multiple symbols and time windows.
If indicator (trend-follow) shows no positive deflated-Sharpe across 3+
symbols and 2+ windows, it does not have edge. Mean-reversion's expected
domain is low-ADX regimes — backtest specifically in choppy windows. If
funding-harvest shows positive total P&L net of costs over 60+ days, the
regime persists and small live size makes sense.
"""
from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from src.services.backtest import (
    FundingBacktestParams,
    backtest_funding,
    backtest_indicator,
    backtest_mean_reversion,
    format_stats,
)
from src.tools.binance_client import BinanceClient


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategy",
        choices=["indicator", "meanrev", "funding", "both", "all"],
        default="indicator",
    )
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--htf", default="1h")
    ap.add_argument("--bars", type=int, default=5000, help="bars for indicator backtest")
    ap.add_argument("--days", type=int, default=30, help="days for funding backtest")
    ap.add_argument("--funding-notional", type=float, default=100.0)
    ap.add_argument("--funding-entry-bps", type=float, default=10.0)
    ap.add_argument("--funding-avg-entry-bps", type=float, default=5.0)
    ap.add_argument("--funding-exit-bps", type=float, default=2.0)
    args = ap.parse_args()

    b = BinanceClient()
    await b.start()
    try:
        if args.strategy in ("indicator", "both", "all"):
            stats, trades = await backtest_indicator(
                b, symbol=args.symbol, tf=args.tf, htf=args.htf, bars=args.bars,
            )
            print(format_stats(stats))
            print("last 10 indicator trades:")
            for t in trades[-10:]:
                exit_px = f"{t.exit_price:.4f}" if t.exit_price else "?"
                pnl = f"{t.pnl_usd:+.2f}" if t.pnl_usd is not None else "?"
                print(f"  {t.side:5s} {t.entry_price:.4f} -> {exit_px} "
                      f"({t.exit_reason}) pnl=${pnl}")

        if args.strategy in ("meanrev", "all"):
            stats, trades = await backtest_mean_reversion(
                b, symbol=args.symbol, tf=args.tf, htf=args.htf, bars=args.bars,
            )
            print(format_stats(stats))
            print("last 10 mean-rev trades:")
            for t in trades[-10:]:
                exit_px = f"{t.exit_price:.4f}" if t.exit_price else "?"
                pnl = f"{t.pnl_usd:+.2f}" if t.pnl_usd is not None else "?"
                print(f"  {t.side:5s} {t.entry_price:.4f} -> {exit_px} "
                      f"({t.exit_reason}) pnl=${pnl}")

        if args.strategy in ("funding", "both", "all"):
            params = FundingBacktestParams(
                notional_per_pair_usd=args.funding_notional,
                entry_threshold_bps=args.funding_entry_bps,
                entry_avg_threshold_bps=args.funding_avg_entry_bps,
                exit_threshold_bps=args.funding_exit_bps,
            )
            stats, trades = await backtest_funding(
                b, symbol=args.symbol, days=args.days, params=params,
            )
            print(format_stats(stats))
            print("funding trades:")
            for t in trades:
                pnl = f"{t.pnl_usd:+.2f}" if t.pnl_usd is not None else "?"
                print(f"  pair@{t.entry_price:.2f}: pnl=${pnl} reason={t.exit_reason}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
