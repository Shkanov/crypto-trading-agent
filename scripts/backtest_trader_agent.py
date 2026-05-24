"""Replay historical klines through a real spawned TraderAgent.

Every wake event is a real Opus 4.7 call. Cost scales linearly with
--max-wakes. Tools that lack historical fidelity (orderbook depth,
news sentiment, liquidations) return explicit "not available in
backtest mode" notices so the agent doesn't rely on fabricated data.

Usage (smoke test — recommended for first run):
    BINANCE_TESTNET=false python -m scripts.backtest_trader_agent \\
        --symbol BTCUSDT --tf 5m --htf 1h --bars 200 --max-wakes 10

Requires ANTHROPIC_API_KEY in env (or .env). Without it, prepare()
raises before any spend happens. The trader-agent's TokenBudget
(llm_trader_daily_budget_usd, default $5/day) also gates further calls
once exhausted — replay continues so open trades resolve, but no new
agent invocations.
"""
from __future__ import annotations

import argparse
import asyncio
import json

from dotenv import load_dotenv

from src.services.backtest_trader import TraderBacktestHarness


def _fmt_stats(r) -> str:
    s = r.stats
    return (
        f"trades={s.trades}  win_rate={s.win_rate*100:.1f}%  "
        f"pnl=${s.total_pnl_usd:+.2f}  "
        f"sharpe={s.sharpe:+.2f}  dSh={s.deflated_sharpe:+.2f}  "
        f"mDD%={s.max_drawdown_pct:.1f}  ann%={s.annualized_pct:+.1f}"
    )


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--htf", default="1h")
    ap.add_argument("--bars", type=int, default=200,
                     help="closed bars on the trigger TF to replay")
    ap.add_argument("--max-wakes", type=int, default=10,
                     help="hard cap on real agent invocations (cost guard)")
    ap.add_argument("--ledger-json",
                     help="write the full SimTrade ledger to this path as JSON")
    args = ap.parse_args()

    h = TraderBacktestHarness(
        symbol=args.symbol, tf=args.tf, htf=args.htf,
        bars=args.bars, max_wakes=args.max_wakes,
    )
    try:
        await h.prepare()
        result = await h.run()
    finally:
        await h.close()

    print("\n== TraderAgent Backtest ==")
    print(f"symbol={args.symbol}  tf={args.tf}  htf={args.htf}  "
          f"bars={args.bars}  max_wakes={args.max_wakes}")
    print(_fmt_stats(result))
    print(f"wakes_invoked={result.wakes_invoked}  "
          f"tool_calls={result.total_tool_calls}  "
          f"usd_spent=${result.total_usd_spent:.4f}")
    print("wake_counts:")
    for k, v in sorted(result.wake_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:>20s}: {v}")

    if args.ledger_json:
        ledger = [
            {
                "symbol": t.symbol, "side": t.side, "qty": t.qty,
                "entry": t.entry_price, "stop": t.stop, "tp": t.tp,
                "exit": t.exit_price, "reason": t.exit_reason,
                "pnl_usd": t.pnl_usd,
                "entry_ts_ms": t.entry_ts_ms, "exit_ts_ms": t.exit_ts_ms,
            } for t in result.closed_trades
        ]
        with open(args.ledger_json, "w") as f:
            json.dump(ledger, f, indent=2, default=str)
        print(f"\nledger -> {args.ledger_json}")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
