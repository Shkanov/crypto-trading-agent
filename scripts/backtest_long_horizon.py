"""Long-horizon (~1y) backtest of indicator / mean-reversion / funding across
a multi-symbol universe (top-N USDT spot by 24h volume, plus aktradescalp's
most-traded symbols when they exist on Binance spot with enough history).

Writes results to data/research/long_horizon/long_horizon_<timestamp>.json
and prints a per-symbol table + universe aggregate.

Cascade joint sim is intentionally excluded: it needs futures-OI history, and
Binance's futures_open_interest_hist only retains ~30 days, so a 1y joint sim
is not physically available from the API.

Usage:
  .venv/bin/python -m scripts.backtest_long_horizon
  .venv/bin/python -m scripts.backtest_long_horizon --bars 35000 --funding-days 365
  .venv/bin/python -m scripts.backtest_long_horizon --symbols BTCUSDT,ETHUSDT,SOLUSDT
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.services.backtest import (
    FundingBacktestParams,
    backtest_funding,
    backtest_indicator,
    backtest_mean_reversion,
)
from src.tools.binance_client import BinanceClient


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data/research/long_horizon"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CALLS_PATH = REPO / "data/research/aktradescalp/aktradescalp_calls.json"

STABLE_SUFFIX_TOKENS = ("FDUSD", "USDC", "EUR", "USD1", "TUSD", "BUSD", "DAI", "PYUSD")


async def build_universe(b: BinanceClient, top_n: int = 20) -> list[str]:
    """Top-N spot USDT pairs by 24h quote volume, stables filtered out,
    augmented with aktradescalp's most-called symbols that exist on spot."""
    info = await b.client.get_ticker()
    spot_universe = [r["symbol"] for r in (await b.client.get_exchange_info())["symbols"]
                     if r.get("status") == "TRADING" and r.get("quoteAsset") == "USDT"]
    spot_set = set(spot_universe)

    usdt = [
        r for r in info
        if r["symbol"].endswith("USDT")
        and r["symbol"] in spot_set
        and not any(tok in r["symbol"][:-4] for tok in STABLE_SUFFIX_TOKENS)
        and not any(suf in r["symbol"] for suf in ("UPUSDT", "DOWNUSDT", "BEARUSDT", "BULLUSDT"))
    ]
    usdt.sort(key=lambda r: float(r["quoteVolume"]), reverse=True)
    top = [r["symbol"] for r in usdt[:top_n]]

    # Add aktradescalp's symbols with >=2 calls that exist on Binance spot
    if CALLS_PATH.exists():
        calls = json.loads(CALLS_PATH.read_text())
        sym_counts = Counter(c["symbol"] for c in calls
                             if c.get("symbol") and c.get("side") in ("long", "short"))
        for sym, cnt in sym_counts.most_common():
            if cnt < 2:
                break
            if sym in spot_set and sym not in top:
                top.append(sym)

    return top


async def run_for_symbol(
    b: BinanceClient, symbol: str,
    bars: int, funding_days: int,
) -> dict:
    """Run all three backtests for one symbol. Returns a result row."""
    row: dict = {"symbol": symbol}

    # Indicator confluence (15m / 1h)
    try:
        t0 = time.time()
        stats, _trades = await backtest_indicator(
            b, symbol=symbol, tf="15m", htf="1h", bars=bars,
        )
        row["indicator"] = {**asdict(stats), "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:
        row["indicator"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}

    # Mean reversion (15m / 1h)
    try:
        t0 = time.time()
        stats, _trades = await backtest_mean_reversion(
            b, symbol=symbol, tf="15m", htf="1h", bars=bars,
        )
        row["meanrev"] = {**asdict(stats), "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:
        row["meanrev"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}

    # Funding harvest (perp)
    try:
        t0 = time.time()
        stats, _trades = await backtest_funding(
            b, symbol=symbol, days=funding_days, params=FundingBacktestParams(),
        )
        row["funding"] = {**asdict(stats), "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:
        row["funding"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}

    return row


def _short(stats: dict) -> str:
    if "error" in stats:
        return f"  ERROR: {stats['error']}"
    return (f"  trades={stats['trades']:4d}  pnl=${stats['total_pnl_usd']:+10.2f}  "
            f"wr={stats['win_rate']*100:5.1f}%  sharpe={stats['sharpe']:+5.2f}  "
            f"defl={stats['deflated_sharpe']:+5.2f}  dd%={stats['max_drawdown_pct']:5.2f}  "
            f"ann={stats['annualized_pct']:+6.1f}%  ({stats['elapsed_s']}s)")


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("LONG-HORIZON BACKTEST SUMMARY")
    print("=" * 100)
    for r in results:
        print(f"\n{r['symbol']}")
        for strat_name in ("indicator", "meanrev", "funding"):
            s = r.get(strat_name, {})
            print(f" {strat_name:10s}{_short(s)}")

    print("\n" + "-" * 100)
    print("UNIVERSE AGGREGATE")
    print("-" * 100)
    for strat_name in ("indicator", "meanrev", "funding"):
        ok = [r[strat_name] for r in results
              if strat_name in r and "error" not in r[strat_name]]
        if not ok:
            print(f"{strat_name:10s} no successful runs")
            continue
        n_sym = len(ok)
        total_pnl = sum(s["total_pnl_usd"] for s in ok)
        n_pos = sum(1 for s in ok if s["total_pnl_usd"] > 0)
        n_trades = sum(s["trades"] for s in ok)
        avg_sharpe = sum(s["sharpe"] for s in ok) / n_sym
        avg_dsharpe = sum(s["deflated_sharpe"] for s in ok) / n_sym
        print(f"{strat_name:10s}  symbols={n_sym}  +pnl={n_pos}/{n_sym}  "
              f"total_pnl=${total_pnl:+.2f}  trades={n_trades}  "
              f"avg_sharpe={avg_sharpe:+.2f}  avg_defl={avg_dsharpe:+.2f}")


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None,
                    help="Comma-separated symbol list; overrides --top-n")
    ap.add_argument("--top-n", type=int, default=20,
                    help="Top-N USDT spot pairs by 24h volume (default 20)")
    ap.add_argument("--bars", type=int, default=35_000,
                    help="Number of 15m bars (~1y = 35040)")
    ap.add_argument("--funding-days", type=int, default=365)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    b = BinanceClient()
    await b.start()
    try:
        if args.symbols:
            universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        else:
            universe = await build_universe(b, top_n=args.top_n)

        print(f"universe ({len(universe)} symbols):")
        for s in universe:
            print(f"  {s}")
        print(f"\nbars={args.bars} (~{args.bars*15/60/24:.0f} days at 15m)")
        print(f"funding_days={args.funding_days}")
        print()

        results: list[dict] = []
        total_t0 = time.time()
        for i, sym in enumerate(universe, 1):
            t0 = time.time()
            print(f"[{i}/{len(universe)}] {sym} ...", flush=True)
            row = await run_for_symbol(
                b, sym, bars=args.bars, funding_days=args.funding_days,
            )
            results.append(row)
            for strat_name in ("indicator", "meanrev", "funding"):
                s = row.get(strat_name, {})
                print(f"   {strat_name:10s}{_short(s)}")
            print(f"   total: {time.time() - t0:.1f}s "
                  f"(elapsed: {time.time() - total_t0:.0f}s)")

        print_summary(results)

        # Write JSON
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.out_tag}" if args.out_tag else ""
        out_path = OUT_DIR / f"long_horizon_{ts}{tag}.json"
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bars": args.bars,
            "funding_days": args.funding_days,
            "universe": universe,
            "results": results,
        }, indent=2))
        print(f"\nwrote {out_path}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
