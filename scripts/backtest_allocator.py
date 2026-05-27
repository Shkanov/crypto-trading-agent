"""Compare {equal, inverse_vol, hrp} multi-strategy allocator methods on the
validated 4-strategy basket.

Strategies (basket of CPCV-validated configs from the strategy-tuning sprint):

  1. **carry** — cross-sectional funding carry, weekly rebalance. Source:
     existing `funding_carry_*_pit.json` weekly P&L, spread evenly across
     the 7 days of each rebalance window.
  2. **pairs_ETHBTC** — cointegrated pairs trade on ETH/BTC. Source:
     existing `pairs_*.json` trade list, bucketed by exit_ts_ms to UTC days.
  3. **meanrev_FET** and **meanrev_SOL** — separate slots so the allocator
     sees 4 strategies (better HRP signal than 3). Re-runs the winning
     `rsi_oversold=25/30, atr_stop_mult=1.5, baseline` configs.

For each allocator method, the harness walks the daily return matrix one
day at a time. At every `rebalance_days` boundary it rebalances via the
production `allocate()` (same code path as `Orchestrator._rebalance_allocator`),
honouring the `lookback_days` window and turnover threshold. Combined daily
return = Σ w_i * r_i with the current weights. The output is a side-by-side
table of portfolio Sharpe / max-DD / total-return per method.

Note: this is a **single-realisation comparison** on one historical window.
For statistical significance against this sprint's CPCV+PBO bar, treat each
method's Sharpe difference with caution unless the gap is ≥0.5 SR.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.backtest_allocator
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from src.config.settings import get_settings
from src.services.backtest import backtest_mean_reversion
from src.services.portfolio import allocate
from src.strategies.mean_reversion import MeanReversionConfig
from src.tools.binance_client import BinanceClient

log = structlog.get_logger(__name__)

DAY_MS = 86_400_000


def _utc_midnight(ts_ms: int) -> int:
    return (int(ts_ms) // DAY_MS) * DAY_MS


# ---------------------------------------------------------------------------
# Per-strategy daily-PnL builders

def carry_weekly_to_daily(weekly: list[dict]) -> dict[int, float]:
    """Spread each weekly bar's total_pnl_usd evenly across the 7 days
    following its `rebalance_ts_ms`. Reasonable proxy for daily attribution
    when the underlying strategy holds positions for the whole week."""
    out: dict[int, float] = {}
    for w in weekly:
        start = _utc_midnight(int(w["rebalance_ts_ms"]))
        per_day = float(w["total_pnl_usd"]) / 7.0
        for i in range(7):
            d = start + i * DAY_MS
            out[d] = out.get(d, 0.0) + per_day
    return out


def trade_exits_to_daily(trades: list[dict], pnl_field: str = "pnl_usd",
                          exit_field: str = "exit_ts_ms") -> dict[int, float]:
    """Bucket trade exits by UTC day. Open trades (no exit_ts_ms) excluded."""
    out: dict[int, float] = {}
    for t in trades:
        ts = t.get(exit_field) if isinstance(t, dict) else getattr(t, exit_field, None)
        pnl = t.get(pnl_field) if isinstance(t, dict) else getattr(t, pnl_field, None)
        if ts is None or pnl is None:
            continue
        d = _utc_midnight(int(ts))
        out[d] = out.get(d, 0.0) + float(pnl)
    return out


async def run_meanrev_strategy(
    binance: BinanceClient, symbol: str, bars: int, tf: str, htf: str,
    rsi_oversold: float, atr_stop_mult: float,
) -> list:
    """Run the baseline mean-rev backtest and return its SimTrade list."""
    cfg = MeanReversionConfig(
        allowed_symbols=[symbol],
        rsi_oversold=rsi_oversold,
        atr_stop_mult=atr_stop_mult,
        htf_timeframe=htf,
        use_strict_regime_gate=False,
        use_triple_barrier=False,
    )
    _stats, trades = await backtest_mean_reversion(
        binance, symbol, tf=tf, htf=htf, bars=bars, cfg=cfg,
    )
    return trades


# ---------------------------------------------------------------------------
# Allocator simulation

def build_returns_matrix(
    strategies: dict[str, dict[int, float]], day_grid: list[int],
    reference_equity_usd: float,
) -> dict[str, np.ndarray]:
    """Per-strategy daily-return-% array, aligned to a shared day grid."""
    denom = reference_equity_usd if reference_equity_usd > 0 else 1.0
    return {
        name: np.array(
            [day_pnl.get(d, 0.0) / denom * 100.0 for d in day_grid],
            dtype=float,
        )
        for name, day_pnl in strategies.items()
    }


def simulate_allocator_walk(
    returns_matrix: dict[str, np.ndarray],
    method: str,
    fallback: str,
    rebalance_days: int,
    lookback_days: int,
    turnover_threshold: float,
) -> tuple[np.ndarray, list[dict], dict[str, float]]:
    """Day-by-day walk: rebalance at every `rebalance_days` boundary using the
    trailing `lookback_days` window, compute daily portfolio return as
    Σ w_i * r_i. Returns (portfolio_returns_pct, rebalance_log, final_weights).

    First period uses equal-weight while history accumulates."""
    syms = list(returns_matrix.keys())
    n_days = len(next(iter(returns_matrix.values())))
    weights = {s: 1.0 / len(syms) for s in syms}
    rebalance_log: list[dict] = []
    portfolio_returns = np.zeros(n_days, dtype=float)

    for day_idx in range(n_days):
        # Rebalance at day boundaries past the first.
        if day_idx > 0 and day_idx % rebalance_days == 0:
            window_start = max(0, day_idx - lookback_days)
            window = {s: returns_matrix[s][window_start:day_idx] for s in syms}
            result = allocate(
                window, method=method, fallback=fallback,
                turnover_threshold=turnover_threshold,
                prev_weights=weights,
            )
            weights = dict(result.weights)
            rebalance_log.append({
                "day_idx": day_idx,
                "method_used": result.method_used,
                "weights": weights,
                "turnover": result.turnover,
                "reason": result.reason,
            })
        # Combined return for this day.
        portfolio_returns[day_idx] = sum(
            weights.get(s, 0.0) * returns_matrix[s][day_idx] for s in syms
        )
    return portfolio_returns, rebalance_log, weights


def portfolio_metrics(returns_pct: np.ndarray) -> dict:
    """Annualised Sharpe + max drawdown + total-return on daily-return-% input."""
    n = len(returns_pct)
    mean_r = float(returns_pct.mean()) if n else 0.0
    std_r = float(returns_pct.std(ddof=1)) if n > 1 else 0.0
    sharpe_d = mean_r / std_r if std_r > 0 else 0.0
    sharpe_annual = sharpe_d * math.sqrt(365.0)
    # Equity curve from log-additive compounding of pct returns
    eq = np.cumprod(1.0 + returns_pct / 100.0)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
    max_dd_pct = float(dd.max() * 100.0) if n else 0.0
    total_return_pct = float((eq[-1] - 1.0) * 100.0) if n else 0.0
    annualised = (
        float((eq[-1] ** (365.0 / n) - 1.0) * 100.0) if n and eq[-1] > 0 else 0.0
    )
    calmar = (annualised / max_dd_pct) if max_dd_pct > 0 else 0.0
    return {
        "n_days": n,
        "mean_daily_pct": mean_r,
        "std_daily_pct": std_r,
        "sharpe_annual": sharpe_annual,
        "max_drawdown_pct": max_dd_pct,
        "total_return_pct": total_return_pct,
        "annualized_pct": annualised,
        "calmar": calmar,
    }


# ---------------------------------------------------------------------------
# Driver

async def amain() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--carry-json",
        default="data/research/strategy_tuning/funding_carry_20260527_124532_pit.json",
        help="PIT-corrected funding_carry weekly JSON",
    )
    ap.add_argument(
        "--pairs-json",
        default="data/research/strategy_tuning/pairs_20260527_105505.json",
        help="Pairs backtest JSON (we use the ETH/BTC pair).",
    )
    ap.add_argument("--meanrev-fet", default="FETUSDT")
    ap.add_argument("--meanrev-sol", default="SOLUSDT")
    ap.add_argument("--bars", type=int, default=8760,
                     help="1h bars for mean-rev backtests (~1y).")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--htf", default="4h")
    ap.add_argument("--lookback-days", type=int, default=90)
    ap.add_argument("--rebalance-days", type=int, default=30)
    ap.add_argument("--turnover-threshold", type=float, default=0.5)
    ap.add_argument("--reference-equity", type=float, default=10_000.0)
    ap.add_argument("--out-dir", default="data/research/strategy_tuning")
    args = ap.parse_args()

    # ----- Carry -----
    carry_raw = json.loads(Path(args.carry_json).read_text())
    carry_daily = carry_weekly_to_daily(carry_raw["weekly"])
    print(f"carry: {len(carry_raw['weekly'])} weekly bars, "
          f"{len(carry_daily)} unique days, "
          f"total PnL ${sum(carry_daily.values()):+.2f}")

    # ----- Pairs ETH/BTC -----
    pairs_raw = json.loads(Path(args.pairs_json).read_text())
    ethbtc = next((r for r in pairs_raw["results"] if r["pair"] == "ETHUSDT/BTCUSDT"), None)
    if ethbtc is None:
        raise SystemExit("No ETHUSDT/BTCUSDT in pairs JSON")
    pairs_daily = trade_exits_to_daily(ethbtc["trades"])
    print(f"pairs_ETHBTC: {len(ethbtc['trades'])} trades, "
          f"{len(pairs_daily)} unique exit days, "
          f"total PnL ${sum(pairs_daily.values()):+.2f}")

    # ----- Mean-rev FET & SOL (baseline mode) -----
    b = BinanceClient()
    await b.start()
    try:
        # rsi25 for FET, rsi30 for SOL — IS-best from cpcv_meanrev sweep.
        print(f"meanrev_FET ({args.meanrev_fet}) — re-running baseline ...")
        fet_trades = await run_meanrev_strategy(
            b, args.meanrev_fet, args.bars, args.tf, args.htf,
            rsi_oversold=25.0, atr_stop_mult=1.5,
        )
        fet_daily = trade_exits_to_daily(
            [{"pnl_usd": t.pnl_usd, "exit_ts_ms": t.exit_ts_ms} for t in fet_trades],
        )
        print(f"  {len(fet_trades)} trades, "
              f"{len(fet_daily)} unique exit days, "
              f"total PnL ${sum(fet_daily.values()):+.2f}")
        print(f"meanrev_SOL ({args.meanrev_sol}) — re-running baseline ...")
        sol_trades = await run_meanrev_strategy(
            b, args.meanrev_sol, args.bars, args.tf, args.htf,
            rsi_oversold=30.0, atr_stop_mult=1.5,
        )
        sol_daily = trade_exits_to_daily(
            [{"pnl_usd": t.pnl_usd, "exit_ts_ms": t.exit_ts_ms} for t in sol_trades],
        )
        print(f"  {len(sol_trades)} trades, "
              f"{len(sol_daily)} unique exit days, "
              f"total PnL ${sum(sol_daily.values()):+.2f}")
    finally:
        await b.close()

    # ----- Align to common day grid -----
    strategies = {
        "carry": carry_daily,
        "pairs_ETHBTC": pairs_daily,
        "meanrev_FET": fet_daily,
        "meanrev_SOL": sol_daily,
    }
    all_days = set()
    for d in strategies.values():
        all_days.update(d.keys())
    if not all_days:
        raise SystemExit("No day coverage in any strategy.")
    day_grid = sorted(all_days)
    print(f"\ncombined window: {len(day_grid)} days from "
          f"{day_grid[0]} to {day_grid[-1]}")

    returns_matrix = build_returns_matrix(
        strategies, day_grid, args.reference_equity,
    )

    # ----- Run 3 allocator methods -----
    fallback = "inverse_vol"
    methods = ["equal", "inverse_vol", "hrp"]
    results: dict[str, dict] = {}
    for method in methods:
        portfolio_returns, rebalance_log, final_weights = simulate_allocator_walk(
            returns_matrix, method=method, fallback=fallback,
            rebalance_days=args.rebalance_days,
            lookback_days=args.lookback_days,
            turnover_threshold=args.turnover_threshold,
        )
        metrics = portfolio_metrics(portfolio_returns)
        # Average L1 turnover across rebalances (excludes any fallback)
        turnovers = [r["turnover"] for r in rebalance_log if r.get("turnover") is not None]
        metrics["avg_turnover"] = float(np.mean(turnovers)) if turnovers else 0.0
        metrics["n_rebalances"] = len(rebalance_log)
        metrics["final_weights"] = final_weights
        # How many rebalances took the fallback path (HRP only)
        metrics["fallbacks"] = sum(
            1 for r in rebalance_log if r.get("method_used") != method
        )
        results[method] = metrics
        print(
            f"\n{method:>11}: SR={metrics['sharpe_annual']:+.2f}  "
            f"DD={metrics['max_drawdown_pct']:.1f}%  "
            f"annualised={metrics['annualized_pct']:+.1f}%  "
            f"Calmar={metrics['calmar']:.2f}  "
            f"avg_turnover={metrics['avg_turnover']:.3f}  "
            f"fallbacks={metrics['fallbacks']}/{metrics['n_rebalances']}"
        )

    # ----- Summary + save -----
    print("\n" + "=" * 78)
    print(f"{'method':>11}  {'Sharpe':>7}  {'DD%':>6}  {'Ann%':>7}  "
          f"{'Calmar':>7}  {'TO':>5}")
    print("=" * 78)
    for method in methods:
        m = results[method]
        print(
            f"{method:>11}  {m['sharpe_annual']:>+7.2f}  "
            f"{m['max_drawdown_pct']:>6.1f}  {m['annualized_pct']:>+7.1f}  "
            f"{m['calmar']:>7.2f}  {m['avg_turnover']:>5.3f}"
        )
    print("=" * 78)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"allocator_compare_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "generated_at": int(time.time() * 1000),
        "args": {
            "carry_json": args.carry_json,
            "pairs_json": args.pairs_json,
            "meanrev_fet": args.meanrev_fet,
            "meanrev_sol": args.meanrev_sol,
            "bars": args.bars,
            "lookback_days": args.lookback_days,
            "rebalance_days": args.rebalance_days,
            "turnover_threshold": args.turnover_threshold,
            "reference_equity_usd": args.reference_equity,
        },
        "n_days": len(day_grid),
        "first_day_ms": day_grid[0],
        "last_day_ms": day_grid[-1],
        "strategies": {
            name: {
                "n_days_with_pnl": sum(1 for v in series.values() if v != 0),
                "total_pnl_usd": float(sum(series.values())),
            }
            for name, series in strategies.items()
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nresult JSON: {out_path}")


if __name__ == "__main__":
    asyncio.run(amain())
