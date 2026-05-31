"""Combinatorial Purged CV + PBO validation driver (sprint item #10).

Sweeps a parameter grid for a strategy, runs the full 1y backtest per
config, and produces two complementary statistics:

  1. **CPCV(N=10, k=2)** — for the IS-best config (and a sample of others)
     we compute Sharpe on each of the 45 test-fold combinations. The result
     is a distribution of OOS Sharpes per config, not a single number — and
     its variance is what tells us whether the headline Sharpe is real.

  2. **PBO (Bailey-Borwein-LdP-Zhu 2017)** — across the full grid, partition
     the daily-bucketed returns matrix (T=days × N=configs) into S=16
     subsamples, iterate over all C(16,8)=12,870 in-sample/out-of-sample
     splits, and measure how often the IS-best config lands in the bottom
     half OOS. PBO > 0.5 = selection bias worse than random → reject the
     family. PBO < 0.5 is the gate the recommendations doc requires.

Currently supports `--strategy funding`. The funding harvest is the only
strategy with proven positive in-sample edge, so it's the right place to
ask "does that edge survive PBO?". Other strategies (indicator, meanrev)
can be wired in by adding a sweep generator + a per-config backtest call.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate \\
      --strategy funding --symbol API3USDT --days 365
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate \\
      --strategy funding --symbol KATUSDT --days 365 --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from src.services.backtest import (
    FundingBacktestParams,
    SimTrade,
    backtest_funding,
)
from src.services.cpcv import (
    cpcv_oos_sharpes,
    daily_bucket_pnls,
    pbo,
    select_is_best_idx,
    sharpe_per_column,
)
from src.tools.binance_client import BinanceClient


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data/research/strategy_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Param grids per strategy

@dataclass
class FundingConfig:
    """One parameter configuration in the funding sweep."""
    z_entry_sigma: float
    z_exit_sigma: float
    z_window_cycles: int

    @property
    def label(self) -> str:
        return (f"ze{self.z_entry_sigma:.1f}_zx{self.z_exit_sigma:.1f}_"
                f"w{self.z_window_cycles}")

    def to_params(self) -> FundingBacktestParams:
        # use_rolling_z=True activates the z-scored entry/exit logic; the
        # base thresholds (entry_threshold_bps etc.) are ignored in that mode.
        return FundingBacktestParams(
            use_rolling_z=True,
            z_entry_sigma=self.z_entry_sigma,
            z_exit_sigma=self.z_exit_sigma,
            z_stop_sigma=2.0,
            z_window_cycles=self.z_window_cycles,
            z_min_window=60,
        )


def funding_grid() -> list[FundingConfig]:
    """3×3×3 = 27 configurations covering the realistic range for funding
    z-thresholds. z_entry ∈ {1.0, 1.5, 2.0} (decile to 2-sigma), z_exit ∈
    {-0.3, 0.3, 0.5} (slightly-negative through median band), z_window ∈
    {120, 180, 240} cycles (40d, 60d, 80d at 8h cadence)."""
    z_entries = [1.0, 1.5, 2.0]
    z_exits = [-0.3, 0.3, 0.5]
    windows = [120, 180, 240]
    return [
        FundingConfig(ze, zx, w)
        for ze, zx, w in product(z_entries, z_exits, windows)
    ]


# ---------------------------------------------------------------------------
# Per-config evaluation

@dataclass
class ConfigResult:
    label: str
    config: dict
    trades: int
    total_pnl_usd: float
    in_sample_sharpe: float
    cpcv_oos_sharpe_mean: float = 0.0
    cpcv_oos_sharpe_std: float = 0.0
    cpcv_oos_sharpes: list[float] = field(default_factory=list)


async def evaluate_funding_config(
    b: BinanceClient, symbol: str, days: int, cfg: FundingConfig,
    sem: asyncio.Semaphore,
) -> tuple[FundingConfig, list[SimTrade]]:
    async with sem:
        _stats, trades = await backtest_funding(
            b, symbol=symbol, days=days, params=cfg.to_params(),
        )
        return cfg, trades


# ---------------------------------------------------------------------------
# Reporting

def _decision(pbo_val: float) -> str:
    if pbo_val < 0.5:
        return "PASS"
    return "REJECT"


def print_report(
    symbol: str, days: int, configs: list[ConfigResult],
    pbo_result, day0_ms: int,
) -> None:
    print("\n" + "=" * 100)
    print(f"CPCV + PBO VALIDATION REPORT  ·  {symbol}  ·  {days}d")
    print("=" * 100)

    # Header summary
    n_pos = sum(1 for c in configs if c.in_sample_sharpe > 0)
    print(f"configs:           {len(configs)}")
    print(f"day0 (UTC):        {datetime.fromtimestamp(day0_ms/1000, tz=timezone.utc).date()}")
    print(f"IS-Sharpe > 0:     {n_pos}/{len(configs)}")
    best = max(configs, key=lambda c: c.in_sample_sharpe)
    worst = min(configs, key=lambda c: c.in_sample_sharpe)
    print(f"best IS:           {best.label}   SR={best.in_sample_sharpe:+.3f}   "
          f"trades={best.trades}  pnl=${best.total_pnl_usd:+.2f}")
    print(f"worst IS:          {worst.label}  SR={worst.in_sample_sharpe:+.3f}  "
          f"trades={worst.trades}  pnl=${worst.total_pnl_usd:+.2f}")

    # CPCV OOS distribution for the IS-best
    if best.cpcv_oos_sharpes:
        print(f"\nCPCV(N=10, k=2) on IS-best config — 45 OOS Sharpe samples:")
        print(f"  mean:        {best.cpcv_oos_sharpe_mean:+.3f}")
        print(f"  std:         {best.cpcv_oos_sharpe_std:.3f}")
        print(f"  IS / OOS:    {best.in_sample_sharpe:+.3f} → "
              f"{best.cpcv_oos_sharpe_mean:+.3f}  "
              f"(deflation: {best.in_sample_sharpe - best.cpcv_oos_sharpe_mean:+.3f})")

    # PBO verdict
    print(f"\nPBO (S=16, C(16,8)={pbo_result.n_partitions} partitions):")
    print(f"  PBO:                  {pbo_result.pbo:.3f}")
    print(f"  mean logit:           {pbo_result.mean_logit:+.3f}")
    print(f"  mean OOS rank pct:    {pbo_result.median_oos_rank_pct:.3f}  "
          "(0.5 = median; higher is better)")
    print(f"  DECISION:             {_decision(pbo_result.pbo)}  "
          f"(gate: PBO < 0.5)")


# ---------------------------------------------------------------------------
# CLI

async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=("funding",), default="funding",
                    help="Strategy to validate. Only `funding` wired so far.")
    ap.add_argument("--symbol", default="API3USDT",
                    help="Symbol to backtest on (default API3USDT — known +DSR).")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--concurrency", type=int, default=2,
                    help="Concurrent backtests (be gentle on the API).")
    ap.add_argument("--n-folds", type=int, default=10,
                    help="CPCV fold count (default 10)")
    ap.add_argument("--k-test", type=int, default=2,
                    help="CPCV holdout size (default 2 → C(10,2)=45 combos)")
    ap.add_argument("--s-subsamples", type=int, default=16,
                    help="PBO subsample count (default 16 → C(16,8)=12870)")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    if args.strategy != "funding":
        raise SystemExit(f"--strategy {args.strategy} not wired yet")

    grid = funding_grid()
    print(f"strategy={args.strategy}  symbol={args.symbol}  days={args.days}")
    print(f"grid: {len(grid)} configs  ·  CPCV(N={args.n_folds}, k={args.k_test})  "
          f"·  PBO S={args.s_subsamples}")

    b = BinanceClient()
    await b.start()
    try:
        sem = asyncio.Semaphore(args.concurrency)
        t0 = time.time()
        # Run all configs concurrently with bounded parallelism.
        tasks = [
            evaluate_funding_config(b, args.symbol, args.days, cfg, sem)
            for cfg in grid
        ]
        per_cfg: list[tuple[FundingConfig, list[SimTrade]]] = []
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            res = await fut
            per_cfg.append(res)
            cfg, trades = res
            n_pnl = sum(1 for t in trades if t.pnl_usd is not None)
            print(f"   [{i}/{len(tasks)}] {cfg.label}  trades={n_pnl}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

        # Common bucketing window: day0 = (now - days) at UTC midnight.
        now_ms = int(time.time() * 1000)
        day_ms = 86_400_000
        day0_ms = (now_ms - args.days * day_ms) // day_ms * day_ms

        columns: list[np.ndarray] = []
        results: list[ConfigResult] = []
        # Preserve grid order in results (asyncio.as_completed scrambles).
        per_cfg.sort(key=lambda x: grid.index(x[0]))
        for cfg, trades in per_cfg:
            ts = [t.entry_ts_ms for t in trades if t.pnl_usd is not None]
            pnls = [t.pnl_usd for t in trades if t.pnl_usd is not None]
            col = daily_bucket_pnls(ts, pnls, day0_ms=day0_ms, n_days=args.days)
            columns.append(col)
            sr = float(sharpe_per_column(col.reshape(-1, 1))[0])
            results.append(ConfigResult(
                label=cfg.label,
                config=asdict(cfg),
                trades=len(pnls),
                total_pnl_usd=float(sum(pnls)) if pnls else 0.0,
                in_sample_sharpe=sr,
            ))

        matrix = np.column_stack(columns)
        print(f"\nmatrix shape: {matrix.shape}  (days × configs)")

        # CPCV OOS distribution for the IS-best config (and others, optional).
        is_best_idx = select_is_best_idx(
            [r.in_sample_sharpe for r in results], [r.trades for r in results])
        for idx, r in enumerate(results):
            oos = cpcv_oos_sharpes(matrix[:, idx], n_folds=args.n_folds,
                                    k=args.k_test, periods_per_year=365.0)
            r.cpcv_oos_sharpes = oos
            if oos:
                r.cpcv_oos_sharpe_mean = float(np.mean(oos))
                r.cpcv_oos_sharpe_std = float(np.std(oos, ddof=1)) if len(oos) > 1 else 0.0

        # PBO across the full grid.
        pbo_res = pbo(matrix, s=args.s_subsamples, periods_per_year=365.0)

        print_report(args.symbol, args.days, results, pbo_res, day0_ms)

        # JSON dump.
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.out_tag}" if args.out_tag else ""
        out_path = OUT_DIR / f"cpcv_{args.strategy}_{args.symbol}_{ts}{tag}.json"
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategy": args.strategy,
            "symbol": args.symbol,
            "days": args.days,
            "day0_ms": day0_ms,
            "n_folds": args.n_folds,
            "k_test": args.k_test,
            "s_subsamples": args.s_subsamples,
            "is_best_idx": is_best_idx,
            "configs": [asdict(r) for r in results],
            "pbo": {
                "pbo": pbo_res.pbo,
                "n_partitions": pbo_res.n_partitions,
                "n_trials": pbo_res.n_trials,
                "mean_logit": pbo_res.mean_logit,
                "mean_oos_rank_pct": pbo_res.median_oos_rank_pct,
                "decision": _decision(pbo_res.pbo),
            },
        }, indent=2, default=str))
        print(f"\nwrote {out_path}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
