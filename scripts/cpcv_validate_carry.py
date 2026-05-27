"""CPCV + PBO validation for the cross-sectional funding-carry overlay.

After PIT-correcting the 1y carry backtest (commit e167c6b + this branch's
PIT wire-up), the single-config headline result was Sharpe 0.81 / DSR 0
on top-30, top_n=3, 25%/side — close to Fan et al.'s reported 0.74 but
failing the multiple-testing gate. This script asks the harder question:

  When we sweep a parameter grid over (top_n × book_pct_per_side),
  does the IS-best config systematically beat OOS, or does PBO say the
  selection is no better than random?

Approach:
  1. Fetch funding + 8h-close histories once for the top-N USDT perp
     universe (PIT-filtered at runtime).
  2. For each config in the grid, run `simulate_carry` → weekly PnL list.
  3. Stack into a (T_weeks × N_configs) matrix.
  4. Compute per-config IS Sharpe, CPCV(N=10, k=2) OOS Sharpe distribution
     (45 holdouts per config), and PBO across all C(s, s/2) partitions.

PBO uses S=8 by default (C(8,4)=70 partitions, ~6.5 weeks per subsample)
because T_weeks ≈ 52 — S=16 would push subsample size to ~3 weeks, where
per-subsample Sharpe is too noisy to be meaningful. The pbo() function
itself is unbiased at any S, but small-sample variance is high.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_carry
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_carry \\
      --days 365 --top-n-universe 30 --s-subsamples 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from scripts.backtest_funding_carry import (
    SymbolHistory,
    build_universe,
    fetch_funding_history,
    fetch_perp_closes_8h,
    simulate_carry,
)
from src.scanners.universe_pit import SymbolListing, load_pit_log
from src.services.cpcv import (
    cpcv_oos_sharpes,
    pbo,
    sharpe_per_column,
)
from src.services.costs import Costs
from src.strategies.funding_carry import CarryParams
from src.tools.binance_client import BinanceClient


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data/research/strategy_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Sweep grid

@dataclass(frozen=True)
class CarrySweepConfig:
    top_n: int
    book_pct_per_side: float

    @property
    def label(self) -> str:
        return f"tn{self.top_n}_bk{int(self.book_pct_per_side*100):02d}"

    def to_params(self) -> CarryParams:
        return CarryParams(top_n=self.top_n,
                           book_pct_per_side=self.book_pct_per_side)


def carry_grid() -> list[CarrySweepConfig]:
    """Default 3×4 = 12 configurations spanning the plausible range.
    top_n ∈ {2, 3, 4}: thinner / canonical / wider basket.
    book_pct_per_side ∈ {0.10, 0.20, 0.25, 0.30}: 10% / 20% / 25% / 30% per leg.
    """
    return [
        CarrySweepConfig(top_n=tn, book_pct_per_side=bk)
        for tn, bk in product([2, 3, 4], [0.10, 0.20, 0.25, 0.30])
    ]


# ---------------------------------------------------------------------------
# Per-config evaluation

@dataclass
class ConfigResult:
    label: str
    config: dict
    weeks: int
    total_pnl_usd: float
    in_sample_sharpe: float
    cpcv_oos_sharpe_mean: float = 0.0
    cpcv_oos_sharpe_std: float = 0.0
    cpcv_oos_sharpes: list[float] = field(default_factory=list)


def _decision(pbo_val: float) -> str:
    return "PASS" if pbo_val < 0.5 else "REJECT"


# ---------------------------------------------------------------------------
# CLI

async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--top-n-universe", type=int, default=30)
    ap.add_argument("--rebalance-hours", type=int, default=168)
    ap.add_argument("--equity-usd", type=float, default=1_000.0)
    ap.add_argument("--pit-log",
                    default="data/research/universe/binance_delistings.json")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--k-test", type=int, default=2)
    ap.add_argument("--s-subsamples", type=int, default=8,
                    help="PBO subsample count. S=8 with T≈52 → 6.5 wk/subsample. "
                         "S=16 would push to 3 wk/sub — too noisy on weekly data.")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    # Load PIT log
    pit_log: Optional[dict[str, SymbolListing]] = None
    if args.pit_log:
        pit_path = Path(args.pit_log)
        if not pit_path.is_absolute():
            pit_path = REPO / pit_path
        pit_log = load_pit_log(pit_path)
        if not pit_log:
            print(f"WARNING: --pit-log={pit_path} empty; running WITHOUT survivorship correction.")
            pit_log = None
        else:
            print(f"PIT log: {len(pit_log)} symbols loaded")

    grid = carry_grid()
    print(f"sweep: {len(grid)} configs · CPCV(N={args.n_folds}, k={args.k_test}) "
          f"· PBO S={args.s_subsamples}")

    costs = Costs()
    b = BinanceClient()
    await b.start()
    try:
        # Build universe + fetch histories ONCE (shared across all configs).
        universe = await build_universe(b, top_n_universe=args.top_n_universe)
        print(f"universe ({len(universe)} symbols): "
              f"{', '.join(universe[:8])}{'...' if len(universe) > 8 else ''}")

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - args.days * 86_400_000
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
                      f"funding={len(funding)} closes={len(closes)}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

        # Run each config against the shared histories.
        per_config_pnls: list[np.ndarray] = []
        results: list[ConfigResult] = []
        max_weeks = 0
        for cfg in grid:
            weekly = simulate_carry(
                histories, start_ms=start_ms, end_ms=now_ms,
                p=cfg.to_params(), start_equity=args.equity_usd,
                costs=costs, pit_log=pit_log,
            )
            pnls = np.array([w.total_pnl_usd for w in weekly], dtype=float)
            per_config_pnls.append(pnls)
            max_weeks = max(max_weeks, len(pnls))
            print(f"   {cfg.label}  weeks={len(pnls)}  pnl=${pnls.sum():+.2f}",
                  flush=True)

        # Pad to common length (rebalances aligned by ts, so all should match;
        # take the minimum to be safe against off-by-ones).
        common_len = min(len(p) for p in per_config_pnls)
        matrix = np.column_stack([p[:common_len] for p in per_config_pnls])
        print(f"\nmatrix shape: {matrix.shape}  (weeks × configs)")

        # Per-config IS Sharpe (weekly periods → ~52 periods/year).
        sharpes = sharpe_per_column(matrix, periods_per_year=52.0)
        for i, cfg in enumerate(grid):
            results.append(ConfigResult(
                label=cfg.label, config=asdict(cfg),
                weeks=common_len,
                total_pnl_usd=float(matrix[:, i].sum()),
                in_sample_sharpe=float(sharpes[i]),
            ))

        # CPCV OOS distribution per config.
        for idx, r in enumerate(results):
            oos = cpcv_oos_sharpes(matrix[:, idx], n_folds=args.n_folds,
                                    k=args.k_test, periods_per_year=52.0)
            r.cpcv_oos_sharpes = oos
            if oos:
                r.cpcv_oos_sharpe_mean = float(np.mean(oos))
                r.cpcv_oos_sharpe_std = float(np.std(oos, ddof=1)) if len(oos) > 1 else 0.0

        # PBO across the full grid.
        pbo_res = pbo(matrix, s=args.s_subsamples, periods_per_year=52.0)

        # ---- Report ----
        print("\n" + "=" * 100)
        print(f"CPCV + PBO VALIDATION  ·  funding-carry  ·  top-{args.top_n_universe} universe  "
              f"·  {args.days}d  ·  PIT={'on' if pit_log else 'off'}")
        print("=" * 100)
        n_pos = sum(1 for r in results if r.in_sample_sharpe > 0)
        print(f"configs:           {len(results)}")
        print(f"IS-Sharpe > 0:     {n_pos}/{len(results)}")
        best = max(results, key=lambda r: r.in_sample_sharpe)
        worst = min(results, key=lambda r: r.in_sample_sharpe)
        print(f"best IS:           {best.label}   SR={best.in_sample_sharpe:+.3f}  "
              f"pnl=${best.total_pnl_usd:+.2f}")
        print(f"worst IS:          {worst.label}  SR={worst.in_sample_sharpe:+.3f}  "
              f"pnl=${worst.total_pnl_usd:+.2f}")

        if best.cpcv_oos_sharpes:
            print(f"\nCPCV(N={args.n_folds}, k={args.k_test}) on IS-best — "
                  f"{len(best.cpcv_oos_sharpes)} OOS Sharpe samples:")
            print(f"  mean:        {best.cpcv_oos_sharpe_mean:+.3f}")
            print(f"  std:         {best.cpcv_oos_sharpe_std:.3f}")
            print(f"  IS → OOS:    {best.in_sample_sharpe:+.3f} → "
                  f"{best.cpcv_oos_sharpe_mean:+.3f}  "
                  f"(deflation: {best.in_sample_sharpe - best.cpcv_oos_sharpe_mean:+.3f})")

        print(f"\nPBO (S={args.s_subsamples}, C({args.s_subsamples},{args.s_subsamples // 2})"
              f"={pbo_res.n_partitions} partitions):")
        print(f"  PBO:                  {pbo_res.pbo:.3f}")
        print(f"  mean logit:           {pbo_res.mean_logit:+.3f}")
        print(f"  mean OOS rank pct:    {pbo_res.median_oos_rank_pct:.3f}  "
              "(0.5 = median; higher is better)")
        print(f"  DECISION:             {_decision(pbo_res.pbo)}  (gate: PBO < 0.5)")

        # JSON dump
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.out_tag}" if args.out_tag else ""
        out_path = OUT_DIR / f"cpcv_carry_{ts}{tag}.json"
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "days": args.days,
            "pit_corrected": pit_log is not None,
            "universe": universe,
            "n_folds": args.n_folds,
            "k_test": args.k_test,
            "s_subsamples": args.s_subsamples,
            "is_best_idx": int(np.argmax([r.in_sample_sharpe for r in results])),
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
