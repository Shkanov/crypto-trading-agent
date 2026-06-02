"""CPCV + PBO validation for carry × momentum double-sort (Card 2).

Thesis: within the standard cross-sectional carry long/short buckets, keep only
names where trailing price momentum *agrees* with the carry direction.
  Long bucket:  top-N by funding level  ∩  positive trailing momentum
  Short bucket: bottom-N by funding level  ∩  negative trailing momentum
Breadth-loss from the momentum filter is intentional — the unmatched slots
leave capital undeployed (cash) rather than forcing low-conviction names.

Motivation: Ethena/BFUSD structural-short inventory compressed carry yields on
pure-yield names (no organic demand). A long with rising funding AND rising price
is more likely to be driven by genuine demand; the momentum filter should purge
the names where the yield is crowded-out. (Cao SSRN 6365329 two-factor model:
log-basis + price-volume explains all 63 significant perp strategies.)

Sweep grid:
  momentum_lookback ∈ {off (plain carry), 7d, 30d}
  × top_n ∈ {3, 5}
  × book_pct ∈ {0.10, 0.25}
  = 12 configs

The `off` configs are plain carry (no momentum filter). Including them means
CPCV computes both within the same run, so the comparison is apples-to-apples.

Pass gate (stricter than Card 1):
  PBO < 0.5  AND  CPCV-OOS-mean > best-`off`-config OOS-mean
The momentum overlay must BEAT plain carry (not just be positive). If it
doesn't lift above carry, the breadth loss is not worth it.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_momentum_carry
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_momentum_carry \\
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
    select_is_best_idx,
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
class MomentumCarrySweepConfig:
    momentum_lookback_hours: Optional[int]   # None = plain carry (off)
    top_n: int
    book_pct_per_side: float

    @property
    def label(self) -> str:
        mom = "off" if self.momentum_lookback_hours is None else f"m{self.momentum_lookback_hours}h"
        return f"{mom}_tn{self.top_n}_bk{int(self.book_pct_per_side*100):02d}"

    @property
    def is_plain_carry(self) -> bool:
        return self.momentum_lookback_hours is None

    def to_params(self) -> CarryParams:
        return CarryParams(top_n=self.top_n,
                           book_pct_per_side=self.book_pct_per_side)


def momentum_carry_grid() -> list[MomentumCarrySweepConfig]:
    """12 configs: momentum_lookback ∈ {None, 7d=168h, 30d=720h}
    × top_n ∈ {3, 5} × book_pct ∈ {0.10, 0.25}."""
    lookbacks = [None, 7 * 24, 30 * 24]
    return [
        MomentumCarrySweepConfig(
            momentum_lookback_hours=lb, top_n=tn, book_pct_per_side=bk,
        )
        for lb, tn, bk in product(lookbacks, [3, 5], [0.10, 0.25])
    ]


# ---------------------------------------------------------------------------
# Per-config result

@dataclass
class ConfigResult:
    label: str
    config: dict
    is_plain_carry: bool
    weeks: int
    total_pnl_usd: float
    price_pnl_usd: float
    funding_pnl_usd: float
    fee_pnl_usd: float
    in_sample_sharpe: float
    cpcv_oos_sharpe_mean: float = 0.0
    cpcv_oos_sharpe_std: float = 0.0
    cpcv_oos_sharpes: list[float] = field(default_factory=list)


def _combined_decision(pbo_val: float, oos_mean: float,
                       carry_baseline_oos: float) -> str:
    """Card 2 gate: PBO < 0.5 AND OOS-mean > carry baseline OOS-mean.
    Must beat plain carry, not just be positive."""
    if pbo_val >= 0.5:
        return "REJECT [PBO ≥ 0.5]"
    if oos_mean <= carry_baseline_oos:
        return f"REJECT [OOS-mean {oos_mean:+.3f} ≤ carry baseline {carry_baseline_oos:+.3f}]"
    return "PASS"


# ---------------------------------------------------------------------------
# CLI

async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description="CPCV+PBO for carry × momentum double-sort (Card 2)."
    )
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--top-n-universe", type=int, default=30)
    ap.add_argument("--equity-usd", type=float, default=1_000.0)
    ap.add_argument("--pit-log",
                    default="data/research/universe/binance_delistings.json")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--k-test", type=int, default=2)
    ap.add_argument("--s-subsamples", type=int, default=8)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    pit_log: Optional[dict[str, SymbolListing]] = None
    if args.pit_log:
        pit_path = Path(args.pit_log)
        if not pit_path.is_absolute():
            pit_path = REPO / pit_path
        pit_log = load_pit_log(pit_path)
        if not pit_log:
            print(f"WARNING: --pit-log={pit_path} empty; running WITHOUT survivorship correction.")
        else:
            print(f"PIT log: {len(pit_log)} symbols loaded")

    grid = momentum_carry_grid()
    plain_carry_labels = [c.label for c in grid if c.is_plain_carry]
    print(f"sweep: {len(grid)} configs  "
          f"({len(plain_carry_labels)} plain-carry baselines + "
          f"{len(grid)-len(plain_carry_labels)} momentum-filtered)")
    print(f"CPCV(N={args.n_folds}, k={args.k_test}) · PBO S={args.s_subsamples}")
    print("Pass gate: PBO < 0.5 AND OOS-mean > best plain-carry OOS-mean")

    costs = Costs()
    b = BinanceClient()
    await b.start()
    try:
        universe = await build_universe(b, top_n_universe=args.top_n_universe)
        print(f"universe ({len(universe)} symbols): "
              f"{', '.join(universe[:8])}{'...' if len(universe) > 8 else ''}")

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - args.days * 86_400_000
        bars_8h = math.ceil(args.days * 3) + 10

        # Need closes back to start_ms - max_lookback for momentum computation.
        # fetch_perp_closes_8h already fetches `bars` bars from now, so the
        # extra lookback is covered as long as bars_8h >= days*3 + lookback*3.
        max_lookback_hours = max(
            c.momentum_lookback_hours for c in grid
            if c.momentum_lookback_hours is not None
        )
        bars_8h_needed = math.ceil((args.days + max_lookback_hours / 24) * 3) + 10

        histories: dict[str, SymbolHistory] = {}
        t0 = time.time()
        for i, sym in enumerate(universe, 1):
            funding, closes = await asyncio.gather(
                fetch_funding_history(b, sym, start_ms, now_ms),
                fetch_perp_closes_8h(b, sym, bars_8h_needed),
            )
            histories[sym] = SymbolHistory(funding=funding, closes_8h=closes)
            if i % 5 == 0 or i == len(universe):
                print(f"   [{i}/{len(universe)}] {sym}  "
                      f"funding={len(funding)} closes={len(closes)}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

        # Run each config.
        per_config_pnls: list[np.ndarray] = []
        results: list[ConfigResult] = []

        for cfg in grid:
            weekly = simulate_carry(
                histories, start_ms=start_ms, end_ms=now_ms,
                p=cfg.to_params(), start_equity=args.equity_usd,
                costs=costs, pit_log=pit_log,
                momentum_lookback_hours=cfg.momentum_lookback_hours,
            )
            pnls = np.array([w.total_pnl_usd for w in weekly], dtype=float)
            price_pnl = sum(w.long_price_pnl + w.short_price_pnl for w in weekly)
            funding_pnl = sum(w.long_funding_pnl + w.short_funding_pnl for w in weekly)
            fee_pnl = sum(w.fee_pnl for w in weekly)
            per_config_pnls.append(pnls)
            tag = "" if cfg.is_plain_carry else "📊"
            print(f"   {tag}{cfg.label:<30}  weeks={len(pnls)}  "
                  f"pnl=${pnls.sum():+.2f}  "
                  f"price=${price_pnl:+.2f}  fees=${fee_pnl:+.2f}", flush=True)

            results.append(ConfigResult(
                label=cfg.label,
                config={
                    "momentum_lookback_hours": cfg.momentum_lookback_hours,
                    "top_n": cfg.top_n,
                    "book_pct_per_side": cfg.book_pct_per_side,
                },
                is_plain_carry=cfg.is_plain_carry,
                weeks=len(pnls),
                total_pnl_usd=float(pnls.sum()),
                price_pnl_usd=float(price_pnl),
                funding_pnl_usd=float(funding_pnl),
                fee_pnl_usd=float(fee_pnl),
                in_sample_sharpe=0.0,
            ))

        # Align to common length.
        common_len = min(len(p) for p in per_config_pnls)
        matrix = np.column_stack([p[:common_len] for p in per_config_pnls])
        print(f"\nmatrix shape: {matrix.shape}  (weeks × configs)")

        sharpes = sharpe_per_column(matrix, periods_per_year=52.0)
        for i, r in enumerate(results):
            r.in_sample_sharpe = float(sharpes[i])
            r.weeks = common_len

        for idx, r in enumerate(results):
            oos = cpcv_oos_sharpes(matrix[:, idx], n_folds=args.n_folds,
                                    k=args.k_test, periods_per_year=52.0)
            r.cpcv_oos_sharpes = oos
            if oos:
                r.cpcv_oos_sharpe_mean = float(np.mean(oos))
                r.cpcv_oos_sharpe_std = float(np.std(oos, ddof=1)) if len(oos) > 1 else 0.0

        pbo_res = pbo(matrix, s=args.s_subsamples, periods_per_year=52.0)

        # Carry baseline = best IS plain-carry config's CPCV OOS-mean.
        plain_carry_results = [r for r in results if r.is_plain_carry]
        best_carry_idx = select_is_best_idx(
            [r.in_sample_sharpe for r in plain_carry_results],
            [r.weeks for r in plain_carry_results],
        )
        carry_baseline_oos = (
            plain_carry_results[best_carry_idx].cpcv_oos_sharpe_mean
            if best_carry_idx is not None else 0.0
        )
        carry_baseline_label = (
            plain_carry_results[best_carry_idx].label
            if best_carry_idx is not None else "none"
        )

        # IS-best across ALL configs (incl. plain carry).
        best_idx = select_is_best_idx(
            [r.in_sample_sharpe for r in results],
            [r.weeks for r in results],
        )
        best = results[best_idx] if best_idx is not None else max(
            results, key=lambda r: r.in_sample_sharpe
        )

        # IS-best among momentum-only configs.
        mom_results = [r for r in results if not r.is_plain_carry]
        best_mom_idx = select_is_best_idx(
            [r.in_sample_sharpe for r in mom_results],
            [r.weeks for r in mom_results],
        ) if mom_results else None
        best_mom = mom_results[best_mom_idx] if best_mom_idx is not None else None

        # ---- Report ----
        print("\n" + "=" * 100)
        print(f"CPCV + PBO  ·  carry × momentum  ·  top-{args.top_n_universe} universe  "
              f"·  {args.days}d  ·  PIT={'on' if pit_log else 'off'}")
        print("=" * 100)
        n_pos = sum(1 for r in results if r.in_sample_sharpe > 0)
        n_pos_mom = sum(1 for r in mom_results if r.in_sample_sharpe > 0)
        print(f"all configs IS>0:    {n_pos}/{len(results)}")
        print(f"momentum-only IS>0:  {n_pos_mom}/{len(mom_results)}")

        print(f"\nPlain-carry baseline: {carry_baseline_label}  "
              f"OOS-mean={carry_baseline_oos:+.3f}")

        print(f"\nTop 5 configs by IS-Sharpe:")
        for r in sorted(results, key=lambda r: r.in_sample_sharpe, reverse=True)[:5]:
            tag = "[carry]" if r.is_plain_carry else "[+mom] "
            print(f"  {tag} {r.label:<30}  IS={r.in_sample_sharpe:+.3f}  "
                  f"OOS={r.cpcv_oos_sharpe_mean:+.3f}±{r.cpcv_oos_sharpe_std:.3f}  "
                  f"pnl=${r.total_pnl_usd:+.2f}")

        if best_mom:
            print(f"\nBest momentum config: {best_mom.label}")
            print(f"  IS  SR:        {best_mom.in_sample_sharpe:+.3f}")
            print(f"  price PnL:     ${best_mom.price_pnl_usd:+.2f}")
            print(f"  CPCV OOS mean: {best_mom.cpcv_oos_sharpe_mean:+.3f} ± "
                  f"{best_mom.cpcv_oos_sharpe_std:.3f}")
            mom_decision = _combined_decision(
                pbo_res.pbo, best_mom.cpcv_oos_sharpe_mean, carry_baseline_oos
            )
            print(f"  Decision:      {mom_decision}")

        print(f"\nPBO (S={args.s_subsamples}, across all {len(results)} configs):")
        print(f"  PBO:                  {pbo_res.pbo:.3f}")
        print(f"  dead configs:         {pbo_res.n_dead_columns}")
        print(f"  mean logit:           {pbo_res.mean_logit:+.3f}")
        print(f"  mean OOS rank pct:    {pbo_res.median_oos_rank_pct:.3f}")

        overall_decision = _combined_decision(
            pbo_res.pbo,
            best_mom.cpcv_oos_sharpe_mean if best_mom else -999,
            carry_baseline_oos,
        )
        print(f"\n  OVERALL DECISION (best momentum vs carry baseline):")
        print(f"  {overall_decision}")

        # JSON dump
        ts_out = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.out_tag}" if args.out_tag else ""
        out_path = OUT_DIR / f"cpcv_momentum_carry_{ts_out}{tag}.json"
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategy": "momentum_carry_double_sort",
            "days": args.days,
            "pit_corrected": pit_log is not None,
            "universe": universe,
            "n_folds": args.n_folds,
            "k_test": args.k_test,
            "s_subsamples": args.s_subsamples,
            "carry_baseline": {
                "label": carry_baseline_label,
                "oos_mean": carry_baseline_oos,
            },
            "is_best_idx": best_idx,
            "is_best_momentum_idx": (
                results.index(best_mom) if best_mom else None
            ),
            "configs": [asdict(r) for r in results],
            "pbo": {
                "pbo": pbo_res.pbo,
                "n_partitions": pbo_res.n_partitions,
                "n_dead_columns": pbo_res.n_dead_columns,
                "n_trials": pbo_res.n_trials,
                "mean_logit": pbo_res.mean_logit,
                "mean_oos_rank_pct": pbo_res.median_oos_rank_pct,
                "overall_decision": overall_decision,
            },
        }, indent=2, default=str))
        print(f"\nwrote {out_path}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
