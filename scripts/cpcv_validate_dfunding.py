"""CPCV + PBO validation for the Δfunding cross-sectional spread (Card 1).

Δfunding signal: at each weekly rebalance, rank the PIT-filtered perp universe
by the *change* in funding over the trailing window:
  dfund_i = mean(funding_i, recent window) − mean(funding_i, prior window)
Long the top-N risers (funding accelerating), short the top-N fallers.
Dollar-neutral. Same execution as validated carry.

Mechanism: funding levels are ~0.97-0.99 autocorrelated (near-unit-root), so
the level is near-arbitraged by Ethena/BFUSD structural shorts. The first
difference (the repricing surprise) is near-orthogonal to the level and harder
to arbitrage away (funding_edges_2026-05-29.md Card 1).

Sweep grid: window_cycles ∈ {14, 21, 42} × top_n ∈ {2, 3, 5} × book_pct ∈
{0.10, 0.25} = 18 configs.  CPCV(N=10, k=2), PBO at S=8.

Pass gate (conjunction, same as all validated strategies):
  PBO < 0.5  AND  CPCV-OOS-mean > 0

Carry baseline for comparison: PIT-corrected carry CPCV PASS
  tn2_bk10  OOS-mean +1.42 ± 2.50, PBO 0.114
Δfunding earns a slot only when it is additive/diversifying vs that baseline.

Usage:
  # Cheap first test (window=21 cycles, top3, 10%/side) before full CPCV:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_dfunding \\
      --cheap-first-test

  # Full 18-config CPCV sweep:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_dfunding

  # With explicit args:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_dfunding \\
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
from functools import partial
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
from src.strategies.funding_carry import CarryParams, funding_window_change
from src.tools.binance_client import BinanceClient


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data/research/strategy_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Validated carry baseline for benchmark comparison (cpcv_carry_20260527_125455.json).
CARRY_BASELINE_OOS_MEAN = 1.42
CARRY_BASELINE_LABEL = "tn2_bk10 (validated carry, PIT-corrected)"


# ---------------------------------------------------------------------------
# Sweep grid

@dataclass(frozen=True)
class DFundingSweepConfig:
    window_cycles: int      # number of 8h funding cycles per window
    top_n: int
    book_pct_per_side: float

    @property
    def window_hours(self) -> int:
        return self.window_cycles * 8

    @property
    def label(self) -> str:
        return f"w{self.window_cycles}_tn{self.top_n}_bk{int(self.book_pct_per_side*100):02d}"

    def to_params(self) -> CarryParams:
        return CarryParams(top_n=self.top_n,
                           book_pct_per_side=self.book_pct_per_side)

    def signal_fn(self):
        """Returns a signal_fn compatible with simulate_carry's signature."""
        return partial(funding_window_change, window_hours=self.window_hours)


def dfunding_grid() -> list[DFundingSweepConfig]:
    """18 configs: window_cycles ∈ {14, 21, 42} × top_n ∈ {2, 3, 5}
    × book_pct ∈ {0.10, 0.25}."""
    return [
        DFundingSweepConfig(
            window_cycles=wc, top_n=tn, book_pct_per_side=bk,
        )
        for wc, tn, bk in product([14, 21, 42], [2, 3, 5], [0.10, 0.25])
    ]


CHEAP_FIRST_CONFIG = DFundingSweepConfig(
    window_cycles=21, top_n=3, book_pct_per_side=0.10
)


# ---------------------------------------------------------------------------
# Per-config result

@dataclass
class ConfigResult:
    label: str
    config: dict
    weeks: int
    total_pnl_usd: float
    price_pnl_usd: float
    funding_pnl_usd: float
    fee_pnl_usd: float
    in_sample_sharpe: float
    cpcv_oos_sharpe_mean: float = 0.0
    cpcv_oos_sharpe_std: float = 0.0
    cpcv_oos_sharpes: list[float] = field(default_factory=list)


def _combined_decision(pbo_val: float, oos_mean: float) -> str:
    """Conjunction gate: PBO < 0.5 AND CPCV-OOS-mean > 0.
    PBO alone passes uniformly-losing families — the conjunction is non-optional
    (lesson from funding 0/38 and pairs CPCV runs)."""
    if pbo_val >= 0.5:
        return "REJECT [PBO ≥ 0.5]"
    if oos_mean <= 0:
        return "REJECT [OOS-mean ≤ 0 (no edge)]"
    return "PASS"


# ---------------------------------------------------------------------------
# CLI

async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description="CPCV+PBO validation for the Δfunding cross-sectional spread."
    )
    ap.add_argument("--cheap-first-test", action="store_true",
                    help="Run a single cheap config (w21_tn3_bk10) and stop. "
                         "Use to check whether price_pnl > costs before committing "
                         "to the full 18-config CPCV sweep.")
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

    # Load PIT log
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

    grid = [CHEAP_FIRST_CONFIG] if args.cheap_first_test else dfunding_grid()
    if args.cheap_first_test:
        print("=== CHEAP FIRST TEST ===  (window=21 cycles=168h, top3, 10%/side)")
        print("If price_pnl + funding_pnl < |fees|, stop here — edge doesn't survive costs.")
    else:
        print(f"sweep: {len(grid)} configs · CPCV(N={args.n_folds}, k={args.k_test}) "
              f"· PBO S={args.s_subsamples}")

    costs = Costs()
    b = BinanceClient()
    await b.start()
    try:
        universe = await build_universe(b, top_n_universe=args.top_n_universe)
        print(f"universe ({len(universe)} symbols): "
              f"{', '.join(universe[:8])}{'...' if len(universe) > 8 else ''}")

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - args.days * 86_400_000

        # For Δfunding we need 2×max_window of prior funding history before
        # start_ms so that the first rebalance can compute both the recent and
        # prior windows. Max window in the grid is 42 cycles × 8h = 336h.
        max_window_hours = max(cfg.window_hours for cfg in grid)
        funding_start_ms = start_ms - 2 * max_window_hours * 3_600_000

        bars_8h = math.ceil(args.days * 3) + 10
        histories: dict[str, SymbolHistory] = {}
        t0 = time.time()
        for i, sym in enumerate(universe, 1):
            funding, closes = await asyncio.gather(
                fetch_funding_history(b, sym, funding_start_ms, now_ms),
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

        for cfg in grid:
            weekly = simulate_carry(
                histories, start_ms=start_ms, end_ms=now_ms,
                p=cfg.to_params(), start_equity=args.equity_usd,
                costs=costs, pit_log=pit_log,
                signal_fn=cfg.signal_fn(),
            )
            pnls = np.array([w.total_pnl_usd for w in weekly], dtype=float)
            price_pnl = sum(w.long_price_pnl + w.short_price_pnl for w in weekly)
            funding_pnl = sum(w.long_funding_pnl + w.short_funding_pnl for w in weekly)
            fee_pnl = sum(w.fee_pnl for w in weekly)
            per_config_pnls.append(pnls)
            print(f"   {cfg.label}  weeks={len(pnls)}  "
                  f"pnl=${pnls.sum():+.2f}  "
                  f"price=${price_pnl:+.2f}  funding=${funding_pnl:+.2f}  "
                  f"fees=${fee_pnl:+.2f}", flush=True)

            if args.cheap_first_test:
                print()
                print("--- CHEAP FIRST TEST RESULT ---")
                print(f"  price_pnl:    ${price_pnl:+.2f}  "
                      f"({'POSITIVE — proceed to CPCV' if price_pnl > 0 else 'NEGATIVE — edge below costs'})")
                gross = price_pnl + funding_pnl
                print(f"  gross PnL:    ${gross:+.2f}  (price + funding, before fees)")
                print(f"  fees:         ${fee_pnl:+.2f}")
                print(f"  net PnL:      ${pnls.sum():+.2f}")
                if pnls.sum() <= 0:
                    print("\n  STOP: net PnL ≤ 0 on cheap first test. "
                          "Run full CPCV only if you want to verify the negative "
                          "is not a single-window artefact.")
                else:
                    print("\n  Net positive — run full CPCV sweep next "
                          "(remove --cheap-first-test flag).")
                ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                tag = f"_{args.out_tag}" if args.out_tag else ""
                out = OUT_DIR / f"dfunding_cheap_{ts_tag}{tag}.json"
                out.write_text(json.dumps({
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "mode": "cheap_first_test",
                    "config": asdict(cfg),
                    "days": args.days,
                    "pit_corrected": pit_log is not None,
                    "universe": universe,
                    "weeks": len(weekly),
                    "price_pnl_usd": price_pnl,
                    "funding_pnl_usd": funding_pnl,
                    "fee_pnl_usd": fee_pnl,
                    "net_pnl_usd": float(pnls.sum()),
                }, indent=2, default=str))
                print(f"\nwrote {out}")
                return

            results.append(ConfigResult(
                label=cfg.label,
                config={"window_cycles": cfg.window_cycles, "top_n": cfg.top_n,
                        "book_pct_per_side": cfg.book_pct_per_side},
                weeks=len(pnls),
                total_pnl_usd=float(pnls.sum()),
                price_pnl_usd=float(price_pnl),
                funding_pnl_usd=float(funding_pnl),
                fee_pnl_usd=float(fee_pnl),
                in_sample_sharpe=0.0,
            ))

        # Align to common length (off-by-ones across window sizes), run CPCV.
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

        # ---- Report ----
        print("\n" + "=" * 100)
        print(f"CPCV + PBO  ·  Δfunding cross-sectional spread  ·  "
              f"top-{args.top_n_universe} universe  ·  {args.days}d  ·  "
              f"PIT={'on' if pit_log else 'off'}")
        print("=" * 100)
        n_pos_is = sum(1 for r in results if r.in_sample_sharpe > 0)
        print(f"configs:           {len(results)}")
        print(f"IS-Sharpe > 0:     {n_pos_is}/{len(results)}")

        best_idx = select_is_best_idx(
            [r.in_sample_sharpe for r in results],
            [r.weeks for r in results],
        )
        best = results[best_idx] if best_idx is not None else max(results, key=lambda r: r.in_sample_sharpe)

        print(f"\nTop configs by IS-Sharpe:")
        sorted_res = sorted(results, key=lambda r: r.in_sample_sharpe, reverse=True)
        for r in sorted_res[:5]:
            print(f"  {r.label:<28}  IS SR={r.in_sample_sharpe:+.3f}  "
                  f"pnl=${r.total_pnl_usd:+.2f}  "
                  f"price=${r.price_pnl_usd:+.2f}  "
                  f"OOS={r.cpcv_oos_sharpe_mean:+.3f}±{r.cpcv_oos_sharpe_std:.3f}")

        print(f"\nIS-best config: {best.label}")
        print(f"  IS  SR:        {best.in_sample_sharpe:+.3f}")
        print(f"  price PnL:     ${best.price_pnl_usd:+.2f}  "
              f"({'positive' if best.price_pnl_usd > 0 else 'NEGATIVE — Δfunding has no price-drift alpha'})")
        if best.cpcv_oos_sharpes:
            print(f"  CPCV OOS mean: {best.cpcv_oos_sharpe_mean:+.3f} ± {best.cpcv_oos_sharpe_std:.3f}")
            print(f"  IS → OOS:      {best.in_sample_sharpe:+.3f} → {best.cpcv_oos_sharpe_mean:+.3f}  "
                  f"(deflation: {best.in_sample_sharpe - best.cpcv_oos_sharpe_mean:+.3f})")

        print(f"\nPBO (S={args.s_subsamples}):")
        print(f"  PBO:                  {pbo_res.pbo:.3f}")
        print(f"  dead configs:         {pbo_res.n_dead_columns}")
        print(f"  mean logit:           {pbo_res.mean_logit:+.3f}")
        print(f"  mean OOS rank pct:    {pbo_res.median_oos_rank_pct:.3f}")

        decision = _combined_decision(pbo_res.pbo, best.cpcv_oos_sharpe_mean)
        print(f"\n  DECISION:   {decision}")
        print(f"  Carry baseline: {CARRY_BASELINE_LABEL}  OOS-mean={CARRY_BASELINE_OOS_MEAN:+.3f}")
        if "PASS" in decision:
            beats = best.cpcv_oos_sharpe_mean > CARRY_BASELINE_OOS_MEAN
            print(f"  Beats carry:    {'YES' if beats else 'NO — diversifier only'}")

        # JSON dump
        ts_out = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.out_tag}" if args.out_tag else ""
        out_path = OUT_DIR / f"cpcv_dfunding_{ts_out}{tag}.json"
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategy": "dfunding_cross_sectional",
            "days": args.days,
            "pit_corrected": pit_log is not None,
            "universe": universe,
            "n_folds": args.n_folds,
            "k_test": args.k_test,
            "s_subsamples": args.s_subsamples,
            "is_best_idx": best_idx,
            "configs": [asdict(r) for r in results],
            "pbo": {
                "pbo": pbo_res.pbo,
                "n_partitions": pbo_res.n_partitions,
                "n_dead_columns": pbo_res.n_dead_columns,
                "n_trials": pbo_res.n_trials,
                "mean_logit": pbo_res.mean_logit,
                "mean_oos_rank_pct": pbo_res.median_oos_rank_pct,
                "decision": decision,
            },
            "carry_baseline": {
                "label": CARRY_BASELINE_LABEL,
                "oos_mean": CARRY_BASELINE_OOS_MEAN,
            },
        }, indent=2, default=str))
        print(f"\nwrote {out_path}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
