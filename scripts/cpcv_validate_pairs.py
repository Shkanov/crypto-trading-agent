"""CPCV + PBO validation for the cointegrated-pairs strategy.

The single-config 1y smoke (commit 01ca969) found:
  ETHUSDT/BTCUSDT  28 trades   Sharpe −0.41  dd 12.5%
  SOLUSDT/ETHUSDT 174 trades   Sharpe −10.2  dd 169%

The driver is correct; the strategy doesn't earn at the canonical
60d-lookback / z_entry=2 / z_exit=0.5 / z_stop=3.5 thresholds. This
script asks the stronger question: is the parameterisation family
fundamentally broken, or is the chosen point bad?

Approach (same template as scripts/cpcv_validate_carry.py):
  1. Fetch each pair's aligned 1h closes ONCE.
  2. Sweep a 4×3×3 = 36-config grid over (lookback, z_entry, z_exit) with
     z_stop fixed per-config at z_entry + 1.5 (sensible scaling).
  3. For each config, run simulate_pair → trade list. Daily-bucket trade
     pnls into a per-config column.
  4. Stack into (T_days, N_configs) per pair. CPCV(N=10, k=2) per config
     gives a 45-sample OOS Sharpe distribution; PBO at S=16 over the
     full matrix gives the selection-bias estimate.

Note on PIT: ETH/BTC and SOL/ETH have been listed since 2017 / 2020;
survivorship is not the bottleneck for this universe. A future
"pair scanner" sweeping across many symbols would need PIT correction;
this fixed-pair driver does not.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_pairs
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_pairs \\
      --pairs ETHUSDT:BTCUSDT --bars 8760 --s-subsamples 16
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

import numpy as np
from dotenv import load_dotenv

from scripts.backtest_pairs import (
    AlignedSeries,
    fetch_aligned_pair,
    simulate_pair,
)
from src.services.costs import Costs
from src.services.cpcv import (
    cpcv_oos_sharpes,
    daily_bucket_pnls,
    pbo,
    sharpe_per_column,
)
from src.strategies.pairs_cointegration import PairsParams
from src.tools.binance_client import BinanceClient


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data/research/strategy_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Sweep grid

@dataclass(frozen=True)
class PairsSweepConfig:
    lookback_bars: int
    z_entry: float
    z_exit: float
    persist_refits: int = 1

    @property
    def label(self) -> str:
        base = f"lb{self.lookback_bars}_ze{self.z_entry:.1f}_zx{self.z_exit:.1f}"
        return base + (f"_pr{self.persist_refits}" if self.persist_refits > 1 else "")

    def to_params(self) -> PairsParams:
        # z_stop scales with z_entry so wider entries get correspondingly
        # wider stops — otherwise a higher z_entry is always tighter to its
        # stop, biasing the sweep toward early-stop variants.
        return PairsParams(
            lookback_bars=self.lookback_bars,
            refit_every_bars=168,          # weekly, as in the canonical run
            coint_pvalue_max=0.05,
            z_entry=self.z_entry,
            z_exit=self.z_exit,
            z_stop=self.z_entry + 1.5,
            persist_refits=self.persist_refits,
        )


def pairs_grid(persist_values: tuple[int, ...] = (1,)) -> list[PairsSweepConfig]:
    """(4×3×3)×|persist| configs over (lookback, z_entry, z_exit, persist).
       lookback in {360,720,1440,2160} bars = {15d, 30d, 60d, 90d} on 1h.
       z_entry in {1.5, 2.0, 2.5}; z_exit in {0.0, 0.3, 0.5}.
       persist (health gate) defaults to (1,) = legacy; pass e.g. (1,2,4) to
       put the persistence gate INTO the PBO family so selection bias on it is
       measured rather than assumed.
    """
    return [
        PairsSweepConfig(lookback_bars=lb, z_entry=ze, z_exit=zx, persist_refits=pr)
        for lb, ze, zx, pr in product(
            [360, 720, 1440, 2160],
            [1.5, 2.0, 2.5],
            [0.0, 0.3, 0.5],
            persist_values,
        )
    ]


# ---------------------------------------------------------------------------
# Per-pair evaluation

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


@dataclass
class PairValidationResult:
    pair: str
    n_configs: int
    configs: list[ConfigResult] = field(default_factory=list)
    pbo: float = 0.0
    pbo_n_partitions: int = 0
    pbo_mean_logit: float = 0.0
    pbo_mean_oos_rank: float = 0.0
    decision: str = ""
    is_best_label: str = ""
    is_best_idx: int = 0


def _decision(pbo_val: float) -> str:
    return "PASS" if pbo_val < 0.5 else "REJECT"


# ---------------------------------------------------------------------------
# Pair validation

def validate_pair(
    series: AlignedSeries,
    sym_a: str, sym_b: str,
    grid: list[PairsSweepConfig],
    notional_per_leg: float,
    start_equity: float,
    costs: Costs,
    n_folds: int, k_test: int, s_subsamples: int,
) -> PairValidationResult:
    """Run all configs against the shared aligned series, build the
    matrix, compute CPCV + PBO."""
    pair_label = f"{sym_a}/{sym_b}"

    # Common day0 for daily-bucketing all configs' trades — anchored at
    # the first aligned close-time, rounded down to UTC midnight.
    if series.close_times.size == 0:
        return PairValidationResult(pair=pair_label, n_configs=len(grid))
    t0_ms = int(series.close_times[0])
    day_ms = 86_400_000
    day0_ms = (t0_ms // day_ms) * day_ms
    last_ms = int(series.close_times[-1])
    n_days = max(2, (last_ms - day0_ms) // day_ms + 1)

    columns: list[np.ndarray] = []
    results: list[ConfigResult] = []
    half_spread = costs.half_spread_bps_default
    for cfg in grid:
        res = simulate_pair(
            series, pair_label=pair_label, sym_a=sym_a, sym_b=sym_b,
            p=cfg.to_params(), notional_per_leg=notional_per_leg,
            start_equity=start_equity, costs=costs,
            half_spread_bps=half_spread, interval="1h",
        )
        ts = [t.entry_ts_ms for t in res.trades if t.pnl_usd is not None]
        pnls = [t.pnl_usd for t in res.trades if t.pnl_usd is not None]
        col = daily_bucket_pnls(ts, pnls, day0_ms=day0_ms, n_days=n_days)
        columns.append(col)
        sr = float(sharpe_per_column(col.reshape(-1, 1), periods_per_year=365.0)[0])
        results.append(ConfigResult(
            label=cfg.label, config=asdict(cfg),
            trades=len(pnls),
            total_pnl_usd=float(sum(pnls)) if pnls else 0.0,
            in_sample_sharpe=sr,
        ))

    matrix = np.column_stack(columns)

    # CPCV OOS distribution per config.
    for idx, r in enumerate(results):
        oos = cpcv_oos_sharpes(matrix[:, idx], n_folds=n_folds,
                                k=k_test, periods_per_year=365.0)
        r.cpcv_oos_sharpes = oos
        if oos:
            r.cpcv_oos_sharpe_mean = float(np.mean(oos))
            r.cpcv_oos_sharpe_std = float(np.std(oos, ddof=1)) if len(oos) > 1 else 0.0

    # PBO across the family.
    pbo_res = pbo(matrix, s=s_subsamples, periods_per_year=365.0)
    is_best_idx = int(np.argmax([r.in_sample_sharpe for r in results]))

    return PairValidationResult(
        pair=pair_label,
        n_configs=len(grid),
        configs=results,
        pbo=pbo_res.pbo,
        pbo_n_partitions=pbo_res.n_partitions,
        pbo_mean_logit=pbo_res.mean_logit,
        pbo_mean_oos_rank=pbo_res.median_oos_rank_pct,
        decision=_decision(pbo_res.pbo),
        is_best_label=results[is_best_idx].label,
        is_best_idx=is_best_idx,
    )


# ---------------------------------------------------------------------------
# Reporting

def print_pair_report(args, v: PairValidationResult) -> None:
    print("\n" + "=" * 100)
    print(f"CPCV + PBO  ·  {v.pair}  ·  {len(v.configs)} configs  ·  "
          f"{args.bars} bars")
    print("=" * 100)
    n_pos = sum(1 for c in v.configs if c.in_sample_sharpe > 0)
    print(f"IS-Sharpe > 0:     {n_pos}/{len(v.configs)}")
    best = v.configs[v.is_best_idx]
    worst = min(v.configs, key=lambda c: c.in_sample_sharpe)
    print(f"best IS:           {best.label}   SR={best.in_sample_sharpe:+.3f}  "
          f"trades={best.trades}  pnl=${best.total_pnl_usd:+.2f}")
    print(f"worst IS:          {worst.label}  SR={worst.in_sample_sharpe:+.3f}  "
          f"trades={worst.trades}  pnl=${worst.total_pnl_usd:+.2f}")
    if best.cpcv_oos_sharpes:
        print(f"\nCPCV(N={args.n_folds}, k={args.k_test}) on IS-best — "
              f"{len(best.cpcv_oos_sharpes)} OOS Sharpe samples:")
        print(f"  mean:        {best.cpcv_oos_sharpe_mean:+.3f}")
        print(f"  std:         {best.cpcv_oos_sharpe_std:.3f}")
        print(f"  IS → OOS:    {best.in_sample_sharpe:+.3f} → "
              f"{best.cpcv_oos_sharpe_mean:+.3f}  "
              f"(deflation: {best.in_sample_sharpe - best.cpcv_oos_sharpe_mean:+.3f})")
    print(f"\nPBO (S={args.s_subsamples}, {v.pbo_n_partitions} partitions):")
    print(f"  PBO:                  {v.pbo:.3f}")
    print(f"  mean logit:           {v.pbo_mean_logit:+.3f}")
    print(f"  mean OOS rank pct:    {v.pbo_mean_oos_rank:.3f}  "
          "(0.5 = median; higher is better)")
    print(f"  DECISION:             {v.decision}  (gate: PBO < 0.5)")


# ---------------------------------------------------------------------------
# CLI

DEFAULT_PAIRS = "ETHUSDT:BTCUSDT,SOLUSDT:ETHUSDT"


def _parse_pairs(arg: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if ":" not in chunk:
            raise ValueError(f"pair must be 'A:B' (got {chunk!r})")
        a, b = chunk.split(":", 1)
        out.append((a.strip().upper(), b.strip().upper()))
    return out


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=DEFAULT_PAIRS,
                    help=f"comma-separated 'A:B' pairs (default {DEFAULT_PAIRS})")
    ap.add_argument("--bars", type=int, default=8_760,
                    help="bars per leg (~1y at 1h = 8760)")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--notional", type=float, default=1_000.0,
                    help="USD per B-leg; A-leg scaled by abs(β)·notional")
    ap.add_argument("--equity-usd", type=float, default=1_000.0)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--k-test", type=int, default=2)
    ap.add_argument("--s-subsamples", type=int, default=16,
                    help="PBO subsample count (default 16 → C(16,8)=12870)")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--persist-grid", default="1",
                    help="comma-separated persistence values for the health gate, "
                         "e.g. '1,2,4'. Default '1' = legacy (no gate).")
    args = ap.parse_args()

    persist_values = tuple(int(x) for x in args.persist_grid.split(","))
    grid = pairs_grid(persist_values)
    pairs = _parse_pairs(args.pairs)
    print(f"pairs: {len(pairs)} · sweep: {len(grid)} configs · "
          f"CPCV(N={args.n_folds}, k={args.k_test}) · PBO S={args.s_subsamples}")

    costs = Costs()
    b = BinanceClient()
    await b.start()
    try:
        validations: list[PairValidationResult] = []
        for i, (sa, sb) in enumerate(pairs, 1):
            label = f"{sa}/{sb}"
            print(f"\n[{i}/{len(pairs)}] {label} fetching ...", flush=True)
            t0 = time.time()
            try:
                series = await fetch_aligned_pair(b, sa, sb, bars=args.bars,
                                                  interval=args.interval)
            except Exception as e:  # noqa: BLE001
                print(f"   FETCH FAILED: {type(e).__name__}: {e}")
                continue
            span_days = (series.close_times[-1] - series.close_times[0]) / 86_400_000
            print(f"   {len(series.close_times)} aligned bars, span={span_days:.0f}d, "
                  f"fetch={time.time()-t0:.1f}s")
            t1 = time.time()
            v = validate_pair(
                series=series, sym_a=sa, sym_b=sb, grid=grid,
                notional_per_leg=args.notional,
                start_equity=args.equity_usd, costs=costs,
                n_folds=args.n_folds, k_test=args.k_test,
                s_subsamples=args.s_subsamples,
            )
            print(f"   sweep={time.time()-t1:.1f}s ({len(grid)} configs)")
            validations.append(v)
            print_pair_report(args, v)

        # JSON dump
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.out_tag}" if args.out_tag else ""
        out_path = OUT_DIR / f"cpcv_pairs_{ts}{tag}.json"
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bars": args.bars,
            "interval": args.interval,
            "n_folds": args.n_folds, "k_test": args.k_test,
            "s_subsamples": args.s_subsamples,
            "pairs": [{
                "pair": v.pair,
                "n_configs": v.n_configs,
                "is_best_label": v.is_best_label,
                "is_best_idx": v.is_best_idx,
                "pbo": v.pbo,
                "pbo_n_partitions": v.pbo_n_partitions,
                "pbo_mean_logit": v.pbo_mean_logit,
                "pbo_mean_oos_rank": v.pbo_mean_oos_rank,
                "decision": v.decision,
                "configs": [asdict(c) for c in v.configs],
            } for v in validations],
        }, indent=2, default=str))
        print(f"\nwrote {out_path}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
