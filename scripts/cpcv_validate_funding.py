"""Multi-symbol CPCV+PBO validation for the funding-harvest strategy.

`scripts/cpcv_validate.py` (sprint #10) validates ONE symbol at a time
across a 27-config grid. This driver lifts that into the same template
used for carry (`cpcv_validate_carry.py`) and pairs
(`cpcv_validate_pairs.py`):

  1. Build a candidate universe of top-N USDT perps (majors ex'd —
     Ethena dominates BTC/ETH).
  2. PIT-filter by coverage_fraction over the backtest window so we
     only validate symbols that were continuously live across [start, end].
  3. Per surviving symbol, run the 27-config sweep:
        z_entry ∈ {1.0, 1.5, 2.0}
        z_exit  ∈ {-0.3, 0.3, 0.5}
        z_window ∈ {120, 180, 240} cycles
     Daily-bucket per-config trade pnls → (T_days, 27) matrix.
     Compute CPCV(N=10, k=2) OOS Sharpe for IS-best + PBO at S=16.
  4. Apply the **combined gate** (lesson from pairs validation):
        PASS iff PBO < 0.5 AND CPCV_OOS_mean(IS-best) > 0
     PBO alone passes families where every config loses uniformly
     (selection isn't biased only because there's nothing to bias toward).
  5. Aggregate report: per-symbol verdict + expected portfolio PnL
     from PASS-only symbols.

Universe scoring is symbol-level — funding harvest is per-symbol mean
reversion of the funding rate, not a cross-sectional play. So this
script answers "which symbols deserve to be in the live universe?"

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_funding
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_funding \\
      --days 365 --top-n-universe 15 --concurrency 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from scripts.backtest_funding_carry import build_universe
from scripts.cpcv_validate import (
    FundingConfig,
    evaluate_funding_config,
    funding_grid,
)
from src.scanners.universe_pit import (
    SymbolListing,
    filter_universe_for_span,
    load_pit_log,
)
from src.services.cpcv import (
    cpcv_oos_sharpes,
    daily_bucket_pnls,
    pbo,
    sharpe_per_column,
)
from src.tools.binance_client import BinanceClient


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data/research/strategy_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-symbol result

@dataclass
class SymbolConfigResult:
    label: str
    config: dict
    trades: int
    total_pnl_usd: float
    in_sample_sharpe: float
    cpcv_oos_sharpe_mean: float = 0.0
    cpcv_oos_sharpe_std: float = 0.0


@dataclass
class SymbolValidationResult:
    symbol: str
    n_configs: int
    configs: list[SymbolConfigResult] = field(default_factory=list)
    is_best_label: str = ""
    is_best_idx: int = 0
    is_best_sharpe: float = 0.0
    is_best_oos_mean: float = 0.0
    pbo: float = 0.0
    pbo_n_partitions: int = 0
    pbo_mean_logit: float = 0.0
    pbo_mean_oos_rank: float = 0.0
    decision: str = ""
    decision_reason: str = ""


def _combined_decision(pbo_val: float, oos_mean: float) -> tuple[str, str]:
    """Conjunction gate established by the pairs validation: PBO<0.5 AND
    OOS-mean>0. Pure PBO passes uniformly-losing families because
    selection has no bias to surface; we need positive OOS mean to confirm
    the underlying edge."""
    pbo_ok = pbo_val < 0.5
    oos_ok = oos_mean > 0
    if pbo_ok and oos_ok:
        return "PASS", ""
    if not pbo_ok and not oos_ok:
        return "REJECT", "PBO ≥ 0.5 AND OOS-mean ≤ 0"
    if not pbo_ok:
        return "REJECT", f"PBO {pbo_val:.3f} ≥ 0.5"
    return "REJECT", f"OOS-mean {oos_mean:+.3f} ≤ 0 (no edge)"


# ---------------------------------------------------------------------------
# Per-symbol validation

async def validate_symbol(
    b: BinanceClient, symbol: str, days: int,
    grid: list[FundingConfig], concurrency: int,
    n_folds: int, k_test: int, s_subsamples: int,
) -> SymbolValidationResult:
    """Run the 27-config sweep against one symbol, compute CPCV + PBO."""
    sem = asyncio.Semaphore(concurrency)
    tasks = [evaluate_funding_config(b, symbol, days, cfg, sem) for cfg in grid]
    per_cfg: list[tuple[FundingConfig, list]] = []
    for fut in asyncio.as_completed(tasks):
        per_cfg.append(await fut)
    per_cfg.sort(key=lambda x: grid.index(x[0]))

    # Daily bucket each config's pnls.
    now_ms = int(time.time() * 1000)
    day_ms = 86_400_000
    day0_ms = (now_ms - days * day_ms) // day_ms * day_ms

    columns: list[np.ndarray] = []
    results: list[SymbolConfigResult] = []
    for cfg, trades in per_cfg:
        ts = [t.entry_ts_ms for t in trades if t.pnl_usd is not None]
        pnls = [t.pnl_usd for t in trades if t.pnl_usd is not None]
        col = daily_bucket_pnls(ts, pnls, day0_ms=day0_ms, n_days=days)
        columns.append(col)
        sr = float(sharpe_per_column(col.reshape(-1, 1), periods_per_year=365.0)[0])
        results.append(SymbolConfigResult(
            label=cfg.label, config=asdict(cfg),
            trades=len(pnls),
            total_pnl_usd=float(sum(pnls)) if pnls else 0.0,
            in_sample_sharpe=sr,
        ))
    matrix = np.column_stack(columns)

    # CPCV OOS distribution (only the IS-best is the relevant decision point)
    is_best_idx = int(np.argmax([r.in_sample_sharpe for r in results]))
    for idx, r in enumerate(results):
        oos = cpcv_oos_sharpes(matrix[:, idx], n_folds=n_folds,
                                k=k_test, periods_per_year=365.0)
        if oos:
            r.cpcv_oos_sharpe_mean = float(np.mean(oos))
            r.cpcv_oos_sharpe_std = float(np.std(oos, ddof=1)) if len(oos) > 1 else 0.0

    pbo_res = pbo(matrix, s=s_subsamples, periods_per_year=365.0)
    is_best = results[is_best_idx]
    decision, reason = _combined_decision(pbo_res.pbo, is_best.cpcv_oos_sharpe_mean)

    return SymbolValidationResult(
        symbol=symbol, n_configs=len(grid),
        configs=results,
        is_best_label=is_best.label, is_best_idx=is_best_idx,
        is_best_sharpe=is_best.in_sample_sharpe,
        is_best_oos_mean=is_best.cpcv_oos_sharpe_mean,
        pbo=pbo_res.pbo, pbo_n_partitions=pbo_res.n_partitions,
        pbo_mean_logit=pbo_res.mean_logit,
        pbo_mean_oos_rank=pbo_res.median_oos_rank_pct,
        decision=decision, decision_reason=reason,
    )


# ---------------------------------------------------------------------------
# Reporting

def print_summary(validations: list[SymbolValidationResult]) -> None:
    print("\n" + "=" * 110)
    print("CPCV + PBO VALIDATION  ·  funding-harvest  ·  multi-symbol")
    print("=" * 110)
    print(f"{'symbol':<14}{'best cfg':<26}{'IS SR':>7}{'OOS mean':>10}"
          f"{'OOS σ':>8}{'PBO':>7}{'rank':>7}  decision")
    print("-" * 110)
    n_pass = 0
    pass_pnls = 0.0
    for v in validations:
        is_best = v.configs[v.is_best_idx] if v.configs else None
        pnl = is_best.total_pnl_usd if is_best else 0.0
        oos_std = is_best.cpcv_oos_sharpe_std if is_best else 0.0
        flag = "✓" if v.decision == "PASS" else " "
        print(f"{v.symbol:<14}{v.is_best_label:<26}"
              f"{v.is_best_sharpe:+7.3f}{v.is_best_oos_mean:+10.3f}"
              f"{oos_std:>8.2f}{v.pbo:>7.3f}{v.pbo_mean_oos_rank:>7.3f}  "
              f"{flag} {v.decision}"
              f"{('  ('+v.decision_reason+')') if v.decision_reason else ''}")
        if v.decision == "PASS":
            n_pass += 1
            pass_pnls += pnl

    print("-" * 110)
    print(f"PASSed:          {n_pass}/{len(validations)}")
    print(f"PASS-only PnL:   ${pass_pnls:+.2f}  "
          f"(IS-best config × symbol, summed)")


# ---------------------------------------------------------------------------
# CLI

async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--top-n-universe", type=int, default=20,
                    help="candidate USDT perp universe before PIT filter")
    ap.add_argument("--skip-top", type=int, default=0,
                    help="drop the leading N highest-volume symbols (e.g. "
                         "--skip-top 30 --top-n-universe 100 → mid-cap "
                         "ranks 31..100 by 24h volume)")
    ap.add_argument("--max-symbols", type=int, default=12,
                    help="cap on PIT-survivors actually validated")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="concurrent backtests within each symbol's 27-cfg sweep")
    ap.add_argument("--pit-log",
                    default="data/research/universe/binance_delistings.json")
    ap.add_argument("--pit-min-coverage", type=float, default=0.95,
                    help="min fraction of backtest window the symbol must have been live")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--k-test", type=int, default=2)
    ap.add_argument("--s-subsamples", type=int, default=16)
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
            print(f"WARNING: PIT log empty at {pit_path}; skipping coverage filter.")
            pit_log = None
        else:
            print(f"PIT log: {len(pit_log)} symbols loaded")

    grid = funding_grid()
    print(f"per-symbol sweep: {len(grid)} configs · "
          f"CPCV(N={args.n_folds}, k={args.k_test}) · PBO S={args.s_subsamples}")

    b = BinanceClient()
    await b.start()
    try:
        candidates = await build_universe(b, top_n_universe=args.top_n_universe)
        if args.skip_top:
            n_before = len(candidates)
            candidates = candidates[args.skip_top:]
            print(f"\ncandidate universe ({len(candidates)} symbols after "
                  f"skipping top-{args.skip_top} of {n_before} by 24h volume)")
        else:
            print(f"\ncandidate universe ({len(candidates)} symbols)")

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - args.days * 86_400_000

        # PIT-filter for full-window coverage.
        if pit_log is not None:
            survivors = filter_universe_for_span(
                pit_log, start_ms=start_ms, end_ms=now_ms,
                candidates=candidates, min_coverage=args.pit_min_coverage,
            )
            dropped = [s for s in candidates if s not in survivors]
            print(f"PIT-survivors ({len(survivors)}/{len(candidates)} "
                  f"≥{args.pit_min_coverage*100:.0f}% coverage)")
            if dropped:
                print(f"  dropped (insufficient history): {', '.join(dropped[:10])}"
                      f"{'...' if len(dropped) > 10 else ''}")
            candidates = survivors

        if args.max_symbols:
            candidates = candidates[: args.max_symbols]
        print(f"validating {len(candidates)} symbols: {', '.join(candidates)}")

        validations: list[SymbolValidationResult] = []
        t0 = time.time()
        for i, sym in enumerate(candidates, 1):
            t1 = time.time()
            print(f"\n[{i}/{len(candidates)}] {sym} sweeping {len(grid)} configs ...",
                  flush=True)
            try:
                v = await validate_symbol(
                    b, sym, args.days, grid, args.concurrency,
                    args.n_folds, args.k_test, args.s_subsamples,
                )
            except Exception as e:  # noqa: BLE001
                print(f"   FAILED: {type(e).__name__}: {e}")
                continue
            validations.append(v)
            print(f"   IS-best {v.is_best_label}  SR={v.is_best_sharpe:+.2f}  "
                  f"OOS-mean={v.is_best_oos_mean:+.2f}  PBO={v.pbo:.3f}  "
                  f"→ {v.decision}  ({time.time()-t1:.0f}s, total {time.time()-t0:.0f}s)",
                  flush=True)

        print_summary(validations)

        # JSON dump
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.out_tag}" if args.out_tag else ""
        out_path = OUT_DIR / f"cpcv_funding_multi_{ts}{tag}.json"
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "days": args.days,
            "n_folds": args.n_folds, "k_test": args.k_test,
            "s_subsamples": args.s_subsamples,
            "pit_min_coverage": args.pit_min_coverage,
            "candidates": candidates,
            "n_pass": sum(1 for v in validations if v.decision == "PASS"),
            "validations": [{
                "symbol": v.symbol,
                "is_best_label": v.is_best_label,
                "is_best_sharpe": v.is_best_sharpe,
                "is_best_oos_mean": v.is_best_oos_mean,
                "pbo": v.pbo,
                "pbo_n_partitions": v.pbo_n_partitions,
                "pbo_mean_oos_rank": v.pbo_mean_oos_rank,
                "decision": v.decision,
                "decision_reason": v.decision_reason,
                "configs": [asdict(c) for c in v.configs],
            } for v in validations],
        }, indent=2, default=str))
        print(f"\nwrote {out_path}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
