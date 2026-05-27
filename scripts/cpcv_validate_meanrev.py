"""Multi-symbol CPCV+PBO validation for the mean-reversion strategy.

Continues the carry/pairs/funding template. The sprint baseline had
mean-reversion losing −$1.1k on a 23-symbol mid-cap universe at 15m;
sprint #13 added the Hurst+VR+OU regime gate (Chan ch.2 + Macrosynergy
2023) and triple-barrier exits (López de Prado AFML ch.3) which "kill
60-80% of false positives in directional crypto regimes" and bound the
loss tail. The validation question:

  Did the sprint #13 additions rescue the strategy, or is it still REJECT?

Approach:
  1. Top-N spot USDT universe (reuses `backtest_long_horizon.build_universe`).
  2. PIT-filter at ≥95% coverage.
  3. Per symbol, sweep a 3×3×3 = 27-config grid:
       rsi_oversold     ∈ {25, 30, 35}    (overbought = 100 - oversold)
       atr_stop_mult    ∈ {1.0, 1.5, 2.0}
       operating_mode   ∈ {baseline, strict_gate, triple_barrier}
     where:
       baseline       = original ADX<20 + fixed ATR exits
       strict_gate    = Hurst+VR+OU regime gate + fixed ATR exits (sprint #13a)
       triple_barrier = ADX<20 + σ-scaled barriers + OU time stop (sprint #13b)
     The two sprint #13 additions are orthogonal; combining both
     ({strict_gate, triple_barrier}=True) is also possible but kept out
     of the headline grid to keep configs manageable.
  4. CPCV(N=10, k=2) per config, PBO at S=16 on the (T_days, 27) matrix.
  5. Combined gate: PASS iff PBO < 0.5 AND CPCV_OOS_mean(IS-best) > 0.

Defaults: 1y of 1h bars (8760), HTF 4h. The original sprint baseline
used 15m/1h; 1h trades fewer signals but is the canonical Chan
regime-aware MR setup AND is 4× lighter on data fetch, which matters
because mean-rev backtest re-fetches klines per config (the production
function takes `binance, symbol, tf, htf, bars` — no shared-data path
yet, so each config call re-paginates the kline history).

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_meanrev
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_meanrev \\
      --top-n 10 --tf 1h --htf 4h --bars 8760 --concurrency 2
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

from scripts.backtest_long_horizon import build_universe
from src.scanners.universe_pit import (
    SymbolListing,
    filter_universe_for_span,
    load_pit_log,
)
from src.services.backtest import (
    SimTrade,
    backtest_mean_reversion,
)
from src.services.cpcv import (
    cpcv_oos_sharpes,
    daily_bucket_pnls,
    pbo,
    sharpe_per_column,
)
from src.strategies.mean_reversion import MeanReversionConfig
from src.tools.binance_client import BinanceClient


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data/research/strategy_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Sweep grid

OPERATING_MODES = ("baseline", "strict_gate", "triple_barrier")


@dataclass(frozen=True)
class MeanRevSweepConfig:
    rsi_oversold: float
    atr_stop_mult: float
    operating_mode: str             # one of OPERATING_MODES

    @property
    def label(self) -> str:
        return (f"rsi{int(self.rsi_oversold):02d}_atr{self.atr_stop_mult:.1f}_"
                f"{self.operating_mode}")

    def to_cfg(self, symbol: str, htf: str) -> MeanReversionConfig:
        return MeanReversionConfig(
            allowed_symbols=[symbol],
            rsi_oversold=self.rsi_oversold,
            rsi_overbought=100.0 - self.rsi_oversold,
            atr_stop_mult=self.atr_stop_mult,
            htf_timeframe=htf,
            use_strict_regime_gate=(self.operating_mode == "strict_gate"),
            use_triple_barrier=(self.operating_mode == "triple_barrier"),
        )


def meanrev_grid() -> list[MeanRevSweepConfig]:
    """3×3×3 = 27 configs."""
    return [
        MeanRevSweepConfig(rsi_oversold=ro, atr_stop_mult=at, operating_mode=om)
        for ro, at, om in product(
            [25.0, 30.0, 35.0],
            [1.0, 1.5, 2.0],
            OPERATING_MODES,
        )
    ]


# ---------------------------------------------------------------------------
# Per-symbol result types

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
    """The lesson from pairs + funding validation: PBO alone passes
    uniformly-losing families. Combined gate is PBO<0.5 AND OOS-mean>0."""
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

async def _run_one_config(
    b: BinanceClient, symbol: str, tf: str, htf: str, bars: int,
    cfg: MeanRevSweepConfig, sem: asyncio.Semaphore,
) -> tuple[MeanRevSweepConfig, list[SimTrade]]:
    async with sem:
        try:
            _stats, trades = await backtest_mean_reversion(
                b, symbol=symbol, tf=tf, htf=htf, bars=bars,
                cfg=cfg.to_cfg(symbol, htf),
            )
            return cfg, trades
        except Exception as e:  # noqa: BLE001
            print(f"   {symbol}/{cfg.label}: backtest failed: "
                  f"{type(e).__name__}: {e}")
            return cfg, []


async def validate_symbol(
    b: BinanceClient, symbol: str, tf: str, htf: str, bars: int,
    grid: list[MeanRevSweepConfig], concurrency: int,
    n_folds: int, k_test: int, s_subsamples: int,
) -> SymbolValidationResult:
    """Run the 27-config sweep against one symbol."""
    sem = asyncio.Semaphore(concurrency)
    tasks = [_run_one_config(b, symbol, tf, htf, bars, cfg, sem) for cfg in grid]
    per_cfg: list[tuple[MeanRevSweepConfig, list[SimTrade]]] = []
    for fut in asyncio.as_completed(tasks):
        per_cfg.append(await fut)
    # Preserve grid order
    per_cfg.sort(key=lambda x: grid.index(x[0]))

    # Day-0 anchor: start of the backtest window
    now_ms = int(time.time() * 1000)
    bar_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
              "4h": 14_400_000}.get(tf, 3_600_000)
    n_days = max(2, int(bars * bar_ms / 86_400_000) + 1)
    day0_ms = (now_ms - n_days * 86_400_000) // 86_400_000 * 86_400_000

    columns: list[np.ndarray] = []
    results: list[SymbolConfigResult] = []
    for cfg, trades in per_cfg:
        ts = [t.entry_ts_ms for t in trades if t.pnl_usd is not None]
        pnls = [t.pnl_usd for t in trades if t.pnl_usd is not None]
        col = daily_bucket_pnls(ts, pnls, day0_ms=day0_ms, n_days=n_days)
        columns.append(col)
        sr = float(sharpe_per_column(col.reshape(-1, 1), periods_per_year=365.0)[0])
        results.append(SymbolConfigResult(
            label=cfg.label, config=asdict(cfg),
            trades=len(pnls),
            total_pnl_usd=float(sum(pnls)) if pnls else 0.0,
            in_sample_sharpe=sr,
        ))
    matrix = np.column_stack(columns)

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
    print("CPCV + PBO VALIDATION  ·  mean-reversion  ·  multi-symbol")
    print("=" * 110)
    print(f"{'symbol':<14}{'best cfg':<32}{'IS SR':>7}{'OOS mean':>10}"
          f"{'OOS σ':>8}{'PBO':>7}{'rank':>7}  decision")
    print("-" * 110)
    n_pass = 0
    pass_pnls = 0.0
    for v in validations:
        is_best = v.configs[v.is_best_idx] if v.configs else None
        pnl = is_best.total_pnl_usd if is_best else 0.0
        oos_std = is_best.cpcv_oos_sharpe_std if is_best else 0.0
        flag = "✓" if v.decision == "PASS" else " "
        print(f"{v.symbol:<14}{v.is_best_label:<32}"
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

    # Operating-mode tally: which mode produced the IS-best config across
    # all symbols? Useful even when 0/N pass — tells us whether sprint #13's
    # additions ARE the best operating point on this universe.
    mode_counts: dict[str, int] = {m: 0 for m in OPERATING_MODES}
    for v in validations:
        if v.configs:
            best_cfg = v.configs[v.is_best_idx].config
            mode_counts[best_cfg["operating_mode"]] += 1
    print(f"\nIS-best by operating mode:")
    for m, c in mode_counts.items():
        print(f"  {m:<18}{c}/{len(validations)}")


# ---------------------------------------------------------------------------
# CLI

async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=20,
                    help="candidate spot universe size before PIT filter")
    ap.add_argument("--max-symbols", type=int, default=10,
                    help="cap on PIT-survivors actually validated")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--htf", default="4h")
    ap.add_argument("--bars", type=int, default=8_760,
                    help="bars per kline fetch (1y at 1h = 8760)")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="concurrent backtests within each symbol's 27-cfg sweep")
    ap.add_argument("--pit-log",
                    default="data/research/universe/binance_delistings.json")
    ap.add_argument("--pit-min-coverage", type=float, default=0.95)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--k-test", type=int, default=2)
    ap.add_argument("--s-subsamples", type=int, default=16)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

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

    grid = meanrev_grid()
    print(f"per-symbol sweep: {len(grid)} configs · {args.tf}/{args.htf} · "
          f"CPCV(N={args.n_folds}, k={args.k_test}) · PBO S={args.s_subsamples}")

    b = BinanceClient()
    await b.start()
    try:
        candidates = await build_universe(b, top_n=args.top_n)
        print(f"\nspot candidate universe ({len(candidates)} symbols)")

        now_ms = int(time.time() * 1000)
        bar_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
                  "4h": 14_400_000}.get(args.tf, 3_600_000)
        days_in_window = int(args.bars * bar_ms / 86_400_000)
        start_ms = now_ms - days_in_window * 86_400_000

        if pit_log is not None:
            survivors = filter_universe_for_span(
                pit_log, start_ms=start_ms, end_ms=now_ms,
                candidates=candidates, min_coverage=args.pit_min_coverage,
            )
            dropped = [s for s in candidates if s not in survivors]
            print(f"PIT-survivors ({len(survivors)}/{len(candidates)} "
                  f"≥{args.pit_min_coverage*100:.0f}% coverage)")
            if dropped:
                print(f"  dropped: {', '.join(dropped[:10])}"
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
                    b, sym, args.tf, args.htf, args.bars,
                    grid, args.concurrency,
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

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.out_tag}" if args.out_tag else ""
        out_path = OUT_DIR / f"cpcv_meanrev_{ts}{tag}.json"
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tf": args.tf, "htf": args.htf, "bars": args.bars,
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
