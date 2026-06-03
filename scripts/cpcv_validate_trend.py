"""CPCV + PBO validation for the Donchian55 trend-following strategy.

The indicator-confluence strategy was −$8.4k / SR −6.24 on the original
23-symbol mid-cap 15m universe. This script tests the recommendations-doc
fixes (recommendations_2026_05_27.md §2.1):

  entry_rule  = "donchian55"       Turtle-style H1: close > Donchian(55)_prior
  exit_rule   = "chandelier"       ATR trailing stop, no fixed TP
  require_strong_trend_regime=True ADX(14) on 4H > adx_min AND Chop(14) < chop_max
  timeframe   = 1H (NOT 15m)       Trend strategies live at 4H–daily; 15m = chop
  universe    = top-8 majors       BTCUSDT/ETHUSDT/SOLUSDT/BNBUSDT/XRPUSDT/DOGEUSDT/
                                   ADAUSDT/AVAXUSDT (Begušić/Kostanjčar arXiv 1904.00890:
                                   momentum is concentrated in liquid majors)

The key efficiency gain over the mean-rev validator: klines are fetched ONCE
per symbol and shared across all configs via the new `simulate_indicator` API
(no re-fetching per config).

Grid: 3 × 3 × 2 = 18 configs
  adx_strong_min  ∈ {20, 25, 30}     how strict the trend gate is
  chop_max        ∈ {45, 50, 55}     Dreiss choppiness ceiling
  chandelier_mult ∈ {2.5, 3.0}       trailing stop tightness (StratBase 2024)
All configs use entry_rule="donchian55", exit_rule="chandelier",
require_strong_trend_regime=True, htf="4h".

Pass gate: PBO < 0.5 AND CPCV-OOS-mean > 0 (same conjunction as all prior runs).

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_trend
  BINANCE_TESTNET=false .venv/bin/python -m scripts.cpcv_validate_trend \\
      --bars 8760 --n-folds 10 --s-subsamples 16
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

from src.models.types import Kline, StrategyConfig
from src.services.backtest import SimTrade, simulate_indicator
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

MAJORS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
]


# ---------------------------------------------------------------------------
# Grid

@dataclass(frozen=True)
class TrendSweepConfig:
    adx_strong_min: float
    chop_max: float
    chandelier_atr_mult: float

    @property
    def label(self) -> str:
        return (f"adx{int(self.adx_strong_min)}_"
                f"chop{int(self.chop_max)}_"
                f"chand{self.chandelier_atr_mult:.1f}")

    def to_strategy_cfg(self, symbol: str) -> StrategyConfig:
        return StrategyConfig(
            allowed_symbols=[symbol],
            entry_rule="donchian55",
            exit_rule="chandelier",
            require_strong_trend_regime=True,
            adx_strong_min=self.adx_strong_min,
            chop_max=self.chop_max,
            chandelier_atr_mult=self.chandelier_atr_mult,
            htf_timeframe="4h",
        )


def trend_grid() -> list[TrendSweepConfig]:
    """3 × 3 × 2 = 18 configs."""
    return [
        TrendSweepConfig(
            adx_strong_min=adx, chop_max=chop, chandelier_atr_mult=chand,
        )
        for adx, chop, chand in product(
            [20.0, 25.0, 30.0],
            [45.0, 50.0, 55.0],
            [2.5, 3.0],
        )
    ]


# ---------------------------------------------------------------------------
# Result types

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
class SymbolResult:
    symbol: str
    configs: list[SymbolConfigResult] = field(default_factory=list)
    is_best_label: str = ""
    is_best_sharpe: float = 0.0
    is_best_oos_mean: float = 0.0
    pbo: float = 0.0
    n_dead: int = 0
    decision: str = ""


def _combined_decision(pbo_val: float, oos_mean: float) -> str:
    if pbo_val >= 0.5:
        return "REJECT [PBO ≥ 0.5]"
    if oos_mean <= 0:
        return f"REJECT [OOS-mean {oos_mean:+.3f} ≤ 0]"
    return "PASS"


# ---------------------------------------------------------------------------
# Kline fetch helpers

def _rows_to_klines(rows: list, symbol: str, tf: str) -> list[Kline]:
    return [Kline(
        symbol=symbol, timeframe=tf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in rows]


async def fetch_symbol(b: BinanceClient, symbol: str,
                       bars_1h: int, bars_4h: int) -> tuple[list[Kline], list[Kline]]:
    raw_1h, raw_4h = await asyncio.gather(
        b.fetch_klines_paginated(symbol, "1h", total=bars_1h, market="spot"),
        b.fetch_klines_paginated(symbol, "4h", total=bars_4h, market="spot"),
    )
    return _rows_to_klines(raw_1h, symbol, "1h"), _rows_to_klines(raw_4h, symbol, "4h")


# ---------------------------------------------------------------------------
# Per-symbol validation

def validate_symbol(
    symbol: str,
    ks_1h: list[Kline],
    ks_4h: list[Kline],
    grid: list[TrendSweepConfig],
    n_folds: int,
    k_test: int,
    s_subsamples: int,
    bars: int,
) -> SymbolResult:
    """Run all configs on pre-fetched klines, CPCV+PBO the result."""
    now_ms = int(time.time() * 1000)
    n_days = bars // 24 + 2     # 1h bars → days
    day0_ms = (now_ms - n_days * 86_400_000) // 86_400_000 * 86_400_000

    columns: list[np.ndarray] = []
    cfg_results: list[SymbolConfigResult] = []

    for cfg in grid:
        try:
            _, trades = simulate_indicator(
                symbol, ks_1h, ks_4h,
                cfg=cfg.to_strategy_cfg(symbol),
            )
        except Exception as e:
            print(f"   {symbol}/{cfg.label}: {type(e).__name__}: {e}")
            trades = []

        ts = [t.entry_ts_ms for t in trades if t.pnl_usd is not None]
        pnls = [t.pnl_usd for t in trades if t.pnl_usd is not None]
        col = daily_bucket_pnls(ts, pnls, day0_ms=day0_ms, n_days=n_days)
        columns.append(col)

        sr = float(sharpe_per_column(col.reshape(-1, 1), periods_per_year=365.0)[0])
        cfg_results.append(SymbolConfigResult(
            label=cfg.label,
            config={"adx_strong_min": cfg.adx_strong_min,
                    "chop_max": cfg.chop_max,
                    "chandelier_atr_mult": cfg.chandelier_atr_mult},
            trades=len(pnls),
            total_pnl_usd=float(sum(pnls)) if pnls else 0.0,
            in_sample_sharpe=sr,
        ))
        print(f"   {cfg.label:<30}  trades={len(pnls):3d}  "
              f"pnl=${sum(pnls):+.2f}  SR={sr:+.3f}", flush=True)

    matrix = np.column_stack(columns)
    is_best_idx = select_is_best_idx(
        [r.in_sample_sharpe for r in cfg_results],
        [r.trades for r in cfg_results],
    )

    for idx, r in enumerate(cfg_results):
        oos = cpcv_oos_sharpes(matrix[:, idx], n_folds=n_folds,
                                k=k_test, periods_per_year=365.0)
        if oos:
            r.cpcv_oos_sharpe_mean = float(np.mean(oos))
            r.cpcv_oos_sharpe_std = float(np.std(oos, ddof=1)) if len(oos) > 1 else 0.0

    pbo_res = pbo(matrix, s=s_subsamples, periods_per_year=365.0)

    best_idx = is_best_idx if is_best_idx is not None else int(
        np.argmax([r.in_sample_sharpe for r in cfg_results]))
    best = cfg_results[best_idx]
    decision = _combined_decision(pbo_res.pbo, best.cpcv_oos_sharpe_mean)

    return SymbolResult(
        symbol=symbol,
        configs=cfg_results,
        is_best_label=best.label,
        is_best_sharpe=best.in_sample_sharpe,
        is_best_oos_mean=best.cpcv_oos_sharpe_mean,
        pbo=pbo_res.pbo,
        n_dead=pbo_res.n_dead_columns,
        decision=decision,
    )


# ---------------------------------------------------------------------------
# CLI

async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description="CPCV+PBO for Donchian55 trend-following on top-8 majors."
    )
    ap.add_argument("--bars", type=int, default=8760,
                    help="1H bars per symbol (8760 = 1 year)")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--k-test", type=int, default=2)
    ap.add_argument("--s-subsamples", type=int, default=16,
                    help="PBO subsamples. S=16 on T=365 daily buckets → "
                         "~23 days/subsample. More stable than S=8 for daily data.")
    ap.add_argument("--symbols", nargs="+", default=MAJORS,
                    help="Override symbol list")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    grid = trend_grid()
    print(f"Donchian55 trend CPCV — {len(args.symbols)} symbols × {len(grid)} configs")
    print(f"CPCV(N={args.n_folds}, k={args.k_test}) · PBO S={args.s_subsamples}")
    print(f"Symbols: {', '.join(args.symbols)}")

    bars_4h = max(300, args.bars // 4 + 50)
    b = BinanceClient()
    await b.start()
    try:
        all_results: list[SymbolResult] = []

        for sym in args.symbols:
            print(f"\n─── {sym} ───")
            t0 = time.time()
            ks_1h, ks_4h = await fetch_symbol(b, sym, args.bars + 50, bars_4h)
            print(f"  fetched 1h={len(ks_1h)} 4h={len(ks_4h)} ({time.time()-t0:.1f}s)")
            result = validate_symbol(
                sym, ks_1h, ks_4h, grid,
                n_folds=args.n_folds, k_test=args.k_test,
                s_subsamples=args.s_subsamples, bars=args.bars,
            )
            all_results.append(result)
            print(f"  IS-best: {result.is_best_label}  "
                  f"IS={result.is_best_sharpe:+.3f}  "
                  f"OOS={result.is_best_oos_mean:+.3f}  "
                  f"PBO={result.pbo:.3f}  → {result.decision}")

        # ── Summary ──
        n_pass = sum(1 for r in all_results if "PASS" in r.decision)
        print(f"\n{'='*80}")
        print(f"TREND-FOLLOWING CPCV SUMMARY  ·  {args.bars//24}d  ·  "
              f"Donchian55 + ADX/Chop gate + Chandelier")
        print(f"{'='*80}")
        print(f"Symbols validated: {len(all_results)}")
        print(f"PASS:              {n_pass}/{len(all_results)}")
        print()
        print(f"{'Symbol':<14}  {'IS-best config':<30}  {'IS SR':>7}  "
              f"{'OOS SR':>7}  {'PBO':>6}  {'Decision'}")
        print(f"{'-'*14}  {'-'*30}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*20}")
        for r in all_results:
            print(f"{r.symbol:<14}  {r.is_best_label:<30}  "
                  f"{r.is_best_sharpe:>+7.3f}  "
                  f"{r.is_best_oos_mean:>+7.3f}  "
                  f"{r.pbo:>6.3f}  {r.decision}")

        # JSON dump
        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.out_tag}" if args.out_tag else ""
        out_path = OUT_DIR / f"cpcv_trend_{ts_tag}{tag}.json"
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategy": "donchian55_trend",
            "bars": args.bars,
            "symbols": args.symbols,
            "n_folds": args.n_folds,
            "k_test": args.k_test,
            "s_subsamples": args.s_subsamples,
            "n_pass": n_pass,
            "results": [asdict(r) for r in all_results],
        }, indent=2, default=str))
        print(f"\nwrote {out_path}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
