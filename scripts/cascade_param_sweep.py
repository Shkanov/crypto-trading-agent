"""Execution-parameter sweep for the cascade-breakout strategy (P3).

Sweep over (hard_time_stop_bars, tp1_r_multiple, stop_atr_mult,
trail_atr_mult) on corpus replay. Objective: maximize Sharpe subject to
≥20 trades AND positive PnL in each walk-forward window (else the
result is regime-specific overfit).

Methodology:
  1. For each parameter combination, run corpus replay (his 36 actual
     entries × our execution rules with these params).
  2. Compute aggregate Sharpe, total PnL, trade count, max DD.
  3. Also compute per-window PnL (W1, W2, W3) — gate-out combinations
     where any window has negative aggregate PnL.
  4. Print top-10 by aggregate Sharpe (filtered by gates).
  5. For the top-3 surviving combinations, re-run the joint sim with
     those params to verify the gains transfer.

Usage:
  .venv/bin/python -m scripts.cascade_param_sweep
"""
from __future__ import annotations

import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.config.settings import get_settings
from src.models.types import Kline
from src.scanners.aktradescalp_scanner import (
    ScannerParams,
    UniverseParams,
)
from src.services.backtest import (
    BacktestStats,
    CascadeBacktestParams,
    SimTrade,
    _stats_from_trades,
    simulate_cascade_breakout,
)
from scripts.cascade_validate import simulate_entries
from scripts.joint_sim import (
    load_15m_histories,
    load_1h_universe,
    precompute_scanner_approvals,
    snap_to_m15_close_times,
)


REPO = Path(__file__).resolve().parent.parent
CALLS_PATH = REPO / "data/research/aktradescalp/aktradescalp_calls.json"

WINDOWS = [
    ("W1", "2026-04-03T00:00:00+00:00", "2026-04-21T00:00:00+00:00"),
    ("W2", "2026-04-22T00:00:00+00:00", "2026-05-11T00:00:00+00:00"),
    ("W3", "2026-05-12T00:00:00+00:00", "2026-05-26T00:00:00+00:00"),
]


def main() -> None:
    load_dotenv()
    print("loading M15 histories...")
    t0 = time.time()
    histories_15m = load_15m_histories()
    print(f"  {len(histories_15m)} symbols in {time.time()-t0:.1f}s")

    calls = json.load(open(CALLS_PATH))
    calls = [c for c in calls if c["side"] in ("long", "short")]
    his_entries = [{
        "symbol": c["symbol"],
        "ts_ms": int(datetime.fromisoformat(c["dt_iso"]).timestamp() * 1000),
        "side": c["side"],
    } for c in calls]

    window_ms = [(label,
                  int(datetime.fromisoformat(s).timestamp() * 1000),
                  int(datetime.fromisoformat(e).timestamp() * 1000))
                 for label, s, e in WINDOWS]

    # Sweep grid
    grid = {
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "tp1_r_multiple": [1.0, 1.5, 2.0, 3.0],
        "hard_time_stop_bars": [24, 48, 96, 192],     # 6h, 12h, 24h, 48h
        "trail_atr_mult": [0.5, 1.0, 2.0],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\nsweeping {len(combos)} combinations on corpus replay...")

    results = []
    t0 = time.time()
    for i, combo in enumerate(combos):
        p_dict = dict(zip(keys, combo))
        p_bt = CascadeBacktestParams(cost_bps_override=15.0, **p_dict)
        stats, trades = simulate_entries(histories_15m, his_entries, p_bt)

        # Per-window PnL
        per_win = {}
        for label, s_ms, e_ms in window_ms:
            win_pnl = sum((t.pnl_usd or 0.0) for t in trades
                          if t.entry_ts_ms and s_ms <= t.entry_ts_ms < e_ms)
            win_n = sum(1 for t in trades
                        if t.entry_ts_ms and s_ms <= t.entry_ts_ms < e_ms)
            per_win[label] = {"pnl": win_pnl, "n": win_n}

        results.append({
            **p_dict,
            "trades": stats.trades, "wr": stats.win_rate,
            "avg_pnl": stats.avg_pnl_usd, "total_pnl": stats.total_pnl_usd,
            "sharpe": stats.sharpe, "deflated": stats.deflated_sharpe,
            "max_dd_pct": stats.max_drawdown_pct,
            "annualized": stats.annualized_pct,
            "w1_pnl": per_win["W1"]["pnl"], "w1_n": per_win["W1"]["n"],
            "w2_pnl": per_win["W2"]["pnl"], "w2_n": per_win["W2"]["n"],
            "w3_pnl": per_win["W3"]["pnl"], "w3_n": per_win["W3"]["n"],
            "min_window_pnl": min(per_win[w]["pnl"] for w in per_win),
        })
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(combos)} ({time.time()-t0:.1f}s)")

    print(f"  sweep complete: {time.time()-t0:.1f}s")

    # Filter: ≥20 trades AND positive PnL in EACH window
    survivors = [r for r in results
                  if r["trades"] >= 20 and r["min_window_pnl"] > 0]
    print(f"\nresults: {len(results)} total, {len(survivors)} survive gates"
          f" (≥20 trades + positive each window)")

    survivors.sort(key=lambda r: r["sharpe"], reverse=True)

    print(f"\n=== TOP 10 (filtered, by Sharpe) ===")
    print(f"{'stop':>5s} {'tp1R':>5s} {'TSbars':>7s} {'trail':>6s}  "
          f"{'n':>3s} {'WR':>6s}  {'avg$':>7s}  {'total':>7s}  {'Sharpe':>6s}  "
          f"{'defl':>5s}  {'W1$':>6s} {'W2$':>6s} {'W3$':>6s}")
    for r in survivors[:10]:
        print(f"{r['stop_atr_mult']:>5.1f} {r['tp1_r_multiple']:>5.1f} "
              f"{r['hard_time_stop_bars']:>7d} {r['trail_atr_mult']:>6.1f}  "
              f"{r['trades']:>3d} {r['wr']:>5.1%}  {r['avg_pnl']:>+7.2f}  "
              f"{r['total_pnl']:>+7.2f}  {r['sharpe']:>+6.2f}  "
              f"{r['deflated']:>+5.2f}  "
              f"{r['w1_pnl']:>+6.2f} {r['w2_pnl']:>+6.2f} {r['w3_pnl']:>+6.2f}")

    # ─── Validate top-3 on joint sim ───
    if not survivors:
        print("\n  ⚠ no parameter combination survives the walk-forward gate")
    else:
        print(f"\n=== JOINT-SIM VALIDATION (top 3) ===")
        print("  precomputing scanner approvals (rank=10, score_min=1.0)...")
        histories_1h, _ = load_1h_universe()
        his_symbols = set(histories_15m.keys())
        s_dict = {k: getattr(ScannerParams(), k)
                  for k in ScannerParams().__dataclass_fields__}
        s_dict["score_min"] = 1.0
        s_relaxed = ScannerParams(**s_dict)
        approvals = precompute_scanner_approvals(
            histories_1h, his_symbols, UniverseParams(), s_relaxed, rank_cutoff=10)
        approvals = snap_to_m15_close_times(approvals, histories_15m)

        for rk, r in enumerate(survivors[:3]):
            p_dict = {k: r[k] for k in keys}
            p_bt = CascadeBacktestParams(cost_bps_override=15.0, **p_dict)
            all_trades = []
            for sym, ks in histories_15m.items():
                _, trades = simulate_cascade_breakout(
                    sym, ks, params=p_bt,
                    approved_timestamps=approvals.get(sym, set()))
                all_trades.extend(trades)
            span_days = (max(t.entry_ts_ms for t in all_trades)
                          - min(t.entry_ts_ms for t in all_trades)
                          if all_trades else 1) / 1000 / 86400
            j_stats = _stats_from_trades("joint", all_trades,
                                          get_settings().account_equity_usd,
                                          span_days)
            print(f"\n  rank {rk+1}:  stop={r['stop_atr_mult']}, tp1R={r['tp1_r_multiple']}, "
                  f"TS={r['hard_time_stop_bars']}, trail={r['trail_atr_mult']}")
            print(f"    corpus:  n={r['trades']}  Sharpe={r['sharpe']:+.2f}  "
                  f"avg=${r['avg_pnl']:+.2f}  total=${r['total_pnl']:+.2f}")
            print(f"    joint:   n={j_stats.trades}  Sharpe={j_stats.sharpe:+.2f}  "
                  f"avg=${j_stats.avg_pnl_usd:+.2f}  total=${j_stats.total_pnl_usd:+.2f}")

    out = REPO / "data/research/aktradescalp/cascade_param_sweep.json"
    out.write_text(json.dumps({
        "n_combinations": len(results),
        "n_survivors": len(survivors),
        "top_survivors": survivors[:20],
        "all_results": results,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
