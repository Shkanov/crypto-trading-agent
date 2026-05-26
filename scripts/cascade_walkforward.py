"""Walk-forward stability check for the cascade-breakout v2 strategy.

Splits the ~80d corpus window (Apr 3 – May 22 2026) into 3 non-
overlapping sub-windows of roughly equal calendar length. For each:
runs (a) corpus replay restricted to his calls in that window, and
(b) joint sim with scanner approvals and detector entries restricted
to that window. Reports per-window stats.

Pass criterion: Sharpe > 1 in EACH window for the joint sim — if the
edge collapses in one window, the +37% annualized is regime-specific,
not robust.

Usage:
  .venv/bin/python -m scripts.cascade_walkforward
"""
from __future__ import annotations

import json
from collections import defaultdict
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
    ("W1", "2026-04-03", "2026-04-21"),
    ("W2", "2026-04-22", "2026-05-11"),
    ("W3", "2026-05-12", "2026-05-26"),
]


def _to_ms(date_str: str) -> int:
    return int(datetime.fromisoformat(date_str + "T00:00:00+00:00").timestamp() * 1000)


def _filter_ks(ks: list[Kline], start_ms: int, end_ms: int) -> list[Kline]:
    return [k for k in ks if start_ms <= k.close_time < end_ms]


def _filter_approvals(approvals: dict[str, set[int]],
                       start_ms: int, end_ms: int) -> dict[str, set[int]]:
    return {sym: {ts for ts in v if start_ms <= ts < end_ms}
            for sym, v in approvals.items()}


def _aggregate(stats_list: list[BacktestStats]) -> BacktestStats:
    """Combine multiple BacktestStats into one (weighted sums)."""
    out = BacktestStats(strategy="aggregate")
    if not stats_list:
        return out
    out.trades = sum(s.trades for s in stats_list)
    out.wins = sum(s.wins for s in stats_list)
    out.losses = sum(s.losses for s in stats_list)
    out.total_pnl_usd = sum(s.total_pnl_usd for s in stats_list)
    if out.trades > 0:
        out.avg_pnl_usd = out.total_pnl_usd / out.trades
        out.win_rate = out.wins / out.trades
    return out


def run_corpus_in_window(histories_15m: dict[str, list[Kline]],
                          calls_in_window: list[dict],
                          params: CascadeBacktestParams) -> BacktestStats:
    if not calls_in_window:
        return BacktestStats(strategy="corpus")
    stats, _ = simulate_entries(histories_15m, calls_in_window, params)
    return stats


def run_joint_in_window(histories_15m: dict[str, list[Kline]],
                         approvals_window: dict[str, set[int]],
                         start_ms: int, end_ms: int,
                         params: CascadeBacktestParams,
                         ) -> tuple[BacktestStats, int]:
    """Run joint sim per symbol, but only over bars in [start_ms, end_ms)."""
    all_trades: list[SimTrade] = []
    span_days = (end_ms - start_ms) / 1000 / 86400
    settings = get_settings()
    for sym, ks in histories_15m.items():
        ks_win = _filter_ks(ks, start_ms - 7 * 86_400_000, end_ms)
        # We keep a 7-day buffer before window start for warmup but only
        # accept entries inside [start_ms, end_ms). The simulator's
        # approved_timestamps filter handles the time-window gate.
        approved_in_window = approvals_window.get(sym, set())
        if not approved_in_window or len(ks_win) < 150:
            continue
        # Filter the approved set strictly to the window — already done by caller
        s, trades = simulate_cascade_breakout(
            sym, ks_win, params=params,
            approved_timestamps=approved_in_window,
        )
        all_trades.extend(trades)
    return _stats_from_trades("joint_window", all_trades,
                                settings.account_equity_usd, span_days), len(all_trades)


def main() -> None:
    load_dotenv()
    print("loading 1h universe + funding + OI...")
    histories_1h, _ = load_1h_universe()
    print(f"  {len(histories_1h)} symbols\n")

    print("loading M15 histories...")
    histories_15m = load_15m_histories()
    his_symbols = set(histories_15m.keys())
    print(f"  {len(his_symbols)} symbols\n")

    print("precomputing scanner approvals (rank=10, score_min=1.0)...")
    s = ScannerParams()
    s_dict = {k: getattr(s, k) for k in s.__dataclass_fields__}
    s_dict["score_min"] = 1.0
    s_relaxed = ScannerParams(**s_dict)
    approvals = precompute_scanner_approvals(
        histories_1h, his_symbols, UniverseParams(), s_relaxed, rank_cutoff=10)
    approvals = snap_to_m15_close_times(approvals, histories_15m)
    print()

    calls = json.load(open(CALLS_PATH))
    calls = [c for c in calls if c["side"] in ("long", "short")]
    call_entries = [{
        "symbol": c["symbol"],
        "ts_ms": int(datetime.fromisoformat(c["dt_iso"]).timestamp() * 1000),
        "side": c["side"],
    } for c in calls]

    p_bt = CascadeBacktestParams(cost_bps_override=15.0)

    print(f"\n{'='*72}\nWALK-FORWARD ANALYSIS  (3 non-overlapping windows)\n{'='*72}")
    print(f"{'window':5s}  {'span':24s}  {'src':6s}  {'n':>3s}  {'WR':>6s}  "
          f"{'avg$':>7s}  {'total$':>8s}  {'Sharpe':>6s}")
    print(f"{'-'*5}  {'-'*24}  {'-'*6}  {'-'*3}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*6}")

    results = []
    for label, start_iso, end_iso in WINDOWS:
        start_ms = _to_ms(start_iso)
        end_ms = _to_ms(end_iso)

        # Corpus replay
        win_calls = [e for e in call_entries if start_ms <= e["ts_ms"] < end_ms]
        c_stats = run_corpus_in_window(histories_15m, win_calls, p_bt)
        print(f"{label:5s}  {start_iso} → {end_iso}  corpus  {c_stats.trades:>3d}  "
              f"{c_stats.win_rate:>5.1%}  {c_stats.avg_pnl_usd:>+7.2f}  "
              f"{c_stats.total_pnl_usd:>+8.2f}  {c_stats.sharpe:>+6.2f}")

        # Joint sim restricted to window
        win_approvals = _filter_approvals(approvals, start_ms, end_ms)
        j_stats, n_trades = run_joint_in_window(
            histories_15m, win_approvals, start_ms, end_ms, p_bt)
        print(f"{label:5s}  {start_iso} → {end_iso}  joint   {j_stats.trades:>3d}  "
              f"{j_stats.win_rate:>5.1%}  {j_stats.avg_pnl_usd:>+7.2f}  "
              f"{j_stats.total_pnl_usd:>+8.2f}  {j_stats.sharpe:>+6.2f}")

        results.append({
            "window": label, "start": start_iso, "end": end_iso,
            "corpus": {"trades": c_stats.trades, "wr": c_stats.win_rate,
                       "avg": c_stats.avg_pnl_usd, "total": c_stats.total_pnl_usd,
                       "sharpe": c_stats.sharpe},
            "joint": {"trades": j_stats.trades, "wr": j_stats.win_rate,
                      "avg": j_stats.avg_pnl_usd, "total": j_stats.total_pnl_usd,
                      "sharpe": j_stats.sharpe},
        })

    # Aggregates
    print()
    corpus_agg = _aggregate([
        run_corpus_in_window(histories_15m,
            [e for e in call_entries if _to_ms(w[1]) <= e["ts_ms"] < _to_ms(w[2])],
            p_bt) for w in WINDOWS
    ])
    print(f"{'agg':5s}  full corpus replay        corpus  {corpus_agg.trades:>3d}  "
          f"{corpus_agg.win_rate:>5.1%}  {corpus_agg.avg_pnl_usd:>+7.2f}  "
          f"{corpus_agg.total_pnl_usd:>+8.2f}  {'—':>6s}")

    # Verdict
    print(f"\n{'='*72}\nVERDICT\n{'='*72}")
    joint_sharpes = [r["joint"]["sharpe"] for r in results]
    corpus_sharpes = [r["corpus"]["sharpe"] for r in results]
    print(f"  joint Sharpe per window:   {[f'{s:+.2f}' for s in joint_sharpes]}")
    print(f"  corpus Sharpe per window:  {[f'{s:+.2f}' for s in corpus_sharpes]}")
    min_joint = min(joint_sharpes)
    min_corpus = min(corpus_sharpes)
    print(f"\n  min joint Sharpe:   {min_joint:+.2f}  "
          f"(gate: > +1.0 = {'PASS' if min_joint > 1.0 else 'FAIL'})")
    print(f"  min corpus Sharpe:  {min_corpus:+.2f}  "
          f"(gate: > +1.0 = {'PASS' if min_corpus > 1.0 else 'FAIL'})")

    if min_joint < 0 or min_corpus < 0:
        print(f"\n  ⚠ NEGATIVE Sharpe in at least one window — edge is regime-specific")
    elif min_joint < 1.0:
        print(f"\n  ⚠ Joint Sharpe < 1 in at least one window — sample-size or"
              f" regime sensitivity")
    else:
        print(f"\n  ✓ Strategy is consistently profitable across walk-forward windows")

    out = REPO / "data/research/aktradescalp/cascade_walkforward.json"
    out.write_text(json.dumps({"windows": results, "min_joint_sharpe": min_joint,
                                "min_corpus_sharpe": min_corpus}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
