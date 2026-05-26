"""Joint-system simulator: scanner + detector + execution rules.

The production setup: the detector fires only when the cross-sectional
scanner has approved the symbol+hour. This is what we'd actually run
live, and the answer it produces is what determines whether the v2
strategy has real edge once selection isn't perfect (corpus replay) or
absent (detector-only).

Pipeline:
  1. Pre-compute scanner approvals at every HOUR in 07-12 UTC over the
     validation window, using cached 1h klines + funding + OI from
     scanner_cache/.
  2. For each symbol, expand approved hours → set of M15 close_times.
  3. Run simulate_cascade_breakout per symbol with approved_timestamps
     set — the simulator skips bars outside the approval.
  4. Aggregate. Compare to the corpus replay (upper bound) and
     detector-only (lower bound) from cascade_validate.

Usage:
  .venv/bin/python -m scripts.joint_sim
  .venv/bin/python -m scripts.joint_sim --rank-cutoff 10
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.config.settings import get_settings
from src.models.types import Kline
from src.scanners.aktradescalp_scanner import (
    ScannerParams,
    SymbolHistory,
    UniverseParams,
    compute_features,
    score_universe,
)
from src.services.backtest import (
    BacktestStats,
    CascadeBacktestParams,
    SimTrade,
    _stats_from_trades,
    format_stats,
    simulate_cascade_breakout,
)


REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data/research/aktradescalp/scanner_cache"
CALLS_PATH = REPO / "data/research/aktradescalp/aktradescalp_calls.json"

SESSION_START_UTC = 7
SESSION_END_UTC = 12             # inclusive
HOUR_MS = 60 * 60 * 1000
M15_MS = 15 * 60 * 1000


def _row_to_kline_1h(r, symbol: str) -> Kline:
    return Kline(
        symbol=symbol, timeframe="1h",
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    )


def _row_to_kline_15m(r, symbol: str) -> Kline:
    return Kline(
        symbol=symbol, timeframe="15m",
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    )


def load_1h_universe() -> tuple[dict[str, SymbolHistory], dict[str, int]]:
    """Load 1h klines + funding + OI for all symbols in scanner_cache.
    Returns (histories, onboards)."""
    onboards: dict[str, int] = {}
    info_path = CACHE / "exchange_info.json"
    if info_path.exists():
        onboards = {k: int(v) for k, v in json.loads(info_path.read_text()).items()}

    histories: dict[str, SymbolHistory] = {}
    for kp in sorted(CACHE.glob("klines_1h_*.json")):
        sym = kp.stem.replace("klines_1h_", "")
        rows = json.loads(kp.read_text())
        if len(rows) < 30 * 24:
            continue
        klines = [_row_to_kline_1h(r, sym) for r in rows]
        fr = (json.loads((CACHE / f"funding_{sym}.json").read_text())
              if (CACHE / f"funding_{sym}.json").exists() else [])
        oi = (json.loads((CACHE / f"oi_{sym}.json").read_text())
              if (CACHE / f"oi_{sym}.json").exists() else [])
        histories[sym] = SymbolHistory(
            symbol=sym, klines_1h=klines,
            funding_rates=[(int(r["fundingTime"]), float(r["fundingRate"])) for r in fr],
            oi_history=[(int(r["timestamp"]), float(r["sumOpenInterest"])) for r in oi],
            listing_date_ms=onboards.get(sym, 0),
        )
    return histories, onboards


def load_15m_histories() -> dict[str, list[Kline]]:
    out: dict[str, list[Kline]] = {}
    for p in sorted(CACHE.glob("klines_15m_*.json")):
        sym = p.stem.replace("klines_15m_", "")
        rows = json.loads(p.read_text())
        if len(rows) >= 100:
            out[sym] = [_row_to_kline_15m(r, sym) for r in rows]
    return out


def precompute_scanner_approvals(
    histories_1h: dict[str, SymbolHistory],
    his_symbols: set[str],
    universe: UniverseParams,
    scanner: ScannerParams,
    rank_cutoff: int = 5,
) -> dict[str, set[int]]:
    """At each HOUR in 07-12 UTC across the data window, compute scanner
    output and record which symbols passed (rank ≤ rank_cutoff).

    Returns {symbol: set[m15_close_time_ms]} — the symbol is approved for
    any M15 bar whose close-time falls in an approved hour."""
    # Find the time range of the data
    all_times = []
    for h in histories_1h.values():
        if h.klines_1h:
            all_times.append(h.klines_1h[0].close_time)
            all_times.append(h.klines_1h[-1].close_time)
    if not all_times:
        return {}
    min_ts = min(all_times) + 30 * 86_400_000     # allow 30d for baselines
    max_ts = max(all_times)

    # Walk hour by hour
    cur = (min_ts // HOUR_MS) * HOUR_MS
    end = (max_ts // HOUR_MS) * HOUR_MS
    approvals_by_symbol: dict[str, set[int]] = defaultdict(set)
    hours_checked = 0
    hours_with_candidates = 0
    total_candidates_emitted = 0

    while cur <= end:
        h = datetime.fromtimestamp(cur / 1000, tz=timezone.utc).hour
        if SESSION_START_UTC <= h <= SESSION_END_UTC:
            hours_checked += 1
            features = {sym: compute_features(hist, cur)
                        for sym, hist in histories_1h.items()}
            ranked = score_universe(features, cur, universe, scanner)
            if ranked:
                hours_with_candidates += 1
                # Approved set: top-N by score (default 5)
                approved_syms = {c.symbol for c in ranked[:rank_cutoff]}
                total_candidates_emitted += len(approved_syms)
                # Mark all 4 M15 close_times in this hour as approved for each sym
                for sym in approved_syms:
                    if sym not in his_symbols:
                        continue
                    for off in range(4):
                        m15_close = cur + (off + 1) * M15_MS - 1
                        # Binance M15 close_time format: e.g. for hour H,
                        # the bars close at H+15min-1, H+30min-1, etc.
                        # Snap to actual M15 boundary:
                        approvals_by_symbol[sym].add(m15_close)
        cur += HOUR_MS
    print(f"  hours checked:           {hours_checked}")
    print(f"  hours with candidates:   {hours_with_candidates}  "
          f"({100*hours_with_candidates/hours_checked:.1f}%)")
    print(f"  total candidates emitted: {total_candidates_emitted}")
    return dict(approvals_by_symbol)


def snap_to_m15_close_times(approvals: dict[str, set[int]],
                             histories_15m: dict[str, list[Kline]]
                             ) -> dict[str, set[int]]:
    """The hour-start arithmetic above only approximates M15 close_times.
    Snap each approved-hour to the ACTUAL 4 M15 close_times within it."""
    out: dict[str, set[int]] = {}
    for sym, fake_times in approvals.items():
        ks = histories_15m.get(sym, [])
        if not ks:
            continue
        # Build a quick set of all M15 close_times for fast lookup
        m15_ct = [k.close_time for k in ks]
        # For each fake time, find the nearest M15 close_time in same hour
        hour_to_m15s: dict[int, list[int]] = defaultdict(list)
        for ct in m15_ct:
            hour_start = (ct // HOUR_MS) * HOUR_MS
            hour_to_m15s[hour_start].append(ct)
        snapped: set[int] = set()
        for ft in fake_times:
            hour_start = (ft // HOUR_MS) * HOUR_MS
            snapped.update(hour_to_m15s.get(hour_start, []))
        out[sym] = snapped
    return out


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-cutoff", type=int, default=5,
                        help="Approve top-N candidates per hour (default 5)")
    parser.add_argument("--score-min", type=float, default=None,
                        help="Override ScannerParams.score_min")
    args = parser.parse_args()

    print("loading 1h universe + funding + OI from cache...")
    t0 = time.time()
    histories_1h, _ = load_1h_universe()
    print(f"  {len(histories_1h)} symbols loaded in {time.time()-t0:.1f}s")

    print("loading M15 histories...")
    t0 = time.time()
    histories_15m = load_15m_histories()
    print(f"  {len(histories_15m)} symbols loaded in {time.time()-t0:.1f}s")

    his_symbols = set(histories_15m.keys())  # we only simulate his symbols

    u = UniverseParams()
    s = ScannerParams()
    if args.score_min is not None:
        s = ScannerParams(**{**{k: getattr(s, k) for k in s.__dataclass_fields__},
                              "score_min": args.score_min})

    print(f"\nprecomputing scanner approvals (rank ≤ {args.rank_cutoff})...")
    t0 = time.time()
    approvals = precompute_scanner_approvals(
        histories_1h, his_symbols, u, s, args.rank_cutoff)
    approvals = snap_to_m15_close_times(approvals, histories_15m)
    n_approved_bars = sum(len(v) for v in approvals.values())
    print(f"  precompute: {time.time()-t0:.1f}s")
    print(f"  symbols with any approval: {sum(1 for v in approvals.values() if v)}"
          f" / {len(his_symbols)}")
    print(f"  total approved M15 bars:   {n_approved_bars}")

    # ─── Run the joint simulator per symbol ───
    print(f"\nrunning joint simulator across {len(histories_15m)} symbols...")
    p = CascadeBacktestParams(cost_bps_override=15.0)
    all_trades: list[SimTrade] = []
    total_pnl = 0.0
    span_start_ms: Optional[int] = None
    span_end_ms: Optional[int] = None
    for sym, ks in histories_15m.items():
        approved = approvals.get(sym, set())
        stats, trades = simulate_cascade_breakout(
            sym, ks, params=p, approved_timestamps=approved)
        all_trades.extend(trades)
        total_pnl += stats.total_pnl_usd
        if ks:
            ct_first = ks[0].close_time
            ct_last = ks[-1].close_time
            span_start_ms = (ct_first if span_start_ms is None
                              else min(span_start_ms, ct_first))
            span_end_ms = (ct_last if span_end_ms is None
                            else max(span_end_ms, ct_last))

    span_days = ((span_end_ms - span_start_ms) / 1000 / 86400
                 if span_start_ms and span_end_ms else 1.0)
    joint = _stats_from_trades("cascade_joint", all_trades,
                                get_settings().account_equity_usd, span_days)
    print(format_stats(joint))

    # ─── Compare to corpus replay + detector-only ───
    cmp_path = REPO / "data/research/aktradescalp/cascade_strategy_validation.json"
    if cmp_path.exists():
        cmp = json.loads(cmp_path.read_text())
        c = cmp["corpus_replay"]
        d = cmp["detector_only"]
        rb = cmp["random_baseline"]
        r_avg_pnl = sum(r["avg_pnl_usd"] for r in rb) / len(rb)
        r_wr = sum(r["win_rate"] for r in rb) / len(rb)
        r_trades = sum(r["trades"] for r in rb) // len(rb)

        print(f"\n{'='*68}\nCOMPARISON — joint vs corpus / random / detector-only\n{'='*68}")
        print(f"{'metric':22s}  {'joint':>10s}  {'corpus':>10s}  {'random':>10s}  "
              f"{'det-only':>10s}")
        print(f"{'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        print(f"{'trades':22s}  {joint.trades:>10d}  {c['trades']:>10d}"
              f"  {r_trades:>10d}  {d['trades']:>10d}")
        print(f"{'win rate':22s}  {joint.win_rate:>9.1%}  {c['win_rate']:>9.1%}"
              f"  {r_wr:>9.1%}  {d['win_rate']:>9.1%}")
        print(f"{'avg P&L per trade':22s}  {joint.avg_pnl_usd:>+10.3f}  "
              f"{c['avg_pnl_usd']:>+10.3f}  {r_avg_pnl:>+10.3f}  {d['avg_pnl']:>+10.3f}")
        print(f"{'total P&L':22s}  {joint.total_pnl_usd:>+10.2f}  "
              f"{c['total_pnl_usd']:>+10.2f}  ---  {d['total_pnl']:>+10.2f}")
        print(f"{'Sharpe':22s}  {joint.sharpe:>+10.2f}  {c['sharpe']:>+10.2f}  "
              f"---  ---")
        joint_vs_random_bps = ((joint.avg_pnl_usd - r_avg_pnl)
                                / get_settings().account_equity_usd * 10_000)
        print(f"\nedge vs random:  {joint.avg_pnl_usd - r_avg_pnl:+.3f}$/trade  "
              f"= {joint_vs_random_bps:+.1f} bps on equity")

    out = REPO / "data/research/aktradescalp/joint_sim_validation.json"
    out.write_text(json.dumps({
        "rank_cutoff": args.rank_cutoff,
        "approved_bars_total": n_approved_bars,
        "joint": {
            "trades": joint.trades, "wins": joint.wins, "losses": joint.losses,
            "win_rate": joint.win_rate, "total_pnl_usd": joint.total_pnl_usd,
            "avg_pnl_usd": joint.avg_pnl_usd, "sharpe": joint.sharpe,
            "deflated_sharpe": joint.deflated_sharpe,
            "max_drawdown_pct": joint.max_drawdown_pct,
            "annualized_pct": joint.annualized_pct,
        },
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
