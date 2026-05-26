"""Diagnose where joint sim's PnL comes from — his-matched trades vs
scanner+detector-only trades."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.config.settings import get_settings
from src.scanners.aktradescalp_scanner import ScannerParams, UniverseParams
from src.services.backtest import CascadeBacktestParams, simulate_cascade_breakout
from scripts.joint_sim import (
    load_15m_histories,
    load_1h_universe,
    precompute_scanner_approvals,
    snap_to_m15_close_times,
)


REPO = Path(__file__).resolve().parent.parent
CALLS_PATH = REPO / "data/research/aktradescalp/aktradescalp_calls.json"


def main():
    load_dotenv()
    print("loading data + computing scanner approvals (rank=10, score_min=1.0)...")
    histories_1h, _ = load_1h_universe()
    histories_15m = load_15m_histories()
    his_symbols = set(histories_15m.keys())

    s = ScannerParams()
    s_dict = {k: getattr(s, k) for k in s.__dataclass_fields__}
    s_dict["score_min"] = 1.0
    s_relaxed = ScannerParams(**s_dict)
    approvals = precompute_scanner_approvals(
        histories_1h, his_symbols, UniverseParams(), s_relaxed, rank_cutoff=10)
    approvals = snap_to_m15_close_times(approvals, histories_15m)

    # His call timestamps (within ±2 bars of each = "near" his call)
    calls = json.load(open(CALLS_PATH))
    calls = [c for c in calls if c["side"] in ("long", "short")]
    his_ts_by_sym: dict[str, list[tuple[int, str]]] = {}
    for c in calls:
        sym = c["symbol"]
        ts = int(datetime.fromisoformat(c["dt_iso"]).timestamp() * 1000)
        his_ts_by_sym.setdefault(sym, []).append((ts, c["side"]))

    NEAR_MS = 2 * 15 * 60 * 1000  # ±2 M15 bars

    p = CascadeBacktestParams(cost_bps_override=15.0)
    print(f"\nrunning joint sim across {len(histories_15m)} symbols...")
    matched_pnls = []
    unmatched_pnls = []
    for sym, ks in histories_15m.items():
        approved = approvals.get(sym, set())
        stats, trades = simulate_cascade_breakout(
            sym, ks, params=p, approved_timestamps=approved)
        his_list = his_ts_by_sym.get(sym, [])
        for t in trades:
            # Find nearest his-call for this symbol+side
            best = None
            for his_ts, his_side in his_list:
                if his_side != t.side:
                    continue
                d = abs((t.entry_ts_ms or 0) - his_ts)
                if best is None or d < best:
                    best = d
            if best is not None and best <= NEAR_MS:
                matched_pnls.append((sym, t.side, t.pnl_usd or 0, best))
            else:
                nearest_any_side = None
                for his_ts, his_side in his_list:
                    d = abs((t.entry_ts_ms or 0) - his_ts)
                    if nearest_any_side is None or d < nearest_any_side:
                        nearest_any_side = d
                unmatched_pnls.append((sym, t.side, t.pnl_usd or 0,
                                       nearest_any_side))

    print(f"\n=== JOINT-SIM trades split by his-call proximity ===")
    print(f"  matched (within ±2 bars, same side):  {len(matched_pnls)}")
    print(f"  unmatched:                            {len(unmatched_pnls)}")

    if matched_pnls:
        m_pnls = [p for _, _, p, _ in matched_pnls]
        m_total = sum(m_pnls)
        m_avg = m_total / len(m_pnls)
        m_wr = sum(1 for p in m_pnls if p > 0) / len(m_pnls)
        print(f"\n  MATCHED group:")
        print(f"    n trades:     {len(m_pnls)}")
        print(f"    win rate:     {m_wr:.1%}")
        print(f"    total P&L:    ${m_total:+.2f}")
        print(f"    avg/trade:    ${m_avg:+.3f}")

    if unmatched_pnls:
        u_pnls = [p for _, _, p, _ in unmatched_pnls]
        u_total = sum(u_pnls)
        u_avg = u_total / len(u_pnls)
        u_wr = sum(1 for p in u_pnls if p > 0) / len(u_pnls)
        print(f"\n  UNMATCHED group:")
        print(f"    n trades:     {len(u_pnls)}")
        print(f"    win rate:     {u_wr:.1%}")
        print(f"    total P&L:    ${u_total:+.2f}")
        print(f"    avg/trade:    ${u_avg:+.3f}")

    print(f"\n  -> matched trades carry  ${m_total:+.2f} / ${m_total + u_total:+.2f}"
          f" = {100 * m_total / (m_total + u_total):.0f}% of joint PnL"
          if matched_pnls and unmatched_pnls else "")

    print(f"\n=== Per-trade detail ===")
    print(f"\nMATCHED (his_call ±2 bars):")
    for sym, side, pnl, dist_ms in sorted(matched_pnls, key=lambda x: x[2], reverse=True):
        bars = dist_ms / (15 * 60 * 1000)
        print(f"  {sym:12s} {side:5s}  PnL=${pnl:+7.2f}  dist={bars:+.1f} M15 bars")
    print(f"\nUNMATCHED:")
    for sym, side, pnl, dist_ms in sorted(unmatched_pnls, key=lambda x: x[2], reverse=True):
        bars = (dist_ms / (15 * 60 * 1000)) if dist_ms else float("inf")
        print(f"  {sym:12s} {side:5s}  PnL=${pnl:+7.2f}  nearest_his={bars:.0f} M15 bars away")


if __name__ == "__main__":
    main()
