"""Sweep joint-sim filter strictness: min_confluence × allowed_modes ×
scanner_score_min. Find the combination that maximizes Sharpe on the
hypothesis that tighter joint gates recover more 'matched-class'
quality (where matched joint trades made +$5.49/trade vs +$0.52 for
unmatched)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.config.settings import get_settings
from src.models.types import Kline
from src.scanners.aktradescalp_scanner import ScannerParams, UniverseParams
from src.services.backtest import (
    BacktestStats,
    CascadeBacktestParams,
    _stats_from_trades,
    simulate_cascade_breakout,
)
from scripts.joint_sim import (
    load_15m_histories,
    load_1h_universe,
    precompute_scanner_approvals,
    snap_to_m15_close_times,
)


REPO = Path(__file__).resolve().parent.parent


def run_joint(histories_15m, approvals, params):
    settings = get_settings()
    all_trades = []
    for sym, ks in histories_15m.items():
        _, trades = simulate_cascade_breakout(
            sym, ks, params=params,
            approved_timestamps=approvals.get(sym, set()))
        all_trades.extend(trades)
    span_days = ((max(t.entry_ts_ms for t in all_trades)
                   - min(t.entry_ts_ms for t in all_trades))
                  / 1000 / 86400 if all_trades else 1.0)
    return _stats_from_trades("joint", all_trades,
                                settings.account_equity_usd, span_days), all_trades


def main():
    load_dotenv()
    histories_1h, _ = load_1h_universe()
    histories_15m = load_15m_histories()
    his_symbols = set(histories_15m.keys())

    print(f"{'='*80}\n  JOINT-SIM STRICTNESS SWEEP\n{'='*80}")
    print(f"{'score':>5s} {'rank':>4s} {'minC':>4s} {'modes':24s} "
          f"{'n':>3s} {'WR':>6s} {'avg$':>7s} {'total$':>8s} {'Sharpe':>6s}  "
          f"{'defl':>5s}")

    base_scanner = ScannerParams()

    mode_configs = [
        ("all", None),
        ("cont+rev", frozenset({"continuation", "reversal"})),
        ("cont only", frozenset({"continuation"})),
        ("rev only", frozenset({"reversal"})),
    ]

    for score_min in (1.0, 1.5, 2.0):
        for rank in (5, 10):
            s_dict = {k: getattr(base_scanner, k)
                       for k in base_scanner.__dataclass_fields__}
            s_dict["score_min"] = score_min
            s = ScannerParams(**s_dict)
            approvals = precompute_scanner_approvals(
                histories_1h, his_symbols, UniverseParams(), s, rank_cutoff=rank)
            approvals = snap_to_m15_close_times(approvals, histories_15m)

            for min_conf in (2, 3, 4):
                for mode_label, allowed in mode_configs:
                    p = CascadeBacktestParams(
                        cost_bps_override=15.0,
                        min_confluence=min_conf,
                        allowed_modes=allowed,
                    )
                    stats, _ = run_joint(histories_15m, approvals, p)
                    print(f"{score_min:>5.1f} {rank:>4d} {min_conf:>4d} "
                          f"{mode_label:24s} "
                          f"{stats.trades:>3d} {stats.win_rate:>5.1%} "
                          f"{stats.avg_pnl_usd:>+7.3f} "
                          f"{stats.total_pnl_usd:>+8.2f} "
                          f"{stats.sharpe:>+6.2f}  {stats.deflated_sharpe:>+5.2f}")


if __name__ == "__main__":
    main()
