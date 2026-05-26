"""Integration validator for the cascade-breakout v2 strategy (M4).

Three tests, each on the same M15 kline cache his_symbols × ~80d range:

  1. CORPUS REPLAY    — open trades at HIS exact call timestamps+sides,
                        apply our execution rules. Answers: "is his
                        selection profitable when we plug in our entry +
                        stop + TP + scratch rules?"

  2. DETECTOR-ONLY    — run simulate_cascade_breakout per symbol with no
                        scanner filter. Establishes the detector's pure
                        signal quality.

  3. RANDOM BASELINE  — same number of entries as (1), random bars in the
                        07-12 UTC session window, same execution rules.
                        Lift over random is the edge claim.

Output: per-test BacktestStats + a comparison summary. Gate (M4 PASS):
corpus replay beats random by ≥+100 bps/trade NET of costs.

Usage:
  .venv/bin/python -m scripts.cascade_validate
"""
from __future__ import annotations

import asyncio
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.config.settings import get_settings
from src.models.types import Kline
from src.services.backtest import (
    BacktestStats,
    CascadeBacktestParams,
    SimTrade,
    _close_cascade_trade,
    _scratch_check,
    _stats_from_trades,
    _take_partial,
    format_stats,
    simulate_cascade_breakout,
)
from src.strategies.cascade_breakout import _atr


REPO = Path(__file__).resolve().parent.parent
CALLS_PATH = REPO / "data/research/aktradescalp/aktradescalp_calls.json"
CACHE = REPO / "data/research/aktradescalp/scanner_cache"

SESSION_START_UTC = 7
SESSION_END_UTC = 12


# ─────────────────── data loading ─────────────────────────────────────────


def _row_to_kline(r, symbol: str) -> Kline:
    return Kline(
        symbol=symbol, timeframe="15m",
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    )


def load_klines() -> dict[str, list[Kline]]:
    out: dict[str, list[Kline]] = {}
    for p in CACHE.glob("klines_15m_*.json"):
        sym = p.stem.replace("klines_15m_", "")
        rows = json.loads(p.read_text())
        if len(rows) >= 100:
            out[sym] = [_row_to_kline(r, sym) for r in rows]
    return out


# ─────────────── corpus / random-baseline simulator ───────────────────────


def _open_trade_at(
    symbol: str, ks: list[Kline], entry_idx: int, side: str,
    atr: float, params: CascadeBacktestParams, settings, fee_bps: float,
) -> Optional[SimTrade]:
    """Build a SimTrade as if a pattern fired at bar `entry_idx`. Stop is
    placed using a structural reference from the last 20 bars (most recent
    swing low for long, swing high for short)."""
    bar = ks[entry_idx]
    # Find structural reference in last 20 bars
    lookback = ks[max(0, entry_idx - 20): entry_idx]
    if not lookback:
        return None
    if side == "long":
        struct = min(b.low for b in lookback)
        stop = max(struct - params.stop_struct_buffer_atr * atr,
                   bar.close - params.stop_atr_mult * atr)
    else:
        struct = max(b.high for b in lookback)
        stop = min(struct + params.stop_struct_buffer_atr * atr,
                   bar.close + params.stop_atr_mult * atr)

    risk_per_unit = abs(bar.close - stop)
    if risk_per_unit <= 0:
        return None

    entry_slip = bar.close * (settings.slippage_bps / 10_000)
    entry_px = bar.close + entry_slip if side == "long" else bar.close - entry_slip
    tp1 = (entry_px + params.tp1_r_multiple * risk_per_unit) if side == "long" \
        else (entry_px - params.tp1_r_multiple * risk_per_unit)

    risk_pct = (params.risk_per_trade_pct or settings.risk_per_trade_pct) / 100.0
    risk_usd = settings.account_equity_usd * risk_pct
    qty = min(risk_usd / risk_per_unit, settings.max_notional_usd / entry_px)
    if qty <= 0:
        return None

    t = SimTrade(
        symbol=symbol, strategy="cascade_replay",
        side=side, qty=qty,
        entry_price=entry_px, stop=stop, tp=tp1,
        entry_ts_ms=bar.close_time,
    )
    t.phase = "pre_tp1"                  # type: ignore[attr-defined]
    t.tp1_price = tp1                    # type: ignore[attr-defined]
    t.partial_pnl = 0.0                  # type: ignore[attr-defined]
    t.partial_qty = 0.0                  # type: ignore[attr-defined]
    t.bars_held = 0                      # type: ignore[attr-defined]
    t.atr_at_entry = atr                 # type: ignore[attr-defined]
    t.trail_extreme = None               # type: ignore[attr-defined]
    t.entry_high = bar.high              # type: ignore[attr-defined]
    t.entry_low = bar.low                # type: ignore[attr-defined]
    return t


def _resolve_one_trade(
    t: SimTrade, ks: list[Kline], entry_idx: int,
    params: CascadeBacktestParams, settings, fee_bps: float,
) -> None:
    """Step through bars after entry until trade closes."""
    stop_slip = settings.paper_stop_slippage_bps / 10_000
    tp_slip = settings.paper_tp_slippage_bps / 10_000

    for j in range(entry_idx + 1, len(ks)):
        k = ks[j]
        t.bars_held += 1  # type: ignore[attr-defined]
        side = t.side
        phase = t.phase  # type: ignore[attr-defined]

        stop_hit = ((side == "long" and k.low <= t.stop)
                    or (side == "short" and k.high >= t.stop))

        if phase == "pre_tp1":
            tp1 = t.tp1_price  # type: ignore[attr-defined]
            tp1_hit = ((side == "long" and k.high >= tp1)
                       or (side == "short" and k.low <= tp1))
            if stop_hit:
                _close_cascade_trade(t, t.stop, "stop", k.close_time,
                                      stop_slip, fee_bps, full=True)
                return
            if tp1_hit:
                _take_partial(t, tp1, k.close_time, tp_slip, fee_bps,
                              params.tp1_exit_fraction)
                t.phase = "post_tp1"      # type: ignore[attr-defined]
                t.stop = t.entry_price
                t.trail_extreme = (       # type: ignore[attr-defined]
                    k.high if side == "long" else k.low)
                continue
            if t.bars_held >= params.scratch_bars and _scratch_check(t, k, params):  # type: ignore[attr-defined]
                _close_cascade_trade(t, k.close, "scratch", k.close_time,
                                      0.0, fee_bps, full=True)
                return
            if t.bars_held >= params.hard_time_stop_bars:                            # type: ignore[attr-defined]
                _close_cascade_trade(t, k.close, "time_stop", k.close_time,
                                      0.0, fee_bps, full=True)
                return

        elif phase == "post_tp1":
            if side == "long":
                t.trail_extreme = max(t.trail_extreme, k.high)                       # type: ignore[attr-defined]
                trail_stop = t.trail_extreme - params.trail_atr_mult * t.atr_at_entry  # type: ignore[attr-defined]
                t.stop = max(t.stop, trail_stop)
            else:
                t.trail_extreme = min(t.trail_extreme, k.low)                        # type: ignore[attr-defined]
                trail_stop = t.trail_extreme + params.trail_atr_mult * t.atr_at_entry  # type: ignore[attr-defined]
                t.stop = min(t.stop, trail_stop)

            stop_hit_post = ((side == "long" and k.low <= t.stop)
                              or (side == "short" and k.high >= t.stop))
            if stop_hit_post:
                _close_cascade_trade(t, t.stop, "trail_stop", k.close_time,
                                      stop_slip, fee_bps, full=False)
                return
            if t.bars_held >= params.hard_time_stop_bars:                            # type: ignore[attr-defined]
                _close_cascade_trade(t, k.close, "time_stop", k.close_time,
                                      0.0, fee_bps, full=False)
                return

    # Ran out of bars
    if t.pnl_usd is None:
        _close_cascade_trade(t, ks[-1].close, "eod", ks[-1].close_time,
                              0.0, fee_bps,
                              full=(t.phase == "pre_tp1"))  # type: ignore[attr-defined]


def simulate_entries(
    histories: dict[str, list[Kline]],
    entries: list[dict],
    params: CascadeBacktestParams,
) -> tuple[BacktestStats, list[SimTrade]]:
    """`entries` is a list of {symbol, ts_ms, side}. Opens a trade at each
    using our execution rules. Returns aggregate stats."""
    settings = get_settings()
    if params.cost_bps_override is not None:
        fee_bps = params.cost_bps_override
    else:
        fee_bps = settings.perps_taker_fee_bps + settings.slippage_bps

    trades: list[SimTrade] = []
    span_start_ms: Optional[int] = None
    span_end_ms: Optional[int] = None

    for e in entries:
        ks = histories.get(e["symbol"])
        if not ks:
            continue
        # find entry_idx
        entry_idx = None
        for i, k in enumerate(ks):
            if k.close_time > e["ts_ms"]:
                entry_idx = i
                break
        if entry_idx is None or entry_idx < 20 or entry_idx + 2 >= len(ks):
            continue
        atr = _atr(ks[: entry_idx + 1], 14)
        if atr is None or atr <= 0:
            continue
        t = _open_trade_at(e["symbol"], ks, entry_idx, e["side"], atr,
                            params, settings, fee_bps)
        if t is None:
            continue
        _resolve_one_trade(t, ks, entry_idx, params, settings, fee_bps)
        trades.append(t)
        ct = ks[entry_idx].close_time
        if span_start_ms is None or ct < span_start_ms:
            span_start_ms = ct
        if span_end_ms is None or ct > span_end_ms:
            span_end_ms = ct

    span_days = ((span_end_ms - span_start_ms) / 1000 / 86400
                 if span_start_ms and span_end_ms else 1.0)
    return _stats_from_trades("cascade_replay", trades,
                               settings.account_equity_usd, span_days), trades


def make_random_entries(
    histories: dict[str, list[Kline]],
    call_template: list[dict],
    rng: random.Random,
) -> list[dict]:
    """For each call in `call_template`, pick a random session-window bar in
    the same symbol at least 1 day away from the actual call. Same side."""
    out = []
    for c in call_template:
        ks = histories.get(c["symbol"])
        if not ks:
            continue
        avoid_lo = c["ts_ms"] - 86_400_000
        avoid_hi = c["ts_ms"] + 86_400_000
        eligible = []
        for i in range(20, len(ks) - 25):
            ct = ks[i].close_time
            if avoid_lo <= ct <= avoid_hi:
                continue
            h = datetime.fromtimestamp(ct / 1000, tz=timezone.utc).hour
            if SESSION_START_UTC <= h <= SESSION_END_UTC:
                eligible.append(i)
        if not eligible:
            continue
        idx = rng.choice(eligible)
        out.append({
            "symbol": c["symbol"], "ts_ms": ks[idx].close_time,
            "side": c["side"],
        })
    return out


# ─────────────────────────── main ──────────────────────────────────────────


def main() -> None:
    load_dotenv()
    histories = load_klines()
    print(f"loaded {len(histories)} symbols")

    calls = json.load(open(CALLS_PATH))
    his_entries = []
    for c in calls:
        if c["side"] not in ("long", "short"):
            continue
        his_entries.append({
            "symbol": c["symbol"],
            "ts_ms": int(datetime.fromisoformat(c["dt_iso"]).timestamp() * 1000),
            "side": c["side"],
        })
    print(f"loaded {len(his_entries)} his calls")

    # Cost model: 15 bps one-leg → 30 bps round-trip (research synthesis)
    p = CascadeBacktestParams(cost_bps_override=15.0)

    # ─── Test 1: corpus replay (his entries + our execution) ───
    print(f"\n{'='*60}\nTEST 1: CORPUS REPLAY  (his entries + our execution)\n{'='*60}")
    s1, trades1 = simulate_entries(histories, his_entries, p)
    print(format_stats(s1))
    if trades1:
        avg_bps = (s1.total_pnl_usd / get_settings().account_equity_usd
                   / (0.01 * (p.risk_per_trade_pct
                              or get_settings().risk_per_trade_pct))
                   / s1.trades * 10_000)
        print(f"  per-trade in bps on risk:  {avg_bps:+.1f}")

    # ─── Test 2: random baseline ───
    print(f"\n{'='*60}\nTEST 2: RANDOM BASELINE  (random session bars)\n{'='*60}")
    rng = random.Random(42)
    n_seeds = 5
    rand_results = []
    for seed in range(n_seeds):
        rng_seed = random.Random(seed)
        rand_entries = make_random_entries(histories, his_entries, rng_seed)
        s, _ = simulate_entries(histories, rand_entries, p)
        rand_results.append(s)
    avg_pnl_per_trade = sum(r.avg_pnl_usd for r in rand_results) / n_seeds
    avg_wr = sum(r.win_rate for r in rand_results) / n_seeds
    avg_total = sum(r.total_pnl_usd for r in rand_results) / n_seeds
    print(f"  averaged over {n_seeds} seeds:")
    print(f"    trades:      {sum(r.trades for r in rand_results) // n_seeds}")
    print(f"    win rate:    {avg_wr:.1%}")
    print(f"    total P&L:   ${avg_total:+.2f}")
    print(f"    avg P&L/tr:  ${avg_pnl_per_trade:+.3f}")

    # ─── Test 3: detector-only across all symbols ───
    print(f"\n{'='*60}\nTEST 3: DETECTOR-ONLY  (no scanner filter)\n{'='*60}")
    det_total_pnl = 0.0
    det_trades = 0
    det_wins = 0
    for sym, ks in histories.items():
        s, _ = simulate_cascade_breakout(sym, ks, params=p)
        det_total_pnl += s.total_pnl_usd
        det_trades += s.trades
        det_wins += s.wins
    det_wr = det_wins / det_trades if det_trades else 0.0
    det_avg = det_total_pnl / det_trades if det_trades else 0.0
    print(f"  symbols simulated:  {len(histories)}")
    print(f"  total trades:       {det_trades}")
    print(f"  win rate:           {det_wr:.1%}")
    print(f"  total P&L:          ${det_total_pnl:+.2f}")
    print(f"  avg P&L/trade:      ${det_avg:+.3f}")

    # ─── Comparison ───
    print(f"\n{'='*60}\nCOMPARISON\n{'='*60}")
    print(f"{'metric':30s}  {'corpus':>10s}  {'random':>10s}  {'detector':>10s}")
    print(f"{'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"{'trades':30s}  {s1.trades:>10d}  "
          f"{sum(r.trades for r in rand_results)//n_seeds:>10d}  {det_trades:>10d}")
    print(f"{'win rate':30s}  {s1.win_rate:>9.1%}  {avg_wr:>9.1%}  {det_wr:>9.1%}")
    print(f"{'avg P&L per trade ($)':30s}  {s1.avg_pnl_usd:>+10.3f}  "
          f"{avg_pnl_per_trade:>+10.3f}  {det_avg:>+10.3f}")
    print(f"{'total P&L ($)':30s}  {s1.total_pnl_usd:>+10.2f}  "
          f"{avg_total:>+10.2f}  {det_total_pnl:>+10.2f}")
    edge_vs_random_bps = ((s1.avg_pnl_usd - avg_pnl_per_trade)
                          / get_settings().account_equity_usd * 10_000)
    print(f"\nedge vs random:  {s1.avg_pnl_usd - avg_pnl_per_trade:+.3f} $/trade"
          f"  =  {edge_vs_random_bps:+.1f} bps on equity")
    print(f"GATE (≥+100 bps lift):  "
          f"{'PASS' if edge_vs_random_bps >= 100 else 'FAIL'}")

    out = REPO / "data/research/aktradescalp/cascade_strategy_validation.json"
    out.write_text(json.dumps({
        "corpus_replay": _stats_to_dict(s1),
        "random_baseline": [_stats_to_dict(r) for r in rand_results],
        "detector_only": {"trades": det_trades, "wins": det_wins,
                          "win_rate": det_wr, "total_pnl": det_total_pnl,
                          "avg_pnl": det_avg},
        "edge_vs_random_bps_on_equity": edge_vs_random_bps,
    }, indent=2))
    print(f"\nwrote {out}")


def _stats_to_dict(s: BacktestStats) -> dict:
    return {
        "strategy": s.strategy, "trades": s.trades, "wins": s.wins, "losses": s.losses,
        "total_pnl_usd": s.total_pnl_usd, "avg_pnl_usd": s.avg_pnl_usd,
        "win_rate": s.win_rate, "sharpe": s.sharpe, "deflated_sharpe": s.deflated_sharpe,
        "max_drawdown_pct": s.max_drawdown_pct, "annualized_pct": s.annualized_pct,
    }


if __name__ == "__main__":
    main()
