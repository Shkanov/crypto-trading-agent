"""Backtest driver for cointegrated-pairs stat-arb (sprint item #5).

Pipeline per pair (A, B):
  1. Fetch 1h klines for A and B (Binance spot, paginated).
  2. Align by close_time → log(A_t), log(B_t).
  3. Weekly Engle-Granger refit on the trailing `lookback_bars`:
       log(B) = α + β · log(A) + ε      ;  test ε for stationarity (ADF).
  4. On every closed bar, call `evaluate_pair(...)`:
       z = (ε_t − μ̂) / σ̂  using THE LATEST FIT (no lookahead beyond it).
       |z| ≥ z_entry  → enter dollar-hedged: notional $L on B-leg, β·$L on A-leg.
       |z| < z_exit   → close.
       |z| > z_stop   → stop.
       refit fails    → stop.
  5. Per-trade PnL uses the realistic cost model from `src.services.costs`:
       per-leg taker fee + Almgren-Chriss sqrt-impact + half-spread.

Outputs JSON to data/research/strategy_tuning/pairs_<ts>.json, prints a summary
table and the per-pair refit log.

Usage:
  .venv/bin/python -m scripts.backtest_pairs
  .venv/bin/python -m scripts.backtest_pairs --pairs ETHUSDT:BTCUSDT,SOLUSDT:ETHUSDT
  .venv/bin/python -m scripts.backtest_pairs --bars 8760 --notional 1000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from src.config.settings import get_settings
from src.services.backtest import (
    _deflated_sharpe,
    _max_drawdown,
    _equity_curve,
    _sharpe_from_pnls_and_span,
)
from src.services.costs import (
    Costs,
    adjust_entry_price,
    adjust_exit_price,
    impact_k_for_symbol,
    taker_fee_usd,
)
from src.strategies.pairs_cointegration import (
    PairsParams,
    evaluate_pair,
    fit_engle_granger,
)
from src.tools.binance_client import BinanceClient


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data/research/strategy_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PAIRS = "ETHUSDT:BTCUSDT,SOLUSDT:ETHUSDT"


@dataclass
class PairTrade:
    """One cointegration-cycle trade: entry → exit/stop on both legs."""
    side: str                    # "long_a_short_b" | "long_b_short_a"
    entry_ts_ms: int
    exit_ts_ms: Optional[int] = None
    entry_z: float = 0.0
    exit_z: float = 0.0
    a_entry: float = 0.0
    b_entry: float = 0.0
    a_exit: float = 0.0
    b_exit: float = 0.0
    beta: float = 1.0
    notional_a: float = 0.0
    notional_b: float = 0.0
    exit_reason: str = ""
    pnl_usd: Optional[float] = None
    bars_held: int = 0


@dataclass
class PairStats:
    pair: str                    # e.g. "ETHUSDT/BTCUSDT"
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_usd: float = 0.0
    avg_pnl_usd: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    deflated_sharpe: float = 0.0
    max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    annualized_pct: float = 0.0
    starting_equity_usd: float = 0.0
    ending_equity_usd: float = 0.0
    refits_total: int = 0
    refits_cointegrated: int = 0
    avg_adf_p: float = 0.0
    avg_bars_held: float = 0.0


# ---------------------------------------------------------------------------
# Data loading

@dataclass
class AlignedSeries:
    close_times: np.ndarray       # int ms
    closes_a: np.ndarray
    closes_b: np.ndarray
    qv_a: np.ndarray              # quote volume per bar, A
    qv_b: np.ndarray              # quote volume per bar, B


async def fetch_aligned_pair(
    b: BinanceClient, sym_a: str, sym_b: str, bars: int, interval: str = "1h",
) -> AlignedSeries:
    """Fetch both legs and inner-join on close_time."""
    raw_a, raw_b = await asyncio.gather(
        b.fetch_klines_paginated(sym_a, interval, total=bars, market="spot"),
        b.fetch_klines_paginated(sym_b, interval, total=bars, market="spot"),
    )

    def to_map(raw: list[list]) -> dict[int, tuple[float, float]]:
        # close_time → (close, quote_volume)
        return {int(r[6]): (float(r[4]), float(r[7])) for r in raw}

    map_a, map_b = to_map(raw_a), to_map(raw_b)
    common = sorted(set(map_a) & set(map_b))
    if not common:
        raise RuntimeError(f"no overlapping closes between {sym_a} and {sym_b}")
    cts = np.asarray(common, dtype=np.int64)
    ca = np.asarray([map_a[t][0] for t in common], dtype=float)
    cb = np.asarray([map_b[t][0] for t in common], dtype=float)
    va = np.asarray([map_a[t][1] for t in common], dtype=float)
    vb = np.asarray([map_b[t][1] for t in common], dtype=float)
    return AlignedSeries(close_times=cts, closes_a=ca, closes_b=cb, qv_a=va, qv_b=vb)


# ---------------------------------------------------------------------------
# Per-bar PnL helpers

_INTERVAL_MS = {"1h": 3_600_000, "15m": 900_000, "4h": 14_400_000, "1d": 86_400_000}


def _adv_5m_usd_at(qv: np.ndarray, idx: int, interval: str, window: int = 24) -> float:
    """Trailing avg quote-volume scaled to a 5-minute equivalent. Mirrors
    `_adv_5m_usd` from backtest.py but vectorized over a numpy series."""
    if idx <= 0:
        return 0.0
    lo = max(0, idx - window)
    sl = qv[lo:idx]
    if sl.size == 0:
        return 0.0
    bar_dur_min = _INTERVAL_MS.get(interval, 3_600_000) / 60_000.0
    if bar_dur_min <= 0:
        return 0.0
    return float(sl.mean()) * (5.0 / bar_dur_min)


def _close_pair_trade(
    trade: PairTrade,
    a_close_raw: float,
    b_close_raw: float,
    adv5m_a: float,
    adv5m_b: float,
    impact_k_a: float,
    impact_k_b: float,
    half_spread_bps: float,
    costs: Costs,
    reason: str,
    z_now: float,
    ts_ms: int,
    stop_slippage_mult: float = 5.0,
) -> None:
    """Compute pair PnL = leg-A return + leg-B return − fees − slippage.

    `trade.side` records the long leg of the pair. We model entry slippage
    already paid into trade.a_entry / trade.b_entry, and exit slippage here.
    Stop fills get an inflated half-spread to model fast-move execution.
    """
    eff_hs = half_spread_bps * stop_slippage_mult if reason == "stop" else half_spread_bps

    # Direction by leg given the pair side.
    if trade.side == "long_a_short_b":
        side_a, side_b = "long", "short"
    else:
        side_a, side_b = "short", "long"

    notional_a_exit = abs(trade.notional_a)
    notional_b_exit = abs(trade.notional_b)
    a_exit = adjust_exit_price(a_close_raw, side_a, notional_a_exit, adv5m_a, impact_k_a, eff_hs)
    b_exit = adjust_exit_price(b_close_raw, side_b, notional_b_exit, adv5m_b, impact_k_b, eff_hs)
    trade.a_exit, trade.b_exit = a_exit, b_exit
    trade.exit_reason = reason
    trade.exit_z = z_now
    trade.exit_ts_ms = ts_ms

    # Returns per leg = (notional) * (price_change / entry_price). Long positive
    # when price up; short positive when price down.
    ret_a = trade.notional_a * (a_exit - trade.a_entry) / trade.a_entry
    if side_a == "short":
        ret_a = -ret_a
    ret_b = trade.notional_b * (b_exit - trade.b_entry) / trade.b_entry
    if side_b == "short":
        ret_b = -ret_b

    # Exit fees (entry fees already accounted at entry).
    fee_a_exit = taker_fee_usd(notional_a_exit, "spot", costs)
    fee_b_exit = taker_fee_usd(notional_b_exit, "spot", costs)
    trade.pnl_usd = ret_a + ret_b - fee_a_exit - fee_b_exit


# ---------------------------------------------------------------------------
# Per-pair simulator

@dataclass
class PairResult:
    pair: str
    stats: PairStats
    trades: list[PairTrade] = field(default_factory=list)
    refit_log: list[dict] = field(default_factory=list)


def _refit_indices(n: int, lookback: int, refit_every: int) -> list[int]:
    """Bar indices at which to (re)fit. First fit at idx=lookback, then every
    refit_every bars. The fit at idx covers data [idx-lookback : idx]."""
    out: list[int] = []
    i = lookback
    while i <= n - 1:
        out.append(i)
        i += refit_every
    return out


def simulate_pair(
    series: AlignedSeries,
    pair_label: str,
    sym_a: str,
    sym_b: str,
    p: PairsParams,
    notional_per_leg: float,
    start_equity: float,
    costs: Costs,
    half_spread_bps: float,
    interval: str = "1h",
) -> PairResult:
    """Run the cointegration sim on aligned closes. Returns trades + stats."""
    n = len(series.close_times)
    log_a = np.log(series.closes_a)
    log_b = np.log(series.closes_b)

    impact_k_a = impact_k_for_symbol(sym_a)
    impact_k_b = impact_k_for_symbol(sym_b)

    fit_points = _refit_indices(n, p.lookback_bars, p.refit_every_bars)
    if not fit_points:
        return PairResult(pair=pair_label, stats=PairStats(pair=pair_label,
                                                           starting_equity_usd=start_equity,
                                                           ending_equity_usd=start_equity))

    refit_log: list[dict] = []
    trades: list[PairTrade] = []
    open_trade: Optional[PairTrade] = None
    fit = None
    recent_pass: list[bool] = []   # per-refit ADF pass/fail, for persistence gate
    next_fit_at = fit_points[0]
    fit_iter = iter(fit_points)
    _ = next(fit_iter)  # consume the first; we'll fit immediately when we hit it

    for i in range(p.lookback_bars, n):
        # Refit boundary?
        if i >= next_fit_at:
            new_fit = fit_engle_granger(log_a[i - p.lookback_bars: i],
                                        log_b[i - p.lookback_bars: i], p)
            if new_fit is not None:
                fit = new_fit
                recent_pass.append(new_fit.is_cointegrated)
                refit_log.append({
                    "idx": i,
                    "ts_ms": int(series.close_times[i]),
                    "alpha": fit.alpha,
                    "beta": fit.beta,
                    "adf_p": fit.adf_p,
                    "is_coint": fit.is_cointegrated,
                    "resid_mean": fit.resid_mean,
                    "resid_std": fit.resid_std,
                })
            # Advance refit cursor.
            try:
                next_fit_at = next(fit_iter)
            except StopIteration:
                next_fit_at = n + 1  # no more refits

        if fit is None:
            continue

        sig = evaluate_pair(
            fit,
            log_a_t=float(log_a[i]),
            log_b_t=float(log_b[i]),
            current_side=(open_trade.side if open_trade is not None else None),
            p=p,
        )

        a_raw = float(series.closes_a[i])
        b_raw = float(series.closes_b[i])
        adv_a = _adv_5m_usd_at(series.qv_a, i, interval)
        adv_b = _adv_5m_usd_at(series.qv_b, i, interval)

        # Handle exits first.
        if open_trade is not None and (sig.is_exit or sig.is_stop):
            reason = "stop" if sig.is_stop else "exit"
            _close_pair_trade(
                open_trade, a_raw, b_raw, adv_a, adv_b,
                impact_k_a, impact_k_b, half_spread_bps, costs,
                reason=reason, z_now=sig.z, ts_ms=int(series.close_times[i]),
            )
            open_trade.bars_held = i - _bar_index_of(open_trade.entry_ts_ms, series.close_times)
            trades.append(open_trade)
            open_trade = None
            continue  # don't enter on the same bar as exit

        # No entry while holding.
        if open_trade is not None:
            continue

        if sig.side is None:
            continue

        # Health gate: only OPEN when the regime is established — last
        # `persist_refits` refits all passed the ADF gate. (Exits unaffected.)
        if p.persist_refits > 1:
            if (len(recent_pass) < p.persist_refits
                    or not all(recent_pass[-p.persist_refits:])):
                continue

        # Build a new entry. Dollar-hedge: B-leg notional = L, A-leg notional = β·L.
        # β can be negative in degenerate fits — abs() it so we always hedge the
        # right magnitude, then let `side` carry the direction information.
        hedge = max(0.1, min(10.0, abs(fit.beta)))
        notional_b = notional_per_leg
        notional_a = hedge * notional_per_leg

        side_a = "long" if sig.side == "long_a_short_b" else "short"
        side_b = "short" if sig.side == "long_a_short_b" else "long"
        a_entry = adjust_entry_price(a_raw, side_a, notional_a, adv_a, impact_k_a, half_spread_bps)
        b_entry = adjust_entry_price(b_raw, side_b, notional_b, adv_b, impact_k_b, half_spread_bps)

        open_trade = PairTrade(
            side=sig.side,
            entry_ts_ms=int(series.close_times[i]),
            entry_z=sig.z,
            a_entry=a_entry, b_entry=b_entry,
            beta=fit.beta,
            notional_a=notional_a,
            notional_b=notional_b,
        )

    # Force-close any still-open trade at the last bar (cost: same as a normal exit).
    if open_trade is not None:
        last = n - 1
        adv_a = _adv_5m_usd_at(series.qv_a, last, interval)
        adv_b = _adv_5m_usd_at(series.qv_b, last, interval)
        _close_pair_trade(
            open_trade,
            float(series.closes_a[last]), float(series.closes_b[last]),
            adv_a, adv_b, impact_k_a, impact_k_b, half_spread_bps, costs,
            reason="eod", z_now=0.0, ts_ms=int(series.close_times[last]),
        )
        open_trade.bars_held = last - _bar_index_of(open_trade.entry_ts_ms, series.close_times)
        trades.append(open_trade)
        open_trade = None

    # Subtract entry fees per trade. Exit fees were already booked inside
    # _close_pair_trade — entry fees are charged here so PnL = leg-returns
    # − fees(entry+exit) − slippage(entry+exit).
    for t in trades:
        entry_fees = (
            taker_fee_usd(abs(t.notional_a), "spot", costs)
            + taker_fee_usd(abs(t.notional_b), "spot", costs)
        )
        if t.pnl_usd is not None:
            t.pnl_usd -= entry_fees

    # Stats.
    stats = _compute_stats(pair_label, trades, series.close_times, start_equity,
                           refits=refit_log)
    return PairResult(pair=pair_label, stats=stats, trades=trades, refit_log=refit_log)


def _bar_index_of(ts_ms: int, close_times: np.ndarray) -> int:
    idx = int(np.searchsorted(close_times, ts_ms))
    return min(max(idx, 0), len(close_times) - 1)


def _compute_stats(
    pair_label: str, trades: list[PairTrade], close_times: np.ndarray,
    start_equity: float, refits: list[dict],
) -> PairStats:
    out = PairStats(pair=pair_label, starting_equity_usd=start_equity)
    pnls = [t.pnl_usd for t in trades if t.pnl_usd is not None]
    if close_times.size >= 2:
        span_days = (int(close_times[-1]) - int(close_times[0])) / (1000 * 86400)
    else:
        span_days = 0.0

    if pnls:
        out.trades = len(pnls)
        out.wins = sum(1 for p in pnls if p > 0)
        out.losses = sum(1 for p in pnls if p < 0)
        out.total_pnl_usd = float(sum(pnls))
        out.avg_pnl_usd = out.total_pnl_usd / out.trades
        out.win_rate = out.wins / out.trades
        out.sharpe = _sharpe_from_pnls_and_span(pnls, span_days)
        out.deflated_sharpe = _deflated_sharpe(out.sharpe, out.trades, pnls=pnls)
        curve = _equity_curve(pnls, start_equity)
        out.ending_equity_usd = float(curve[-1])
        out.max_drawdown_usd, out.max_drawdown_pct = _max_drawdown(curve)
        if span_days > 0 and start_equity > 0:
            out.annualized_pct = (out.total_pnl_usd / start_equity) * (365.0 / span_days) * 100.0
        out.avg_bars_held = float(np.mean([t.bars_held for t in trades])) if trades else 0.0
    else:
        out.ending_equity_usd = start_equity

    out.refits_total = len(refits)
    out.refits_cointegrated = sum(1 for r in refits if r["is_coint"])
    if refits:
        out.avg_adf_p = float(np.mean([r["adf_p"] for r in refits]))
    return out


# ---------------------------------------------------------------------------
# Reporting

def _short(s: PairStats) -> str:
    return (f"  trades={s.trades:4d}  pnl=${s.total_pnl_usd:+10.2f}  "
            f"wr={s.win_rate*100:5.1f}%  sharpe={s.sharpe:+5.2f}  "
            f"defl={s.deflated_sharpe:+5.2f}  dd%={s.max_drawdown_pct:5.2f}  "
            f"ann={s.annualized_pct:+6.1f}%  "
            f"coint={s.refits_cointegrated}/{s.refits_total}  "
            f"adf_p̄={s.avg_adf_p:.3f}  hold̄={s.avg_bars_held:.0f}bars")


def print_summary(results: list[PairResult]) -> None:
    print("\n" + "=" * 110)
    print("COINTEGRATED-PAIRS BACKTEST SUMMARY")
    print("=" * 110)
    for r in results:
        print(f"\n{r.pair}")
        print(_short(r.stats))


# ---------------------------------------------------------------------------
# CLI

def _parse_pairs(arg: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"pair must be 'A:B' (got {chunk!r})")
        a, b = chunk.split(":", 1)
        out.append((a.strip().upper(), b.strip().upper()))
    return out


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=DEFAULT_PAIRS,
                    help=f"Comma-separated 'A:B' pairs (default {DEFAULT_PAIRS})")
    ap.add_argument("--bars", type=int, default=8_760,
                    help="Bars per leg (~1y at 1h = 8760)")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--lookback", type=int, default=PairsParams.lookback_bars,
                    help="Refit lookback in bars (default 1440 ≈ 60d at 1h)")
    ap.add_argument("--refit-every", type=int, default=PairsParams.refit_every_bars,
                    help="Refit cadence in bars (default 168 = weekly at 1h)")
    ap.add_argument("--z-entry", type=float, default=PairsParams.z_entry)
    ap.add_argument("--z-exit", type=float, default=PairsParams.z_exit)
    ap.add_argument("--z-stop", type=float, default=PairsParams.z_stop)
    ap.add_argument("--notional", type=float, default=1_000.0,
                    help="USD per leg-B; leg-A scaled by abs(β)·notional")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    pairs = _parse_pairs(args.pairs)
    p = PairsParams(
        lookback_bars=args.lookback,
        refit_every_bars=args.refit_every,
        z_entry=args.z_entry, z_exit=args.z_exit, z_stop=args.z_stop,
    )
    costs = Costs()
    half_spread = costs.half_spread_bps_default
    settings = get_settings()
    start_equity = settings.account_equity_usd

    b = BinanceClient()
    await b.start()
    try:
        results: list[PairResult] = []
        total_t0 = time.time()
        for i, (sa, sb) in enumerate(pairs, 1):
            label = f"{sa}/{sb}"
            t0 = time.time()
            print(f"[{i}/{len(pairs)}] {label} fetching ...", flush=True)
            try:
                series = await fetch_aligned_pair(b, sa, sb, bars=args.bars,
                                                  interval=args.interval)
            except Exception as e:
                print(f"   FETCH FAILED: {type(e).__name__}: {e}")
                continue
            print(f"   {len(series.close_times)} aligned bars, "
                  f"span={(series.close_times[-1]-series.close_times[0])/(1000*86400):.0f}d, "
                  f"fetch={time.time()-t0:.1f}s")
            t1 = time.time()
            res = simulate_pair(
                series, pair_label=label, sym_a=sa, sym_b=sb, p=p,
                notional_per_leg=args.notional, start_equity=start_equity,
                costs=costs, half_spread_bps=half_spread, interval=args.interval,
            )
            results.append(res)
            print(f"   sim={time.time()-t1:.1f}s")
            print(_short(res.stats))
            print(f"   elapsed total: {time.time()-total_t0:.0f}s")

        print_summary(results)

        # JSON dump.
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.out_tag}" if args.out_tag else ""
        out_path = OUT_DIR / f"pairs_{ts}{tag}.json"
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "interval": args.interval,
            "bars": args.bars,
            "pair_params": asdict(p),
            "notional_per_leg": args.notional,
            "results": [
                {
                    "pair": r.pair,
                    "stats": asdict(r.stats),
                    "refits": r.refit_log,
                    "trades": [asdict(t) for t in r.trades],
                }
                for r in results
            ],
        }, indent=2, default=str))
        print(f"\nwrote {out_path}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
