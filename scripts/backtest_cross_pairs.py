"""Native crypto-cross vs synthetic 2-leg execution — head-to-head.

Question (user, 2026-05-31): for a crypto-vs-crypto relative-value trade, is it
better to trade the NATIVE cross instrument (e.g. ETHBTC spot, one leg, one fee,
thin book) or the SYNTHETIC 2-leg spread (ETHUSDT + BTCUSDT, dollar-neutral,
deep books but double the fees)?

We hold the SIGNAL identical so the only thing that varies is execution cost.
Signal = β=1 ratio reversion (the structural form the native cross forces):

    ratio_t   = ETHUSDT_close / BTCUSDT_close        (≈ ETHBTC, the cross rate)
    z_t       = (log ratio_t − rolling_mean) / rolling_std    (causal, closed bars)
    entry     |z| >= z_entry   (short ratio if z>0 "ETH rich", long if z<0)
    exit      |z| <= z_exit
    stop      |z| >= z_stop
    one position at a time.

Execution A (native ETHBTC): trade the cross as ONE instrument with `notional`
USD at risk. PnL = notional · (ratio %move), sign by side. Cost = 2 fills ·
(spot taker + ETHBTC half-spread + impact).

Execution B (synthetic): long-leg + short-leg, `notional` USD each, dollar-
neutral. PnL = notional · (ret_long − ret_short). Cost = 4 fills · (spot taker +
per-leg half-spread + impact).

Half-spreads are calibrated LIVE from bookTicker at startup (falls back to the
2026-05-31 snapshot if the call fails). Fees: spot taker 10bp (matches the
validated `backtest_pairs` sim). A `--perp-synthetic` line is printed as a bonus
because USDT perps (5bp taker, shortable) change the verdict — ETHBTC has no perp.

Caveat printed at runtime: BOTH spot variants require margin to short; the
native cross can only be shorted via cross-margin borrow (interest not modeled),
the synthetic via perp/margin. This compares execution COST, not borrow access.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.backtest_cross_pairs \\
      --base ETH --quote BTC --bars 8760
"""
from __future__ import annotations

import argparse
import asyncio
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from src.services.backtest import BacktestStats, _stats_from_trades, SimTrade
from src.services.costs import (
    Costs,
    IMPACT_K_MAJOR,
    impact_k_for_symbol,
    taker_fee_usd,
)
from src.strategies.pairs_cointegration import (
    PairsParams,
    current_residual,
    current_zscore,
    fit_engle_granger,
)
from src.tools.binance_client import BinanceClient

# 2026-05-31 bookTicker snapshot (half-spread bps) — fallback only.
FALLBACK_HS = {"ETHBTC": 1.83, "ETHUSDT": 0.02, "BTCUSDT": 0.00,
               "SOLETH": 1.23, "SOLUSDT": 0.61, "SOLBTC": 1.5}


@dataclass
class Aligned:
    ts: list[int]
    base_usdt: list[float]   # e.g. ETHUSDT close
    quote_usdt: list[float]  # e.g. BTCUSDT close
    cross: list[float]       # native cross close (ETHBTC)
    qv_base: list[float]     # quote-vol USD for base leg
    qv_quote: list[float]
    qv_cross_usd: list[float]


async def _fetch_closes(b: BinanceClient, sym: str, tf: str, bars: int):
    raw = await b.fetch_klines_paginated(sym, tf, total=bars, market="spot")
    return {int(r[6]): (float(r[4]), float(r[7])) for r in raw}  # close_time -> (close, quoteVol)


async def fetch_aligned(b: BinanceClient, base: str, quote: str, tf: str,
                        bars: int) -> Aligned:
    base_sym, quote_sym, cross_sym = f"{base}USDT", f"{quote}USDT", f"{base}{quote}"
    db = await _fetch_closes(b, base_sym, tf, bars)
    dq = await _fetch_closes(b, quote_sym, tf, bars)
    dc = await _fetch_closes(b, cross_sym, tf, bars)
    common = sorted(set(db) & set(dq) & set(dc))
    quote_px_usd = [dq[t][0] for t in common]  # BTCUSDT price, to dollarize cross vol
    return Aligned(
        ts=common,
        base_usdt=[db[t][0] for t in common],
        quote_usdt=[dq[t][0] for t in common],
        cross=[dc[t][0] for t in common],
        qv_base=[db[t][1] for t in common],
        qv_quote=[dq[t][1] for t in common],
        # cross quoteVol is in `quote` units (BTC); ×BTCUSD → USD
        qv_cross_usd=[dc[t][1] * qp for t, qp in zip(common, quote_px_usd)],
    )


async def calibrate_half_spreads(b: BinanceClient, syms: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in syms:
        try:
            bt = await b.client.get_orderbook_ticker(symbol=s)
            bid, ask = float(bt["bidPrice"]), float(bt["askPrice"])
            mid = (bid + ask) / 2
            out[s] = (ask - bid) / 2 / mid * 1e4 if mid > 0 else FALLBACK_HS.get(s, 3.0)
        except Exception:
            out[s] = FALLBACK_HS.get(s, 3.0)
    return out


def zscore_signals(a: Aligned, p: PairsParams) -> list[Optional[int]]:
    """Causal rolling z-score of log(ratio). Returns per-bar target state:
    +1 = long ratio, -1 = short ratio, 0 = flat/no-signal. None during warmup."""
    log_ratio = np.log(np.array(a.base_usdt) / np.array(a.quote_usdt))
    n = len(log_ratio)
    win = p.lookback_bars
    states: list[Optional[int]] = [None] * n
    for i in range(n):
        if i < max(p.min_lookback_for_z, 1):
            continue
        lo = max(0, i - win)
        hist = log_ratio[lo:i]  # strictly prior bars — no lookahead
        if len(hist) < p.min_lookback_for_z:
            continue
        mu, sd = hist.mean(), hist.std(ddof=1)
        if sd <= 0:
            continue
        z = (log_ratio[i] - mu) / sd
        states[i] = z  # store raw z; entry/exit logic applied in the sim loop
    return states  # type: ignore[return-value]


def coint_signals(a: Aligned, p: PairsParams, gate_p: float = 0.05,
                  beta_window: int = 0, beta_tol: float = 0.0,
                  persist_refits: int = 1):
    """Full Engle-Granger cointegration with weekly refit + ADF gate. Returns
    (z_ser, beta_ser, coint_ser): per-bar z-score of the β-residual, the fitted
    β (hedge ratio), and whether the latest refit passed the health gate.

    Fit log(base) = α + β·log(quote); z>0 ⇒ base rich ⇒ short ratio. Native
    execution ignores β (structurally β=1); synthetic sizes the quote leg by β.

    HEALTH GATE (default off → behaves like the strategy's p<0.05 gate):
      - `gate_p`: ADF threshold.
      - `persist_refits` N: require the last N refits to ALL pass `gate_p`
        before trading. Targets the ADF-timeline finding directly — stand aside
        at a regime's flickering ONSET (wait N weeks for it to establish), and
        exit on the FIRST failing refit at a break. N=1 = the loose default.
      - `beta_window` > 0: also require β STABILITY — last `beta_window` refit
        βs have relative std (std/|mean|) <= `beta_tol`."""
    log_base = np.log(np.asarray(a.base_usdt))
    log_quote = np.log(np.asarray(a.quote_usdt))
    n = len(log_base)
    z_ser: list[Optional[float]] = [None] * n
    beta_ser: list[float] = [1.0] * n
    coint_ser: list[bool] = [False] * n
    fit = None
    recent_betas: list[float] = []   # raw refit βs, for stability check
    recent_pass: list[bool] = []     # per-refit gate_p pass/fail, for persistence

    def _beta_stable() -> bool:
        if beta_window <= 0:
            return True
        if len(recent_betas) < beta_window:
            return False
        w = recent_betas[-beta_window:]
        mean = float(np.mean(w))
        if abs(mean) < 1e-9:
            return False
        return float(np.std(w)) / abs(mean) <= beta_tol

    def _persisted() -> bool:
        if len(recent_pass) < persist_refits:
            return False
        return all(recent_pass[-persist_refits:])

    for i in range(n):
        if i < p.min_lookback_for_z:
            continue
        if fit is None or (i % p.refit_every_bars == 0):
            lo = max(0, i - p.lookback_bars)
            fit = fit_engle_granger(log_quote[lo:i], log_base[lo:i], p)  # a=quote, b=base
            if fit is not None:
                recent_betas.append(fit.beta)
                recent_pass.append(fit.adf_p < gate_p)
        if fit is None:
            continue
        r_t = current_residual(fit, float(log_quote[i]), float(log_base[i]))
        z = current_zscore(fit, r_t)
        if z is None:
            continue
        z_ser[i] = z
        beta_ser[i] = max(0.1, min(10.0, abs(fit.beta)))
        coint_ser[i] = _persisted() and _beta_stable()
    return z_ser, beta_ser, coint_ser


def _adv5m(a: Aligned, qv_series, i):
    """Turnover-based 5-min $ volume at bar i (causal). Used only for the
    sqrt-impact SENSITIVITY line — see the impact caveat in _run."""
    lo = max(0, i - 288)
    sl = qv_series[lo:i] or [qv_series[i]]
    bar_min = (a.ts[i] - a.ts[i - 1]) / 60000.0 if i > 0 else 60.0
    return float(np.mean(sl)) * (5.0 / bar_min) if bar_min > 0 else 0.0


def _run(a: Aligned, zser, p: PairsParams, mode: str, notional: float,
         costs: Costs, hs: dict[str, float], base: str, quote: str,
         synth_venue: str = "spot", beta_ser=None, coint_ser=None) -> tuple[BacktestStats, dict]:
    """Decompose every trade into raw_gross (no costs), spread cost, fee cost,
    and a separately-tracked turnover-impact estimate.

    The HEADLINE PnL uses raw_gross − spread − fee (the honest cost for a small
    retail ticket: cross the quoted spread + pay commission). The Almgren
    sqrt-impact term is reported separately because for a tight-spread but
    low-turnover cross (ETHBTC: 1.83bp spread, ~$11M/day) it explodes on a
    $1000 ticket — low turnover ≠ thin top-of-book for an MM-quoted cross."""
    base_sym, quote_sym, cross_sym = f"{base}USDT", f"{quote}USDT", f"{base}{quote}"
    ik_base = impact_k_for_symbol(base_sym)
    ik_quote = impact_k_for_symbol(quote_sym)
    ik_cross = IMPACT_K_MAJOR  # ETH/BTC are both majors; tight 1.83bp spread

    trades: list[SimTrade] = []
    pos: Optional[int] = None
    entry_i, entry_ratio, entry_beta = 0, 0.0, 1.0
    fee_tot = spread_tot = impact_tot = raw_gross_tot = 0.0
    span_days = (a.ts[-1] - a.ts[0]) / 1000 / 86400
    venue = "spot" if mode == "native" else synth_venue

    def ratio_at(i):
        return a.base_usdt[i] / a.quote_usdt[i]

    def _leg_costs(side, px_entry, px_exit, hs_bps, ik, qv, leg_notional):
        """Return (raw_gross, spread_cost, impact_cost) for one leg sized
        `leg_notional`. raw_gross uses unslipped prices."""
        rg = leg_notional * (px_exit - px_entry) / px_entry
        if side == "short":
            rg = -rg
        spread_cost = leg_notional * (hs_bps / 1e4) * 2       # cross spread on 2 fills
        adv = _adv5m(a, qv, entry_i)
        imp_bps = ik * math.sqrt(leg_notional / adv) * 1e4 if adv > 0 else 0.0
        impact_cost = leg_notional * (imp_bps / 1e4) * 2
        return rg, spread_cost, impact_cost

    def close_position(i, reason):
        nonlocal pos, fee_tot, spread_tot, impact_tot, raw_gross_tot
        if mode == "native":
            side = "long" if pos == 1 else "short"
            rg, sc, ic = _leg_costs(side, entry_ratio, ratio_at(i),
                                    hs[cross_sym], ik_cross, a.qv_cross_usd, notional)
            fee = taker_fee_usd(notional, venue, costs) * 2
        else:
            b_side = "long" if pos == 1 else "short"
            q_side = "short" if pos == 1 else "long"
            q_notional = notional * entry_beta   # β-hedge the quote leg (β=1 in ratio mode)
            rg_b, sc_b, ic_b = _leg_costs(b_side, a.base_usdt[entry_i], a.base_usdt[i],
                                          hs[base_sym], ik_base, a.qv_base, notional)
            rg_q, sc_q, ic_q = _leg_costs(q_side, a.quote_usdt[entry_i], a.quote_usdt[i],
                                          hs[quote_sym], ik_quote, a.qv_quote, q_notional)
            rg, sc, ic = rg_b + rg_q, sc_b + sc_q, ic_b + ic_q
            fee = (taker_fee_usd(notional, venue, costs) * 2
                   + taker_fee_usd(q_notional, venue, costs) * 2)
        raw_gross_tot += rg
        spread_tot += sc
        impact_tot += ic
        fee_tot += fee
        pnl = rg - sc - fee  # headline net excludes turnover-impact
        trades.append(SimTrade(symbol=cross_sym, strategy=f"xpair_{mode}",
                               side="long" if pos == 1 else "short", qty=0.0,
                               entry_price=entry_ratio, stop=0.0, tp=0.0,
                               entry_ts_ms=a.ts[entry_i], exit_price=ratio_at(i),
                               exit_reason=reason, exit_ts_ms=a.ts[i], pnl_usd=pnl))
        pos = None

    for i in range(len(a.ts)):
        z = zser[i]
        if z is None:
            continue
        if pos is not None:
            if abs(z) <= p.z_exit:
                close_position(i, "exit")
            elif abs(z) >= p.z_stop:
                close_position(i, "stop")
            elif coint_ser is not None and not coint_ser[i]:
                close_position(i, "stop")  # cointegration lost mid-hold
        # Entry gated by ADF cointegration when in coint mode.
        entry_ok = coint_ser is None or coint_ser[i]
        if pos is None and entry_ok and abs(z) >= p.z_entry:
            pos = -1 if z > 0 else 1   # z>0 base rich → short ratio
            entry_i, entry_ratio = i, ratio_at(i)
            entry_beta = beta_ser[i] if beta_ser is not None else 1.0
    if pos is not None:
        close_position(len(a.ts) - 1, "eod")

    stats = _stats_from_trades(f"xpair_{mode}", trades, 1000.0, span_days)
    return stats, {"fee": fee_tot, "spread": spread_tot, "impact": impact_tot,
                   "raw_gross": raw_gross_tot}


def _print(label: str, stats: BacktestStats, info: dict):
    print("-" * 78)
    print(f"  {label}")
    print(f"    round trips:   {stats.trades}")
    print(f"    raw gross:     ${info['raw_gross']:+.2f}   (signal PnL, no costs)")
    print(f"    − spread:      ${info['spread']:.2f}")
    print(f"    − fees:        ${info['fee']:.2f}")
    print(f"    = NET PnL:     ${stats.total_pnl_usd:+.2f}   "
          f"({stats.annualized_pct:+.1f}%/yr on $1000)")
    print(f"    win rate:      {stats.win_rate*100:.0f}%   Sharpe {stats.sharpe:+.2f}")
    print(f"    [turnover-impact sensitivity (NOT in net): ${info['impact']:.2f}]")


async def amain() -> None:
    load_dotenv("/Users/BulatShkanov/Downloads/crypto-trading-agent/.env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="ETH")
    ap.add_argument("--quote", default="BTC")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--bars", type=int, default=8760)
    ap.add_argument("--notional", type=float, default=1000.0)
    ap.add_argument("--z-entry", type=float, default=2.0)
    ap.add_argument("--z-exit", type=float, default=0.5)
    ap.add_argument("--z-stop", type=float, default=3.5)
    ap.add_argument("--lookback-bars", type=int, default=24 * 60)
    ap.add_argument("--signal", choices=["ratio", "coint"], default="ratio",
                    help="ratio = β=1 z-reversion; coint = full Engle-Granger β-fit + ADF gate")
    ap.add_argument("--gate-p", type=float, default=0.05,
                    help="ADF p-value gate (coint mode). 0.01 = strict health gate")
    ap.add_argument("--beta-window", type=int, default=0,
                    help="health gate: require β stability over last N refits (0=off)")
    ap.add_argument("--beta-tol", type=float, default=0.15,
                    help="health gate: max relative std of recent refit βs")
    ap.add_argument("--persist-refits", type=int, default=1,
                    help="health gate: require last N refits to all pass gate-p (1=off)")
    args = ap.parse_args()

    p = PairsParams(lookback_bars=args.lookback_bars, z_entry=args.z_entry,
                    z_exit=args.z_exit, z_stop=args.z_stop)
    costs = Costs()
    base, quote = args.base, args.quote
    syms = [f"{base}USDT", f"{quote}USDT", f"{base}{quote}"]

    b = BinanceClient()
    await b.start()
    try:
        hs = await calibrate_half_spreads(b, syms)
        a = await fetch_aligned(b, base, quote, args.tf, args.bars)
        span = (a.ts[-1] - a.ts[0]) / 1000 / 86400
        if args.signal == "coint":
            zser, beta_ser, coint_ser = coint_signals(
                a, p, gate_p=args.gate_p, beta_window=args.beta_window,
                beta_tol=args.beta_tol, persist_refits=args.persist_refits)
            n_coint = sum(coint_ser)
            betas = [beta_ser[i] for i in range(len(coint_ser)) if coint_ser[i]]
            beta_med = float(np.median(betas)) if betas else 1.0
            gate_desc = f"ADF p<{args.gate_p}"
            if args.persist_refits > 1:
                gate_desc += f" × {args.persist_refits} consecutive refits"
            if args.beta_window > 0:
                gate_desc += f" + β-stable(last {args.beta_window} refits, relstd≤{args.beta_tol})"
            sig_desc = (f"Engle-Granger β-fit, health gate [{gate_desc}] (refit/{p.refit_every_bars}b, "
                        f"lookback {p.lookback_bars}b)  ·  tradeable {n_coint}/{len(coint_ser)} "
                        f"bars ({n_coint/len(coint_ser)*100:.0f}%), median β={beta_med:.2f}")
        else:
            zser = zscore_signals(a, p)
            beta_ser = coint_ser = None
            sig_desc = (f"β=1 ratio z-reversion (entry|z|>={p.z_entry}, exit<={p.z_exit}, "
                        f"stop>={p.z_stop})")
        st_n, in_n = _run(a, zser, p, "native", args.notional, costs, hs, base, quote,
                          beta_ser=beta_ser, coint_ser=coint_ser)
        st_s, in_s = _run(a, zser, p, "synthetic", args.notional, costs, hs, base, quote, "spot",
                          beta_ser=beta_ser, coint_ser=coint_ser)
        st_p, in_p = _run(a, zser, p, "synthetic", args.notional, costs, hs, base, quote, "perp",
                          beta_ser=beta_ser, coint_ser=coint_ser)

        print("\n" + "=" * 78)
        print(f"  CROSS-PAIR EXECUTION HEAD-TO-HEAD  ·  {base}/{quote}  ·  {args.tf}  "
              f"·  {len(a.ts)} bars (~{span:.0f}d)")
        print(f"  signal: {sig_desc}  ·  notional ${args.notional:.0f}/leg")
        print(f"  half-spreads (bp): " + "  ".join(f"{s}={hs[s]:.2f}" for s in syms))
        print("=" * 78)
        _print(f"A) NATIVE {base}{quote} spot  (1 instrument, 2 fills/RT, {hs[syms[2]]:.2f}bp spread)",
               st_n, in_n)
        _print(f"B) SYNTHETIC {base}USDT/{quote}USDT spot  (2 legs, 4 fills/RT)",
               st_s, in_s)
        _print(f"C) SYNTHETIC {base}USDT/{quote}USDT PERP  (2 legs, 4 fills/RT, 5bp taker; ETHBTC has no perp)",
               st_p, in_p)
        print("=" * 78)
        best = max([("native spot", st_n.total_pnl_usd),
                    ("synthetic spot", st_s.total_pnl_usd),
                    ("synthetic perp", st_p.total_pnl_usd)], key=lambda x: x[1])
        print(f"  WINNER (net of costs): {best[0]}  (${best[1]:+.2f})")
        print(f"  NOTE: shorting the ratio needs margin/perp; native cross short = "
              f"cross-margin borrow (interest NOT modeled). Funding on perp legs NOT modeled.")
        print("=" * 78)
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
