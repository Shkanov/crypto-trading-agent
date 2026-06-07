"""Investigate the large-cap Δfunding SIGN FLIP.

On the 53-name large/mid-cap study pool the LIVE direction (long rising-funding,
short falling-funding) is raw-negative, and reversing it looks strongly positive.
This probe asks: is the FLIP a *real* contrarian edge on large caps, or a
full-sample / outlier artifact?  No ML — just per-leg price + funding + net,
sliced by sign, config, sub-period, and concentration.

Builds the 2y panel ONCE and replays dfunding_events per (window, top_n); every
other cut is in-memory, so this is fast on a warm cache.

  uv run --extra research python -m research.ml_meta._probe_sign

Per leg, with stored side = the LIVE (normal) side:
    price_normal = lab.ret           (side-adjusted directional return)
    fund_normal  = _funding_return(..., stored_side)
    net_normal   = price_normal + fund_normal - cost
    FLIP reverses the position:  price_flip = -price_normal,
    fund_flip = -fund_normal,  net_flip = -price_normal - fund_normal - cost
"""
import time

import numpy as np
import pandas as pd

from research.ml_meta.data import build_funding_panel, build_panel
from research.ml_meta.funding_events import DFundingParams, dfunding_events
from research.ml_meta.labeling import triple_barrier_labels
from research.ml_meta.run_dfunding import UNIVERSE, _funding_return

END = (int(time.time() * 1000) // 3_600_000) * 3_600_000 - 3_600_000 * 6
DAYS = 730
START = END - DAYS * 24 * 3_600_000
COST = 10 / 1e4


def _legs(price, funding, wc, tn):
    """One row per leg over the full window: (t0, sym, price_normal, fund_normal).
    FLIP economics are derived from these, so events are replayed only once."""
    ev = dfunding_events(funding, price, DFundingParams(window_cycles=wc, top_n=tn))
    rows = []
    for sym, e in ev.items():
        lab = triple_barrier_labels(price[sym]["close"], e, pt=0, sl=0, cost_bps=0)
        for t0 in e.index:
            t1 = lab.t1.loc[t0]
            side = float(e.loc[t0, "side"])
            t0ms = int(t0.value // 1_000_000)
            t1ms = int(pd.Timestamp(t1).value // 1_000_000)
            rows.append((t0, sym,
                         float(lab.ret.loc[t0]),                       # price_normal
                         _funding_return(funding[sym], t0ms, t1ms, side)))  # fund_normal
    return pd.DataFrame(rows, columns=["t0", "sym", "price", "fund"])


def _econ(df, flip):
    """(n, price_bps, fund_bps, net_bps, t_stat, win_frac) for a leg frame."""
    if df.empty:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0
    sgn = -1.0 if flip else 1.0
    price = sgn * df["price"].to_numpy()
    fund = sgn * df["fund"].to_numpy()
    net = price + fund - COST
    t = net.mean() / (net.std(ddof=1) / np.sqrt(len(net))) if len(net) > 1 else 0.0
    return (len(net), price.mean() * 1e4, fund.mean() * 1e4,
            net.mean() * 1e4, t, float((net > 0).mean()))


def main():
    print(f"build 2y panel once  ({DAYS}d, {len(UNIVERSE)} candidates)")
    price = build_panel(UNIVERSE, "1h", START, END)
    mb = max(len(d) for d in price.values())
    price = {s: d for s, d in price.items() if len(d) >= 0.95 * mb}
    funding = build_funding_panel(list(price), START, END)
    print(f"kept {len(price)} symbols\n")

    mid = pd.to_datetime(END - (DAYS // 2) * 24 * 3_600_000, unit="ms", utc=True)

    # ---- 1. config × sign grid, full window -------------------------------
    print("=== full 2y: normal (LIVE) vs FLIP ===")
    hdr = f'{"cfg":<12}{"n":>5}{"price":>8}{"fund":>8}{"net":>8}{"t":>7}{"win%":>7}'
    print("NORMAL  " + hdr[8:])
    legs_by_cfg = {}
    for wc in (21, 42):
        for tn in (2, 3):
            legs = _legs(price, funding, wc, tn)
            legs_by_cfg[(wc, tn)] = legs
            for flip in (False, True):
                n, pp, ff, nn, t, w = _econ(legs, flip)
                tag = f'w{wc} tn{tn} {"FLIP" if flip else "norm"}'
                print(f'{tag:<12}{n:>5}{pp:>8.1f}{ff:>8.1f}{nn:>8.1f}{t:>7.2f}{w*100:>7.1f}')

    # ---- 2. temporal stability of the FLIP at the default config ----------
    legs = legs_by_cfg[(21, 3)]
    h1 = legs[legs["t0"] < mid]
    h2 = legs[legs["t0"] >= mid]
    print("\n=== FLIP temporal stability (w21 tn3) ===")
    for name, d in (("year1", h1), ("year2", h2), ("full", legs)):
        n, pp, ff, nn, t, w = _econ(d, flip=True)
        print(f'{name:<8}{n:>5}  net {nn:>7.1f}bps  t {t:>5.2f}  win {w*100:>5.1f}%  '
              f'(price {pp:+.1f} fund {ff:+.1f})')

    # ---- 3. concentration: is the FLIP edge from a few symbols? -----------
    print("\n=== FLIP net by symbol (w21 tn3, top/bottom 6 by total) ===")
    g = legs.assign(net=-legs["price"] - legs["fund"] - COST)
    by = g.groupby("sym")["net"].agg(["count", "mean", "sum"]).sort_values("sum")
    by["mean_bps"] = by["mean"] * 1e4
    tot = by["sum"].sum()
    head = pd.concat([by.head(6), by.tail(6)])
    for sym, r in head.iterrows():
        print(f'  {sym:<12} n={int(r["count"]):>3}  mean {r["mean_bps"]:>7.1f}bps  '
              f'sum {r["sum"]*1e4:>8.1f}bps  ({r["sum"]/tot*100:>5.1f}% of total)')
    print(f'  total FLIP sum across all symbols: {tot*1e4:.1f}bps over {len(legs)} legs')


if __name__ == "__main__":
    main()
