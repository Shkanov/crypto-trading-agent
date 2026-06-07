"""Does the raw Δfunding edge reproduce on the LIVE universe (newer alts)?

The 2y-clean-history harness shows raw Δfunding at -75bps/leg (majors in) /
-126bps/leg (majors out). The validated CPCV PASS (w21_tn3_bk10, OOS +0.978,
price +$63 on a $1000 book over 365d) used a DIFFERENT universe: the top-30 USDT
perps by 24h volume as of 2026-06-01, majors excluded — dominated by names that
LISTED in the last 6-18 months (HYPE, ASTER, LAB, PLAY, ALLO, AIA, GUN, HOME,
ONDO, TAO, WLD...). The harness's 0.95*max_bars history filter throws all of
those out, so the two runs never tested the same universe.

This probe feeds the harness's PIT-correct replay the *validated* universe over a
matched ~400d window, with a relaxed history floor (a name is eligible once it
has real history; PIT volume rank + the 2-funding-window check gate entry). It
reports RAW per-leg economics only (no ML) — the question is purely "is raw
Δfunding positive in the universe the live strategy actually trades?"

  uv run --extra research python -m research.ml_meta._probe_universe

Caveats this DELIBERATELY does NOT fix (they bias TOWARD finding an edge, so a
negative result here is conservative/strong):
  * The candidate pool is today's top-30 by volume applied across the window —
    the same current-universe look-ahead the validated run had. PIT volume rank
    only re-orders within this already-hindsight pool.
  * Tokenized-equity perps (XAU/XAG) and the unfetchable CJK-symbol meme are
    dropped; live excludes the former too.
"""
import time

import numpy as np
import pandas as pd

from research.ml_meta.data import build_funding_panel, build_panel
from research.ml_meta.funding_events import DFundingParams, dfunding_events
from research.ml_meta.labeling import triple_barrier_labels
from research.ml_meta.run_dfunding import _funding_return

# Validated universe (cpcv_dfunding_20260601_070409.json), fetchable crypto-perp
# subset: drop XAUUSDT/XAGUSDT (tokenized equity, live excludes) and the
# CJK-symbol meme. 1000PEPEUSDT is Binance's listed contract for PEPE.
VALIDATED_UNIVERSE = [
    "HYPEUSDT", "ZECUSDT", "LABUSDT", "XLMUSDT", "PORTALUSDT", "STGUSDT",
    "ALLOUSDT", "WLDUSDT", "NEARUSDT", "PLAYUSDT", "HUSDT", "DOGEUSDT",
    "SUIUSDT", "CLUSDT", "HOMEUSDT", "ASTERUSDT", "TONUSDT", "1000PEPEUSDT",
    "AIAUSDT", "ADAUSDT", "MUUSDT", "ONDOUSDT", "FETUSDT", "TAOUSDT",
    "HIVEUSDT", "GUNUSDT", "BCHUSDT",
]

DAYS = 400
COST = 10 / 1e4
MIN_BARS = 60 * 24          # require >=60 days of hourly history to be a candidate


def _legs(price, funding, wc, tn):
    ev = dfunding_events(funding, price, DFundingParams(window_cycles=wc, top_n=tn))
    rows = []
    for sym, e in ev.items():
        lab = triple_barrier_labels(price[sym]["close"], e, pt=0, sl=0, cost_bps=0)
        for t0 in e.index:
            t1 = lab.t1.loc[t0]
            side = float(e.loc[t0, "side"])
            t0ms = int(t0.value // 1_000_000)
            t1ms = int(pd.Timestamp(t1).value // 1_000_000)
            rows.append((t0, sym, side, float(lab.ret.loc[t0]),
                         _funding_return(funding[sym], t0ms, t1ms, side)))
    return pd.DataFrame(rows, columns=["t0", "sym", "side", "price", "fund"])


def _econ(df):
    if df.empty:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0
    price = df["price"].to_numpy()
    fund = df["fund"].to_numpy()
    net = price + fund - COST
    t = net.mean() / (net.std(ddof=1) / np.sqrt(len(net))) if len(net) > 1 else 0.0
    return (len(net), price.mean() * 1e4, fund.mean() * 1e4,
            net.mean() * 1e4, t, float((net > 0).mean()))


def main():
    end = (int(time.time() * 1000) // 3_600_000) * 3_600_000 - 3_600_000 * 6
    start = end - DAYS * 24 * 3_600_000
    print(f"=== raw Δfunding on the LIVE (validated) universe — {DAYS}d ===")
    print(f"candidates: {len(VALIDATED_UNIVERSE)} names, majors already excluded\n")
    price = build_panel(VALIDATED_UNIVERSE, "1h", start, end)
    # Relaxed floor: a name is a candidate once it has >=MIN_BARS history. PIT
    # volume rank + the 2-funding-window check still gate weekly entry, so newly
    # listed names enter only when they actually have data — no look-ahead.
    price = {s: d for s, d in price.items() if len(d) >= MIN_BARS}
    funding = build_funding_panel(list(price), start, end)
    print(f"kept {len(price)} symbols with >= {MIN_BARS//24}d history:")
    for s in sorted(price, key=lambda s: len(price[s])):
        print(f"  {s:<14} {len(price[s])//24:>4}d")

    mid = pd.to_datetime(end - (DAYS // 2) * 24 * 3_600_000, unit="ms", utc=True)

    print("\n=== config grid, RAW (live direction), net per leg ===")
    print(f'{"cfg":<12}{"n":>5}{"price":>8}{"fund":>8}{"net":>8}{"t":>7}{"win%":>7}')
    legs_by_cfg = {}
    for wc in (14, 21, 42):
        for tn in (2, 3, 5):
            legs = _legs(price, funding, wc, tn)
            legs_by_cfg[(wc, tn)] = legs
            n, pp, ff, nn, t, w = _econ(legs)
            print(f'w{wc} tn{tn:<7}{n:>5}{pp:>8.1f}{ff:>8.1f}{nn:>8.1f}{t:>7.2f}{w*100:>7.1f}')

    legs = legs_by_cfg[(21, 3)]
    print("\n=== validated config w21_tn3: temporal stability ===")
    for name, d in (("half1", legs[legs["t0"] < mid]),
                    ("half2", legs[legs["t0"] >= mid]),
                    ("full", legs)):
        n, pp, ff, nn, t, w = _econ(d)
        print(f'{name:<8}{n:>5}  net {nn:>7.1f}bps  t {t:>5.2f}  win {w*100:>5.1f}%  '
              f'(price {pp:+.1f} fund {ff:+.1f})')

    print("\n=== w21_tn3 net by symbol (concentration; top/bottom 6 by sum) ===")
    g = legs.assign(net=legs["price"] + legs["fund"] - COST)
    by = g.groupby("sym")["net"].agg(["count", "mean", "sum"]).sort_values("sum")
    tot = by["sum"].sum()
    for sym, r in pd.concat([by.head(6), by.tail(6)]).iterrows():
        print(f'  {sym:<14} n={int(r["count"]):>3}  mean {r["mean"]*1e4:>7.1f}bps  '
              f'sum {r["sum"]*1e4:>8.1f}bps  ({(r["sum"]/tot*100 if tot else 0):>5.1f}% of total)')
    print(f'  total net across symbols: {tot*1e4:.1f}bps over {len(legs)} legs')


if __name__ == "__main__":
    main()
