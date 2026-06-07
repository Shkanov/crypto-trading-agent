"""Phase-2 driver: does meta-labeling beat raw Δfunding OOS, after costs?

    uv run --extra research python -m research.ml_meta.run_dfunding

Same harness as Phase 1, new primary. Δfunding is the one strategy with a
VALIDATED raw positive edge (CPCV PASS top-30/50), so this asks the question
meta-labeling is actually built for: can a non-linear filter skip the legs the
rule picks that won't pay, and lift the surviving book's OOS economics?

Label economics are COMPLETE for a carry leg held one week:
    ret_net = directional price return        (triple-barrier, pure vertical)
            − round-trip trading cost
            + funding return  (−side × Σ funding over the hold; the leg PAYS
                               funding when it's on the wrong side of the rate)
y = 1 iff ret_net > 0. The meta-model is trained to predict that, so it can't
"win" by ignoring the funding drag this book actually carries.

Caveat (printed at runtime): the universe is the FIXED 16-symbol cache, not the
live PIT-by-volume top-30. That weakens absolute breadth/survivorship realism
but not the meta-vs-raw relative comparison, which is the verdict here.
"""
from __future__ import annotations

import time

import pandas as pd

from research.ml_meta.cv import purged_cpcv
from research.ml_meta.data import build_funding_panel, build_panel
from research.ml_meta.evaluate import evaluate_cpcv, summarize
from research.ml_meta.features import build_features, feature_cols
from research.ml_meta.funding_events import (
    SIGNAL_COLS,
    DFundingParams,
    dfunding_events,
)
from research.ml_meta.labeling import average_uniqueness, triple_barrier_labels

# Broad candidate pool of USDⓈ-M perps that have existed across the full 2y
# window. dfunding_events ranks these PIT-by-volume into the top-`universe_size`
# each rebalance, so this is a CANDIDATE pool, not the per-rebalance universe.
# Symbols without near-full history are dropped in build_dataset.
UNIVERSE = [
    # large caps
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "ATOMUSDT",
    "NEARUSDT", "FETUSDT", "INJUSDT", "UNIUSDT", "BCHUSDT", "ETCUSDT",
    "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "AAVEUSDT", "SUIUSDT",
    # mid/lower-cap higher-OI alts (where the carry edge is documented to live)
    "TIAUSDT", "SEIUSDT", "RUNEUSDT", "GALAUSDT", "SANDUSDT", "MANAUSDT",
    "ALGOUSDT", "ICPUSDT", "IMXUSDT", "GRTUSDT", "LDOUSDT", "AXSUSDT",
    "CHZUSDT", "CRVUSDT", "COMPUSDT", "SNXUSDT", "DYDXUSDT", "GMTUSDT",
    "APEUSDT", "ENJUSDT", "1INCHUSDT", "KAVAUSDT", "STXUSDT", "ORDIUSDT",
    "WIFUSDT", "PEPEUSDT", "ENAUSDT", "JUPUSDT", "PYTHUSDT", "JTOUSDT",
]

FEATURE_COLS = feature_cols(SIGNAL_COLS, include_time=False)   # weekly clock → no hour/dow


def _funding_return(events: list[tuple[int, float]],
                    t0_ms: int, t1_ms: int, side: float) -> float:
    """Funding PnL as a fraction of notional over the hold (t0, t1].
    A long (side +1) PAYS when funding>0, so its funding return is −Σ rate."""
    s = sum(r for (t, r) in events if t0_ms < t <= t1_ms)
    return -side * s


def build_dataset(symbols, start_ms, end_ms, *, interval="1h",
                  trade_cost_bps=10.0, p: DFundingParams | None = None):
    p = p or DFundingParams()
    print("price panel:")
    price = build_panel(symbols, interval, start_ms, end_ms)
    # Keep only near-full-history names so the PIT volume rank isn't distorted by
    # symbols that only listed partway through (their early "volume" is zero).
    if price:
        max_bars = max(len(df) for df in price.values())
        price = {s: df for s, df in price.items() if len(df) >= 0.95 * max_bars}
    print(f"kept {len(price)} symbols with >=95% history (of {len(symbols)} candidates)")
    print("funding panel:")
    funding = build_funding_panel(list(price), start_ms, end_ms)

    events_by_symbol = dfunding_events(funding, price, p)
    if not events_by_symbol:
        raise SystemExit("no Δfunding events — universe too small or no history")

    cost = trade_cost_bps / 1e4
    label_rows = []
    for sym, ev in events_by_symbol.items():
        # Pure vertical barrier (pt=sl=0): hold the leg to the next rebalance,
        # exactly like the live strategy. ret = directional price return.
        lab = triple_barrier_labels(price[sym]["close"], ev, pt=0.0, sl=0.0,
                                    cost_bps=0.0)
        fe = funding[sym]
        for t0 in ev.index:
            t1 = lab.t1.loc[t0]
            side = float(ev.loc[t0, "side"])
            t0_ms = int(t0.value // 1_000_000)
            t1_ms = int(pd.Timestamp(t1).value // 1_000_000)
            fund_ret = _funding_return(fe, t0_ms, t1_ms, side)
            ret_net = float(lab.ret.loc[t0]) - cost + fund_ret
            label_rows.append({"sym": sym, "t0": t0, "t1": t1,
                               "y": int(ret_net > 0), "ret_net": ret_net,
                               "fund_ret": fund_ret})

    feats = build_features(events_by_symbol, price, signal_cols=SIGNAL_COLS)
    labels = pd.DataFrame(label_rows)
    ds = (feats.merge(labels, on=["sym", "t0"], how="inner")
          .sort_values("t0").reset_index(drop=True))

    grid = None
    for k in price.values():
        grid = k.index if grid is None else grid.union(k.index)
    ds["w"] = average_uniqueness(grid, ds["t0"], ds["t1"]).values
    return ds, grid


def main():
    end = (int(time.time() * 1000) // 3_600_000) * 3_600_000 - 3_600_000 * 6
    start = end - 2 * 365 * 24 * 3_600_000     # 2 years
    print("=== Phase 2: meta-labeling the Δfunding primary ===")
    print("NOTE: fixed 16-symbol universe (not live PIT-by-volume top-30); "
          "relative meta-vs-raw verdict is unaffected.\n")

    ds, grid = build_dataset(UNIVERSE, start, end)
    n_long = int((ds.side > 0).sum())
    print(f"\nevents: {len(ds)}  win-rate(base, net)={ds['y'].mean():.3f}  "
          f"longs={n_long} shorts={len(ds) - n_long}")
    print(f"raw mean net/leg: {ds['ret_net'].mean()*1e4:+.1f} bps   "
          f"(price {((ds['ret_net']-ds['fund_ret']).mean())*1e4:+.1f} + "
          f"funding {ds['fund_ret'].mean()*1e4:+.1f})")

    splits = purged_cpcv(ds["t0"], ds["t1"], grid, n_folds=6, k=2, embargo_frac=0.01)
    print(f"CPCV splits: {len(splits)} (purged + embargoed)")

    for thr in (0.50, 0.55, 0.60):
        res = evaluate_cpcv(ds, splits, threshold=thr, feature_cols=FEATURE_COLS)
        s = summarize(res)
        if not s.get("folds"):
            print(f"thr={thr}: no usable folds"); continue
        print(f"\n=== threshold {thr} ===")
        print(f" usable folds: {s['folds']}   avg kept: {s['avg_kept_frac']:.2f}")
        print(f" RAW : IR {s['raw_ir_mean']:+.3f}±{s['raw_ir_std']:.3f}  "
              f"win {s['raw_win']:.3f}  mean_ret {s['raw_mean_ret']:+.5f}")
        print(f" META: IR {s['meta_ir_mean']:+.3f}±{s['meta_ir_std']:.3f}  "
              f"win {s['meta_win']:.3f}  mean_ret {s['meta_mean_ret']:+.5f}")
        print(f" IR uplift: {s['ir_uplift_mean']:+.3f}  "
              f"(positive in {s['ir_uplift_pos_frac']*100:.0f}% of folds)")
        # Honest gate: "OOS-positive" must clear BOTH an economic floor and a
        # fold-stability bar — otherwise a hard threshold can abstain its way to
        # a near-zero mean_ret that is pure fold noise and falsely reads as a
        # win. MIN_EDGE: net edge must beat a meaningful slice of round-trip
        # cost, not a fraction of a tick. STABLE: across-fold IR mean must
        # exceed its own std (crude t>1) so the sign isn't one lucky fold.
        MIN_EDGE = 5e-4                              # 5 bps/leg net, post-cost
        econ_ok = s["meta_mean_ret"] > MIN_EDGE
        stable = s["meta_ir_mean"] > s["meta_ir_std"]
        if econ_ok and stable and s["ir_uplift_pos_frac"] > 0.5:
            verdict = "DEPLOYABLE: meta OOS-positive (econ-significant + fold-stable)"
        elif s["meta_mean_ret"] > 0 and not (econ_ok and stable):
            verdict = ("INCONCLUSIVE: meta nominally >0 but inside fold noise "
                       f"(need mean_ret>{MIN_EDGE:.0e} AND IR>IR_std)")
        elif s["raw_mean_ret"] > 0 and s["meta_mean_ret"] > s["raw_mean_ret"]:
            verdict = "PROMISING: raw already positive, meta lifts it further"
        elif s["ir_uplift_mean"] > 0:
            verdict = "NOT deployable: improves raw but still OOS-negative (lose less)"
        else:
            verdict = "NOT deployable: no improvement over raw"
        print(f" -> {verdict}")


if __name__ == "__main__":
    main()
