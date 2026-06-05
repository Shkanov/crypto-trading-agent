"""Phase-1 driver: does meta-labeling beat raw mean-rev OOS, after costs?

    uv run --extra research python -m research.ml_meta.run

Builds the dataset (events -> cost-aware triple-barrier labels -> PIT features ->
uniqueness weights), runs the LightGBM meta-model through span-purged CPCV, and
prints the meta-filtered vs raw-primary OOS comparison. Falsification-first: if
meta doesn't beat raw, that's the result.
"""
from __future__ import annotations

import time

import pandas as pd

from research.ml_meta.cv import purged_cpcv
from research.ml_meta.data import build_panel
from research.ml_meta.evaluate import evaluate_cpcv, summarize
from research.ml_meta.events import MeanRevParams, mean_reversion_events
from research.ml_meta.features import build_features
from research.ml_meta.labeling import average_uniqueness, triple_barrier_labels

UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "ATOMUSDT",
    "NEARUSDT", "FETUSDT", "INJUSDT", "UNIUSDT",
]


def build_dataset(symbols, start_ms, end_ms, *, interval="1h",
                  cost_bps=12.0, pt=1.0, sl=1.0, mr=None):
    panel = build_panel(symbols, interval, start_ms, end_ms)
    mr = mr or MeanRevParams()
    cost = cost_bps / 1e4
    events_by_symbol, label_rows = {}, []
    for sym, k in panel.items():
        ev = mean_reversion_events(k, mr)
        if len(ev) == 0:
            continue
        events_by_symbol[sym] = ev
        lab = triple_barrier_labels(k["close"], ev, pt=pt, sl=sl, cost_bps=cost_bps)
        for t0 in ev.index:
            label_rows.append({"sym": sym, "t0": t0,
                               "y": int(lab.bin.loc[t0]),
                               "t1": lab.t1.loc[t0],
                               "ret_net": float(lab.ret.loc[t0]) - cost})
    feats = build_features(events_by_symbol, panel)
    labels = pd.DataFrame(label_rows)
    ds = feats.merge(labels, on=["sym", "t0"], how="inner").sort_values("t0").reset_index(drop=True)
    grid = None
    for k in panel.values():
        grid = k.index if grid is None else grid.union(k.index)
    ds["w"] = average_uniqueness(grid, ds["t0"], ds["t1"]).values
    return ds, grid


def main():
    end = (int(time.time() * 1000) // 3_600_000) * 3_600_000 - 3_600_000 * 6
    start = end - 2 * 365 * 24 * 3_600_000     # 2 years
    print(f"building dataset: {len(UNIVERSE)} symbols, 2y of 1h ...")
    ds, grid = build_dataset(UNIVERSE, start, end)
    print(f"events: {len(ds)}  win-rate(base)={ds['y'].mean():.3f}  "
          f"longs={int((ds.side>0).sum())} shorts={int((ds.side<0).sum())}")

    splits = purged_cpcv(ds["t0"], ds["t1"], grid, n_folds=6, k=2, embargo_frac=0.01)
    print(f"CPCV splits: {len(splits)} (purged + embargoed)")

    for thr in (0.50, 0.55, 0.60):
        res = evaluate_cpcv(ds, splits, threshold=thr)
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
        # Honest gate: a DEPLOYABLE edge must be OOS-POSITIVE, not merely better
        # than a losing baseline. Beating raw while still negative = "lose less".
        if s["meta_mean_ret"] > 0 and s["ir_uplift_pos_frac"] > 0.5:
            verdict = "DEPLOYABLE: meta is OOS-positive and beats raw"
        elif s["ir_uplift_mean"] > 0:
            verdict = "NOT deployable: improves raw but still OOS-negative (lose less)"
        else:
            verdict = "NOT deployable: no improvement over raw"
        print(f" -> {verdict}")


if __name__ == "__main__":
    main()
