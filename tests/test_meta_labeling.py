"""Tests for research/ml_meta/labeling.py — triple-barrier + uniqueness.

Every case uses a hand-built price path whose barrier outcome is known by
construction, so a regression in the labeling logic (i.e. a potential leak or a
sign error) fails loudly.
"""
import numpy as np
import pandas as pd
import pytest

from research.ml_meta.labeling import average_uniqueness, triple_barrier_labels


def _idx(n):
    return pd.date_range("2026-01-01", periods=n, freq="1h")


def _events(t0, side, trgt, t_vert):
    return pd.DataFrame({"side": [side], "trgt": [trgt], "t_vertical": [t_vert]},
                        index=pd.Index([t0]))


def test_long_hits_profit_barrier():
    idx = _idx(7)
    close = pd.Series([100, 101, 102, 103, 104, 105, 106], index=idx, dtype=float)
    ev = _events(idx[0], side=+1, trgt=0.01, t_vert=idx[5])  # pt at +0.02
    r = triple_barrier_labels(close, ev, pt=2, sl=2)
    assert r.touch.iloc[0] == "pt"
    assert r.t1.iloc[0] == idx[2]          # +0.02 first reached at bar 2
    assert r.ret.iloc[0] == pytest.approx(0.02)
    assert r.bin.iloc[0] == 1


def test_long_hits_stop_barrier():
    idx = _idx(7)
    close = pd.Series([100, 99, 98, 97, 96, 95, 94], index=idx, dtype=float)
    ev = _events(idx[0], side=+1, trgt=0.01, t_vert=idx[5])  # sl at -0.02
    r = triple_barrier_labels(close, ev, pt=2, sl=2)
    assert r.touch.iloc[0] == "sl"
    assert r.t1.iloc[0] == idx[2]
    assert r.ret.iloc[0] == pytest.approx(-0.02)
    assert r.bin.iloc[0] == 0


def test_short_profits_when_price_falls():
    idx = _idx(7)
    close = pd.Series([100, 99, 98, 97, 96, 95, 94], index=idx, dtype=float)
    ev = _events(idx[0], side=-1, trgt=0.01, t_vert=idx[5])
    r = triple_barrier_labels(close, ev, pt=2, sl=2)
    assert r.touch.iloc[0] == "pt"            # short's profit barrier
    assert r.ret.iloc[0] == pytest.approx(0.02)
    assert r.bin.iloc[0] == 1


def test_flat_path_hits_vertical_and_loses_after_cost():
    idx = _idx(7)
    close = pd.Series([100] * 7, index=idx, dtype=float)
    ev = _events(idx[0], side=+1, trgt=0.01, t_vert=idx[5])
    r = triple_barrier_labels(close, ev, pt=2, sl=2, cost_bps=10)
    assert r.touch.iloc[0] == "vertical"
    assert r.ret.iloc[0] == pytest.approx(0.0)
    assert r.bin.iloc[0] == 0                  # 0 - cost < 0


def test_cost_flips_a_marginal_win_to_a_loss():
    idx = _idx(7)
    # +0.5% by the vertical barrier, no horizontal touch (pt=2*0.01=0.02 not reached)
    close = pd.Series([100, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6], index=idx, dtype=float)
    ev = _events(idx[0], side=+1, trgt=0.01, t_vert=idx[5])
    cheap = triple_barrier_labels(close, ev, pt=2, sl=2, cost_bps=10)    # 0.1%
    pricey = triple_barrier_labels(close, ev, pt=2, sl=2, cost_bps=100)  # 1.0%
    assert cheap.touch.iloc[0] == "vertical"
    assert cheap.bin.iloc[0] == 1     # 0.5% - 0.1% > 0
    assert pricey.bin.iloc[0] == 0    # 0.5% - 1.0% < 0


def test_entry_is_decision_bar_and_scan_excludes_it():
    # close[t0]=100 is the entry; the very next bar already +2% must trigger pt,
    # proving the return is measured from close[t0] and the scan starts AFTER t0.
    idx = _idx(5)
    close = pd.Series([100, 102, 102, 102, 102], index=idx, dtype=float)
    ev = _events(idx[0], side=+1, trgt=0.01, t_vert=idx[3])
    r = triple_barrier_labels(close, ev, pt=2, sl=2)
    assert r.t1.iloc[0] == idx[1]
    assert r.ret.iloc[0] == pytest.approx(0.02)
    assert r.touch.iloc[0] == "pt"


def test_vertical_must_be_after_t0():
    idx = _idx(5)
    close = pd.Series([100, 101, 102, 103, 104], index=idx, dtype=float)
    bad = pd.DataFrame({"side": [1], "trgt": [0.01], "t_vertical": [idx[0]]},
                       index=pd.Index([idx[0]]))
    with pytest.raises(ValueError):
        triple_barrier_labels(close, bad, pt=2, sl=2)


def test_uniqueness_non_overlapping_is_one():
    idx = _idx(6)
    t0 = pd.Series([idx[0], idx[3]])
    t1 = pd.Series([idx[2], idx[5]])
    w = average_uniqueness(idx, t0, t1)
    assert w.iloc[0] == pytest.approx(1.0)
    assert w.iloc[1] == pytest.approx(1.0)


def test_uniqueness_full_overlap_is_half():
    idx = _idx(6)
    t0 = pd.Series([idx[0], idx[0]])
    t1 = pd.Series([idx[2], idx[2]])
    w = average_uniqueness(idx, t0, t1)
    assert w.iloc[0] == pytest.approx(0.5)
    assert w.iloc[1] == pytest.approx(0.5)


def test_uniqueness_partial_overlap():
    # e1: bars 0..3, e2: bars 2..5. Overlap on bars 2,3 (concurrency 2 there).
    idx = _idx(6)
    t0 = pd.Series([idx[0], idx[2]])
    t1 = pd.Series([idx[3], idx[5]])
    w = average_uniqueness(idx, t0, t1)
    # e1 window bars 0,1 (conc1) + 2,3 (conc2) -> mean(1,1,.5,.5)=0.75
    assert w.iloc[0] == pytest.approx(0.75)
    # e2 window bars 2,3 (conc2) + 4,5 (conc1) -> mean(.5,.5,1,1)=0.75
    assert w.iloc[1] == pytest.approx(0.75)
