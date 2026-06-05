"""Tests for research/ml_meta/cv.py — purged + embargoed event-span CPCV.

The decisive test is `test_no_train_event_overlaps_any_test_fold`: the entire
point of purging is that a training event's label window must never touch a test
fold, so we assert that invariant across every combinatorial split.
"""
import numpy as np
import pandas as pd
import pytest

from research.ml_meta.cv import (
    PurgedSplit,
    event_grid_positions,
    purged_cpcv,
    purged_cpcv_splits,
)
from src.services.cpcv import make_folds


def test_split_count_and_disjoint():
    # 8 events, one per fold-ish; n_folds=5, k=2 -> C(5,2)=10 splits.
    t0 = np.array([0, 2, 4, 6, 8, 10, 12, 14])
    t1 = t0.copy()
    splits = purged_cpcv_splits(t0, t1, n_grid=15, n_folds=5, k=2)
    assert len(splits) == 10
    for s in splits:
        assert isinstance(s, PurgedSplit)
        assert set(s.train_idx).isdisjoint(set(s.test_idx))


def test_span_purge_removes_overlapping_train_event():
    # grid 12, folds of 3: [0,3) [3,6) [6,9) [9,12)
    # A: t0=2 (fold0) but label window reaches t1=4 (into fold1)
    # B: t0=5 (fold1); C: t0=7 (fold2)
    t0 = np.array([2, 5, 7]); t1 = np.array([4, 5, 7])
    splits = {s.test_folds: s for s in purged_cpcv_splits(t0, t1, 12, n_folds=4, k=1)}
    s1 = splits[(1,)]                      # test = fold1
    assert set(s1.test_idx) == {1}        # B is the only event with t0 in fold1
    assert 0 not in set(s1.train_idx)     # A purged: its window touches fold1
    assert 2 in set(s1.train_idx)         # C kept


def test_embargo_drops_event_right_after_test_fold():
    # D = event 0 at pos1 (fold0); E = event 1 at pos4 (fold1).
    t0 = np.array([1, 4]); t1 = np.array([1, 4])
    no_emb = {s.test_folds: s for s in purged_cpcv_splits(t0, t1, 12, n_folds=4, k=1, embargo=0)}
    emb = {s.test_folds: s for s in purged_cpcv_splits(t0, t1, 12, n_folds=4, k=1, embargo=2)}
    # test = fold0 [0,3): D (event 0) is the test event; embargo band is [3,5).
    assert 0 in set(emb[(0,)].test_idx)          # D is fold0's test event
    assert 1 in set(emb[(1,)].test_idx)          # E is fold1's test event
    # with embargo=2, E (pos4 in [3,5)) is purged from fold0's train set
    assert 1 not in set(emb[(0,)].train_idx)
    assert 1 in set(no_emb[(0,)].train_idx)      # without embargo, E trains


def test_no_train_event_overlaps_any_test_fold():
    rng = np.random.RandomState(0)
    n_grid = 240
    n_events = 80
    t0 = np.sort(rng.randint(0, n_grid - 10, size=n_events))
    span = rng.randint(0, 8, size=n_events)            # variable label horizons
    t1 = np.minimum(t0 + span, n_grid - 1)
    folds = make_folds(n_grid, 6)
    splits = purged_cpcv_splits(t0, t1, n_grid, n_folds=6, k=2, embargo=3)
    for s in splits:
        test_ranges = [(folds[i].start, folds[i].end) for i in s.test_folds]
        for ti in s.train_idx:
            for (fs, fe) in test_ranges:
                overlaps = (t0[ti] < fe) and (t1[ti] >= fs)
                assert not overlaps, (
                    f"train event {ti} span [{t0[ti]},{t1[ti]}] overlaps test "
                    f"fold [{fs},{fe})")


def test_event_grid_positions_exact_and_raises():
    grid = pd.date_range("2026-01-01", periods=10, freq="1h")
    times = pd.Series([grid[0], grid[3], grid[9]])
    pos = event_grid_positions(times, grid)
    assert list(pos) == [0, 3, 9]
    off = pd.Series([pd.Timestamp("2026-01-01 00:30")])  # not on the hourly grid
    with pytest.raises(ValueError):
        event_grid_positions(off, grid)


def test_t1_before_t0_raises():
    with pytest.raises(ValueError):
        purged_cpcv_splits(np.array([5]), np.array([3]), 10, n_folds=3, k=1)


def test_timestamp_wrapper_end_to_end():
    grid = pd.date_range("2026-01-01", periods=60, freq="1h")
    t0 = pd.Series([grid[5], grid[20], grid[40]])
    t1 = pd.Series([grid[8], grid[22], grid[45]])
    splits = purged_cpcv(t0, t1, grid, n_folds=4, k=1, embargo_frac=0.02)
    assert len(splits) == 4
    for s in splits:
        assert set(s.train_idx).isdisjoint(set(s.test_idx))
