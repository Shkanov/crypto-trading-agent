"""Purged + embargoed combinatorial cross-validation over EVENT SPANS.

`src/services/cpcv.py` partitions a time grid into combinatorial folds and
purges by row index — correct for the existing per-config PnL evaluation. Meta-
labeling is different: each training sample is an *event* with a label window
``[t0, t1]`` (decision bar → first barrier touch). A training event leaks into a
test fold if its window merely *overlaps* that fold's time range, even when its
decision bar t0 sits outside it (AFML 7.4.1, "purging"). Plus an embargo band
after each test fold to kill forward autocorrelation leakage (AFML 7.4.2).

This module reuses `make_folds` / `cpcv_test_combos` / `fold_membership` for the
combinatorial fold structure and adds the span-aware purge + embargo on top.
Pure / deterministic / no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.services.cpcv import cpcv_test_combos, fold_membership, make_folds


@dataclass(frozen=True)
class PurgedSplit:
    test_idx: np.ndarray     # positions (into the event arrays) of test events
    train_idx: np.ndarray    # positions of train events (purged + embargoed)
    test_folds: tuple[int, ...]


def event_grid_positions(times: pd.Series | pd.DatetimeIndex,
                         grid: pd.DatetimeIndex) -> np.ndarray:
    """Map event timestamps to integer positions on ``grid``.

    Each timestamp must lie on the grid (events are bar-aligned). Uses an exact
    indexer and fails loudly on any miss — a silent nearest-match would smear
    fold boundaries and reintroduce leakage.
    """
    # pd.DatetimeIndex(series) preserves tz; .values would strip it and break a
    # tz-aware grid match.
    idx = times if isinstance(times, pd.DatetimeIndex) else pd.DatetimeIndex(times)
    pos = grid.get_indexer(idx)
    if (pos < 0).any():
        raise ValueError("every event timestamp must lie exactly on `grid`")
    return pos


def purged_cpcv_splits(
    t0_pos: np.ndarray,
    t1_pos: np.ndarray,
    n_grid: int,
    *,
    n_folds: int = 6,
    k: int = 2,
    embargo: int = 0,
) -> list[PurgedSplit]:
    """Combinatorial purged+embargoed splits.

    Parameters
    ----------
    t0_pos, t1_pos : int arrays (len = n_events)
        Grid positions of each event's decision bar (t0) and label-window end
        (t1). ``t1_pos >= t0_pos`` required.
    n_grid : int
        Length of the time grid the folds partition.
    n_folds, k : int
        Combinatorial CV: test set is every C(n_folds, k) tuple of folds.
    embargo : int
        Number of grid bars after each test fold to drop from training.

    Returns one PurgedSplit per fold-combination. Guarantees: train ∩ test = ∅,
    and no training event's label window overlaps any test fold (or its embargo).
    """
    t0_pos = np.asarray(t0_pos, dtype=int)
    t1_pos = np.asarray(t1_pos, dtype=int)
    if t0_pos.shape != t1_pos.shape:
        raise ValueError("t0_pos and t1_pos must be the same length")
    if (t1_pos < t0_pos).any():
        raise ValueError("t1_pos must be >= t0_pos for every event")
    if (t0_pos < 0).any() or (t1_pos >= n_grid).any():
        raise ValueError("event positions out of [0, n_grid) range")

    folds = make_folds(n_grid, n_folds)
    membership = fold_membership(folds, n_grid)
    event_fold = membership[t0_pos]   # which fold each event's t0 belongs to

    splits: list[PurgedSplit] = []
    for combo in cpcv_test_combos(n_folds, k):
        test_mask = np.isin(event_fold, combo)
        train_mask = ~test_mask
        for fi in combo:
            f = folds[fi]
            # Purge: drop train events whose label window [t0,t1] overlaps the
            # test fold's bar range [f.start, f.end).
            overlap = (t0_pos < f.end) & (t1_pos >= f.start)
            train_mask &= ~overlap
            # Embargo: drop train events whose decision bar falls in the band
            # immediately AFTER the test fold.
            if embargo > 0:
                emb_end = min(n_grid, f.end + embargo)
                in_emb = (t0_pos >= f.end) & (t0_pos < emb_end)
                train_mask &= ~in_emb
        splits.append(PurgedSplit(
            test_idx=np.where(test_mask)[0],
            train_idx=np.where(train_mask)[0],
            test_folds=tuple(combo),
        ))
    return splits


def purged_cpcv(
    t0: pd.Series,
    t1: pd.Series,
    grid: pd.DatetimeIndex,
    *,
    n_folds: int = 6,
    k: int = 2,
    embargo_frac: float = 0.01,
) -> list[PurgedSplit]:
    """Timestamp-friendly wrapper: takes event ``t0``/``t1`` timestamps and a
    bar ``grid``, returns purged combinatorial splits. ``embargo_frac`` is the
    embargo as a fraction of the grid length (AFML suggests ~1%)."""
    t0_pos = event_grid_positions(t0, grid)
    t1_pos = event_grid_positions(t1, grid)
    embargo = int(round(embargo_frac * len(grid)))
    return purged_cpcv_splits(t0_pos, t1_pos, len(grid),
                              n_folds=n_folds, k=k, embargo=embargo)
