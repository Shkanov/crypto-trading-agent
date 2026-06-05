"""Tests for research/ml_meta/events.py + tz-aware regressions.

The decisive test is `test_compute_indicators_is_causal`: an indicator value at
bar t must be identical whether computed on the full series or on a prefix ending
at t. If truncating the future changes the past, a feature is peeking ahead.
"""
import numpy as np
import pandas as pd

from research.ml_meta.cv import event_grid_positions
from research.ml_meta.events import MeanRevParams, compute_indicators, mean_reversion_events
from research.ml_meta.labeling import average_uniqueness


def _synthetic_klines(n=1200, seed=7):
    rng = np.random.RandomState(seed)
    close = 100 * np.exp(np.cumsum(rng.randn(n) * 0.02))
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    high = close * (1 + np.abs(rng.randn(n)) * 0.005)
    low = close * (1 - np.abs(rng.randn(n)) * 0.005)
    return pd.DataFrame({"high": high, "low": low, "close": close,
                         "open": close, "volume": 1.0, "quote_volume": 1.0}, index=idx)


def test_compute_indicators_is_causal():
    k = _synthetic_klines()
    p = MeanRevParams()
    full = compute_indicators(k, p)
    cut = 900
    prefix = compute_indicators(k.iloc[:cut], p)
    # On the shared, warmed-up region the values must match exactly.
    shared = full.iloc[200:cut - 1]
    pre = prefix.iloc[200:cut - 1]
    for col in ["rsi", "atr", "adx", "bbl", "bbm", "bbu", "stoch_k"]:
        pd.testing.assert_series_equal(shared[col], pre[col], check_names=False)


def test_events_structure_and_pit_horizon():
    k = _synthetic_klines()
    ev = mean_reversion_events(k, MeanRevParams())
    # Structure invariants hold regardless of how many fire.
    assert set(["side", "trgt", "t_vertical", "stretch", "rsi", "adx"]).issubset(ev.columns)
    if len(ev):
        assert (ev.t_vertical > ev.index).all()          # time barrier strictly forward
        assert set(np.unique(ev.side)).issubset({-1.0, 1.0})
        assert (ev.trgt > 0).all()
        assert (ev.stretch >= 0).all()


def test_events_prefix_matches_full_for_contained_events():
    k = _synthetic_klines()
    p = MeanRevParams()
    full = mean_reversion_events(k, p)
    cut = 1000
    prefix = mean_reversion_events(k.iloc[:cut], p)
    # Events whose whole window fits inside the prefix must be identical.
    contained = full[full.t_vertical <= k.index[cut - 1]]
    common = contained.index.intersection(prefix.index)
    if len(common):
        pd.testing.assert_series_equal(
            full.loc[common, "side"], prefix.loc[common, "side"], check_names=False)
        pd.testing.assert_series_equal(
            full.loc[common, "trgt"], prefix.loc[common, "trgt"], check_names=False)


def test_tz_aware_uniqueness_regression():
    # The bug that real data caught: .values stripped tz and mismatched the grid.
    grid = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
    t0 = pd.Series([grid[0], grid[3]])
    t1 = pd.Series([grid[2], grid[5]])
    w = average_uniqueness(grid, t0, t1)
    assert np.allclose(w.values, [1.0, 1.0])


def test_tz_aware_grid_positions_regression():
    grid = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
    pos = event_grid_positions(pd.Series([grid[1], grid[7]]), grid)
    assert list(pos) == [1, 7]
