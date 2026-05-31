"""Unit tests for src.services.cpcv — pure logic, no network."""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from src.services.cpcv import (
    cpcv_oos_sharpes,
    cpcv_test_combos,
    daily_bucket_pnls,
    fold_membership,
    make_folds,
    pbo,
    select_is_best_idx,
    sharpe_per_column,
    holdout_mask,
    train_mask_with_embargo,
)


# ---------------------------------------------------------------------------
# Fold infrastructure

def test_make_folds_even_partition() -> None:
    folds = make_folds(100, 10)
    assert len(folds) == 10
    assert [f.start for f in folds] == [i * 10 for i in range(10)]
    assert [f.end for f in folds] == [(i + 1) * 10 for i in range(10)]


def test_make_folds_remainder_in_last() -> None:
    folds = make_folds(103, 10)
    # Last fold gets the extra 3 rows
    assert folds[-1].end == 103
    assert folds[-1].start == 90
    assert folds[-2].end == 90


def test_make_folds_rejects_bad_inputs() -> None:
    import pytest
    with pytest.raises(ValueError):
        make_folds(5, 10)
    with pytest.raises(ValueError):
        make_folds(100, 0)


def test_cpcv_test_combos_count_matches_choose() -> None:
    # Canonical CPCV: C(10, 2) = 45
    combos = cpcv_test_combos(10, 2)
    assert len(combos) == 45
    # All combos are sorted, unique, of length k
    assert all(c == tuple(sorted(c)) for c in combos)
    assert all(len(c) == 2 for c in combos)
    assert len(set(combos)) == 45


def test_fold_membership_assigns_each_row() -> None:
    folds = make_folds(20, 4)
    mem = fold_membership(folds, 20)
    # Each row owned by exactly one fold; folds 0..3 each cover 5 rows.
    assert (mem >= 0).all()
    for i in range(4):
        assert (mem == i).sum() == 5


def test_holdout_mask_covers_test_folds() -> None:
    folds = make_folds(100, 10)
    mask = holdout_mask(folds, (2, 7), 100)
    assert mask.sum() == 20
    assert mask[20:30].all() and mask[70:80].all()
    # Nothing else marked
    assert not mask[:20].any() and not mask[30:70].any() and not mask[80:].any()


def test_train_mask_embargo_zeroes_after_each_test_fold() -> None:
    folds = make_folds(100, 10)
    train = train_mask_with_embargo(folds, (2, 7), 100, embargo=3)
    # Test rows are out
    assert not train[20:30].any()
    assert not train[70:80].any()
    # Embargo: 3 rows after each test fold are dropped
    assert not train[30:33].any()
    assert not train[80:83].any()
    # Rest is train
    assert train[33:70].all()


# ---------------------------------------------------------------------------
# Daily bucketing

def test_daily_bucket_pnls_aggregates_by_day() -> None:
    day0 = 1_700_000_000_000
    day_ms = 86_400_000
    ts = [day0 + 1, day0 + day_ms + 100, day0 + 2 * day_ms]
    pnls = [10.0, -5.0, 7.5]
    out = daily_bucket_pnls(ts, pnls, day0_ms=day0, n_days=5)
    assert out[0] == 10.0
    assert out[1] == -5.0
    assert out[2] == 7.5
    assert out[3] == 0.0


def test_daily_bucket_pnls_drops_out_of_range() -> None:
    day0 = 1_700_000_000_000
    day_ms = 86_400_000
    ts = [day0 - day_ms, day0 + 10 * day_ms]
    out = daily_bucket_pnls(ts, [1.0, 1.0], day0_ms=day0, n_days=5)
    assert out.sum() == 0.0


# ---------------------------------------------------------------------------
# Sharpe per column

def test_sharpe_per_column_basic() -> None:
    # Column 0: constant positive returns → high Sharpe (perfectly stable up)
    # Actually std=0 → returns 0 by contract.
    # Column 1: noisy positive drift → finite Sharpe
    rng = np.random.default_rng(42)
    n = 252
    col0 = np.full(n, 0.1)        # zero variance
    col1 = 0.1 + rng.normal(0, 1, n)  # drift + noise
    m = np.column_stack([col0, col1])
    sr = sharpe_per_column(m, periods_per_year=252.0)
    assert sr[0] == 0.0
    assert sr[1] > 0.5


def test_sharpe_per_column_rejects_1d() -> None:
    import pytest
    with pytest.raises(ValueError):
        sharpe_per_column(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# CPCV per-config OOS distribution

def test_cpcv_oos_sharpes_returns_45_for_default_grid() -> None:
    arr = np.full(200, 0.05) + np.random.default_rng(7).normal(0, 1, 200)
    out = cpcv_oos_sharpes(arr, n_folds=10, k=2, periods_per_year=252.0)
    assert len(out) == 45
    assert all(isinstance(x, float) for x in out)


def test_cpcv_oos_sharpes_low_variance_when_returns_uniform() -> None:
    # Stationary positive-drift signal → OOS Sharpe across all 45 combos
    # should cluster tightly.
    rng = np.random.default_rng(13)
    arr = 0.5 + rng.normal(0, 1, 1000)
    out = cpcv_oos_sharpes(arr, n_folds=10, k=2, periods_per_year=252.0)
    sd = float(np.std(out))
    # Spread should be modest — stationary means each holdout sees similar dynamics
    assert sd < 2.0


# ---------------------------------------------------------------------------
# PBO

def test_pbo_rejects_non_2d_or_too_few_trials() -> None:
    import pytest
    with pytest.raises(ValueError):
        pbo(np.zeros((100,)))
    with pytest.raises(ValueError):
        pbo(np.zeros((100, 1)))


def test_pbo_drops_dead_nontrading_columns() -> None:
    """A non-trading config (all-zero column, Sharpe 0) must NOT be eligible as
    IS-best in a family of otherwise-losing configs, and must be dropped from
    PBO — otherwise a dead strategy 'passes' with a spuriously low PBO.
    Regression for the cpcv_validate_pairs null-config bug (2026-05-31)."""
    rng = np.random.default_rng(7)
    n_obs = 1600
    # 20 genuinely-losing configs (negative drift) + 1 dead all-zero column.
    losers = rng.normal(-0.05, 1.0, size=(n_obs, 20))
    dead = np.zeros((n_obs, 1))
    m = np.column_stack([losers, dead])
    res = pbo(m, s=16)
    assert res.n_dead_columns == 1
    assert res.n_trials == 20            # dead column excluded
    # With the dead column gone, PBO is computed over the losing family only.
    # (Pre-fix the all-zero column won every argmax and forced PBO ≈ 0.)
    assert res.n_partitions == math.comb(16, 8)


def test_select_is_best_idx_skips_nontrading() -> None:
    # config 2 has the highest Sharpe but never traded → must be skipped.
    sharpes = [-1.0, -0.5, 0.0, -0.8]
    trades = [10, 4, 0, 7]
    assert select_is_best_idx(sharpes, trades) == 1
    # all dead → fall back to raw argmax (degenerate).
    assert select_is_best_idx([0.0, 0.0], [0, 0]) in (0, 1)
    # no trade info → raw argmax.
    assert select_is_best_idx([1.0, 2.0, 0.5]) == 1


def test_pbo_degenerate_when_all_dead() -> None:
    """If <2 trading configs survive, PBO can't estimate selection bias →
    returns pbo=1.0 (REJECT) with n_partitions=0 instead of a false pass."""
    m = np.zeros((200, 5))
    m[:, 0] = np.random.default_rng(1).normal(0, 1, 200)  # only 1 trades
    res = pbo(m, s=8)
    assert res.n_partitions == 0
    assert res.pbo == 1.0
    assert res.n_dead_columns == 4


def test_pbo_uncorrelated_columns_near_half() -> None:
    """When trial returns are pure iid noise across columns, the IS-best
    column is no better than random OOS → PBO should hover around 0.5.
    With S=16 we have C(16,8)=12,870 partitions which keeps variance low."""
    rng = np.random.default_rng(11)
    n_trials = 30
    n_obs = 1600
    m = rng.normal(0, 1, size=(n_obs, n_trials))
    res = pbo(m, s=16)
    assert res.n_partitions == math.comb(16, 8)
    # Honest no-signal band. The IS-best of N pure-noise trials is unbiased
    # by symmetry → OOS rank distribution is uniform → mean rank_pct ≈ 0.5.
    assert 0.40 <= res.pbo <= 0.60


def test_pbo_dominant_column_low_pbo() -> None:
    """When one trial has uniformly higher mean than the rest, its OOS rank
    is consistently at the top → PBO well under 0.5."""
    rng = np.random.default_rng(17)
    n_trials = 10
    n_obs = 1600
    m = rng.normal(0, 1, size=(n_obs, n_trials))
    m[:, 0] += 2.0   # dominant trial drift
    res = pbo(m, s=16)
    assert res.pbo < 0.05
    # Mean OOS rank of IS-best should be near top (1.0)
    assert res.median_oos_rank_pct > 0.9


def test_pbo_partition_count_matches_choose() -> None:
    rng = np.random.default_rng(0)
    m = rng.normal(0, 1, size=(160, 5))
    res = pbo(m, s=8)
    assert res.n_partitions == math.comb(8, 4)  # = 70
    assert res.n_trials == 5
    assert len(res.logits) == res.n_partitions


def test_pbo_subsampled_partitions_match_request() -> None:
    rng = np.random.default_rng(2)
    m = rng.normal(0, 1, size=(160, 6))
    res = pbo(m, s=8, max_partitions=20, rng_seed=99)
    assert res.n_partitions == 20
