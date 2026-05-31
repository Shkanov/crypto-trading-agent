"""Combinatorial Purged Cross-Validation (CPCV) and Probability of Backtest
Overfitting (PBO).

Replaces the 3-window walk-forward harness with the López de Prado (2018,
*Advances in Financial Machine Learning*, Ch. 12) CPCV procedure plus the
Bailey-Borwein-López de Prado-Zhu (2017) PBO statistic. With only 3 windows
the test-statistic variance is so high that a 60% in-sample-positive result
is statistically indistinguishable from noise; with CPCV(N=10, k=2)=45
combinations we get a real OOS Sharpe distribution per parameter trial.

Two related procedures live here:

  1. **CPCV** — for one parameter configuration, partition the timeline into
     N contiguous folds, hold out every combination of k=2 folds as the test
     set, and compute test-window Sharpe on the trades that fall in the held
     out folds. The result is C(N,k) OOS Sharpe samples per config — a
     distribution, not a single point estimate. Purging (drop train trades
     whose lifespan overlaps a test fold) and embargo (drop train trades
     immediately after a test fold) prevent leakage from autocorrelated
     returns and overlapping trade durations.

  2. **PBO** — across a parameter sweep of N_trials configs, partition T
     observations (rows) into S subsamples. For each C(S, S/2) partition into
     in-sample J / out-of-sample J̄, find the IS-best trial and read its OOS
     rank. PBO = P(rank ≤ median) — the chance that the best parameter set
     in-sample lands in the bottom half out-of-sample. PBO > 0.5 is selection
     bias worse than random and the strategy family should be rejected.

This module is pure logic. Callers run the backtests, daily-bucket the
trade PnLs into a (T_days × N_trials) returns matrix, and pass that here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# CPCV fold infrastructure

@dataclass(frozen=True)
class Fold:
    """One contiguous time-slice. `start`/`end` are row indices (inclusive,
    exclusive) into a returns matrix or trade timeline."""
    idx: int
    start: int
    end: int


def make_folds(n_periods: int, n_folds: int) -> list[Fold]:
    """Partition `n_periods` row indices into `n_folds` contiguous equal-sized
    folds. The last fold absorbs the remainder when `n_periods` is not
    divisible by `n_folds`.
    """
    if n_folds <= 0:
        raise ValueError("n_folds must be > 0")
    if n_periods < n_folds:
        raise ValueError(f"n_periods={n_periods} < n_folds={n_folds}")
    size = n_periods // n_folds
    out: list[Fold] = []
    for i in range(n_folds):
        start = i * size
        end = (i + 1) * size if i < n_folds - 1 else n_periods
        out.append(Fold(idx=i, start=start, end=end))
    return out


def cpcv_test_combos(n_folds: int, k: int) -> list[tuple[int, ...]]:
    """All C(n_folds, k) tuples of fold indices to use as the OOS test set.

    With N=10, k=2 this yields 45 combinations — the canonical CPCV setup.
    """
    if not 1 <= k < n_folds:
        raise ValueError(f"need 1 <= k < n_folds (got k={k}, n_folds={n_folds})")
    return list(combinations(range(n_folds), k))


def fold_membership(folds: list[Fold], n_periods: int) -> np.ndarray:
    """Length-n_periods array where each entry is the fold index that owns it."""
    out = np.full(n_periods, -1, dtype=np.int64)
    for f in folds:
        out[f.start: f.end] = f.idx
    return out


def holdout_mask(folds: list[Fold], test_idxs: tuple[int, ...],
                 n_periods: int) -> np.ndarray:
    """Boolean mask of rows that belong to any of the test folds."""
    out = np.zeros(n_periods, dtype=bool)
    for i in test_idxs:
        f = folds[i]
        out[f.start: f.end] = True
    return out


def train_mask_with_embargo(
    folds: list[Fold], test_idxs: tuple[int, ...],
    n_periods: int, embargo: int = 0,
) -> np.ndarray:
    """Train mask = complement of (test ∪ embargo bands). An embargo of `e`
    drops the next `e` rows AFTER each test fold from training, since
    return autocorrelation can leak information forward.

    Note: this module doesn't itself run training (we evaluate per-config
    stats on each holdout), but the helper is here for callers that do
    machine-learning model fitting on the residual folds.
    """
    is_test = holdout_mask(folds, test_idxs, n_periods)
    is_train = ~is_test
    for i in test_idxs:
        f = folds[i]
        emb_end = min(n_periods, f.end + max(0, embargo))
        is_train[f.end: emb_end] = False
    return is_train


# ---------------------------------------------------------------------------
# Bucketing trade PnLs into a time-aligned returns matrix

def daily_bucket_pnls(
    trade_ts_ms: Iterable[int],
    trade_pnls: Iterable[float],
    day0_ms: int,
    n_days: int,
) -> np.ndarray:
    """Sum trade PnLs into per-day buckets indexed by entry timestamp.

    `day0_ms` is the start of day 0 (UTC midnight is the usual choice).
    Trades before day 0 or beyond `n_days` are dropped. Returns a length
    `n_days` float array — usable directly as one column of the (T, N)
    PBO matrix.
    """
    out = np.zeros(n_days, dtype=float)
    day_ms = 86_400_000
    for ts, pnl in zip(trade_ts_ms, trade_pnls):
        if pnl is None:
            continue
        d = (int(ts) - day0_ms) // day_ms
        if 0 <= d < n_days:
            out[d] += float(pnl)
    return out


def sharpe_per_column(matrix: np.ndarray, periods_per_year: float = 365.0) -> np.ndarray:
    """Annualised Sharpe for each column of `matrix` (rows = time, cols =
    strategies). Zero-volatility columns return 0."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"matrix must be 2-D (got {m.shape})")
    mu = m.mean(axis=0)
    sd = m.std(axis=0, ddof=1)
    out = np.zeros_like(mu)
    # Use a relative + absolute floor: numpy returns ~1e-17 std for arrays
    # filled with a non-representable float like 0.1 due to FP quantization.
    # The IS-best from such a "constant" column would otherwise explode.
    threshold = 1e-12 * (np.abs(mu) + 1.0)
    nz = sd > threshold
    out[nz] = mu[nz] / sd[nz] * math.sqrt(periods_per_year)
    return out


# ---------------------------------------------------------------------------
# Bailey-Borwein-López de Prado-Zhu PBO

@dataclass
class PBOResult:
    """Output of `pbo`. `logits` are log(λ / (1-λ)) per partition; their mean
    is the "stochastic dominance" reading (negative = selection generalises
    poorly). Bailey et al. report both the probability and the logit mean."""
    pbo: float
    n_partitions: int
    n_trials: int
    logits: list[float] = field(default_factory=list)
    mean_logit: float = 0.0
    median_oos_rank_pct: float = 0.0   # mean OOS rank of IS-best, in [0,1]
    n_dead_columns: int = 0            # configs dropped as non-trading (all-zero)


def select_is_best_idx(
    sharpes: Iterable[float],
    trades: Optional[Iterable[int]] = None,
    min_trades: int = 1,
) -> int:
    """Index of the in-sample-best config, EXCLUDING non-trading configs.

    A config that never trades has a flat all-zero PnL column → Sharpe 0. In a
    family where every genuinely-trading config has NEGATIVE Sharpe, that 0
    would win `argmax` and be reported as "IS-best" (and games PBO). Restrict
    the argmax to configs with >= `min_trades` trades. Falls back to the raw
    argmax only when nothing traded (degenerate family)."""
    sr = np.asarray(list(sharpes), dtype=float)
    if trades is None:
        return int(np.argmax(sr))
    tr = list(trades)
    elig = [i for i, t in enumerate(tr) if t >= min_trades]
    if not elig:
        return int(np.argmax(sr))
    return max(elig, key=lambda i: float(sr[i]))


def pbo(
    matrix: np.ndarray,
    s: int = 16,
    periods_per_year: float = 365.0,
    max_partitions: Optional[int] = None,
    rng_seed: Optional[int] = None,
    active_mask: Optional[Iterable[bool]] = None,
) -> PBOResult:
    """Bailey-Borwein-LdP-Zhu (2017) Probability of Backtest Overfitting.

    `matrix` shape (T, N): T = time observations, N = strategy/parameter
    trials. We partition T rows into S subsamples of T/S rows each, then
    iterate over all C(S, S/2) ways to assign half as in-sample J and half
    as out-of-sample J̄:

        1. n* = argmax over trials of Sharpe(J, trial)
        2. ω̄ = rank of n* within Sharpe(J̄, ·)   (1 = worst, N = best)
        3. λ = ω̄ / (N + 1)
        4. logit = ln(λ / (1 - λ))
        5. count this partition as "overfit" if λ ≤ 0.5

    PBO = #overfit / #partitions. For S=16 there are C(16,8)=12,870 partitions;
    pass `max_partitions` and `rng_seed` to subsample when N_trials is large
    or the matrix has many rows.

    NON-TRADING CONFIGS ARE DROPPED. A config that never trades is an all-zero
    column with Sharpe 0; in a family of otherwise-losing configs that 0 wins
    the step-1 argmax every partition and produces a spuriously LOW PBO (a dead
    strategy "passing"). We drop all-zero columns, and additionally any column
    masked False by `active_mask` (e.g. trades < a caller's min-trades floor).
    If fewer than 2 trading configs survive, the family can't be validated:
    returns pbo=1.0 (REJECT) with n_partitions=0 as the degenerate signal.

    Returns a PBOResult; floor on surviving trials is 2. For N<8 the rank
    discretisation makes the metric crude; Bailey et al. recommend N >= 10.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"matrix must be 2-D (got {m.shape})")
    if m.shape[1] < 2:
        raise ValueError("need at least 2 trials for PBO")
    # Drop non-trading / inactive columns (see docstring). This is distinct
    # from the malformed-input check above: the family was well-formed, but
    # some configs never traded and must not be eligible as IS-best.
    dead = np.all(m == 0.0, axis=0)
    if active_mask is not None:
        dead = dead | ~np.asarray(list(active_mask), dtype=bool)
    n_dead = int(dead.sum())
    if n_dead:
        m = m[:, ~dead]
    t, n = m.shape
    if n < 2:
        # Fewer than 2 TRADING configs survived → cannot estimate selection
        # bias. Treat as REJECT (pbo=1.0) with the degenerate n_partitions=0.
        return PBOResult(pbo=1.0, n_partitions=0, n_trials=n,
                         n_dead_columns=n_dead)
    if s < 2 or s % 2 != 0:
        raise ValueError(f"s must be an even integer >= 2 (got {s})")
    if t < s:
        raise ValueError(f"need t >= s rows for {s} subsamples (got t={t})")

    # Trim to a multiple of s so every subsample has equal size.
    rows_per_sub = t // s
    usable = rows_per_sub * s
    m = m[:usable]
    subs = m.reshape(s, rows_per_sub, n)

    all_partitions = list(combinations(range(s), s // 2))
    if max_partitions is not None and len(all_partitions) > max_partitions:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(all_partitions), size=max_partitions, replace=False)
        partitions = [all_partitions[i] for i in idx]
    else:
        partitions = all_partitions

    logits: list[float] = []
    ranks_pct: list[float] = []
    overfit = 0
    for combo in partitions:
        J_mask = np.zeros(s, dtype=bool)
        J_mask[list(combo)] = True
        J = subs[J_mask].reshape(-1, n)
        Jbar = subs[~J_mask].reshape(-1, n)
        sr_J = sharpe_per_column(J, periods_per_year=periods_per_year)
        sr_Jbar = sharpe_per_column(Jbar, periods_per_year=periods_per_year)
        n_star = int(np.argmax(sr_J))
        # rank: 1 = worst, n = best
        order = np.argsort(sr_Jbar)        # ascending by Sharpe
        rank_arr = np.empty(n, dtype=np.int64)
        rank_arr[order] = np.arange(1, n + 1)
        rank_n_star = int(rank_arr[n_star])
        rank_pct = rank_n_star / (n + 1)
        ranks_pct.append(rank_pct)
        if rank_pct <= 0.5:
            overfit += 1
        # logit (clip for log)
        lam = min(max(rank_pct, 1e-9), 1 - 1e-9)
        logits.append(math.log(lam / (1.0 - lam)))

    return PBOResult(
        pbo=overfit / len(partitions),
        n_partitions=len(partitions),
        n_trials=n,
        logits=logits,
        mean_logit=float(np.mean(logits)) if logits else 0.0,
        median_oos_rank_pct=float(np.mean(ranks_pct)) if ranks_pct else 0.0,
        n_dead_columns=n_dead,
    )


# ---------------------------------------------------------------------------
# CPCV-Sharpe-per-config (one strategy's OOS distribution)

def cpcv_oos_sharpes(
    column: np.ndarray,
    n_folds: int = 10,
    k: int = 2,
    periods_per_year: float = 365.0,
) -> list[float]:
    """For ONE parameter config, walk every C(n_folds, k) holdout combo and
    compute Sharpe over the held-out rows. Returns a list of length C(n,k).

    This is the per-config OOS distribution used to (a) compute mean/std of
    OOS Sharpe and (b) discount the in-sample Sharpe through that variance.
    """
    arr = np.asarray(column, dtype=float)
    if arr.ndim != 1:
        raise ValueError("column must be 1-D")
    folds = make_folds(arr.shape[0], n_folds)
    out: list[float] = []
    for combo in cpcv_test_combos(n_folds, k):
        mask = holdout_mask(folds, combo, arr.shape[0])
        seg = arr[mask]
        if seg.size < 2:
            out.append(0.0)
            continue
        sd = seg.std(ddof=1)
        if sd == 0:
            out.append(0.0)
        else:
            out.append(float(seg.mean() / sd * math.sqrt(periods_per_year)))
    return out
