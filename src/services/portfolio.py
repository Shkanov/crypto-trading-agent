"""Multi-strategy portfolio allocator (sprint #17, §3.4).

Three allocation methods:

  1. **equal_weight** — w_i = 1/N. DeMiguel-Garlappi-Uppal 2009 showed that
     for ≤5 strategies, equal-weight typically beats any "fancy" method
     out of sample because parameter estimation error swamps the
     diversification benefit. This is the recommended starting point.

  2. **inverse_vol** — w_i ∝ 1 / σ_i. Each strategy contributes the same
     marginal risk to the portfolio. Robust and fast — the canonical
     fallback when fancier methods misbehave.

  3. **hrp** — López de Prado (2016) Hierarchical Risk Parity. Three steps:
       a. tree clustering on the correlation-distance d_ij = √(½(1−ρ_ij));
       b. quasi-diagonalisation: reorder by the dendrogram leaf order;
       c. recursive bisection: split halves, allocate inverse-variance
          between halves until each leaf has its share.
     Diversifies along the empirical correlation structure without
     requiring matrix inversion, so it's less sensitive than mean-variance
     to near-singular covariance matrices.

The module is **pure logic**: takes a `{strategy_name: return_series}`
dict, returns a `{strategy_name: weight}` dict whose values sum to 1.
The caller (orchestrator + monthly rebalance loop) is responsible for
maintaining the rolling-90d window, deciding when to rebalance, and
applying the weights to live capital.

`allocate(...)` is the high-level dispatcher with a turnover guard:
HRP weights can be jittery when strategy correlations are unstable, so
if the L1 distance from the previous weight vector exceeds
`turnover_threshold` we silently fall back to inverse-vol (or
equal-weight) for that rebalance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from src.services.storage import Storage


DAY_MS = 86_400_000


# ---------------------------------------------------------------------------
# Equal weight

def equal_weight(strategies: list[str]) -> dict[str, float]:
    """w_i = 1/N. Returns {} when `strategies` is empty."""
    if not strategies:
        return {}
    n = len(strategies)
    return {s: 1.0 / n for s in strategies}


# ---------------------------------------------------------------------------
# Inverse vol

def inverse_vol(
    strategy_returns: dict[str, np.ndarray],
    min_vol: float = 1e-12,
) -> dict[str, float]:
    """w_i ∝ 1 / σ_i. Zero-variance strategies are clamped to `min_vol`
    so they don't claim infinite weight; in practice this maps them to
    the largest weight in the basket — equivalent to "treat constant
    return as zero risk" (caller's responsibility to scrub these)."""
    if not strategy_returns:
        return {}
    syms = list(strategy_returns.keys())
    vols = np.array([
        max(float(np.asarray(strategy_returns[s], dtype=float).std(ddof=1)
                   if len(strategy_returns[s]) > 1 else min_vol),
            min_vol)
        for s in syms
    ])
    inv = 1.0 / vols
    inv = inv / inv.sum()
    return {syms[i]: float(inv[i]) for i in range(len(syms))}


# ---------------------------------------------------------------------------
# HRP — López de Prado 2016

def _single_linkage_leaf_order(dist: np.ndarray) -> list[int]:
    """Agglomerative single-linkage hierarchical clustering. Returns the
    leaf-order list used for HRP's quasi-diagonalisation.

    `dist` is the (N×N) distance matrix. The algorithm:
      1. Each item is its own cluster.
      2. Find the two closest clusters by single linkage (min pairwise dist
         between members) and merge their leaf lists by concatenation.
      3. Repeat until one cluster remains.
    The final cluster's leaf list is the quasi-diagonal order.

    O(N³) — fine for N ≤ ~50 strategies. We never need more than ~10.
    """
    n = dist.shape[0]
    if n == 0:
        return []
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > 1:
        # Find argmin of single-linkage distance between every pair.
        best_i = best_j = -1
        best_d = math.inf
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d_ij = min(dist[a, b] for a in clusters[i] for b in clusters[j])
                if d_ij < best_d:
                    best_d = d_ij
                    best_i, best_j = i, j
        merged = clusters[best_i] + clusters[best_j]
        # Remove in reverse-index order to keep indices stable.
        clusters = [c for k, c in enumerate(clusters)
                    if k != best_i and k != best_j]
        clusters.append(merged)
    return clusters[0]


def _cluster_variance(cov: np.ndarray, idxs: list[int]) -> float:
    """Inverse-variance-weighted cluster variance — w_i ∝ 1/σ²_i, then
    σ²_cluster = w' Σ w. Used inside the recursive bisection to compute
    each half's "risk budget"."""
    var_i = np.diag(cov)[idxs]
    # Guard against zero-variance entries
    var_i = np.where(var_i > 0, var_i, 1e-12)
    w = 1.0 / var_i
    w = w / w.sum()
    sub = cov[np.ix_(idxs, idxs)]
    return float(w @ sub @ w)


def hrp(
    strategy_returns: dict[str, np.ndarray],
) -> dict[str, float]:
    """Hierarchical Risk Parity (López de Prado 2016) on the strategy
    return series. Each value is a 1-D numpy array of equal length.

    Returns a {strategy: weight} dict whose values sum to 1. With 1
    strategy → all weight to it; with ≥2 strategies → the full HRP recipe.

    Degenerate cases (no returns, NaN columns) fall back to equal-weight
    so the allocator can never crash the caller.
    """
    if not strategy_returns:
        return {}
    syms = list(strategy_returns.keys())
    n = len(syms)
    if n == 1:
        return {syms[0]: 1.0}

    # Stack returns into a (T, N) matrix.
    series_list = [np.asarray(strategy_returns[s], dtype=float) for s in syms]
    min_len = min(s.size for s in series_list)
    if min_len < 2:
        # Can't compute covariance — equal-weight fallback.
        return equal_weight(syms)
    # Right-align to the shortest series (most recent window).
    M = np.column_stack([s[-min_len:] for s in series_list])
    cov = np.cov(M.T, ddof=1)
    if cov.ndim == 0:                                   # n=1 numerical edge
        return {syms[0]: 1.0}
    # Correlation; clamp anything outside [-1, 1] (numerical noise).
    std = np.sqrt(np.diag(cov))
    if np.any(std == 0):
        # A constant-return strategy has zero variance → equal-weight.
        return equal_weight(syms)
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    # Distance: d_ij = sqrt(0.5 * (1 - ρ_ij))   ∈ [0, 1]
    d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))

    order = _single_linkage_leaf_order(d)

    # Recursive bisection over the ordered list — iterative via stack.
    weights = np.ones(n, dtype=float)
    # Stack holds slices of `order` (lists of original indices).
    stack: list[list[int]] = [order]
    while stack:
        cluster = stack.pop()
        if len(cluster) <= 1:
            continue
        mid = len(cluster) // 2
        left = cluster[:mid]
        right = cluster[mid:]
        v_l = _cluster_variance(cov, left)
        v_r = _cluster_variance(cov, right)
        total = v_l + v_r
        if total <= 0:
            alpha = 0.5
        else:
            alpha = 1.0 - v_l / total
        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= (1.0 - alpha)
        stack.append(left)
        stack.append(right)

    weights = weights / weights.sum()
    return {syms[i]: float(weights[i]) for i in range(n)}


# ---------------------------------------------------------------------------
# Allocator dispatcher with turnover fallback

@dataclass(frozen=True)
class AllocationResult:
    weights: dict[str, float]
    method_used: str
    turnover: float                # L1 distance from prev_weights, 0 if no prev
    reason: str = ""               # populated when a fallback fires


def _turnover_l1(new: dict[str, float], prev: dict[str, float]) -> float:
    """L1 = ½ ∑ |w_new - w_prev|. Range [0, 1]: 0 = identical, 1 = total
    rotation. The ½ matches the standard PM convention so a one-asset swap
    of 50% → 50% counts as 0.5 turnover, not 1.0."""
    keys = set(new) | set(prev)
    return 0.5 * sum(abs(new.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)


def allocate(
    strategy_returns: dict[str, np.ndarray],
    method: str = "equal",
    fallback: str = "inverse_vol",
    turnover_threshold: float = 0.5,
    prev_weights: Optional[dict[str, float]] = None,
) -> AllocationResult:
    """Compute weights with the requested method, falling back if HRP
    turnover exceeds `turnover_threshold`.

    `method` ∈ {"equal", "inverse_vol", "hrp"}.
    `fallback` is the method used when the turnover guard fires (only
    triggers when `method="hrp"`).
    """
    if method not in ("equal", "inverse_vol", "hrp"):
        raise ValueError(f"unknown allocation method: {method!r}")
    if fallback not in ("equal", "inverse_vol"):
        raise ValueError(f"unknown fallback: {fallback!r}")

    syms = list(strategy_returns.keys())
    if method == "equal":
        w = equal_weight(syms)
    elif method == "inverse_vol":
        w = inverse_vol(strategy_returns)
    else:
        w = hrp(strategy_returns)

    turnover = _turnover_l1(w, prev_weights) if prev_weights else 0.0

    if method == "hrp" and prev_weights and turnover > turnover_threshold:
        w_fb = (equal_weight(syms) if fallback == "equal"
                else inverse_vol(strategy_returns))
        turnover_fb = _turnover_l1(w_fb, prev_weights)
        return AllocationResult(
            weights=w_fb,
            method_used=fallback,
            turnover=turnover_fb,
            reason=(f"hrp turnover {turnover:.2f} > {turnover_threshold:.2f}; "
                    f"fell back to {fallback}"),
        )

    return AllocationResult(weights=w, method_used=method, turnover=turnover)


# ---------------------------------------------------------------------------
# Strategy returns builder — feeds `allocate()` from Storage

async def build_strategy_returns(
    storage: "Storage",
    strategy_names: list[str],
    reference_equity_usd: float,
    now_ms: int,
    lookback_days: int = 90,
) -> dict[str, np.ndarray]:
    """Build a {strategy: daily_return_pct_array} dict for the allocator.

    Returns are computed as `daily_pnl_usd / reference_equity_usd * 100`. Same
    denominator across strategies ⇒ correlation/inverse-vol relationships are
    preserved while putting magnitudes on a portfolio-equity scale. We don't
    try to track a separate per-strategy equity base; that would require
    persisting the historical slice schedule and adds little for HRP (which
    is correlation-driven).

    Each output array has `lookback_days` entries. Strategies absent from
    storage rows entirely produce all-zero arrays (which inverse-vol clamps
    to its `min_vol` floor and HRP detects and falls back to equal-weight).
    """
    today_ms = (now_ms // DAY_MS) * DAY_MS
    start_ms = today_ms - lookback_days * DAY_MS
    per_strat = await storage.realized_pnl_by_day_per_strategy(
        start_ms, today_ms + DAY_MS,
    )
    denom = reference_equity_usd if reference_equity_usd > 0 else 1.0
    out: dict[str, np.ndarray] = {}
    days = [start_ms + i * DAY_MS for i in range(lookback_days)]
    for name in strategy_names:
        day_map = per_strat.get(name, {})
        series = np.array(
            [day_map.get(d, 0.0) / denom * 100.0 for d in days],
            dtype=float,
        )
        out[name] = series
    return out
