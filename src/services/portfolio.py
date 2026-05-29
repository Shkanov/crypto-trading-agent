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
# Active-day volatility
#
# The return series fed to the allocator are zero-padded on every day a
# strategy didn't trade (see `build_strategy_returns`). A sleeve that trades
# 20 of 357 days is 94% zeros, so the std of the *padded* series collapses
# toward zero — which makes both inverse-vol (w∝1/σ) and HRP (inverse-variance
# bisection) read "rarely active" as "low risk" and pile capital onto the
# thinnest sleeves. `_active_day_std` measures volatility *conditional on the
# sleeve being active* (std over the non-zero entries), so idle days no longer
# masquerade as low risk.

def _active_day_std(series: np.ndarray, min_vol: float = 1e-12) -> float:
    """Std (ddof=1) over the non-zero entries of `series`, floored at
    `min_vol`. Fewer than 2 active days → `min_vol` (insufficient evidence to
    estimate vol; the evidence floor in `allocate` is the intended guard for
    these). Treats exact-zero days as "did not trade"."""
    arr = np.asarray(series, dtype=float)
    active = arr[arr != 0.0]
    if active.size < 2:
        return min_vol
    return max(float(active.std(ddof=1)), min_vol)


# ---------------------------------------------------------------------------
# Inverse vol

def inverse_vol(
    strategy_returns: dict[str, np.ndarray],
    min_vol: float = 1e-12,
) -> dict[str, float]:
    """w_i ∝ 1 / σ_i, where σ_i is the **active-day** volatility (see
    `_active_day_std`) rather than the std of the zero-padded series. This
    stops idle sleeves from claiming the largest weight just for being
    inactive. Sleeves with <2 active days fall to the `min_vol` floor and
    would otherwise dominate — `allocate`'s evidence floor scrubs those."""
    if not strategy_returns:
        return {}
    syms = list(strategy_returns.keys())
    vols = np.array([_active_day_std(strategy_returns[s], min_vol) for s in syms])
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

    # Per-sleeve risk = volatility *conditional on the sleeve being active*
    # (std over non-zero days), so idle days don't deflate it. Sleeves with no
    # usable risk estimate — never traded in-window or constant — are "dead":
    # exclude them and run HRP on the survivors, rather than letting one
    # zero-variance column collapse the whole basket to equal-weight. With a
    # 90-day lookback a sleeve that trades ~15 days a year is routinely idle for
    # a full window, so this path matters in practice.
    MIN_VOL = 1e-12
    active_var = np.array(
        [_active_day_std(M[:, i], MIN_VOL) ** 2 for i in range(n)], dtype=float,
    )
    live_idx = [i for i in range(n) if active_var[i] > MIN_VOL * MIN_VOL]
    if len(live_idx) < n:
        if len(live_idx) <= 1:
            # 0 or 1 sleeve carries risk → nothing to diversify across.
            return equal_weight(syms)
        live = [syms[i] for i in live_idx]
        sub = hrp({s: strategy_returns[s] for s in live})
        return {s: float(sub.get(s, 0.0)) for s in syms}

    # Drop all-quiet days (every sleeve zero): they carry no cross-sectional
    # information and only deflate variance / inflate spurious co-movement
    # among intermittent sleeves. Correlations come from this reduced basis.
    active_rows = M[~np.all(M == 0.0, axis=1)]
    basis = active_rows if active_rows.shape[0] >= 2 else M
    cov = np.cov(basis.T, ddof=1)
    if cov.ndim == 0:                                   # n=1 numerical edge
        return {syms[0]: 1.0}
    cov = np.array(cov, dtype=float, copy=True)
    # Override the diagonal with active-day variances so the inverse-variance
    # bisection in `_cluster_variance` sees each sleeve's true risk rather than
    # zero-padding-deflated variance.
    np.fill_diagonal(cov, active_var)
    # Correlation; clamp anything outside [-1, 1] (numerical noise + the
    # diagonal override can push a ratio slightly past 1).
    std = np.sqrt(np.diag(cov))
    if np.any(std == 0):
        # Defensive: a constant active basis → equal-weight.
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


def _active_day_count(series: np.ndarray) -> int:
    """Number of days the sleeve actually traded (non-zero return entries)."""
    return int(np.count_nonzero(np.asarray(series, dtype=float)))


def _apply_evidence_floor(
    weights: dict[str, float],
    strategy_returns: dict[str, np.ndarray],
    min_active_days: int,
) -> tuple[dict[str, float], list[str]]:
    """Zero out sleeves with fewer than `min_active_days` active days in the
    window and renormalize the survivors to sum 1. A floored sleeve earns its
    way back once it trades enough. If *no* sleeve clears the floor (or the
    survivors carry no weight) the input is returned unchanged — we never blank
    the whole book. Returns `(weights, floored_names)`."""
    if min_active_days <= 0 or not weights:
        return dict(weights), []
    floored = [
        s for s in weights
        if _active_day_count(strategy_returns.get(s, np.empty(0))) < min_active_days
    ]
    if not floored:
        return dict(weights), []
    kept_total = sum(w for s, w in weights.items() if s not in floored)
    if kept_total <= 0:
        return dict(weights), []          # nothing survives → leave as-is
    out = {
        s: (weights[s] / kept_total if s not in floored else 0.0)
        for s in weights
    }
    return out, floored


def _apply_weight_cap(
    weights: dict[str, float], max_weight: float,
) -> tuple[dict[str, float], bool]:
    """Iteratively cap any sleeve above `max_weight`, redistributing the excess
    pro-rata to the headroom of the uncapped sleeves until all are at or below
    the cap. The cap is raised to 1/k when it would be infeasible for k nonzero
    sleeves to sum to 1 (e.g. a single survivor must hold 100%). Returns
    `(weights, capped_any)`."""
    nonzero = {s: w for s, w in weights.items() if w > 0.0}
    k = len(nonzero)
    if k == 0 or max_weight <= 0.0 or max_weight >= 1.0:
        return dict(weights), False
    cap = max(max_weight, 1.0 / k)        # feasibility floor
    if all(w <= cap + 1e-12 for w in nonzero.values()):
        return dict(weights), False
    w = dict(weights)
    capped_any = False
    for _ in range(k + 1):
        over = [s for s in nonzero if w[s] > cap + 1e-12]
        if not over:
            break
        capped_any = True
        excess = sum(w[s] - cap for s in over)
        for s in over:
            w[s] = cap
        recipients = {s: w[s] for s in nonzero if w[s] < cap - 1e-12}
        room = sum(cap - rw for rw in recipients.values())
        if room <= 0:
            break
        for s in recipients:
            w[s] += excess * (cap - recipients[s]) / room
    return w, capped_any


def allocate(
    strategy_returns: dict[str, np.ndarray],
    method: str = "equal",
    fallback: str = "inverse_vol",
    turnover_threshold: float = 0.5,
    prev_weights: Optional[dict[str, float]] = None,
    min_active_days: int = 0,
    max_weight: float = 1.0,
) -> AllocationResult:
    """Compute weights with the requested method, falling back if HRP
    turnover exceeds `turnover_threshold`, then apply two structural
    constraints to whatever weights result (including the fallback):

      * **evidence floor** — sleeves with fewer than `min_active_days` active
        days in the window are zeroed and the rest renormalized, so a
        thin-track-record sleeve can't dominate the book on a lucky window;
      * **weight cap** — no sleeve exceeds `max_weight` (excess redistributed
        to the others), bounding single-sleeve drawdown contribution.

    `method` ∈ {"equal", "inverse_vol", "hrp"}.
    `fallback` is the method used when the turnover guard fires (only
    triggers when `method="hrp"`). The defaults `min_active_days=0` and
    `max_weight=1.0` are no-ops, preserving the pre-constraint behaviour for
    callers that don't opt in.
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

    method_used = method
    reasons: list[str] = []

    # Turnover guard fires on the *raw* HRP turnover, before constraints.
    raw_turnover = _turnover_l1(w, prev_weights) if prev_weights else 0.0
    if method == "hrp" and prev_weights and raw_turnover > turnover_threshold:
        w = (equal_weight(syms) if fallback == "equal"
             else inverse_vol(strategy_returns))
        method_used = fallback
        reasons.append(
            f"hrp turnover {raw_turnover:.2f} > {turnover_threshold:.2f}; "
            f"fell back to {fallback}"
        )

    # Structural constraints — applied to the chosen weights (fallback too).
    w, floored = _apply_evidence_floor(w, strategy_returns, min_active_days)
    if floored:
        reasons.append(
            f"evidence floor (<{min_active_days} active days) zeroed "
            f"{', '.join(sorted(floored))}"
        )
    w, capped = _apply_weight_cap(w, max_weight)
    if capped:
        reasons.append(f"capped single-sleeve weight at {max_weight:.0%}")

    # Report turnover against the final, constrained weights.
    turnover = _turnover_l1(w, prev_weights) if prev_weights else 0.0
    return AllocationResult(
        weights=w,
        method_used=method_used,
        turnover=turnover,
        reason="; ".join(reasons),
    )


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
