"""Mean-reversion regime gate: Hurst exponent + Variance-Ratio + OU half-life.

Replaces ADX<20 as the regime filter (per `Chan, Algorithmic Trading` ch. 2 +
Macrosynergy 2023 empirical study: ADX is the weakest filter; the 3-test
stack Hurst<0.4 AND VR-reject AND OU-half-life in [4,30] bars eliminates 60-80%
of false-positive entries on directional crypto).

All three tests operate on a price-return series at the strategy timeframe.
Caller-supplied window is typically 100-200 bars (Chan p. 49)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class RegimeGateParams:
    hurst_window: int = 100              # bars used to estimate Hurst
    hurst_max: float = 0.40              # H<0.40 = mean-reverting / anti-persistent
    vr_lags: tuple[int, ...] = (2, 4, 8)
    vr_alpha: float = 0.05               # reject random-walk at 5%
    ou_min_half_life_bars: float = 4.0
    ou_max_half_life_bars: float = 30.0
    ou_window: int = 200                 # bars used to fit OU


def hurst_exponent(prices: Sequence[float], max_lag: int = 20) -> Optional[float]:
    """Hurst via rescaled range / classic log-log regression of variance vs lag.

    Returns H in [0, 1]. H ≈ 0.5 random walk. H < 0.5 anti-persistent / MR.
    H > 0.5 persistent / trending. Returns None when not enough data."""
    n = len(prices)
    if n < max_lag * 2 + 10:
        return None
    arr = np.asarray(prices, dtype=float)
    lags = list(range(2, max_lag + 1))
    tau = []
    for lag in lags:
        diffs = arr[lag:] - arr[:-lag]
        if len(diffs) < 2:
            continue
        sd = float(np.std(diffs, ddof=1))
        if sd <= 0:
            continue
        tau.append(sd)
    if len(tau) < 4:
        return None
    log_lags = np.log(lags[:len(tau)])
    log_tau = np.log(tau)
    slope = float(np.polyfit(log_lags, log_tau, 1)[0])
    return max(0.0, min(1.0, slope))


def variance_ratio_test(returns: Sequence[float], lag: int) -> Optional[float]:
    """Lo-MacKinlay 1988 simple variance ratio at horizon `lag`.

    VR(lag) = Var(r_t + r_{t+1} + ... + r_{t+lag-1}) / (lag * Var(r_t))
    VR < 1 => negative autocorrelation (mean reversion).
    VR > 1 => positive autocorrelation (momentum).
    Caller compares against 1.0 with an asymptotic z-test (we provide just the
    point estimate here; the threshold check is in `passes_variance_ratio`).
    Returns None when insufficient data."""
    arr = np.asarray(returns, dtype=float)
    if len(arr) < lag * 4 or lag < 2:
        return None
    var1 = float(np.var(arr, ddof=1))
    if var1 <= 0:
        return None
    # k-period returns are sums of `lag` consecutive 1-period returns.
    summed = np.array([float(arr[i:i + lag].sum()) for i in range(0, len(arr) - lag + 1)])
    var_k = float(np.var(summed, ddof=1))
    return var_k / (lag * var1)


def passes_variance_ratio(
    returns: Sequence[float],
    lags: tuple[int, ...] = (2, 4, 8),
    alpha: float = 0.05,
) -> bool:
    """True iff all `lags` produce VR < 1 (mean-reverting evidence).

    `alpha` is unused in this simple-form check; including for API symmetry.
    For a strict z-test, multiply VR-1 by sqrt(n) and compare to Φ⁻¹(alpha/2).
    Most practitioners just take VR<1 across multiple lags as evidence."""
    for lag in lags:
        v = variance_ratio_test(returns, lag)
        if v is None or v >= 1.0:
            return False
    return True


def ou_half_life_bars(prices: Sequence[float]) -> Optional[float]:
    """Half-life of an OU process fitted by AR(1) regression of Δprice on price.

    Discrete OU:  Δx_t = -θ (x_{t-1} - μ) dt + σ dW_t
                  Δx_t = β x_{t-1} + α + ε,  β = -θ Δt
                  half_life = log(2) / |β|/dt   ≈  log(2) / (-β) for unit dt

    Returns None when β is non-negative (no mean reversion) or insufficient data.
    """
    arr = np.asarray(prices, dtype=float)
    n = len(arr)
    if n < 30:
        return None
    x = arr[:-1]
    dx = arr[1:] - x
    # OLS: dx = α + β x
    A = np.vstack([np.ones(len(x)), x]).T
    try:
        sol, *_ = np.linalg.lstsq(A, dx, rcond=None)
    except np.linalg.LinAlgError:
        return None
    beta = float(sol[1])
    if beta >= 0:
        return None
    return float(math.log(2.0) / -beta)


def passes_regime_gate(
    prices: Sequence[float],
    p: Optional[RegimeGateParams] = None,
) -> bool:
    """All three checks must pass:
      1. Hurst < hurst_max  (anti-persistent series)
      2. VR<1 across `vr_lags` (negative serial correlation)
      3. OU half-life ∈ [ou_min, ou_max] bars (revert within hold horizon)
    """
    p = p or RegimeGateParams()
    if len(prices) < max(p.hurst_window, p.ou_window):
        return False
    win_h = list(prices[-p.hurst_window:])
    h = hurst_exponent(win_h)
    if h is None or h >= p.hurst_max:
        return False
    rets = list(np.diff(np.asarray(prices[-p.ou_window:], dtype=float)))
    if not passes_variance_ratio(rets, p.vr_lags, p.vr_alpha):
        return False
    win_o = list(prices[-p.ou_window:])
    hl = ou_half_life_bars(win_o)
    if hl is None:
        return False
    return p.ou_min_half_life_bars <= hl <= p.ou_max_half_life_bars
