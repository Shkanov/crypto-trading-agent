"""Engle-Granger cointegrated pairs strategy (Krauss 2017; Tadi & Witzany 2023).

The point: cointegration-based pairs trading is the most-documented positive-edge
mean-reversion technique in crypto (Lintilhac & Tourin 2017 Sharpe ~3 on BTC/LTC
1h; Fischer-Krauss-Deinert 2019 Sharpe ~2.3 on 40-coin residual-MR; Tadi-Witzany
2023 copula variant). The signal does NOT require directional alpha — it harvests
short-run divergence of two cointegrated price series.

Pipeline per refit cycle:
  1. Fetch log-prices for symbol-A and symbol-B over `lookback_days`.
  2. Engle-Granger: OLS of log(B) on log(A); test residuals for stationarity
     via ADF. Accept the pair iff ADF p < 0.05.
  3. Compute z-score of current residual using rolling 60-90d mean/std.
  4. Entry: |z| >= z_entry. Long the underpriced leg + short the overpriced.
  5. Exit when |z| < z_exit. Stop out when |z| > z_stop (regime broke).

The strategy is pure logic — no I/O. Driver scripts fetch prices and pass them
in. Same model can run live (1h refit, intraday z-monitoring) or in backtest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PairsParams:
    """Cointegration entry/exit thresholds. Defaults from Tadi-Witzany 2023."""
    lookback_bars: int = 24 * 60          # 60d at 1h
    refit_every_bars: int = 24 * 7        # weekly refit
    coint_pvalue_max: float = 0.05
    coint_pvalue_violations_to_drop: int = 3  # drop pair if 3 refits fail
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.5
    min_lookback_for_z: int = 200         # need this many bars to score z


@dataclass
class CointegrationFit:
    """Result of one Engle-Granger refit."""
    alpha: float                          # intercept  log(B) ≈ alpha + beta*log(A)
    beta: float                           # hedge ratio (units of A per unit of B)
    residuals: np.ndarray                 # log(B) - alpha - beta*log(A)
    adf_p: float                          # ADF test on residuals
    is_cointegrated: bool                 # adf_p < p.coint_pvalue_max
    resid_mean: float
    resid_std: float


def _adf_pvalue_via_ols(residuals: np.ndarray) -> float:
    """Approximate ADF p-value via OLS regression of Δresid on lagged resid
    (no constant, no trend, no lag terms — the simplest Augmented Dickey-Fuller).

    Returns a one-sided p-value derived from the t-statistic of β in
        Δy_t = β * y_{t-1} + ε_t
    using MacKinnon (1996) critical values approximated as:
      t < -3.43 → p ≈ 0.01
      t < -2.86 → p ≈ 0.05
      t < -2.57 → p ≈ 0.10
    Linearly interpolated outside these thresholds. Good enough for pair-validation;
    use statsmodels.tsa.stattools.adfuller for production calibration.
    """
    y = residuals.astype(float)
    if len(y) < 25:
        return 1.0
    dy = np.diff(y)
    y_lag = y[:-1]
    n = len(y_lag)
    if np.var(y_lag) == 0:
        return 1.0
    # OLS: dy = beta * y_lag + e  (no const)
    beta = float(np.sum(y_lag * dy) / np.sum(y_lag * y_lag))
    eps = dy - beta * y_lag
    sigma2 = float(np.sum(eps * eps) / max(1, n - 1))
    var_beta = sigma2 / np.sum(y_lag * y_lag)
    if var_beta <= 0:
        return 1.0
    t_stat = beta / math.sqrt(var_beta)
    # MacKinnon 1996 critical values for τ_μ (no constant, sample size ~250)
    if t_stat < -3.43:
        return 0.01
    if t_stat < -2.86:
        # linear interp between 0.01 and 0.05
        return 0.01 + (t_stat - -3.43) / (-2.86 - -3.43) * (0.05 - 0.01)
    if t_stat < -2.57:
        return 0.05 + (t_stat - -2.86) / (-2.57 - -2.86) * (0.10 - 0.05)
    if t_stat < -1.62:   # ~10% one-sided normal
        return 0.10 + (t_stat - -2.57) / (-1.62 - -2.57) * (0.50 - 0.10)
    return min(1.0, 0.5 + (t_stat - -1.62) / 3.0)


def fit_engle_granger(
    log_a: Sequence[float],
    log_b: Sequence[float],
    p: Optional[PairsParams] = None,
) -> Optional[CointegrationFit]:
    """Fit log(B) = α + β * log(A) + ε and ADF-test ε.

    Returns None when input series are too short or degenerate.
    """
    p = p or PairsParams()
    a = np.asarray(log_a, dtype=float)
    b = np.asarray(log_b, dtype=float)
    if len(a) != len(b) or len(a) < p.min_lookback_for_z:
        return None
    X = np.vstack([np.ones(len(a)), a]).T
    try:
        sol, *_ = np.linalg.lstsq(X, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    alpha, beta = float(sol[0]), float(sol[1])
    resid = b - alpha - beta * a
    p_val = _adf_pvalue_via_ols(resid)
    return CointegrationFit(
        alpha=alpha, beta=beta, residuals=resid,
        adf_p=p_val,
        is_cointegrated=p_val < p.coint_pvalue_max,
        resid_mean=float(resid.mean()),
        resid_std=float(resid.std(ddof=1)) if len(resid) > 1 else 0.0,
    )


def current_residual(
    fit: CointegrationFit,
    log_a_t: float,
    log_b_t: float,
) -> float:
    """The residual at time t given the fitted (α, β)."""
    return log_b_t - fit.alpha - fit.beta * log_a_t


def current_zscore(fit: CointegrationFit, resid_t: float) -> Optional[float]:
    if fit.resid_std <= 0:
        return None
    return (resid_t - fit.resid_mean) / fit.resid_std


@dataclass
class PairSignal:
    """Output of `evaluate_pair`.

    `side` is the long leg of the pair:
      - 'long_b_short_a': B is undervalued → buy B, sell A. Fired when z < -entry.
      - 'long_a_short_b': A is undervalued → buy A, sell B. Fired when z > +entry.
      - None means hold (no entry).
    """
    side: Optional[str]
    z: float
    is_exit: bool
    is_stop: bool
    fit: CointegrationFit


def evaluate_pair(
    fit: CointegrationFit,
    log_a_t: float,
    log_b_t: float,
    current_side: Optional[str],
    p: Optional[PairsParams] = None,
) -> PairSignal:
    """One-bar decision: enter / exit / hold / stop, given a fitted pair and
    the current state of (a, b).

    `current_side` is the existing leg the strategy is on, or None.
    """
    p = p or PairsParams()
    r_t = current_residual(fit, log_a_t, log_b_t)
    z = current_zscore(fit, r_t)
    if z is None:
        return PairSignal(side=None, z=0.0, is_exit=False, is_stop=False, fit=fit)

    if current_side is not None:
        # Exit if z reverts to median band
        if abs(z) < p.z_exit:
            return PairSignal(side=None, z=z, is_exit=True, is_stop=False, fit=fit)
        # Stop if regime appears broken
        if abs(z) > p.z_stop:
            return PairSignal(side=None, z=z, is_exit=False, is_stop=True, fit=fit)
        # Stop also if pair lost cointegration on most recent refit
        if not fit.is_cointegrated:
            return PairSignal(side=None, z=z, is_exit=False, is_stop=True, fit=fit)
        return PairSignal(side=current_side, z=z, is_exit=False, is_stop=False, fit=fit)

    if not fit.is_cointegrated:
        return PairSignal(side=None, z=z, is_exit=False, is_stop=False, fit=fit)

    if z >= p.z_entry:
        # B is overpriced relative to A → short B, long A
        return PairSignal(side="long_a_short_b", z=z, is_exit=False, is_stop=False, fit=fit)
    if z <= -p.z_entry:
        return PairSignal(side="long_b_short_a", z=z, is_exit=False, is_stop=False, fit=fit)
    return PairSignal(side=None, z=z, is_exit=False, is_stop=False, fit=fit)
