"""Nonlinear autoregressive learners + honest baselines.

ForecastAGLT's measured directional skill came from an LSTM (0.564) and GMDH
(0.551) — a neural and a polynomial nonlinear AR. The gmdh C++ binding segfaults
under numpy-2/py-3.13 on this box, and a single Keras LSTM over 15 scalar lags is
not meaningfully more expressive than other nonlinear regressors for this task.
So the *capability* (nonlinear univariate AR) is exercised by two robust,
already-present model classes:

  * gbm — lightgbm gradient-boosted trees (the tree-nonlinear family),
  * mlp — sklearn MLPRegressor (the neural-nonlinear family, the LSTM's stand-in).

Reporting BOTH is the test: a real edge survives across model classes; an edge
that lives in exactly one config was overfit. Two baselines anchor the read:

  * momentum — predict next return = last return (sign of f{L-1}); the trivial
    "weekly momentum" hypothesis the AR models must beat,
  * always_long / always_short — constant +1 / −1. These bracket the market's
    unconditional drift over the test window. They are the CRITICAL baselines: if
    the universe trended (it did — alts fell over 2024-26), a model that merely
    tilts with the drift scores >0.50 directional with ZERO forecasting skill, and
    always_short will match or beat it. A learned model only demonstrates skill if
    it beats BOTH the drift baselines AND survives de-drifting (scoring on each
    week's cross-sectionally-demeaned returns).

Each learner exposes fit(X, y) / predict(X)->ndarray. X is StandardScaler-ed on
the TRAIN fold only (done by the caller) — harmless for trees, required for MLP.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.neural_network import MLPRegressor


class _Momentum:
    """Predict next return = most-recent lagged return (last feature col)."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Momentum":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, -1]


class _Constant:
    """Constant-sign prediction → always-long (+1) / always-short (−1) book.
    The market-drift baselines a real model must beat."""
    def __init__(self, sign: float):
        self.sign = sign

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Constant":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.sign)


def _gbm() -> lgb.LGBMRegressor:
    # Shallow + regularized: weekly AR samples are few and noisy, so capacity is
    # the first line of defense against overfitting (mirrors the meta-model).
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        verbosity=-1,
        n_jobs=1,
    )


def _mlp(seed: int) -> MLPRegressor:
    # One small hidden layer + L2 + early stopping — the LSTM's lightweight
    # stand-in. Kept small for the same small-data reason.
    return MLPRegressor(
        hidden_layer_sizes=(16,),
        activation="tanh",
        alpha=1e-2,                 # L2
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.15,
        random_state=seed,
    )


MODELS = ("gbm", "mlp", "momentum", "always_long", "always_short")
LEARNED = ("gbm", "mlp")            # the ones that actually fit X→y


def make_model(name: str, seed: int = 42):
    if name == "gbm":
        return _gbm()
    if name == "mlp":
        return _mlp(seed)
    if name == "momentum":
        return _Momentum()
    if name == "always_long":
        return _Constant(1.0)
    if name == "always_short":
        return _Constant(-1.0)
    raise ValueError(f"unknown model {name!r}")
