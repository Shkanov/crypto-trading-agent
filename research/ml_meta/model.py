"""LightGBM meta-classifier: P(this primary signal wins after costs).

Deliberately shallow + regularized — meta-labeling datasets are small and noisy,
so the defense against overfitting is model capacity, not just CV. Trees need no
feature scaling. Hyperparameters are FIXED here; any future tuning must run
inside the CPCV folds and be discounted by PBO (DESIGN.md).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS = dict(
    objective="binary",
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=4,
    min_child_samples=30,     # don't split on a handful of noisy events
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=0.0,
    verbosity=-1,
    n_jobs=1,
)


def train(X: pd.DataFrame, y: pd.Series, w: pd.Series,
          params: dict | None = None) -> lgb.LGBMClassifier:
    m = lgb.LGBMClassifier(**(params or DEFAULT_PARAMS))
    m.fit(X, y, sample_weight=w.values)   # keep feature names consistent
    return m


def predict_win_proba(m: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    return m.predict_proba(X)[:, 1]
