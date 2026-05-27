"""Bailey-López de Prado deflated Sharpe — pure-logic tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.services.backtest import _deflated_sharpe, _norm_cdf, _norm_ppf


def test_norm_ppf_round_trip():
    for p in (0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975):
        x = _norm_ppf(p)
        assert _norm_cdf(x) == pytest.approx(p, abs=1e-4)


def test_norm_ppf_symmetric():
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-6)


def test_deflated_sharpe_too_few_trades():
    assert _deflated_sharpe(2.0, n_trades=1) == 0.0


def test_deflated_sharpe_penalty_grows_with_trials():
    # Same raw SR, more trials → larger E[max_SR] → smaller DSR
    sr = 2.0
    d5 = _deflated_sharpe(sr, n_trades=200, n_trials=5)
    d100 = _deflated_sharpe(sr, n_trades=200, n_trials=100)
    assert d5 > d100


def test_deflated_sharpe_floors_at_zero():
    # Tiny raw SR, big trial count → DSR floors at 0 (no edge after penalty)
    d = _deflated_sharpe(0.1, n_trades=50, n_trials=100)
    assert d == 0.0


def test_deflated_sharpe_passes_raw_sharpe_when_few_trials():
    # SR=3 with only 2 trials should retain most of the raw SR
    d = _deflated_sharpe(3.0, n_trades=200, n_trials=2)
    assert d > 1.5  # most of the raw 3.0 survives


def test_deflated_sharpe_handles_fat_tail_pnls():
    # Skewed/leptokurtic PnL → denominator changes, but our implementation
    # uses the subtractive form so PnL distribution doesn't blow up the
    # result. Verify it remains finite and floored.
    pnls = [1.0] * 50 + [-50.0]  # one big loss
    d = _deflated_sharpe(1.5, n_trades=51, n_trials=20, pnls=pnls)
    assert d >= 0.0
    assert math.isfinite(d)


def test_deflated_sharpe_uses_trial_variance_when_supplied():
    sr_trials = [0.2, 0.5, 1.0, 1.5, 2.0, 0.8, 1.2]  # noisy distribution
    d_with = _deflated_sharpe(
        2.0, n_trades=100, n_trials=20,
        sr_trial_distribution=sr_trials,
    )
    d_without = _deflated_sharpe(2.0, n_trades=100, n_trials=20)
    # Empirical V[SR] of those trials ≈ 0.37, vs default 0.5 → smaller
    # penalty → larger DSR.
    assert d_with > d_without
