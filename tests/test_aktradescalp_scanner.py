"""Pure-logic tests for aktradescalp scanner helpers (no network, no data)."""
from __future__ import annotations

import pytest

from src.scanners.aktradescalp_scanner import _nearest_round_prox_bps


@pytest.mark.parametrize("px,expected_max_bps", [
    # aktrad's announced levels from the corpus — current px is somewhere
    # near the level, scanner must classify it as "close to a round level".
    (0.0498, 50),     # FIDA target was 0.05; px right under should be <50bps
    (0.198,  200),    # UB target was 0.20; px under 0.2 should hit
    (0.495,  150),    # PHA target was 0.50
    (1.02,   300),    # generic px just above 1.0
    (9.7,    400),    # px under 10
    (4.8,    500),    # px under 5
])
def test_nearest_round_prox_is_small_near_aktrad_levels(px, expected_max_bps):
    prox = _nearest_round_prox_bps(px)
    assert prox is not None
    assert prox <= expected_max_bps, f"px={px} prox={prox:.1f}bps > {expected_max_bps}bps"


@pytest.mark.parametrize("px", [0.0, -1.0, -0.5])
def test_nearest_round_prox_invalid_px(px):
    assert _nearest_round_prox_bps(px) is None


def test_nearest_round_prox_far_from_grid():
    # px in the middle of [2, 5] decade — nearest level is 2.0 or 5.0, both
    # >30% away. Helper returns a large bps value.
    px = 3.5
    prox = _nearest_round_prox_bps(px)
    assert prox is not None
    assert prox > 1000, f"px={px} should be far from round levels, got {prox:.1f}bps"
