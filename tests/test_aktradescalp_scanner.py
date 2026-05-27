"""Pure-logic tests for aktradescalp scanner helpers (no network, no data)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.scanners.aktradescalp_scanner import (
    ScannerParams,
    SymbolFeatures,
    UniverseParams,
    _nearest_round_prox_bps,
    score_universe,
)


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


# ─────────────── momentum-primary mode (sprint #15, §2.4.2) ────────────────

def _mid_session_ms() -> int:
    """A Monday 09:00 UTC timestamp — within the default 07-12 session
    window and not a Friday (so friday_multiplier doesn't kick in)."""
    # 2026-05-25 was a Monday (verified)
    return int(datetime(2026, 5, 25, 9, 0, tzinfo=timezone.utc).timestamp() * 1000)


def _make_features(
    sym: str, ts_ms: int, ret_30d_bps: float, ret_7d_bps: float,
    vol_z: float, oi_z: float, funding: float,
) -> SymbolFeatures:
    return SymbolFeatures(
        symbol=sym, ts_ms=ts_ms,
        quote_vol_24h_usd=200_000_000.0,
        days_since_listing=400.0,
        vol_z_1h_sameHour_30d=vol_z,
        oi_z_24h_30d=oi_z,
        ret_24h_bps=ret_7d_bps / 7.0,            # not used in momentum-primary
        ret_30d_bps=ret_30d_bps,
        ret_7d_bps=ret_7d_bps,
        funding_rate_8h=funding,
        history_ok=True,
    )


def _momentum_universe(ts_ms: int, n: int = 20) -> dict[str, SymbolFeatures]:
    """20-symbol universe with monotonic 30d momentum from -2000 bps to
    +2000 bps. Symbols #0..#1 are bottom decile; #18..#19 top decile.
    Vol-z / OI-z / funding are set so that only the EXTREMES have ≥2
    confirmers — middle of the pack would fail."""
    out: dict[str, SymbolFeatures] = {}
    for i in range(n):
        sym = f"SYM{i:02d}USDT"
        # Linear ramp from -2000 to +2000 bps over 30d, with 7d echoing.
        ret_30d = -2000.0 + (4000.0 * i / (n - 1))
        ret_7d = ret_30d * 0.4
        # Tail confirmers: extremes get vol/oi/funding spikes; middle is quiet.
        in_extreme = (i < 2) or (i >= n - 2)
        vol_z = 3.0 if in_extreme else 0.5
        oi_z = 2.5 if in_extreme else 0.3
        # Funding: extremes have |rate| ≈ 0.002 (well above any 95th pct);
        # middle near zero.
        funding = (0.002 if i >= n - 2 else (-0.002 if i < 2 else 0.0001 * i))
        out[sym] = _make_features(sym, ts_ms, ret_30d, ret_7d, vol_z, oi_z, funding)
    return out


def test_momentum_primary_returns_only_top_and_bottom_decile() -> None:
    ts = _mid_session_ms()
    feats = _momentum_universe(ts, n=20)
    s = ScannerParams(use_momentum_primary=True, momentum_top_pct=0.10,
                      confirmers_required=2)
    cands = score_universe(feats, ts, UniverseParams(), s)
    # Top-decile (2 symbols) + bot-decile (2 symbols) = 4 candidates max.
    syms = {c.symbol for c in cands}
    assert syms == {"SYM00USDT", "SYM01USDT", "SYM18USDT", "SYM19USDT"}
    assert len(cands) == 4


def test_momentum_primary_assigns_side_hint_by_direction() -> None:
    ts = _mid_session_ms()
    feats = _momentum_universe(ts, n=20)
    s = ScannerParams(use_momentum_primary=True, momentum_top_pct=0.10,
                      confirmers_required=2)
    cands = score_universe(feats, ts, UniverseParams(), s)
    by_sym = {c.symbol: c for c in cands}
    # Bottom-momentum → short continuation; top-momentum → long continuation
    assert by_sym["SYM00USDT"].side_hint == "short"
    assert by_sym["SYM01USDT"].side_hint == "short"
    assert by_sym["SYM18USDT"].side_hint == "long"
    assert by_sym["SYM19USDT"].side_hint == "long"


def test_momentum_primary_requires_confirmers() -> None:
    """If the extremes have only ONE confirmer (vol_z only, no OI or funding
    extreme), they should be filtered out under confirmers_required=2."""
    ts = _mid_session_ms()
    n = 20
    feats: dict[str, SymbolFeatures] = {}
    for i in range(n):
        sym = f"SYM{i:02d}USDT"
        ret_30d = -2000.0 + (4000.0 * i / (n - 1))
        ret_7d = ret_30d * 0.4
        # ONLY vol_z fires at extremes — oi_z, funding stay quiet
        in_extreme = (i < 2) or (i >= n - 2)
        vol_z = 3.0 if in_extreme else 0.5
        feats[sym] = _make_features(sym, ts, ret_30d, ret_7d,
                                     vol_z=vol_z, oi_z=0.3, funding=0.0)
    s = ScannerParams(use_momentum_primary=True, momentum_top_pct=0.10,
                      confirmers_required=2)
    cands = score_universe(feats, ts, UniverseParams(), s)
    assert cands == [], (
        f"expected 0 candidates with only 1 confirmer; got {[c.symbol for c in cands]}"
    )


def test_momentum_primary_funding_threshold_is_universe_percentile() -> None:
    """When ALL symbols have similar funding (no extremes), nothing should
    clear the 95th-percentile funding gate — but if the top-decile happens
    to also have vol_z+oi_z, they still qualify with confirmers_required=2."""
    ts = _mid_session_ms()
    n = 20
    feats: dict[str, SymbolFeatures] = {}
    for i in range(n):
        sym = f"SYM{i:02d}USDT"
        ret_30d = -2000.0 + (4000.0 * i / (n - 1))
        ret_7d = ret_30d * 0.4
        # All funding equal → no symbol clears the 95th percentile
        in_extreme = (i < 2) or (i >= n - 2)
        vol_z = 3.0 if in_extreme else 0.5
        oi_z = 2.5 if in_extreme else 0.3
        feats[sym] = _make_features(sym, ts, ret_30d, ret_7d,
                                     vol_z=vol_z, oi_z=oi_z, funding=0.0001)
    s = ScannerParams(use_momentum_primary=True, momentum_top_pct=0.10,
                      confirmers_required=2)
    cands = score_universe(feats, ts, UniverseParams(), s)
    # vol_z and oi_z fire (2 confirmers); funding doesn't. Confirmers=2 met.
    syms = {c.symbol for c in cands}
    assert syms == {"SYM00USDT", "SYM01USDT", "SYM18USDT", "SYM19USDT"}
    # Confirmers should not include funding for any of them
    for c in cands:
        assert c.components.get("funding_extreme") is False


def test_momentum_primary_off_uses_legacy_scoring() -> None:
    """Sanity: with use_momentum_primary=False (default), the legacy
    additive scoring is used — symbols with attention ≥ score_min qualify."""
    ts = _mid_session_ms()
    feats = _momentum_universe(ts, n=20)
    s_legacy = ScannerParams(use_momentum_primary=False)
    s_new = ScannerParams(use_momentum_primary=True,
                          momentum_top_pct=0.10, confirmers_required=2)
    legacy = score_universe(feats, ts, UniverseParams(), s_legacy)
    new = score_universe(feats, ts, UniverseParams(), s_new)
    # Legacy mode scores all symbols with ≥ 2 hits; new mode gates by
    # momentum decile first → the two sets should NOT be equal.
    assert {c.symbol for c in legacy} != {c.symbol for c in new}


def test_momentum_primary_session_gate_still_applies() -> None:
    """Outside the configured session window, momentum-primary returns []
    just like the legacy path."""
    # 02:00 UTC is outside the default 07-12 session
    out_of_session = int(datetime(2026, 5, 25, 2, 0, tzinfo=timezone.utc).timestamp() * 1000)
    feats = _momentum_universe(out_of_session, n=20)
    s = ScannerParams(use_momentum_primary=True)
    assert score_universe(feats, out_of_session, UniverseParams(), s) == []


def test_momentum_primary_handles_missing_returns() -> None:
    """Symbols with missing ret_30d or ret_7d are excluded from ranking;
    they don't count toward the universe size or appear as candidates."""
    ts = _mid_session_ms()
    feats = _momentum_universe(ts, n=20)
    # Wipe momentum on one of the top-decile entries
    feats["SYM19USDT"] = SymbolFeatures(
        symbol="SYM19USDT", ts_ms=ts,
        quote_vol_24h_usd=200_000_000.0,
        days_since_listing=400.0,
        vol_z_1h_sameHour_30d=3.0, oi_z_24h_30d=2.5,
        ret_24h_bps=100.0, ret_30d_bps=None, ret_7d_bps=None,
        funding_rate_8h=0.002,
        history_ok=True,
    )
    s = ScannerParams(use_momentum_primary=True, momentum_top_pct=0.10,
                      confirmers_required=2)
    cands = score_universe(feats, ts, UniverseParams(), s)
    syms = {c.symbol for c in cands}
    assert "SYM19USDT" not in syms
