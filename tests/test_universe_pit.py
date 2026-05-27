"""Unit tests for src.scanners.universe_pit — pure-logic, no network."""
from __future__ import annotations

import json
from pathlib import Path

from src.scanners.universe_pit import (
    SymbolListing,
    coverage_fraction,
    eligible_universe_at,
    filter_universe_for_span,
    is_active_at,
    load_pit_log,
    universe_size_over_time,
)


# Synthetic timeline. All timestamps in ms. Year-boundaries roughly:
#   2020-01-01 = 1577836800000
#   2021-01-01 = 1609459200000
#   2022-01-01 = 1640995200000
#   2023-01-01 = 1672531200000

T_2020 = 1_577_836_800_000
T_2021 = 1_609_459_200_000
T_2022 = 1_640_995_200_000
T_2023 = 1_672_531_200_000
T_2024 = 1_704_067_200_000


def _log() -> dict[str, SymbolListing]:
    return {
        # Active throughout
        "BTCUSDT": SymbolListing("BTCUSDT", listed_ms=T_2020, delisted_ms=None),
        # Listed late
        "NEWUSDT": SymbolListing("NEWUSDT", listed_ms=T_2023, delisted_ms=None),
        # Delisted mid-window
        "OLDUSDT": SymbolListing("OLDUSDT", listed_ms=T_2020, delisted_ms=T_2022),
        # Brief window
        "FLYUSDT": SymbolListing("FLYUSDT", listed_ms=T_2021, delisted_ms=T_2022),
    }


def test_is_active_at_boundaries() -> None:
    log = _log()
    # Listed instant is INCLUSIVE
    assert is_active_at(log, "BTCUSDT", T_2020) is True
    # Before listed
    assert is_active_at(log, "NEWUSDT", T_2022) is False
    # Delisted instant is EXCLUSIVE
    assert is_active_at(log, "OLDUSDT", T_2022) is False
    # Mid-window
    assert is_active_at(log, "OLDUSDT", T_2021) is True
    # Missing symbol
    assert is_active_at(log, "MISSING", T_2021) is False


def test_eligible_universe_at_filters_correctly() -> None:
    log = _log()
    # At 2020 only BTC and OLD are live
    assert eligible_universe_at(log, T_2020) == ["BTCUSDT", "OLDUSDT"]
    # At 2021 BTC, OLD, FLY are live (FLY listed at 2021 boundary - inclusive)
    assert eligible_universe_at(log, T_2021) == ["BTCUSDT", "FLYUSDT", "OLDUSDT"]
    # At 2023 only BTC and NEW survive
    assert eligible_universe_at(log, T_2023) == ["BTCUSDT", "NEWUSDT"]


def test_eligible_universe_at_with_subset() -> None:
    log = _log()
    # Subset restricts the candidate pool
    assert eligible_universe_at(log, T_2021, subset=["BTCUSDT", "FLYUSDT"]) == [
        "BTCUSDT", "FLYUSDT",
    ]
    # Subset including a never-active symbol drops it
    assert eligible_universe_at(log, T_2021, subset=["NEWUSDT"]) == []


def test_filter_universe_for_span_full_coverage() -> None:
    log = _log()
    # Window 2020-2024: only BTC continuously live the whole span.
    out = filter_universe_for_span(log, T_2020, T_2024,
                                    candidates=list(log.keys()),
                                    min_coverage=1.0)
    assert out == ["BTCUSDT"]


def test_filter_universe_for_span_partial_coverage() -> None:
    log = _log()
    # Threshold 0.24 (just below 1/4) admits symbols live ≥ ~1y of the 4y window.
    # (Exact 0.25 would fail FLY/NEW by ~0.0002 because of leap-year math.)
    out = filter_universe_for_span(log, T_2020, T_2024,
                                    candidates=list(log.keys()),
                                    min_coverage=0.24)
    # BTC full; OLD ≈50%; NEW ≈25%; FLY ≈25%. All pass at 0.24.
    assert set(out) == {"BTCUSDT", "NEWUSDT", "OLDUSDT", "FLYUSDT"}

    # Tightening to 0.40 drops the two short-window pairs.
    out_tight = filter_universe_for_span(log, T_2020, T_2024,
                                          candidates=list(log.keys()),
                                          min_coverage=0.40)
    assert set(out_tight) == {"BTCUSDT", "OLDUSDT"}


def test_coverage_fraction() -> None:
    log = _log()
    span = T_2024 - T_2020
    # BTC: full coverage
    assert abs(coverage_fraction(log, "BTCUSDT", T_2020, T_2024) - 1.0) < 1e-9
    # OLD live half the window
    cov_old = coverage_fraction(log, "OLDUSDT", T_2020, T_2024)
    assert abs(cov_old - (T_2022 - T_2020) / span) < 1e-9
    # Missing symbol → 0
    assert coverage_fraction(log, "MISSING", T_2020, T_2024) == 0.0


def test_universe_size_over_time() -> None:
    log = _log()
    # Monthly stepping — sample doesn't land exactly on year boundaries because
    # of leap-year arithmetic, so we look up each year via the nearest preceding
    # sample.
    step_ms = 30 * 24 * 3600 * 1000
    out = universe_size_over_time(log, T_2020, T_2024, step_ms=step_ms)
    assert len(out) > 0 and out[0][0] == T_2020

    def n_at(ts: int) -> int:
        return next(n for t, n in reversed(out) if t <= ts)

    # Right after each year boundary, the eligible set has the expected size.
    assert n_at(T_2020 + step_ms) == 2          # BTC + OLD (FLY not yet)
    assert n_at(T_2021 + step_ms) == 3          # + FLY listed
    assert n_at(T_2022 + step_ms) == 1          # FLY + OLD both delisted; NEW not yet
    assert n_at(T_2023 + step_ms) == 2          # BTC + NEW


def test_load_pit_log_handles_missing_file(tmp_path: Path) -> None:
    out = load_pit_log(tmp_path / "does_not_exist.json")
    assert out == {}


def test_load_pit_log_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "log.json"
    p.write_text(json.dumps({
        "BTCUSDT": {"listed_ms": T_2020, "delisted_ms": None},
        "BCCUSDT": {"listed_ms": T_2020, "delisted_ms": T_2022,
                    "source": "manual"},
        "BADENTRY": {"foo": "bar"},  # skipped — no listed_ms
    }))
    log = load_pit_log(p)
    assert set(log.keys()) == {"BTCUSDT", "BCCUSDT"}
    assert log["BTCUSDT"].delisted_ms is None
    assert log["BCCUSDT"].delisted_ms == T_2022


def test_filter_universe_for_span_skips_missing() -> None:
    log = _log()
    out = filter_universe_for_span(log, T_2020, T_2024,
                                    candidates=["UNKNOWN", "BTCUSDT"],
                                    min_coverage=1.0)
    assert out == ["BTCUSDT"]
