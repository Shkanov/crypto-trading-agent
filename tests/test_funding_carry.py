"""Tests for src.strategies.funding_carry — pure logic, no network."""
from __future__ import annotations

import math

from src.strategies.funding_carry import (
    CarryParams,
    build_rebalance,
    cycle_pnl,
    funding_window_change,
    price_momentum,
    rank_for_carry,
    rank_for_carry_momentum,
    trailing_quote_volume,
)


def test_trailing_quote_volume_sums_window_pit_safe() -> None:
    _h = 3_600_000
    ts = 1000 * _h
    vol = {
        ts - 30 * _h: 5.0,   # outside 24h window
        ts - 16 * _h: 10.0,  # inside
        ts - 8 * _h: 20.0,   # inside
        ts + 8 * _h: 999.0,  # future — must be excluded
    }
    v = trailing_quote_volume(vol, ts, window_hours=24)
    assert v is not None and math.isclose(v, 30.0)


def test_trailing_quote_volume_none_when_empty_window() -> None:
    _h = 3_600_000
    ts = 1000 * _h
    vol = {ts - 200 * _h: 7.0}  # only outside the window
    assert trailing_quote_volume(vol, ts, window_hours=24) is None
    assert trailing_quote_volume({}, ts, window_hours=24) is None


# ---------------------------------------------------------------------------
# Price momentum + momentum-conditioned carry ranking (Card 2)


def test_price_momentum_trailing_return() -> None:
    _h = 3_600_000
    ts = 1000 * _h
    closes = {ts - 200 * _h: 100.0, ts - 24 * _h: 110.0, ts: 120.0}
    # 24h lookback: 120/110 - 1
    m = price_momentum(closes, ts, lookback_hours=24)
    assert m is not None and math.isclose(m, 120.0 / 110.0 - 1.0)


def test_price_momentum_pit_safe_uses_close_at_or_before() -> None:
    _h = 3_600_000
    ts = 1000 * _h
    # A future close after ts must be ignored.
    closes = {ts - 48 * _h: 100.0, ts - 24 * _h: 100.0, ts + 8 * _h: 999.0}
    m = price_momentum(closes, ts, lookback_hours=24)
    assert m is not None and math.isclose(m, 0.0)


def test_price_momentum_none_when_missing_endpoint() -> None:
    assert price_momentum({}, 1000, 24) is None


def test_rank_for_carry_momentum_keeps_only_agreeing_names() -> None:
    # Funding: A,B high (long candidates); D,E low (short candidates).
    funding = {"A": 0.01, "B": 0.009, "C": 0.0, "D": -0.009, "E": -0.01}
    # Momentum agrees for A (up) and E (down), disagrees for B (down) and D (up).
    mom = {"A": 0.05, "B": -0.05, "C": 0.0, "D": 0.05, "E": -0.05}
    longs, shorts = rank_for_carry_momentum(funding, mom, CarryParams(top_n=2))
    assert longs == ["A"]          # B dropped: funding-long but momentum-down
    assert shorts == ["E"]         # D dropped: funding-short but momentum-up


def test_rank_for_carry_momentum_drops_missing_momentum() -> None:
    funding = {"A": 0.01, "B": 0.009, "D": -0.009, "E": -0.01}
    mom = {"A": 0.05}              # others missing → treated as disagreeing
    longs, shorts = rank_for_carry_momentum(funding, mom, CarryParams(top_n=2))
    assert longs == ["A"]
    assert shorts == []


# ---------------------------------------------------------------------------
# Δfunding signal (Card 1)

_H = 3_600_000  # 1h in ms


def _events(start_ms: int, n: int, step_h: int, rate: float):
    """n funding events at `rate`, every `step_h` hours from start_ms."""
    return [(start_ms + i * step_h * _H, rate) for i in range(n)]


def test_funding_window_change_positive_when_rising() -> None:
    ts = 100 * 24 * _H
    w = 7 * 24  # 1-week windows
    # prior window [ts-2w, ts-w): low funding; recent [ts-w, ts): high funding.
    prior = _events(ts - 2 * w * _H, 21, 8, 0.0001)    # 8h cycles, 21/week
    recent = _events(ts - w * _H, 21, 8, 0.0010)
    d = funding_window_change(prior + recent, ts, window_hours=w)
    assert d is not None and d > 0
    assert math.isclose(d, 0.0010 - 0.0001, abs_tol=1e-9)


def test_funding_window_change_negative_when_falling() -> None:
    ts = 100 * 24 * _H
    w = 7 * 24
    prior = _events(ts - 2 * w * _H, 21, 8, 0.0010)
    recent = _events(ts - w * _H, 21, 8, 0.0001)
    d = funding_window_change(prior + recent, ts, window_hours=w)
    assert d is not None and d < 0


def test_funding_window_change_none_without_two_windows() -> None:
    ts = 100 * 24 * _H
    w = 7 * 24
    # Only recent window populated → prior empty → None.
    recent = _events(ts - w * _H, 21, 8, 0.0005)
    assert funding_window_change(recent, ts, window_hours=w) is None


def test_funding_window_change_is_pit_safe() -> None:
    # Events at/after ts must be ignored (look-ahead guard).
    ts = 100 * 24 * _H
    w = 7 * 24
    prior = _events(ts - 2 * w * _H, 21, 8, 0.0002)
    recent = _events(ts - w * _H, 21, 8, 0.0002)
    future = _events(ts, 5, 8, 0.5)  # huge future spike — must NOT leak in
    d = funding_window_change(prior + recent + future, ts, window_hours=w)
    assert d is not None and math.isclose(d, 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Ranking

def test_rank_for_carry_picks_extremes() -> None:
    """Build a universe with monotonic funding rates and confirm the
    longs/shorts come from the right ends."""
    funding = {f"SYM{i:02d}": (i - 10) * 0.001 for i in range(20)}
    # SYM00 has -0.010, SYM19 has +0.009
    longs, shorts = rank_for_carry(funding, CarryParams(top_n=3))
    assert longs == ["SYM17", "SYM18", "SYM19"]      # highest funding = longs
    assert shorts == ["SYM00", "SYM01", "SYM02"]     # lowest funding = shorts


def test_rank_for_carry_breaks_ties_alphabetically() -> None:
    # All zero → alphabetical ordering
    funding = {f"SYM{c}": 0.0 for c in "ABCDEFGHIJ"}
    longs, shorts = rank_for_carry(funding, CarryParams(top_n=3))
    assert shorts == ["SYMA", "SYMB", "SYMC"]
    assert longs == ["SYMH", "SYMI", "SYMJ"]


def test_rank_for_carry_drops_missing_funding() -> None:
    funding = {"BTC": 0.001, "ETH": None, "SOL": 0.002, "MISSING": float("nan")}
    longs, shorts = rank_for_carry(funding, CarryParams(top_n=1))
    # Only BTC + SOL are eligible. top_n=1 → 1 long + 1 short, need 2.
    assert longs == ["SOL"]
    assert shorts == ["BTC"]


def test_rank_for_carry_returns_empty_when_too_few_symbols() -> None:
    funding = {"A": 0.001, "B": -0.001, "C": 0.002}  # only 3 symbols
    longs, shorts = rank_for_carry(funding, CarryParams(top_n=3))
    # Need 2*top_n = 6, only 3 → empty
    assert longs == [] and shorts == []


# ---------------------------------------------------------------------------
# build_rebalance

def test_build_rebalance_equal_notional_per_leg() -> None:
    funding = {f"SYM{i:02d}": i * 0.0001 for i in range(20)}
    p = CarryParams(top_n=3, book_pct_per_side=0.20)
    rb = build_rebalance(funding, equity_usd=10_000.0,
                          ts_ms=1_700_000_000_000, p=p)
    assert rb.is_active
    assert len(rb.longs) == 3
    assert len(rb.shorts) == 3
    # 20% of $10k = $2k per leg; / 3 = ~$666.67 per position
    expected = 10_000.0 * 0.20 / 3
    for pos in rb.longs + rb.shorts:
        assert abs(pos.notional_usd - expected) < 1e-9


def test_build_rebalance_skips_small_universe() -> None:
    funding = {f"SYM{i}": i * 0.001 for i in range(5)}
    rb = build_rebalance(funding, equity_usd=10_000.0,
                          ts_ms=1_700_000_000_000,
                          p=CarryParams(top_n=3, min_universe_size=10))
    assert not rb.is_active
    assert "universe too small" in rb.skipped_reason


def test_build_rebalance_skips_zero_equity() -> None:
    funding = {f"SYM{i:02d}": i * 0.0001 for i in range(20)}
    rb = build_rebalance(funding, equity_usd=0.0,
                          ts_ms=1_700_000_000_000, p=CarryParams())
    assert not rb.is_active
    assert "non-positive equity" in rb.skipped_reason


def test_build_rebalance_dollar_neutral() -> None:
    funding = {f"SYM{i:02d}": i * 0.0001 for i in range(20)}
    rb = build_rebalance(funding, equity_usd=10_000.0,
                          ts_ms=1_700_000_000_000,
                          p=CarryParams(top_n=3, book_pct_per_side=0.30))
    long_gross = sum(p.notional_usd for p in rb.longs)
    short_gross = sum(p.notional_usd for p in rb.shorts)
    assert abs(long_gross - short_gross) < 1e-9
    assert abs(long_gross - 10_000.0 * 0.30) < 1e-9


# ---------------------------------------------------------------------------
# cycle_pnl

def _funding_events(t0: int, n_cycles: int, rate: float, cycle_h: int = 8) -> list[tuple[int, float]]:
    return [(t0 + (i + 1) * cycle_h * 3_600_000, rate) for i in range(n_cycles)]


def test_cycle_pnl_long_price_up_gains() -> None:
    t0 = 1_700_000_000_000
    t1 = t0 + 7 * 86_400_000
    pos = type("Pos", (), {})()  # use real CarryPosition for type safety:
    from src.strategies.funding_carry import CarryPosition
    pos = CarryPosition(symbol="X", side="long", notional_usd=1_000.0,
                        entry_funding_rate=0.001)
    r = cycle_pnl(pos, entry_price=100.0, exit_price=105.0,
                   funding_events=[], entry_ts_ms=t0, exit_ts_ms=t1)
    assert abs(r.price_pnl_usd - 50.0) < 1e-9       # 5% on $1k notional
    assert r.funding_pnl_usd == 0.0                 # no funding events provided
    assert r.fee_pnl_usd < 0.0                      # two-sided perp taker
    assert r.total_pnl_usd > 0.0


def test_cycle_pnl_short_price_up_loses() -> None:
    from src.strategies.funding_carry import CarryPosition
    t0 = 1_700_000_000_000
    t1 = t0 + 7 * 86_400_000
    pos = CarryPosition(symbol="X", side="short", notional_usd=1_000.0,
                        entry_funding_rate=-0.001)
    r = cycle_pnl(pos, entry_price=100.0, exit_price=105.0,
                   funding_events=[], entry_ts_ms=t0, exit_ts_ms=t1)
    # Short with price up = adverse
    assert abs(r.price_pnl_usd - (-50.0)) < 1e-9
    assert r.total_pnl_usd < 0.0


def test_cycle_pnl_long_pays_positive_funding() -> None:
    from src.strategies.funding_carry import CarryPosition
    t0 = 1_700_000_000_000
    # 7 days * 3 cycles/day = 21 cycles
    events = _funding_events(t0, 21, rate=0.001)
    t1 = events[-1][0] + 1
    pos = CarryPosition(symbol="X", side="long", notional_usd=1_000.0,
                        entry_funding_rate=0.001)
    r = cycle_pnl(pos, entry_price=100.0, exit_price=100.0,
                   funding_events=events, entry_ts_ms=t0, exit_ts_ms=t1)
    # Long pays 21 cycles × 0.001 × $1000 = $21 (negative for trader)
    assert abs(r.funding_pnl_usd - (-21.0)) < 1e-9


def test_cycle_pnl_short_receives_positive_funding() -> None:
    from src.strategies.funding_carry import CarryPosition
    t0 = 1_700_000_000_000
    events = _funding_events(t0, 21, rate=0.001)
    t1 = events[-1][0] + 1
    pos = CarryPosition(symbol="X", side="short", notional_usd=1_000.0,
                        entry_funding_rate=0.001)
    r = cycle_pnl(pos, entry_price=100.0, exit_price=100.0,
                   funding_events=events, entry_ts_ms=t0, exit_ts_ms=t1)
    # Short with positive funding receives 21 cycles × 0.001 × $1000 = +$21
    assert abs(r.funding_pnl_usd - 21.0) < 1e-9


def test_cycle_pnl_invalid_prices_returns_zero() -> None:
    from src.strategies.funding_carry import CarryPosition
    t0 = 1_700_000_000_000
    pos = CarryPosition(symbol="X", side="long", notional_usd=1_000.0,
                        entry_funding_rate=0.0)
    r = cycle_pnl(pos, entry_price=0.0, exit_price=100.0,
                   funding_events=[], entry_ts_ms=t0, exit_ts_ms=t0 + 1)
    assert r.price_pnl_usd == 0.0
    assert r.fee_pnl_usd == 0.0
    assert r.total_pnl_usd == 0.0


def test_cycle_pnl_fee_scales_with_notional() -> None:
    from src.strategies.funding_carry import CarryPosition
    t0 = 1_700_000_000_000
    small = CarryPosition("A", "long", 100.0, 0.0)
    big = CarryPosition("B", "long", 10_000.0, 0.0)
    r1 = cycle_pnl(small, 100.0, 100.0, [], t0, t0 + 1)
    r2 = cycle_pnl(big, 100.0, 100.0, [], t0, t0 + 1)
    # fees scale linearly with notional
    assert math.isclose(r2.fee_pnl_usd / r1.fee_pnl_usd, 100.0, rel_tol=1e-9)
