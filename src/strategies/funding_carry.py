"""Cross-sectional funding carry overlay (Fan-Jiao-Lu-Tong SSRN 4666425).

The single-symbol funding harvest (`src/strategies/funding_harvest.py`)
trades each symbol's own funding extremes against its own history. This
module is the orthogonal *cross-sectional* play: at each rebalance moment,
RANK the eligible universe by funding rate and LONG the top-N + SHORT the
bottom-N. Equal-weighted within each leg, dollar-neutral overall.

The thesis is the carry premium documented across asset classes — the
high-yield-pays-higher-future-return regularity, here driven by perp/spot
divergence (when funding is positive, the perp is trading above spot
because longs are crowded; high-funding perps tend to outperform after the
crowd thins). Fan et al. report 43.4% p.a. / Sharpe 0.74 on the long-top
minus short-bottom decile spread, concentrated in lower-cap higher-OI alts.

This module is **pure logic**: no I/O. The backtest driver fetches funding
+ price history and walks the rebalances; live trader code does the same
at scheduled rebalance cadence. Both share the same `rank_for_carry` and
`build_rebalance` decisions for live/backtest parity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.services.costs import Costs, funding_accrual_usd, taker_fee_usd


# ---------------------------------------------------------------------------
# Params

@dataclass(frozen=True)
class CarryParams:
    """Defaults follow §2.3 stretch goals of the recommendations doc:
    weekly rebalance, top/bottom 3 funding perps, 25% of book per side
    (50% gross dollar exposure)."""
    top_n: int = 3
    rebalance_period_hours: int = 7 * 24                # weekly
    book_pct_per_side: float = 0.25                     # 25% per leg
    min_universe_size: int = 10                         # need enough breadth
    # Funding cycles are 8h on Binance perps; one week = 21 cycles.
    funding_cycle_hours: int = 8


# ---------------------------------------------------------------------------
# Data shapes

@dataclass(frozen=True)
class CarryPosition:
    symbol: str
    side: str                                            # "long" | "short"
    notional_usd: float                                  # absolute, always > 0
    entry_funding_rate: float                            # rate at entry decision
    entry_price: float = 0.0                             # populated by driver


@dataclass(frozen=True)
class CarryRebalance:
    """Result of one rebalance decision at `ts_ms`."""
    ts_ms: int
    longs: list[CarryPosition] = field(default_factory=list)
    shorts: list[CarryPosition] = field(default_factory=list)
    universe_n: int = 0
    skipped_reason: str = ""

    @property
    def is_active(self) -> bool:
        return bool(self.longs or self.shorts)


@dataclass
class CycleResult:
    """PnL decomposition for one held position over [entry, exit]."""
    symbol: str
    side: str
    notional_usd: float
    price_pnl_usd: float
    funding_pnl_usd: float
    fee_pnl_usd: float

    @property
    def total_pnl_usd(self) -> float:
        return self.price_pnl_usd + self.funding_pnl_usd + self.fee_pnl_usd


# ---------------------------------------------------------------------------
# Ranking

def rank_for_carry(
    funding_by_symbol: dict[str, float],
    p: Optional[CarryParams] = None,
) -> tuple[list[str], list[str]]:
    """Sort `funding_by_symbol` by funding rate. Return (long_symbols,
    short_symbols), each of length `top_n`. Longs are the HIGHEST-funding
    symbols (Fan et al. long-leg); shorts are the LOWEST-funding symbols.

    Symbols whose funding is None / NaN are dropped from the ranking.
    Ties are broken alphabetically for determinism.
    """
    p = p or CarryParams()
    clean: list[tuple[str, float]] = []
    for sym, rate in funding_by_symbol.items():
        if rate is None:
            continue
        # Skip NaNs without importing math (rate is a plain float here).
        if rate != rate:
            continue
        clean.append((sym, float(rate)))
    if not clean:
        return [], []
    # Sort by (rate asc, symbol asc) — lowest funding first, then alphabetical.
    clean.sort(key=lambda x: (x[1], x[0]))
    if len(clean) < 2 * p.top_n:
        # Not enough breadth for clean top/bot legs without overlap.
        return [], []
    shorts = [s for s, _ in clean[: p.top_n]]
    longs = [s for s, _ in clean[-p.top_n:]]
    return longs, shorts


def funding_window_change(
    funding_events: list[tuple[int, float]],
    ts_ms: int,
    window_hours: int = 7 * 24,
) -> Optional[float]:
    """Δfunding signal (Card 1): mean funding over the trailing window
    ``[ts-w, ts)`` minus mean funding over the prior window ``[ts-2w, ts-w)``.

    Positive ⇒ funding has been *rising* into ``ts_ms``. Rationale: funding
    levels are ~0.97-0.99 autocorrelated, so the level is near a unit root and
    carries little new information; the first difference (the repricing
    surprise) is near-orthogonal to the level and harder to arbitrage away than
    the now-compressed static carry yield.

    PIT-safe: uses only funding events strictly before ``ts_ms``. Returns
    ``None`` when either window is empty (driver drops the symbol that cycle),
    so early rebalances require ~2 windows of prior funding history.
    """
    w_ms = window_hours * 3_600_000
    recent_lo = ts_ms - w_ms
    prior_lo = ts_ms - 2 * w_ms
    recent = [r for (t, r) in funding_events if recent_lo <= t < ts_ms]
    prior = [r for (t, r) in funding_events if prior_lo <= t < recent_lo]
    if not recent or not prior:
        return None
    return sum(recent) / len(recent) - sum(prior) / len(prior)


def build_rebalance(
    funding_by_symbol: dict[str, float],
    equity_usd: float,
    ts_ms: int,
    p: Optional[CarryParams] = None,
) -> CarryRebalance:
    """Pick top/bottom N at `ts_ms` and build equal-weighted positions.

    Per-position notional = equity * book_pct_per_side / top_n on each leg.
    A leg with size 0 (when universe is too small) is dropped — driver
    decides whether to skip the cycle.
    """
    p = p or CarryParams()
    if equity_usd <= 0:
        return CarryRebalance(ts_ms=ts_ms, universe_n=len(funding_by_symbol),
                              skipped_reason="non-positive equity")
    n_universe = sum(1 for v in funding_by_symbol.values()
                     if v is not None and v == v)  # NaN check
    if n_universe < p.min_universe_size or n_universe < 2 * p.top_n:
        return CarryRebalance(ts_ms=ts_ms, universe_n=n_universe,
                              skipped_reason=f"universe too small ({n_universe})")

    longs, shorts = rank_for_carry(funding_by_symbol, p)
    if not longs or not shorts:
        return CarryRebalance(ts_ms=ts_ms, universe_n=n_universe,
                              skipped_reason="ranking returned empty legs")

    leg_notional = equity_usd * p.book_pct_per_side
    per_position = leg_notional / p.top_n
    long_pos = [
        CarryPosition(symbol=sym, side="long",
                      notional_usd=per_position,
                      entry_funding_rate=funding_by_symbol[sym])
        for sym in longs
    ]
    short_pos = [
        CarryPosition(symbol=sym, side="short",
                      notional_usd=per_position,
                      entry_funding_rate=funding_by_symbol[sym])
        for sym in shorts
    ]
    return CarryRebalance(ts_ms=ts_ms, longs=long_pos, shorts=short_pos,
                          universe_n=n_universe)


# ---------------------------------------------------------------------------
# Per-cycle PnL

def cycle_pnl(
    position: CarryPosition,
    entry_price: float,
    exit_price: float,
    funding_events: list[tuple[int, float]],
    entry_ts_ms: int,
    exit_ts_ms: int,
    costs: Optional[Costs] = None,
) -> CycleResult:
    """Decompose one position's PnL over [entry, exit] into:
      - price_pnl  = notional * (exit/entry − 1) for long, or −1 × that for short
      - funding_pnl = NEGATIVE of `funding_accrual_usd` (which returns cost from
        the trader's perspective; long with positive funding → COST, so
        funding_pnl is negative)
      - fee_pnl  = −(taker on entry + taker on exit)  (both legs are perp)
    """
    costs = costs or Costs()
    if entry_price <= 0 or exit_price <= 0:
        return CycleResult(
            symbol=position.symbol, side=position.side,
            notional_usd=position.notional_usd,
            price_pnl_usd=0.0, funding_pnl_usd=0.0, fee_pnl_usd=0.0,
        )
    px_ret = (exit_price - entry_price) / entry_price
    if position.side == "short":
        px_ret = -px_ret
    price_pnl = position.notional_usd * px_ret

    funding_cost = funding_accrual_usd(
        side=position.side,                              # type: ignore[arg-type]
        notional_usd=position.notional_usd,
        funding_events=funding_events,
        entry_ts_ms=entry_ts_ms,
        exit_ts_ms=exit_ts_ms,
    )
    funding_pnl = -funding_cost

    fees = 2.0 * taker_fee_usd(position.notional_usd, "perp", costs)
    fee_pnl = -fees

    return CycleResult(
        symbol=position.symbol, side=position.side,
        notional_usd=position.notional_usd,
        price_pnl_usd=price_pnl,
        funding_pnl_usd=funding_pnl,
        fee_pnl_usd=fee_pnl,
    )
