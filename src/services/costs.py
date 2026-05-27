"""Realistic execution-cost model for backtests.

Replaces the flat-slippage assumption in `backtest.py` with:
  - Per-venue taker/maker fees (spot vs perp)
  - Almgren-Chriss sqrt-impact: `impact_bps = 0.5*spread + k * sqrt(notional/ADV_5m)`
    with k ≈ 0.05 for majors, ≈ 0.15 for mid-caps. Calibration from AlphaArchitect
    "Decay of Anomalies" and Robot Wealth crypto execution writeups.
  - Funding-rate accrual on multi-cycle perp holds (long pays positive, short pays
    negative; sign flips for the harvest side).

Round-trip cost on a trade is `entry_impact + exit_impact + 2*taker_fee + funding_accrual`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

Side = Literal["long", "short"]
Venue = Literal["spot", "perp"]


# Empirically calibrated Almgren-Chriss impact constants per liquidity tier.
# Majors = BTC/ETH/SOL/BNB; mid-caps = top-20-by-volume excluding majors.
IMPACT_K_MAJOR = 0.05
IMPACT_K_MID = 0.15
IMPACT_K_SMALL = 0.30        # alts outside top-50 by volume

MAJORS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"})


def impact_k_for_symbol(symbol: str, mid_caps: Optional[Iterable[str]] = None) -> float:
    """Pick the sqrt-impact coefficient by liquidity tier. Mid-caps default to
    a small hand-curated list; callers can override with the actual top-N
    universe from `BinanceClient.get_ticker()`."""
    if symbol in MAJORS:
        return IMPACT_K_MAJOR
    if mid_caps is not None and symbol in mid_caps:
        return IMPACT_K_MID
    return IMPACT_K_SMALL


@dataclass(frozen=True)
class Costs:
    """All cost parameters in basis points. Defaults match Binance retail-VIP-0."""
    fee_taker_perp_bps: float = 5.0     # 0.050%
    fee_maker_perp_bps: float = -2.0    # -0.020% rebate
    fee_taker_spot_bps: float = 10.0    # 0.100%
    fee_maker_spot_bps: float = 10.0    # no rebate
    # Half-spread proxy when actual order book unavailable. 1bp on majors,
    # 3-5bps on mid-caps, 10bps+ on small alts. Caller can override per symbol.
    half_spread_bps_default: float = 3.0
    # Maker fill probability — published numbers from Robot Wealth execution
    # research: ~50% in trending markets, ~80% in chop. Used by simulators
    # that explicitly try to post limits; default sim is taker.
    maker_fill_rate_trending: float = 0.50
    maker_fill_rate_chop: float = 0.80


def slippage_bps(
    notional_usd: float,
    adv_5m_usd: float,
    impact_k: float,
    half_spread_bps: float,
) -> float:
    """Almgren-Chriss sqrt-impact + half-spread crossing.

    `adv_5m_usd` is the 5-minute average dollar volume of the symbol in the
    recent window. The square-root term captures that splitting your order
    has diminishing returns on price impact.

    Returns slippage in basis points (positive = adverse cost). When
    `adv_5m_usd <= 0` we fall back to just the half-spread.
    """
    if notional_usd <= 0:
        return 0.0
    half_spread = max(0.0, half_spread_bps)
    if adv_5m_usd <= 0:
        return half_spread
    participation = notional_usd / adv_5m_usd
    impact = impact_k * math.sqrt(participation) * 10_000.0  # to bps
    return half_spread + impact


def adjust_entry_price(
    raw_price: float,
    side: Side,
    notional_usd: float,
    adv_5m_usd: float,
    impact_k: float,
    half_spread_bps: float,
) -> float:
    """Apply adverse-direction slippage to the intended fill price.

    Longs cross the offer (pay higher); shorts cross the bid (sell lower).
    """
    slip = slippage_bps(notional_usd, adv_5m_usd, impact_k, half_spread_bps) / 10_000.0
    return raw_price * (1.0 + slip) if side == "long" else raw_price * (1.0 - slip)


def adjust_exit_price(
    raw_price: float,
    side: Side,
    notional_usd: float,
    adv_5m_usd: float,
    impact_k: float,
    half_spread_bps: float,
) -> float:
    """Apply adverse-direction slippage to the intended exit price.

    Long exit sells the bid; short exit buys the offer.
    """
    slip = slippage_bps(notional_usd, adv_5m_usd, impact_k, half_spread_bps) / 10_000.0
    return raw_price * (1.0 - slip) if side == "long" else raw_price * (1.0 + slip)


def taker_fee_usd(notional_usd: float, venue: Venue, costs: Costs) -> float:
    """Round-trip taker fee on one fill (entry OR exit, not both)."""
    bps = costs.fee_taker_perp_bps if venue == "perp" else costs.fee_taker_spot_bps
    return abs(notional_usd) * bps / 10_000.0


def funding_accrual_usd(
    side: Side,
    notional_usd: float,
    funding_events: list[tuple[int, float]],
    entry_ts_ms: int,
    exit_ts_ms: int,
) -> float:
    """Net funding paid (positive = cost) over the hold window.

    `funding_events` is a list of (ts_ms, funding_rate_8h_decimal) sorted
    ascending. For each event strictly within (entry_ts_ms, exit_ts_ms], the
    long side pays `notional * rate` and the short side receives the same.

    Returns COST from the trader's perspective — positive number means
    funding ate into PnL; negative means funding paid you.
    """
    if notional_usd <= 0 or not funding_events:
        return 0.0
    sign = 1.0 if side == "long" else -1.0
    cost = 0.0
    for ts, rate in funding_events:
        if entry_ts_ms < ts <= exit_ts_ms:
            cost += sign * notional_usd * rate
    return cost


def round_trip_cost_bps(
    notional_usd: float,
    venue: Venue,
    adv_5m_usd: float,
    impact_k: float,
    half_spread_bps: float,
    costs: Costs,
    n_funding_cycles: int = 0,
    avg_funding_rate: float = 0.0,
    funding_sign: float = 1.0,
) -> float:
    """Round-trip cost estimate in bps of notional, used for entry-gating.

    `funding_sign = +1` for long perp paying positive funding (cost);
    `-1` for short perp receiving positive funding (rebate).
    """
    fee_per_side_bps = (
        costs.fee_taker_perp_bps if venue == "perp" else costs.fee_taker_spot_bps
    )
    slip = slippage_bps(notional_usd, adv_5m_usd, impact_k, half_spread_bps)
    funding_cost_bps = funding_sign * n_funding_cycles * avg_funding_rate * 10_000.0
    return 2.0 * (fee_per_side_bps + slip) + funding_cost_bps
