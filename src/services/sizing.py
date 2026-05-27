"""Vol-targeted position sizing for the backtest harness.

Replaces fixed-$100 notional (Carver ch. 9 critique: "treats a BTC trade at
60% vol identical to an SOL trade at 110%") with a vol-scaled %-of-equity
sizer. Default config follows Carver / Hood-Raughtigan / Hurst-Ooi-Pedersen:

  notional_i = kelly_fraction * equity * target_vol_annual / realized_vol_i
  notional_i = min(notional_i, equity * 2)        # margin reality cap

`realized_vol_i` is annualized EWMA of returns with λ=0.94 (RiskMetrics).
`kelly_fraction = 0.20` is the de-Prado-style fractional Kelly for fat-tailed
crypto returns (Thorp full Kelly is too aggressive; 0.5× still too aggressive
for crypto kurtosis 10+; 0.20× is the practitioner standard).

The portfolio-level vol scalar is applied OUTSIDE this module — see
`portfolio_vol_scalar` in the multi-strategy allocator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class VolTargetConfig:
    """Per-position sizing parameters."""
    target_vol_annual: float = 0.25          # 25% annualised vol per position
    kelly_fraction: float = 0.20             # fractional Kelly haircut
    ewma_lambda: float = 0.94                # RiskMetrics — ~30d half-life
    vol_floor: float = 0.10                  # min realized vol to avoid div-by-zero
    notional_cap_x_equity: float = 2.0       # leverage cap per position
    notional_floor_usd: float = 10.0         # min trade size, smaller is rounding


def ewma_volatility(
    returns: Sequence[float],
    lam: float = 0.94,
) -> float:
    """RiskMetrics EWMA volatility of a return series. Returns volatility of
    one PERIOD — caller annualises by multiplying by sqrt(periods_per_year).

    Walks the sequence backward with the recursive form:
        σ²_t = λ * σ²_{t-1} + (1-λ) * r_t²
    Initialised from the first 30 observations' sample variance to avoid
    cold-start instability. Returns 0 if fewer than 2 returns provided."""
    n = len(returns)
    if n < 2:
        return 0.0
    init_n = min(30, n)
    init_var = sum(r * r for r in returns[:init_n]) / max(1, init_n)
    var = init_var
    for r in returns[init_n:]:
        var = lam * var + (1 - lam) * r * r
    return math.sqrt(var)


def annualize_vol(period_vol: float, bars_per_year: float) -> float:
    """Convert per-bar vol to annualised by sqrt(bars/yr)."""
    return period_vol * math.sqrt(bars_per_year)


def realized_vol_annual_from_klines(
    closes: Sequence[float],
    bars_per_year: float,
    lam: float = 0.94,
) -> float:
    """Convenience: realized annualised vol from a close-price series.

    Uses log-returns (more numerically stable for crypto vol) and the EWMA
    estimator. `bars_per_year` for 5m is 105_120; for 15m is 35_040; for
    1h is 8_760; for 1d is 365.
    """
    if len(closes) < 2:
        return 0.0
    rets = []
    for a, b in zip(closes[:-1], closes[1:]):
        if a > 0 and b > 0:
            rets.append(math.log(b / a))
    if not rets:
        return 0.0
    period_vol = ewma_volatility(rets, lam=lam)
    return annualize_vol(period_vol, bars_per_year)


def vol_target_notional(
    equity_usd: float,
    realized_vol_annual: float,
    cfg: Optional[VolTargetConfig] = None,
) -> float:
    """Notional $ to deploy on the position. See module docstring for formula.

    When `realized_vol_annual` is implausibly small (cold-start, stable-coin),
    we floor it to `cfg.vol_floor` so notional doesn't explode.
    """
    c = cfg or VolTargetConfig()
    rv = max(realized_vol_annual, c.vol_floor)
    if equity_usd <= 0:
        return 0.0
    raw = c.kelly_fraction * equity_usd * c.target_vol_annual / rv
    cap = c.notional_cap_x_equity * equity_usd
    notional = min(raw, cap)
    if notional < c.notional_floor_usd:
        return 0.0
    return notional


def qty_from_notional(notional_usd: float, price: float) -> float:
    """Convert notional dollars to base-asset units."""
    if price <= 0 or notional_usd <= 0:
        return 0.0
    return notional_usd / price
