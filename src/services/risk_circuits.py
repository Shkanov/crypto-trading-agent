"""Trailing drawdown / vol / daily-loss circuit breakers (sprint #12).

The existing `src/tools/risk_gate.py` is a point-in-time gate (does THIS
signal pass?). This module adds **trailing state-aware** circuits that
operate on the equity time series:

  1. **Trailing equity DD.** Peak-aware: if equity is −10% from the rolling
     peak, halve all sizing. If −20%, flatten everything and refuse new
     entries for a 14-day cooloff. The split mirrors AHL's 15%-vol target
     with a −20% soft stop (Carver, *Systematic Trading*, ch. 16).
  2. **Realized portfolio vol.** If 20-day annualized vol > 2× target for
     5 consecutive days, halve sizing — vol-regime de-risking, not binary
     stop-out. From Clenow, *Trading Evolved*, ch. 11.
  3. **Daily PnL.** If today's PnL is < −3% of equity, block new entries
     until the next session (existing positions keep running; this differs
     from `RiskGate.max_daily_loss_pct` which fully halts trading).

The three are independent and compose: the effective `size_multiplier`
is min over the three. `flatten=True` from circuit (1) overrides
everything; `no_new_entries=True` from circuits (1) or (3) blocks new
positions without forcing exits.

The module is **stateless**: callers pass in the equity time series and
any active cooloff timestamp. Use `evaluate_circuits` once per
decision-window (typically once per bar or once per pre-trade check) and
apply the returned `CircuitState` before invoking `RiskGate.check()`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


DAY_MS = 86_400_000


# ---------------------------------------------------------------------------
# Configuration

@dataclass(frozen=True)
class CircuitConfig:
    """Default thresholds match the spec in
    `data/research/strategy_tuning/recommendations_2026_05_27.md` §1.6."""

    # Circuit 1: trailing equity DD
    dd_halve_pct: float = 10.0           # −10% from peak → multiplier 0.5
    dd_flatten_pct: float = 20.0         # −20% from peak → flatten + cooloff
    dd_cooloff_days: int = 14

    # Circuit 2: realized vol regime
    target_vol_pct_annual: float = 15.0  # AHL-style 15% annualized target
    vol_breach_multiple: float = 2.0     # >2× target triggers
    vol_breach_consecutive_days: int = 5
    vol_lookback_days: int = 20          # rolling window for σ̂
    # When the breach condition holds, halve sizes.
    vol_breach_multiplier: float = 0.5

    # Circuit 3: daily PnL cap (blocks new entries; doesn't flatten)
    daily_loss_no_new_pct: float = 3.0


# ---------------------------------------------------------------------------
# Inputs / outputs

@dataclass(frozen=True)
class AccountTimeSeries:
    """Daily equity history fed into the circuits. `equity_curve[-1]` is
    today's mark-to-market equity; `daily_pnl_pct[-1]` is today's % PnL on
    yesterday's equity. The two series must have equal length (today's
    daily_pnl_pct can be 0.0 if the day hasn't closed yet).

    `last_day_ms` is the UTC-midnight timestamp of today's row — used for
    cooloff arithmetic.
    """
    equity_curve: tuple[float, ...]
    daily_pnl_pct: tuple[float, ...]
    last_day_ms: int

    def __post_init__(self) -> None:
        if len(self.equity_curve) != len(self.daily_pnl_pct):
            raise ValueError(
                f"equity_curve len {len(self.equity_curve)} != "
                f"daily_pnl_pct len {len(self.daily_pnl_pct)}"
            )


@dataclass(frozen=True)
class CircuitState:
    """Result of one circuit evaluation. Apply in this order:

      - If `flatten` is True: close all open positions immediately, then
        refuse new entries until `cooloff_until_ms`.
      - If `no_new_entries` is True: don't open new positions but let
        existing ones run.
      - Multiply intended position size by `size_multiplier` (1.0, 0.5, or
        0.0).
    """
    size_multiplier: float
    flatten: bool
    no_new_entries: bool
    cooloff_until_ms: int
    triggered: tuple[str, ...] = field(default_factory=tuple)
    dd_from_peak_pct: float = 0.0
    recent_vol_annual_pct: float = 0.0
    consecutive_high_vol_days: int = 0
    reason: str = ""


# ---------------------------------------------------------------------------
# Circuit primitives

def _trailing_dd_pct(curve: Sequence[float]) -> float:
    """Current drawdown from running peak, in pct (positive = down)."""
    if len(curve) == 0:
        return 0.0
    peaks = np.maximum.accumulate(np.asarray(curve, dtype=float))
    cur = float(curve[-1])
    peak = float(peaks[-1])
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - cur) / peak * 100.0)


def _rolling_vol_annual_pct(daily_pnl_pct: Sequence[float], window: int) -> float:
    """Annualised realised vol of the last `window` daily PnL pct returns.
    Returns 0 when we don't have enough samples."""
    n = len(daily_pnl_pct)
    if n < window:
        return 0.0
    arr = np.asarray(daily_pnl_pct[-window:], dtype=float)
    if arr.std(ddof=1) == 0:
        return 0.0
    return float(arr.std(ddof=1) * math.sqrt(365.0))


def _consecutive_high_vol_days(
    daily_pnl_pct: Sequence[float], window: int,
    threshold_annual_pct: float,
) -> int:
    """How many of the most recent days had rolling-`window`-day annualised
    vol above `threshold_annual_pct`. Counts back from the end until a day
    is below threshold or we run out of history."""
    if len(daily_pnl_pct) < window + 1:
        return 0
    count = 0
    arr = np.asarray(daily_pnl_pct, dtype=float)
    for back in range(len(arr) - window, -1, -1):
        seg = arr[back: back + window]
        if seg.std(ddof=1) == 0:
            break
        vol_ann = seg.std(ddof=1) * math.sqrt(365.0)
        if vol_ann > threshold_annual_pct:
            count += 1
            # Only count the most recent contiguous run — stop on the first
            # quiet day.
        else:
            break
    return count


# ---------------------------------------------------------------------------
# Main evaluator

def evaluate_circuits(
    ts: AccountTimeSeries,
    cfg: Optional[CircuitConfig] = None,
    now_ms: Optional[int] = None,
    active_cooloff_until_ms: int = 0,
) -> CircuitState:
    """Apply all three circuits to the current equity time series.

    `active_cooloff_until_ms` should be carried forward by the caller from
    a previous evaluation that returned a non-zero `cooloff_until_ms`. We
    re-honour it as long as `now_ms < active_cooloff_until_ms`, regardless
    of whether DD has recovered in the meantime.
    """
    cfg = cfg or CircuitConfig()
    now = ts.last_day_ms if now_ms is None else now_ms

    triggered: list[str] = []
    size_mult = 1.0
    flatten = False
    no_new = False
    cooloff_until = active_cooloff_until_ms
    reasons: list[str] = []

    # Carry-forward cooloff: once tripped, stay flat for the rest of the window.
    if active_cooloff_until_ms and now < active_cooloff_until_ms:
        triggered.append("dd_cooloff_active")
        size_mult = 0.0
        no_new = True
        reasons.append(
            f"cooloff active until {active_cooloff_until_ms} "
            f"(remaining {(active_cooloff_until_ms - now) // DAY_MS}d)"
        )

    # ----- Circuit 1: trailing equity DD -----
    dd_pct = _trailing_dd_pct(ts.equity_curve)
    if dd_pct >= cfg.dd_flatten_pct:
        triggered.append("dd_flatten")
        flatten = True
        no_new = True
        size_mult = 0.0
        # Start a fresh cooloff or extend an existing one.
        new_cooloff = now + cfg.dd_cooloff_days * DAY_MS
        cooloff_until = max(cooloff_until, new_cooloff)
        reasons.append(
            f"trailing DD {dd_pct:.1f}% ≥ flatten threshold {cfg.dd_flatten_pct:.1f}%"
        )
    elif dd_pct >= cfg.dd_halve_pct:
        triggered.append("dd_halve")
        size_mult = min(size_mult, 0.5)
        reasons.append(
            f"trailing DD {dd_pct:.1f}% ≥ halve threshold {cfg.dd_halve_pct:.1f}%"
        )

    # ----- Circuit 2: realized vol regime -----
    vol_ann = _rolling_vol_annual_pct(ts.daily_pnl_pct, cfg.vol_lookback_days)
    threshold_vol = cfg.target_vol_pct_annual * cfg.vol_breach_multiple
    consec = _consecutive_high_vol_days(
        ts.daily_pnl_pct, cfg.vol_lookback_days, threshold_vol,
    )
    if consec >= cfg.vol_breach_consecutive_days:
        triggered.append("vol_regime")
        size_mult = min(size_mult, cfg.vol_breach_multiplier)
        reasons.append(
            f"{consec} consecutive days of {cfg.vol_lookback_days}d-vol "
            f"{vol_ann:.1f}% > {threshold_vol:.1f}% (2× target)"
        )

    # ----- Circuit 3: daily PnL block on new entries -----
    if ts.daily_pnl_pct and ts.daily_pnl_pct[-1] <= -cfg.daily_loss_no_new_pct:
        triggered.append("daily_loss_block")
        no_new = True
        reasons.append(
            f"today's PnL {ts.daily_pnl_pct[-1]:+.2f}% ≤ "
            f"−{cfg.daily_loss_no_new_pct:.1f}% block threshold"
        )

    return CircuitState(
        size_multiplier=size_mult,
        flatten=flatten,
        no_new_entries=no_new,
        cooloff_until_ms=cooloff_until,
        triggered=tuple(triggered),
        dd_from_peak_pct=dd_pct,
        recent_vol_annual_pct=vol_ann,
        consecutive_high_vol_days=consec,
        reason="; ".join(reasons) if reasons else "all clear",
    )
