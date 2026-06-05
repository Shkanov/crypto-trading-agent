"""Startup reconciliation — compare local OPEN trades to exchange state.

Two pathological cases this catches:
1. Local DB says we have a perp short on BTCUSDT for 0.01 BTC, but the
   exchange shows no position (was manually closed via Binance UI while
   the agent was down).
2. Exchange shows a position we don't know about (left over from a manual
   trade or previous agent run with a different DB).

On mismatch we HALT the hot loop and require operator resolution. Going
live with state drift is the fast way to lose money on phantom positions.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import structlog

from src.models.types import Trade
from src.services.storage import Storage
from src.tools.binance_client import BinanceClient

log = structlog.get_logger(__name__)


@dataclass
class ReconciliationReport:
    ok: bool
    matched: list[Trade] = field(default_factory=list)
    local_only: list[Trade] = field(default_factory=list)
    exchange_only: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # True when we could NOT reach the exchange to compare state at all
    # (network error / -1003 rate-limit ban) — as opposed to a confirmed
    # local-vs-exchange mismatch. Transient failures must NOT trigger the
    # 24h operator halt; the caller retries until the exchange is reachable.
    transient: bool = False


async def _reconcile_margin(binance: BinanceClient, storage: Storage,
                             local_open: list[Trade],
                             report: ReconciliationReport) -> None:
    """Reconcile SHORT-SPOT legs against the Binance margin account
    (R3 fix). For the negative-funding direction, a short_spot leg is a
    margin-borrowed sell. The margin account exposes `userAssets[].borrowed`
    per asset; we compare against open short-spot Trade rows."""
    short_spot_local = [t for t in local_open if t.market == "spot" and t.side == "short"]
    if not short_spot_local:
        return
    try:
        margin_acct = await binance.client.get_margin_account()
    except Exception as e:
        # If margin is disabled, get_margin_account returns an error; surface
        # as a halt-worthy mismatch since we have local short_spot Trades that
        # imply margin should be active.
        report.ok = False
        report.notes.append(
            f"local has {len(short_spot_local)} short-spot legs but margin "
            f"account fetch failed: {e}"
        )
        return
    borrowed: dict[str, float] = {}
    for asset in (margin_acct.get("userAssets") or []):
        a = asset.get("asset")
        b = float(asset.get("borrowed", 0) or 0)
        if a and b > 0:
            borrowed[a] = b
    # Match local Trade rows by base asset implied from the symbol.
    # Heuristic: BTCUSDT → base = "BTC". Works for *USDT/USDC pairs.
    for t in short_spot_local:
        base = t.symbol.replace("USDT", "").replace("USDC", "")
        borrowed_amt = borrowed.get(base, 0.0)
        if borrowed_amt <= 0:
            report.ok = False
            report.notes.append(
                f"GHOST short-spot: local {t.id[:8]} ({t.symbol} qty={t.qty}) "
                f"has no margin borrow of {base}"
            )
        else:
            rel = abs(borrowed_amt - t.qty) / max(t.qty, 1e-9)
            if rel > 0.05:
                report.ok = False
                report.notes.append(
                    f"QTY MISMATCH short-spot: {t.symbol} local={t.qty} "
                    f"margin borrowed={borrowed_amt} ({rel*100:.1f}% diff)"
                )


async def reconcile_on_boot(binance: BinanceClient, storage: Storage,
                            live: bool) -> ReconciliationReport:
    """Pull open Trades from storage; pull open positions/balances from exchange;
    return a report. In paper mode we skip exchange checks entirely (no positions
    exist on Binance).

    Reconciles BOTH perp positions (`futures_positions`) AND short-spot legs
    (via margin account `userAssets[].borrowed`). Long-spot legs are not
    reconciled because Binance spot can't distinguish "this BTC is part of
    an agent trade" from general inventory."""
    report = ReconciliationReport(ok=True)
    if not live:
        report.notes.append("paper mode — exchange reconciliation skipped")
        return report

    local_open = await storage.list_open_trades()
    if not local_open:
        report.notes.append("no local open trades; clean start")
        # Still check exchange to detect orphan positions.

    # Fetch exchange positions with a few bounded retries. A failure here means
    # we couldn't *see* the exchange (network / -1003 ban), which is a TRANSIENT
    # condition — flag it as such so the caller retries instead of slamming a
    # 24h halt. (binance.futures_positions already waits out a recorded ban via
    # respect_ban; the short backoff covers brief blips before one is recorded.)
    positions = None
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            positions = await binance.futures_positions()
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < 2:
                await asyncio.sleep(2.0 * (attempt + 1))
    if positions is None:
        report.ok = False
        report.transient = True
        report.notes.append(f"could not fetch futures positions: {last_err}")
        return report

    # Keep only nonzero positions
    live_perps = {}
    for p in positions:
        amt = float(p.get("positionAmt", 0) or 0)
        if abs(amt) > 1e-12:
            live_perps[p["symbol"]] = p

    local_perps = [t for t in local_open if t.market == "perps"]
    local_perps_by_symbol = {t.symbol: t for t in local_perps}

    # Local has it, exchange doesn't → ghost trade.
    for t in local_perps:
        live_pos = live_perps.get(t.symbol)
        if not live_pos:
            report.ok = False
            report.local_only.append(t)
            report.notes.append(
                f"GHOST: local trade {t.id[:8]} ({t.symbol} {t.side} {t.qty}) "
                f"has no exchange position"
            )
            continue
        # Direction match check.
        live_amt = float(live_pos.get("positionAmt", 0))
        local_sign = +1 if t.side == "long" else -1
        live_sign = +1 if live_amt > 0 else -1
        if local_sign != live_sign:
            report.ok = False
            report.notes.append(
                f"DIRECTION MISMATCH: {t.symbol} local={t.side} exchange amt={live_amt}"
            )
            continue
        # Quantity within 5% tolerance.
        rel = abs(abs(live_amt) - t.qty) / max(t.qty, 1e-9)
        if rel > 0.05:
            report.ok = False
            report.notes.append(
                f"QTY MISMATCH: {t.symbol} local={t.qty} exchange={abs(live_amt)} "
                f"({rel*100:.1f}% diff)"
            )
            continue
        report.matched.append(t)

    # Exchange has it, local doesn't → orphan from manual trading.
    for sym, p in live_perps.items():
        if sym not in local_perps_by_symbol:
            report.ok = False
            report.exchange_only.append(p)
            report.notes.append(
                f"ORPHAN: exchange has {sym} position amt={p.get('positionAmt')} "
                f"with no local Trade row"
            )

    # R3: reconcile short-spot legs against margin account.
    await _reconcile_margin(binance, storage, local_open, report)

    return report


def format_report(report: ReconciliationReport) -> str:
    lines = [f"Reconciliation: {'OK' if report.ok else 'FAILED'}"]
    lines.append(f"  matched: {len(report.matched)}")
    lines.append(f"  local-only (ghost): {len(report.local_only)}")
    lines.append(f"  exchange-only (orphan): {len(report.exchange_only)}")
    for note in report.notes:
        lines.append(f"  - {note}")
    return "\n".join(lines)
