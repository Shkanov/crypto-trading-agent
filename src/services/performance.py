"""Performance attribution + slippage rollups.

Single source of truth for "did this work?" — queries the Trade and Fill
tables and produces per-strategy, per-symbol, per-time-of-day breakdowns.
Used by Telegram /status, daily digests, and the StrategyAgent's
context payload.

Designed to stay cheap (single SQL pass per question) so it can run on
the housekeeping loop without affecting hot path latency.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import structlog
from sqlalchemy import select

from src.services.storage import FillRow, Storage, TradeRow

log = structlog.get_logger(__name__)


@dataclass
class StratStats:
    strategy: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl_usd: float = 0.0
    avg_pnl_usd: float = 0.0
    win_rate: float = 0.0
    expectancy_usd: float = 0.0
    avg_holding_secs: float = 0.0
    avg_slippage_entry_bps: float = 0.0
    avg_slippage_exit_bps: float = 0.0


@dataclass
class SymbolStats:
    symbol: str
    strategy: str
    trades: int = 0
    realized_pnl_usd: float = 0.0
    win_rate: float = 0.0


@dataclass
class PerformanceReport:
    since: datetime
    by_strategy: list[StratStats]
    by_symbol: list[SymbolStats]
    by_hour_utc: dict[int, float]   # hour-of-day → realized pnl
    overall_pnl_usd: float
    overall_trades: int


def _stats_for_trades(strat: str, rows: list[TradeRow]) -> StratStats:
    out = StratStats(strategy=strat)
    if not rows:
        return out
    pnls: list[float] = []
    holds: list[float] = []
    entry_slips: list[float] = []
    exit_slips: list[float] = []
    for r in rows:
        if r.realized_pnl_usd is None:
            continue
        pnls.append(r.realized_pnl_usd)
        if r.exit_ts_ms and r.entry_ts_ms:
            holds.append((r.exit_ts_ms - r.entry_ts_ms) / 1000.0)
        if r.slippage_bps_entry is not None:
            entry_slips.append(r.slippage_bps_entry)
        if r.slippage_bps_exit is not None:
            exit_slips.append(r.slippage_bps_exit)
    out.trades = len(pnls)
    if not pnls:
        return out
    out.wins = sum(1 for p in pnls if p > 0)
    out.losses = sum(1 for p in pnls if p < 0)
    out.realized_pnl_usd = sum(pnls)
    out.avg_pnl_usd = out.realized_pnl_usd / len(pnls)
    out.win_rate = out.wins / len(pnls)
    out.expectancy_usd = out.avg_pnl_usd
    out.avg_holding_secs = (sum(holds) / len(holds)) if holds else 0.0
    out.avg_slippage_entry_bps = (sum(entry_slips) / len(entry_slips)) if entry_slips else 0.0
    out.avg_slippage_exit_bps = (sum(exit_slips) / len(exit_slips)) if exit_slips else 0.0
    return out


async def build_report(storage: Storage, since: Optional[datetime] = None) -> PerformanceReport:
    """Aggregate closed Trades since `since` (default: 7 days ago)."""
    since = since or (datetime.utcnow() - timedelta(days=7))
    async with storage.session() as s:
        result = await s.execute(
            select(TradeRow).where(
                TradeRow.status == "CLOSED",
                TradeRow.created_at >= since,
            )
        )
        rows = list(result.scalars())

    # By strategy
    strats: dict[str, list[TradeRow]] = {}
    for r in rows:
        strats.setdefault(r.strategy, []).append(r)
    by_strategy = [_stats_for_trades(s, rs) for s, rs in strats.items()]

    # By (symbol, strategy)
    by_symbol_map: dict[tuple[str, str], list[TradeRow]] = {}
    for r in rows:
        by_symbol_map.setdefault((r.symbol, r.strategy), []).append(r)
    by_symbol: list[SymbolStats] = []
    for (sym, strat), rs in by_symbol_map.items():
        pnls = [r.realized_pnl_usd for r in rs if r.realized_pnl_usd is not None]
        if not pnls:
            continue
        by_symbol.append(SymbolStats(
            symbol=sym, strategy=strat,
            trades=len(pnls),
            realized_pnl_usd=sum(pnls),
            win_rate=sum(1 for p in pnls if p > 0) / len(pnls),
        ))

    # By hour-of-day UTC
    by_hour: dict[int, float] = {}
    for r in rows:
        if not r.exit_ts_ms or r.realized_pnl_usd is None:
            continue
        h = datetime.utcfromtimestamp(r.exit_ts_ms / 1000).hour
        by_hour[h] = by_hour.get(h, 0.0) + r.realized_pnl_usd

    overall_pnl = sum((r.realized_pnl_usd or 0.0) for r in rows)
    return PerformanceReport(
        since=since, by_strategy=by_strategy, by_symbol=by_symbol,
        by_hour_utc=by_hour, overall_pnl_usd=overall_pnl, overall_trades=len(rows),
    )


def format_report_markdown(report: PerformanceReport) -> str:
    lines = [f"*Performance since {report.since.strftime('%Y-%m-%d %H:%M UTC')}*"]
    lines.append(f"Total: `{report.overall_trades}` trades, realized `${report.overall_pnl_usd:+.2f}`")
    lines.append("")
    for s in sorted(report.by_strategy, key=lambda x: -x.realized_pnl_usd):
        if s.trades == 0:
            continue
        slip = ""
        if s.avg_slippage_entry_bps or s.avg_slippage_exit_bps:
            slip = f" | slip e{s.avg_slippage_entry_bps:+.1f}bps x{s.avg_slippage_exit_bps:+.1f}bps"
        lines.append(
            f"*{s.strategy}* — n={s.trades} pnl=`${s.realized_pnl_usd:+.2f}` "
            f"win={s.win_rate:.0%} avg=`${s.avg_pnl_usd:+.2f}`{slip}"
        )
    if report.by_symbol:
        lines.append("")
        lines.append("*By symbol:*")
        for sm in sorted(report.by_symbol, key=lambda x: -x.realized_pnl_usd)[:10]:
            lines.append(f"  {sm.symbol} ({sm.strategy}): n={sm.trades} pnl=`${sm.realized_pnl_usd:+.2f}`")
    return "\n".join(lines)
