"""SQLite (via SQLAlchemy async) persistence.

Tables intentionally minimal — we persist what we need to crash-recover and
audit, not raw ticks. Ticks live in memory; durable history lives here.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import structlog
from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.config.settings import get_settings
from src.models.types import Proposal, ProposalStatus, StrategyConfig, Trade, TradeStatus

log = structlog.get_logger(__name__)


class Base(DeclarativeBase):
    pass


class ProposalRow(Base):
    __tablename__ = "proposals"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    market: Mapped[str] = mapped_column(String(8))
    qty: Mapped[float] = mapped_column(Float)
    notional_usd: Mapped[float] = mapped_column(Float)
    leverage: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(24), index=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[str] = mapped_column(Text)  # full Proposal JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FillRow(Base):
    __tablename__ = "fills"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    proposal_id: Mapped[str] = mapped_column(String(64), index=True)
    trade_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    qty: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    signal_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    raw: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class TradeRow(Base):
    __tablename__ = "trades"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    strategy: Mapped[str] = mapped_column(String(32), index=True, default="indicator")
    proposal_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    market: Mapped[str] = mapped_column(String(8))
    side: Mapped[str] = mapped_column(String(8))
    qty: Mapped[float] = mapped_column(Float)
    leverage: Mapped[int] = mapped_column(Integer, default=1)
    entry_price: Mapped[float] = mapped_column(Float)
    entry_ts_ms: Mapped[int] = mapped_column(Integer)
    intended_stop: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    intended_tp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_ts_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    exit_reason: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    fee_total_usd: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    slippage_bps_entry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    slippage_bps_exit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    funding_accrued_usd: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16), index=True, default="OPEN")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class StrategyConfigRow(Base):
    __tablename__ = "strategy_configs"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    payload: Mapped[str] = mapped_column(Text)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PairMetaRow(Base):
    """Pair-level metadata that doesn't fit on a single Trade row: the
    funding direction (+1 long_spot/short_perp, -1 short_spot/long_perp)
    and the perp entry mark used by the F4 safety stop. Keyed by pair_id
    (== proposal_id on the legs)."""
    __tablename__ = "pair_meta"
    pair_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    strategy: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[int] = mapped_column(Integer)
    perp_entry_mark: Mapped[float] = mapped_column(Float, default=0.0)
    last_credited_funding_ms: Mapped[int] = mapped_column(Integer, default=0)
    opened_ms: Mapped[int] = mapped_column(Integer)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class RampStateRow(Base):
    """Persists NotionalRamp state so restarts don't reset accumulated ramp-up."""
    __tablename__ = "ramp_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    current_max_notional_usd: Mapped[float] = mapped_column(Float)
    last_review_ts_ms: Mapped[int] = mapped_column(Integer, default=0)
    consecutive_profitable_weeks: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CircuitStateRow(Base):
    """Persists the trailing-DD cooloff timestamp from `risk_circuits.evaluate_circuits`.
    Without persistence a crash mid-cooloff would reset to the rolling-peak
    after restart and re-enable trading at the worst possible moment."""
    __tablename__ = "circuit_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    cooloff_until_ms: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AllocatorStateRow(Base):
    """Persists multi-strategy allocator state (sprint #17). Without
    persistence, an orchestrator restart would reset to equal-weight on the
    next rebalance check, throwing away the HRP/inverse-vol history."""
    __tablename__ = "allocator_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    last_rebalance_ms: Mapped[int] = mapped_column(Integer, default=0)
    method_used: Mapped[str] = mapped_column(String(16), default="equal")
    # JSON-encoded {strategy_name: weight}; current and previous.
    weights_json: Mapped[str] = mapped_column(Text, default="{}")
    prev_weights_json: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ProposedConfigRow(Base):
    """LLM-proposed configs awaiting user promotion. Advisory-only by default."""
    __tablename__ = "proposed_configs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    base_version: Mapped[int] = mapped_column(Integer)
    payload: Mapped[str] = mapped_column(Text)
    notes: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="PENDING", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class AuditRow(Base):
    __tablename__ = "audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class Storage:
    def __init__(self, database_url: Optional[str] = None) -> None:
        url = database_url or get_settings().database_url
        # SQLite WAL via connect args (no-op on Postgres).
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_async_engine(url, echo=False, connect_args=connect_args)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init(self) -> None:
        import os
        # Ensure local sqlite dir exists.
        url = str(self.engine.url)
        if url.startswith("sqlite") and "///" in url:
            path = url.split("///", 1)[1]
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        # Enable WAL mode for sqlite.
        if url.startswith("sqlite"):
            async with self.engine.begin() as conn:
                await conn.exec_driver_sql("PRAGMA journal_mode=WAL")

    def session(self) -> AsyncSession:
        return self.session_factory()

    # ----- Proposals -----
    async def save_proposal(self, p: Proposal) -> None:
        async with self.session() as s, s.begin():
            existing = await s.get(ProposalRow, p.id)
            if existing is None:
                row = ProposalRow(
                    id=p.id, symbol=p.signal.symbol, side=p.signal.side, market=p.market,
                    qty=p.qty, notional_usd=p.notional_usd, leverage=p.leverage,
                    status=p.status.value, reason=p.reason,
                    payload=p.model_dump_json(),
                )
                s.add(row)
            else:
                existing.status = p.status.value
                existing.reason = p.reason
                existing.payload = p.model_dump_json()
                existing.updated_at = datetime.utcnow()

    async def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        async with self.session() as s:
            row = await s.get(ProposalRow, proposal_id)
            if not row:
                return None
            return Proposal.model_validate_json(row.payload)

    async def list_open_proposals(self) -> list[Proposal]:
        open_states = (
            ProposalStatus.PROPOSED.value,
            ProposalStatus.AWAITING_USER.value,
            ProposalStatus.AUTO_APPROVED.value,
            ProposalStatus.APPROVED.value,
            ProposalStatus.SUBMITTED.value,
        )
        async with self.session() as s:
            result = await s.execute(
                select(ProposalRow).where(ProposalRow.status.in_(open_states))
            )
            return [Proposal.model_validate_json(r.payload) for r in result.scalars()]

    # ----- Fills -----
    async def save_fill(self, proposal_id: str, symbol: str, side: str,
                        qty: float, price: float, fee: float, raw: dict,
                        signal_price: Optional[float] = None,
                        trade_id: Optional[str] = None) -> None:
        async with self.session() as s, s.begin():
            s.add(FillRow(
                proposal_id=proposal_id, trade_id=trade_id, symbol=symbol, side=side,
                qty=qty, price=price, signal_price=signal_price, fee=fee, raw=raw,
            ))

    async def avg_slippage_bps(self, symbol: Optional[str] = None,
                               limit: int = 200) -> Optional[float]:
        async with self.session() as s:
            q = select(FillRow).order_by(FillRow.created_at.desc()).limit(limit)
            if symbol:
                q = q.where(FillRow.symbol == symbol)
            result = await s.execute(q)
            vals: list[float] = []
            for r in result.scalars():
                if r.signal_price and r.signal_price > 0:
                    diff = (r.price - r.signal_price) if r.side == "BUY" else (r.signal_price - r.price)
                    vals.append((diff / r.signal_price) * 10_000)
            return sum(vals) / len(vals) if vals else None

    # ----- Trades (round-trips) -----
    async def open_trade(self, trade: Trade) -> bool:
        """Insert an OPEN Trade. Returns True on insert, False if `trade.id`
        already existed (duplicate — caller should treat this as a bug,
        especially in pair-open paths where it can leave you single-legged)."""
        async with self.session() as s, s.begin():
            existing = await s.get(TradeRow, trade.id)
            if existing is not None:
                log.error("storage.open_trade.duplicate_id",
                          trade_id=trade.id, symbol=trade.symbol,
                          strategy=trade.strategy)
                return False
            s.add(TradeRow(
                id=trade.id, strategy=trade.strategy, proposal_id=trade.proposal_id,
                symbol=trade.symbol, market=trade.market, side=trade.side,
                qty=trade.qty, leverage=trade.leverage,
                entry_price=trade.entry_price, entry_ts_ms=trade.entry_ts_ms,
                intended_stop=trade.intended_stop, intended_tp=trade.intended_tp,
                fee_total_usd=trade.fee_total_usd,
                slippage_bps_entry=trade.slippage_bps_entry,
                funding_accrued_usd=trade.funding_accrued_usd,
                status=TradeStatus.OPEN.value,
            ))
            return True

    async def credit_funding(self, trade_id: str, amount_usd: float) -> Optional[float]:
        """Add `amount_usd` to a Trade's funding_accrued_usd. Returns the new
        running total, or None if the trade is missing/closed."""
        async with self.session() as s, s.begin():
            row = await s.get(TradeRow, trade_id)
            if row is None or row.status != TradeStatus.OPEN.value:
                return None
            row.funding_accrued_usd = (row.funding_accrued_usd or 0.0) + amount_usd
            return row.funding_accrued_usd

    async def close_trade(self, trade_id: str, exit_price: float, exit_reason: str,
                          fee_total_usd: float, slippage_bps_exit: Optional[float] = None
                          ) -> Optional[Trade]:
        async with self.session() as s, s.begin():
            row = await s.get(TradeRow, trade_id)
            if row is None or row.status == TradeStatus.CLOSED.value:
                return None
            row.exit_price = exit_price
            row.exit_ts_ms = int(datetime.utcnow().timestamp() * 1000)
            row.exit_reason = exit_reason
            row.fee_total_usd = (row.fee_total_usd or 0.0) + fee_total_usd
            gross = (exit_price - row.entry_price) * row.qty
            if row.side == "short":
                gross = -gross
            # Realized P&L includes accrued funding (the entire point of
            # delta-neutral pair trades — without this, paper P&L lies).
            row.realized_pnl_usd = gross - row.fee_total_usd + (row.funding_accrued_usd or 0.0)
            row.slippage_bps_exit = slippage_bps_exit
            row.status = TradeStatus.CLOSED.value
            return Trade(
                id=row.id, strategy=row.strategy, proposal_id=row.proposal_id,
                symbol=row.symbol, market=row.market, side=row.side,
                qty=row.qty, leverage=row.leverage,
                entry_price=row.entry_price, entry_ts_ms=row.entry_ts_ms,
                intended_stop=row.intended_stop, intended_tp=row.intended_tp,
                exit_price=row.exit_price, exit_ts_ms=row.exit_ts_ms,
                exit_reason=row.exit_reason, fee_total_usd=row.fee_total_usd,
                realized_pnl_usd=row.realized_pnl_usd,
                slippage_bps_entry=row.slippage_bps_entry,
                slippage_bps_exit=row.slippage_bps_exit,
                funding_accrued_usd=row.funding_accrued_usd or 0.0,
                status=TradeStatus.CLOSED,
            )

    async def list_open_trades(self) -> list[Trade]:
        async with self.session() as s:
            result = await s.execute(select(TradeRow).where(TradeRow.status == TradeStatus.OPEN.value))
            return [self._row_to_trade(r) for r in result.scalars()]

    async def recent_closed_trades(self, since_ms: int) -> list[Trade]:
        cutoff = datetime.utcfromtimestamp(since_ms / 1000)
        async with self.session() as s:
            result = await s.execute(
                select(TradeRow).where(
                    TradeRow.status == TradeStatus.CLOSED.value,
                    TradeRow.created_at >= cutoff,
                )
            )
            return [self._row_to_trade(r) for r in result.scalars()]

    @staticmethod
    def _row_to_trade(row: "TradeRow") -> Trade:
        return Trade(
            id=row.id, strategy=row.strategy, proposal_id=row.proposal_id,
            symbol=row.symbol, market=row.market, side=row.side,
            qty=row.qty, leverage=row.leverage,
            entry_price=row.entry_price, entry_ts_ms=row.entry_ts_ms,
            intended_stop=row.intended_stop, intended_tp=row.intended_tp,
            exit_price=row.exit_price, exit_ts_ms=row.exit_ts_ms,
            exit_reason=row.exit_reason, fee_total_usd=row.fee_total_usd or 0.0,
            realized_pnl_usd=row.realized_pnl_usd,
            slippage_bps_entry=row.slippage_bps_entry,
            slippage_bps_exit=row.slippage_bps_exit,
            funding_accrued_usd=row.funding_accrued_usd or 0.0,
            status=TradeStatus(row.status),
        )

    # ----- StrategyConfig -----
    async def save_strategy_config(self, cfg: StrategyConfig) -> None:
        async with self.session() as s, s.begin():
            existing = await s.get(StrategyConfigRow, cfg.version)
            if existing is None:
                s.add(StrategyConfigRow(
                    version=cfg.version, payload=cfg.model_dump_json(), notes=cfg.notes,
                ))

    async def latest_strategy_config(self) -> Optional[StrategyConfig]:
        async with self.session() as s:
            result = await s.execute(
                select(StrategyConfigRow).order_by(StrategyConfigRow.version.desc()).limit(1)
            )
            row = result.scalar_one_or_none()
            return StrategyConfig.model_validate_json(row.payload) if row else None

    # ----- Proposed configs (advisory) -----
    async def save_proposed_config(self, prop_id: str, base_version: int,
                                    cfg: StrategyConfig, notes: str) -> None:
        async with self.session() as s, s.begin():
            s.add(ProposedConfigRow(
                id=prop_id, base_version=base_version,
                payload=cfg.model_dump_json(), notes=notes,
                status="PENDING",
            ))

    async def get_proposed_config(self, prop_id: str) -> Optional[StrategyConfig]:
        async with self.session() as s:
            row = await s.get(ProposedConfigRow, prop_id)
            if row is None or row.status != "PENDING":
                return None
            return StrategyConfig.model_validate_json(row.payload)

    async def mark_proposed_config(self, prop_id: str, status: str) -> None:
        async with self.session() as s, s.begin():
            row = await s.get(ProposedConfigRow, prop_id)
            if row:
                row.status = status

    async def pending_proposed_configs(self) -> list[tuple[str, str, int]]:
        async with self.session() as s:
            result = await s.execute(
                select(ProposedConfigRow).where(ProposedConfigRow.status == "PENDING")
                .order_by(ProposedConfigRow.created_at.desc())
            )
            return [(r.id, r.notes, r.base_version) for r in result.scalars()]

    # ----- Pair metadata (direction + perp_entry_mark) -----
    async def save_pair_meta(self, pair_id: str, strategy: str, symbol: str,
                              direction: int, perp_entry_mark: float,
                              opened_ms: int) -> None:
        async with self.session() as s, s.begin():
            row = await s.get(PairMetaRow, pair_id)
            if row is None:
                s.add(PairMetaRow(
                    pair_id=pair_id, strategy=strategy, symbol=symbol,
                    direction=direction, perp_entry_mark=perp_entry_mark,
                    opened_ms=opened_ms,
                ))
            else:
                row.direction = direction
                row.perp_entry_mark = perp_entry_mark

    async def update_pair_funding_watermark(self, pair_id: str,
                                             last_credited_funding_ms: int) -> None:
        async with self.session() as s, s.begin():
            row = await s.get(PairMetaRow, pair_id)
            if row is not None:
                row.last_credited_funding_ms = last_credited_funding_ms

    async def close_pair_meta(self, pair_id: str) -> None:
        async with self.session() as s, s.begin():
            row = await s.get(PairMetaRow, pair_id)
            if row is not None:
                row.closed_at = datetime.utcnow()

    async def load_pair_meta(self, pair_id: str) -> Optional[dict]:
        async with self.session() as s:
            row = await s.get(PairMetaRow, pair_id)
            if row is None or row.closed_at is not None:
                return None
            return {
                "pair_id": row.pair_id, "strategy": row.strategy,
                "symbol": row.symbol, "direction": row.direction,
                "perp_entry_mark": row.perp_entry_mark,
                "last_credited_funding_ms": row.last_credited_funding_ms,
                "opened_ms": row.opened_ms,
            }

    # ----- NotionalRamp state -----
    async def save_ramp_state(self, current_max_notional_usd: float,
                              last_review_ts_ms: int,
                              consecutive_profitable_weeks: int) -> None:
        async with self.session() as s, s.begin():
            row = await s.get(RampStateRow, 1)
            if row is None:
                s.add(RampStateRow(
                    id=1,
                    current_max_notional_usd=current_max_notional_usd,
                    last_review_ts_ms=last_review_ts_ms,
                    consecutive_profitable_weeks=consecutive_profitable_weeks,
                ))
            else:
                row.current_max_notional_usd = current_max_notional_usd
                row.last_review_ts_ms = last_review_ts_ms
                row.consecutive_profitable_weeks = consecutive_profitable_weeks
                row.updated_at = datetime.utcnow()

    async def load_ramp_state(self) -> Optional[tuple[float, int, int]]:
        async with self.session() as s:
            row = await s.get(RampStateRow, 1)
            if row is None:
                return None
            return (row.current_max_notional_usd, row.last_review_ts_ms,
                    row.consecutive_profitable_weeks)

    # ----- Circuit-breaker state -----
    async def save_circuit_state(self, cooloff_until_ms: int) -> None:
        async with self.session() as s, s.begin():
            row = await s.get(CircuitStateRow, 1)
            if row is None:
                s.add(CircuitStateRow(id=1, cooloff_until_ms=cooloff_until_ms))
            else:
                row.cooloff_until_ms = cooloff_until_ms
                row.updated_at = datetime.utcnow()

    async def load_circuit_state(self) -> int:
        """Returns the persisted cooloff_until_ms, or 0 if never set."""
        async with self.session() as s:
            row = await s.get(CircuitStateRow, 1)
            return int(row.cooloff_until_ms) if row else 0

    # ----- Allocator state (sprint #17) -----
    async def save_allocator_state(
        self, last_rebalance_ms: int, method_used: str,
        weights: dict[str, float], prev_weights: dict[str, float],
    ) -> None:
        async with self.session() as s, s.begin():
            row = await s.get(AllocatorStateRow, 1)
            w_json = json.dumps(weights)
            p_json = json.dumps(prev_weights)
            if row is None:
                s.add(AllocatorStateRow(
                    id=1, last_rebalance_ms=last_rebalance_ms,
                    method_used=method_used,
                    weights_json=w_json, prev_weights_json=p_json,
                ))
            else:
                row.last_rebalance_ms = last_rebalance_ms
                row.method_used = method_used
                row.weights_json = w_json
                row.prev_weights_json = p_json
                row.updated_at = datetime.utcnow()

    async def load_allocator_state(
        self,
    ) -> Optional[tuple[int, str, dict[str, float], dict[str, float]]]:
        """Returns (last_rebalance_ms, method_used, weights, prev_weights) or
        None if never persisted."""
        async with self.session() as s:
            row = await s.get(AllocatorStateRow, 1)
            if row is None:
                return None
            try:
                w = json.loads(row.weights_json or "{}")
                p = json.loads(row.prev_weights_json or "{}")
            except json.JSONDecodeError:
                w, p = {}, {}
            return (int(row.last_rebalance_ms), str(row.method_used or "equal"),
                    {k: float(v) for k, v in w.items()},
                    {k: float(v) for k, v in p.items()})

    # ----- Audit -----
    async def audit(self, kind: str, payload: dict) -> None:
        try:
            async with self.session() as s, s.begin():
                s.add(AuditRow(kind=kind, payload=payload))
        except Exception as e:  # never let audit kill the hot path
            log.warning("audit.failed", kind=kind, err=str(e))

    async def realized_pnl_since_ms(self, since_ms: int) -> float:
        """Sum realized_pnl_usd of trades CLOSED since `since_ms`.
        Open trades contribute zero — unrealized P&L is computed separately
        (and never used for the daily-loss kill switch)."""
        async with self.session() as s:
            result = await s.execute(
                select(TradeRow).where(
                    TradeRow.status == TradeStatus.CLOSED.value,
                    TradeRow.exit_ts_ms.isnot(None),
                    TradeRow.exit_ts_ms >= since_ms,
                )
            )
            return float(sum((r.realized_pnl_usd or 0.0) for r in result.scalars()))

    async def realized_pnl_today_usd(self) -> float:
        """Realized P&L since UTC midnight."""
        import time as _t
        since_ms = int((_t.time() // 86_400) * 86_400 * 1000)
        return await self.realized_pnl_since_ms(since_ms)

    async def realized_pnl_by_day(self, since_ms: int,
                                   until_ms: int) -> dict[int, float]:
        """Sum realized_pnl_usd of CLOSED trades by UTC day (key = UTC-midnight ms).
        Days with no closes are omitted; callers should fill them with 0.
        Range is half-open: closes with `exit_ts_ms in [since_ms, until_ms)` count."""
        DAY_MS = 86_400_000
        async with self.session() as s:
            result = await s.execute(
                select(TradeRow).where(
                    TradeRow.status == TradeStatus.CLOSED.value,
                    TradeRow.exit_ts_ms.isnot(None),
                    TradeRow.exit_ts_ms >= since_ms,
                    TradeRow.exit_ts_ms < until_ms,
                )
            )
            out: dict[int, float] = {}
            for r in result.scalars():
                if r.realized_pnl_usd is None or r.exit_ts_ms is None:
                    continue
                day_ms = (int(r.exit_ts_ms) // DAY_MS) * DAY_MS
                out[day_ms] = out.get(day_ms, 0.0) + float(r.realized_pnl_usd)
            return out

    async def realized_pnl_by_day_per_strategy(
        self, since_ms: int, until_ms: int,
    ) -> dict[str, dict[int, float]]:
        """Sprint #17: per-strategy daily PnL for the multi-strategy allocator.
        Returns {strategy_name: {utc_midnight_ms: pnl_usd}}. Strategies with
        zero closes in the window are absent from the outer dict; days with
        no closes are absent from the inner dict — callers should treat both
        as zero."""
        DAY_MS = 86_400_000
        async with self.session() as s:
            result = await s.execute(
                select(TradeRow).where(
                    TradeRow.status == TradeStatus.CLOSED.value,
                    TradeRow.exit_ts_ms.isnot(None),
                    TradeRow.exit_ts_ms >= since_ms,
                    TradeRow.exit_ts_ms < until_ms,
                )
            )
            out: dict[str, dict[int, float]] = {}
            for r in result.scalars():
                if r.realized_pnl_usd is None or r.exit_ts_ms is None:
                    continue
                day_ms = (int(r.exit_ts_ms) // DAY_MS) * DAY_MS
                strat = r.strategy or "indicator"
                strat_map = out.setdefault(strat, {})
                strat_map[day_ms] = strat_map.get(day_ms, 0.0) + float(r.realized_pnl_usd)
            return out

    async def consecutive_losses(self, limit: int = 10) -> int:
        """Walk most-recent closed trades, count consecutive losers from the head.
        Wins/break-even reset the counter."""
        async with self.session() as s:
            result = await s.execute(
                select(TradeRow).where(TradeRow.status == TradeStatus.CLOSED.value)
                .order_by(TradeRow.exit_ts_ms.desc()).limit(limit)
            )
            n = 0
            for r in result.scalars():
                if (r.realized_pnl_usd or 0.0) < 0:
                    n += 1
                else:
                    break
            return n
