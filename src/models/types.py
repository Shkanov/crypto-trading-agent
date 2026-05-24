from __future__ import annotations

import time
import uuid
from decimal import Decimal
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

Side = Literal["long", "short", "flat"]
OrderSide = Literal["BUY", "SELL"]
Market = Literal["spot", "perps"]
Timeframe = Literal["1m", "5m", "15m", "1h", "4h", "1d"]

# Deterministic UUID namespace so client_order_id is reproducible across retries.
PROPOSAL_NS = uuid.UUID("8a3bd9d2-1f7e-4f4e-8a2b-6e6f9f1c7d11")


def now_ms() -> int:
    return int(time.time() * 1000)


def stable_proposal_id(symbol: str, side: Side, signal_id: str) -> str:
    return str(uuid.uuid5(PROPOSAL_NS, f"{symbol}|{side}|{signal_id}"))


def client_order_id(proposal_id: str) -> str:
    # Binance limits clientOrderId to ~36 chars; UUID5 is 36 already, so prefix-trim.
    return f"cta_{proposal_id.replace('-', '')[:28]}"


class Kline(BaseModel):
    symbol: str
    timeframe: Timeframe
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    taker_buy_volume: float
    is_closed: bool


class IndicatorSnapshot(BaseModel):
    """Latest indicator values for one (symbol, timeframe). All optional —
    early bars won't have all indicators ready yet."""
    symbol: str
    timeframe: Timeframe
    close: float
    ema21: Optional[float] = None
    ema55: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    rsi14: Optional[float] = None
    atr14: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width_pct_rank: Optional[float] = None
    vwap: Optional[float] = None
    supertrend: Optional[float] = None
    supertrend_dir: Optional[int] = None  # +1 / -1
    cvd: Optional[float] = None
    cvd_slope: Optional[float] = None
    volume_z: Optional[float] = None
    ts_ms: int = Field(default_factory=now_ms)


class FeatureVector(BaseModel):
    """Normalized [-1, 1] features fed to the signal scorer."""
    trend: float = 0.0
    momentum: float = 0.0
    volume: float = 0.0
    volatility: float = 0.0
    pattern: float = 0.0


class Signal(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    symbol: str
    side: Side
    confidence: float
    score: float
    entry: float
    stop: float
    take_profit: float
    edge_bps: float
    features: FeatureVector
    rationale: str = ""
    ts_ms: int = Field(default_factory=now_ms)

    @property
    def rr(self) -> float:
        risk = abs(self.entry - self.stop)
        if risk == 0:
            return 0.0
        return abs(self.take_profit - self.entry) / risk


class ProposalStatus(str, Enum):
    PROPOSED = "PROPOSED"
    AUTO_APPROVED = "AUTO_APPROVED"
    AWAITING_USER = "AWAITING_USER"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    FAILED = "FAILED"


class Proposal(BaseModel):
    id: str
    signal: Signal
    market: Market
    qty: float
    notional_usd: float
    leverage: int = 1
    status: ProposalStatus = ProposalStatus.PROPOSED
    reason: str = ""
    expires_at_ms: int = 0
    created_ms: int = Field(default_factory=now_ms)


class Position(BaseModel):
    symbol: str
    market: Market
    side: Side  # long / short
    qty: float
    entry: float
    stop: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: int = 1
    unrealized_pnl: float = 0.0
    opened_ms: int = Field(default_factory=now_ms)


class Fill(BaseModel):
    proposal_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    fee: float
    signal_price: Optional[float] = None  # for realized-slippage tracking
    ts_ms: int = Field(default_factory=now_ms)

    @property
    def slippage_bps(self) -> Optional[float]:
        if self.signal_price is None or self.signal_price <= 0:
            return None
        # positive = adverse (paid more on a buy, got less on a sell)
        diff = (self.price - self.signal_price) if self.side == "BUY" else (self.signal_price - self.price)
        return (diff / self.signal_price) * 10_000


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


ExitReason = Literal["stop", "tp", "manual", "anomaly", "funding_flip", "basis_breakout"]


class Trade(BaseModel):
    """A closed round-trip (entry+exit) or currently-open position with an
    intended exit. Persisted; daily P&L sums realized_pnl from closed trades."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    strategy: str = "indicator"  # e.g. "indicator" | "funding_harvest"
    proposal_id: str
    symbol: str
    market: Market
    side: Side  # original direction; "long" or "short"
    qty: float
    leverage: int = 1
    entry_price: float
    entry_ts_ms: int = Field(default_factory=now_ms)
    intended_stop: Optional[float] = None
    intended_tp: Optional[float] = None
    exit_price: Optional[float] = None
    exit_ts_ms: Optional[int] = None
    exit_reason: Optional[str] = None
    fee_total_usd: float = 0.0
    realized_pnl_usd: Optional[float] = None
    slippage_bps_entry: Optional[float] = None
    slippage_bps_exit: Optional[float] = None
    # Funding income credited to this leg (positive when collected, negative
    # when paid). Only the short-perp leg of a delta-neutral pair accrues
    # meaningful funding in positive-funding regimes.
    funding_accrued_usd: float = 0.0
    status: TradeStatus = TradeStatus.OPEN

    def compute_realized_pnl(self, exit_price: float) -> float:
        gross = (exit_price - self.entry_price) * self.qty
        if self.side == "short":
            gross = -gross
        return gross - self.fee_total_usd + self.funding_accrued_usd


class PairLeg(BaseModel):
    """One leg of a delta-neutral pair trade (e.g. funding-rate harvest)."""
    symbol: str
    market: Market           # "spot" or "perps"
    side: OrderSide          # "BUY" or "SELL"
    qty: float
    expected_price: float    # for slippage budget
    leverage: int = 1


class PairProposal(BaseModel):
    """A two-leg proposal — both legs must fill or both abort. Routed through
    the orchestrator's approval flow as a single unit; risk gate evaluates
    NET delta + per-coin exposure once.

    `direction` (+1 long_spot/short_perp, -1 short_spot/long_perp) is carried
    on the proposal itself so the strategy doesn't need a side-dictionary
    that can leak on rejection (R9)."""
    id: str
    strategy: str
    legs: list[PairLeg]
    notional_usd: float
    direction: int = 1
    rationale: str = ""
    expected_yield_bps_per_8h: float = 0.0
    expires_at_ms: int = 0
    status: ProposalStatus = ProposalStatus.PROPOSED
    created_ms: int = Field(default_factory=now_ms)


class StrategyConfig(BaseModel):
    """The artifact the StrategyAgent writes. Hot loop reads this every tick.
    Must be deterministic-only — no LLM call paths."""
    version: int = 1
    allowed_symbols: list[str]
    enabled_sides: list[Side] = ["long", "short"]
    feature_weights: dict[str, float] = {
        "trend": 0.35,
        "momentum": 0.25,
        "volume": 0.20,
        "volatility": 0.10,
        "pattern": 0.10,
    }
    long_score_threshold: float = 0.35
    short_score_threshold: float = -0.35
    min_confidence: float = 0.45
    atr_stop_mult: float = 2.0
    rr_target: float = 1.8
    htf_regime_filter: bool = True
    htf_timeframe: Timeframe = "1h"
    notes: str = ""
    created_ms: int = Field(default_factory=now_ms)


class NewsItem(BaseModel):
    id: str
    source: str
    title: str
    url: str
    published_ms: int
    coins: list[str] = []
    raw: dict = {}


class SentimentScore(BaseModel):
    symbol: str
    score: float  # -1..+1
    catalyst: str  # hack / listing / regulation / partnership / macro / flow / other
    confidence: float
    sources: list[str] = []
    ts_ms: int = Field(default_factory=now_ms)


class Anomaly(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    symbol: str
    kind: str  # ws_gap, price_jump, funding_extreme, liq_cluster, news_spike
    detail: str
    severity: Literal["info", "warn", "critical"] = "warn"
    ts_ms: int = Field(default_factory=now_ms)
