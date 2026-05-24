"""Tool surface for the TraderAgent (the "desk trader" loop).

A `TraderToolContext` bundles refs to all data sources the agent reads
(IndicatorEngine, BinanceClient, FundingMonitor, ...) plus callbacks the
orchestrator injects for stateful reads (open positions, account state) and
for the two write tools (propose_trade, propose_close). `build_trader_tools(ctx)`
returns the list of `ToolSpec` that the LLMAgent presents to Claude.

Each handler returns JSON-serializable output capped at ~10KB by the
`LLMAgent` tool-result truncator — keep handler outputs compact.

Tools that need infra not yet present in the repo:
  - get_recent_liquidations: requires a !forceOrder@arr ws subscription
    + rolling store; stubbed for v1 (returns notice).
"""
from __future__ import annotations

import ast
import operator
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import httpx
import structlog

from src.agents.llm_client import ToolSpec
from src.config.settings import Settings, get_settings
from src.models.types import Position
from src.services.basis_monitor import BasisMonitor
from src.services.correlation import CorrelationMatrix
from src.services.funding_monitor import FundingMonitor
from src.services.hodl_benchmark import HodlBenchmark
from src.services.storage import Storage
from src.tools.binance_client import BinanceClient
from src.tools.indicators import IndicatorEngine

log = structlog.get_logger(__name__)


# ---- Safe arithmetic evaluator for the `calc` tool ---------------------------

_CALC_OPS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


# ---- Binance public-endpoint helpers (orderbook, OI) -------------------------
# BinanceClient is signed/private-focused; these are anonymous reads. Kept
# local rather than bloating BinanceClient with desk-trader-specific endpoints.

def _base_url(market: str, settings: Settings) -> str:
    if market == "perps":
        return "https://testnet.binancefuture.com" if settings.binance_testnet \
            else "https://fapi.binance.com"
    return "https://testnet.binance.vision" if settings.binance_testnet \
        else "https://api.binance.com"


async def _fetch_depth(symbol: str, market: str, limit: int,
                       settings: Settings) -> dict[str, Any]:
    path = "/fapi/v1/depth" if market == "perps" else "/api/v3/depth"
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(_base_url(market, settings) + path,
                              params={"symbol": symbol, "limit": limit})
        r.raise_for_status()
        return r.json()


async def _fetch_open_interest(symbol: str, settings: Settings) -> Optional[float]:
    """Current OI in base-asset units (perps only)."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(_base_url("perps", settings) + "/fapi/v1/openInterest",
                              params={"symbol": symbol})
        r.raise_for_status()
        return float(r.json().get("openInterest", 0.0))


async def _fetch_open_interest_hist(symbol: str, period: str, settings: Settings,
                                     limit: int = 12) -> list[dict[str, Any]]:
    """OI history. Only available on mainnet, not testnet."""
    if settings.binance_testnet:
        return []
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get("https://fapi.binance.com/futures/data/openInterestHist",
                              params={"symbol": symbol, "period": period, "limit": limit})
        r.raise_for_status()
        return r.json()


# ---- Context ----------------------------------------------------------------

@dataclass
class TraderToolContext:
    """Refs + callbacks the trader-tools need. Using callbacks for
    orchestrator-owned state (open positions, account state, news subagent
    invocation, write-tool plumbing) avoids a circular import on
    `Orchestrator`."""
    settings: Settings
    binance: BinanceClient
    indicator_engine: IndicatorEngine
    storage: Storage
    funding_monitor: Optional[FundingMonitor] = None
    basis_monitor: Optional[BasisMonitor] = None
    correlation: Optional[CorrelationMatrix] = None
    hodl: Optional[HodlBenchmark] = None
    # Orchestrator-state callbacks
    get_open_positions: Callable[[], list[Position]] = lambda: []
    get_account_state: Callable[[], dict[str, Any]] = lambda: {}
    get_recent_anomalies: Callable[[int], list[dict[str, Any]]] = lambda n: []
    get_last_prices: Callable[[], dict[str, float]] = lambda: {}
    # Subagent: takes a list of symbols, returns sentiment dict
    news_sentiment_subagent: Optional[Callable[[list[str]], Awaitable[dict[str, Any]]]] = None
    # Write tools — orchestrator wires these in
    propose_trade_callback: Optional[Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = None
    propose_close_callback: Optional[Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = None


# ---- Handler implementations ------------------------------------------------

def _make_handlers(ctx: TraderToolContext):
    """Build closures over ctx. Returns the dict of name -> async handler."""
    s = ctx.settings

    async def get_indicator_snapshot(symbol: str, tf: str) -> dict[str, Any]:
        snap = ctx.indicator_engine.latest(symbol, tf)
        if snap is None:
            return {"error": f"no snapshot for {symbol}/{tf}"}
        # IndicatorSnapshot is a pydantic BaseModel
        return snap.model_dump(exclude_none=False)

    async def get_recent_klines(symbol: str, tf: str, n: int = 20) -> dict[str, Any]:
        n = min(max(int(n), 1), 200)
        try:
            raw = await ctx.binance.fetch_klines(symbol, tf, limit=n, market="spot")
        except Exception as e:
            return {"error": f"fetch_klines: {e}"}
        bars = [{
            "t": int(r[0]), "o": float(r[1]), "h": float(r[2]),
            "l": float(r[3]), "c": float(r[4]), "v": float(r[5]),
        } for r in raw]
        return {"symbol": symbol, "tf": tf, "bars": bars}

    async def get_htf_context(symbol: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for tf in ("1h", "4h"):
            snap = ctx.indicator_engine.latest(symbol, tf)
            out[tf] = snap.model_dump(exclude_none=True) if snap else None
        return out

    async def get_funding_basis(symbol: str) -> dict[str, Any]:
        out: dict[str, Any] = {"symbol": symbol}
        if ctx.funding_monitor is not None:
            out["funding_current_bps"] = ctx.funding_monitor.current_bps(symbol)
            out["funding_avg_21_bps"] = ctx.funding_monitor.avg_bps(symbol, 21)
            out["funding_annualized_pct"] = ctx.funding_monitor.annualized_pct(symbol)
        else:
            try:
                fr = await ctx.binance.funding_rate(symbol)
                out["funding_current_bps"] = float(fr) * 10_000 if fr else None
            except Exception as e:
                out["funding_error"] = str(e)
        if ctx.basis_monitor is not None:
            try:
                bs = ctx.basis_monitor.sample(symbol, spot_tf="1m")
                out["basis_bps"] = getattr(bs, "basis_bps", None)
                out["safe_to_open"] = ctx.basis_monitor.safe_to_open(symbol, "1m")
            except Exception as e:
                out["basis_error"] = str(e)
        return out

    async def get_orderbook_snapshot(symbol: str, market: str = "spot",
                                      limit: int = 20) -> dict[str, Any]:
        try:
            depth = await _fetch_depth(symbol, market, min(max(limit, 5), 100), s)
        except Exception as e:
            return {"error": f"depth: {e}"}
        bids = [(float(p), float(q)) for p, q in depth.get("bids", [])[:limit]]
        asks = [(float(p), float(q)) for p, q in depth.get("asks", [])[:limit]]
        if not bids or not asks:
            return {"error": "empty book"}
        bid_top, ask_top = bids[0][0], asks[0][0]
        mid = (bid_top + ask_top) / 2
        spread_bps = (ask_top - bid_top) / mid * 10_000 if mid > 0 else None
        bid_vol = sum(q for _, q in bids)
        ask_vol = sum(q for _, q in asks)
        depth_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if bid_vol + ask_vol > 0 else 0.0
        return {
            "symbol": symbol, "market": market,
            "best_bid": bid_top, "best_ask": ask_top, "mid": mid,
            "spread_bps": spread_bps,
            "bid_vol_top_n": bid_vol, "ask_vol_top_n": ask_vol,
            "depth_imbalance": depth_imbalance,  # +1 bid-heavy, -1 ask-heavy
            "top_levels": {"bids": bids[:5], "asks": asks[:5]},
        }

    async def get_news_sentiment(symbols: Optional[list[str]] = None) -> dict[str, Any]:
        if ctx.news_sentiment_subagent is None:
            return {"error": "news subagent not configured"}
        symbols = symbols or s.symbol_list
        try:
            out = await ctx.news_sentiment_subagent(symbols)
            return out
        except Exception as e:
            return {"error": f"news subagent: {e}"}

    async def get_anomaly_summary(n: int = 5) -> dict[str, Any]:
        items = ctx.get_recent_anomalies(min(max(int(n), 1), 20))
        return {"recent_anomalies": items}

    async def get_position_state() -> dict[str, Any]:
        positions = [p.model_dump() for p in ctx.get_open_positions()]
        acct = ctx.get_account_state()
        return {"positions": positions, "account": acct}

    async def get_recent_fills(n: int = 10) -> dict[str, Any]:
        try:
            since_ms = int(time.time() * 1000) - 24 * 3600 * 1000
            trades = await ctx.storage.recent_closed_trades(since_ms)
        except Exception as e:
            return {"error": str(e)}
        trades = trades[: min(max(int(n), 1), 50)]
        return {"recent_closed_trades": [
            {k: getattr(t, k, None) for k in (
                "symbol", "strategy", "side", "qty", "entry_price",
                "exit_price", "pnl_usd", "exit_reason", "opened_ms", "closed_ms",
            )} for t in trades
        ]}

    async def get_correlation(symbol: str) -> dict[str, Any]:
        if ctx.correlation is None:
            return {"error": "correlation matrix not initialized"}
        beta = ctx.correlation.beta_to_btc(symbol)
        return {
            "symbol": symbol, "beta_to_btc": beta,
            "stale": ctx.correlation.is_stale(max_age=3600),
        }

    async def get_hodl_benchmark(symbol: str = "BTCUSDT") -> dict[str, Any]:
        if ctx.hodl is None:
            return {"error": "hodl benchmark not initialized"}
        btc_now = ctx.get_last_prices().get("BTCUSDT")
        acct = ctx.get_account_state()
        cur_equity = acct.get("equity_usd", s.account_equity_usd)
        if btc_now is None:
            return {"error": "no current BTC price"}
        return {
            "hodl_equity": ctx.hodl.hodl_equity(btc_now),
            "current_equity": cur_equity,
            "outperformance_usd": ctx.hodl.outperformance_usd(cur_equity, btc_now),
            "outperformance_pct": ctx.hodl.outperformance_pct(cur_equity, btc_now),
        }

    async def get_recent_liquidations(symbol: str) -> dict[str, Any]:
        # TODO: subscribe to !forceOrder@arr ws stream and keep a rolling
        # buffer per symbol. Until that lands, this returns a clear notice
        # so the agent knows liquidation data is unavailable.
        return {
            "symbol": symbol,
            "notice": "liquidation feed not yet wired (requires !forceOrder@arr ws subscription); "
                      "do not rely on liquidation flow for this decision",
        }

    async def get_open_interest_change(symbol: str, period: str = "5m") -> dict[str, Any]:
        try:
            current = await _fetch_open_interest(symbol, s)
        except Exception as e:
            return {"error": f"oi current: {e}"}
        out: dict[str, Any] = {"symbol": symbol, "oi_current": current}
        try:
            hist = await _fetch_open_interest_hist(symbol, period, s, limit=12)
        except Exception as e:
            out["hist_error"] = str(e)
            return out
        if not hist:
            out["notice"] = "OI history unavailable on testnet"
            return out
        oldest = float(hist[0].get("sumOpenInterest", 0.0))
        newest = float(hist[-1].get("sumOpenInterest", 0.0))
        out["oi_oldest"] = oldest
        out["oi_newest"] = newest
        out["oi_change_pct"] = ((newest - oldest) / oldest * 100.0) if oldest > 0 else None
        out["period"] = period
        out["samples"] = len(hist)
        return out

    async def calc(expression: str) -> dict[str, Any]:
        try:
            tree = ast.parse(expression, mode="eval")
            return {"result": _safe_eval(tree)}
        except Exception as e:
            return {"error": f"calc: {e}"}

    async def propose_trade(symbol: str, side: str, entry: float, stop: float,
                             take_profit: float, rationale: str,
                             market: str = "spot", leverage: int = 1) -> dict[str, Any]:
        if ctx.propose_trade_callback is None:
            return {"error": "propose_trade not wired"}
        return await ctx.propose_trade_callback({
            "symbol": symbol, "side": side,
            "entry": float(entry), "stop": float(stop),
            "take_profit": float(take_profit),
            "rationale": rationale, "market": market,
            "leverage": int(leverage),
        })

    async def propose_close(position_symbol: str, rationale: str) -> dict[str, Any]:
        if ctx.propose_close_callback is None:
            return {"error": "propose_close not wired"}
        return await ctx.propose_close_callback({
            "symbol": position_symbol, "rationale": rationale,
        })

    return {
        "get_indicator_snapshot": get_indicator_snapshot,
        "get_recent_klines": get_recent_klines,
        "get_htf_context": get_htf_context,
        "get_funding_basis": get_funding_basis,
        "get_orderbook_snapshot": get_orderbook_snapshot,
        "get_news_sentiment": get_news_sentiment,
        "get_anomaly_summary": get_anomaly_summary,
        "get_position_state": get_position_state,
        "get_recent_fills": get_recent_fills,
        "get_correlation": get_correlation,
        "get_hodl_benchmark": get_hodl_benchmark,
        "get_recent_liquidations": get_recent_liquidations,
        "get_open_interest_change": get_open_interest_change,
        "calc": calc,
        "propose_trade": propose_trade,
        "propose_close": propose_close,
    }


# ---- ToolSpec list ----------------------------------------------------------

def build_trader_tools(ctx: TraderToolContext) -> list[ToolSpec]:
    """Return the full TraderAgent tool surface — 14 read tools + 2 write tools."""
    h = _make_handlers(ctx)
    specs: list[ToolSpec] = [
        ToolSpec(
            name="get_indicator_snapshot",
            description="Latest IndicatorSnapshot for (symbol, tf): RSI, StochRSI, BB, ATR, ADX, MACD, OBV, Donchian, etc.",
            input_schema={"type": "object", "required": ["symbol", "tf"], "properties": {
                "symbol": {"type": "string"},
                "tf": {"type": "string", "description": "timeframe e.g. 1m, 5m, 15m, 1h"},
            }},
            handler=h["get_indicator_snapshot"],
        ),
        ToolSpec(
            name="get_recent_klines",
            description="Last N OHLCV bars for a symbol/timeframe (default N=20, max 200).",
            input_schema={"type": "object", "required": ["symbol", "tf"], "properties": {
                "symbol": {"type": "string"}, "tf": {"type": "string"},
                "n": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
            }},
            handler=h["get_recent_klines"],
        ),
        ToolSpec(
            name="get_htf_context",
            description="Higher-timeframe (1h, 4h) indicator snapshots for HTF regime context.",
            input_schema={"type": "object", "required": ["symbol"], "properties": {
                "symbol": {"type": "string"},
            }},
            handler=h["get_htf_context"],
        ),
        ToolSpec(
            name="get_funding_basis",
            description="Funding rate (current + 21-period avg, annualized) and perp-spot basis.",
            input_schema={"type": "object", "required": ["symbol"], "properties": {
                "symbol": {"type": "string"},
            }},
            handler=h["get_funding_basis"],
        ),
        ToolSpec(
            name="get_orderbook_snapshot",
            description="Top-of-book: best bid/ask, spread (bps), top-N depth, depth imbalance.",
            input_schema={"type": "object", "required": ["symbol"], "properties": {
                "symbol": {"type": "string"},
                "market": {"type": "string", "enum": ["spot", "perps"], "default": "spot"},
                "limit": {"type": "integer", "minimum": 5, "maximum": 100, "default": 20},
            }},
            handler=h["get_orderbook_snapshot"],
        ),
        ToolSpec(
            name="get_news_sentiment",
            description="Sentiment scores + summary from the NewsAgent subagent (Haiku). Returns per-symbol scores and a brief market narrative.",
            input_schema={"type": "object", "properties": {
                "symbols": {"type": "array", "items": {"type": "string"}, "description": "symbols to score; defaults to the active universe"},
            }},
            handler=h["get_news_sentiment"],
        ),
        ToolSpec(
            name="get_anomaly_summary",
            description="Last N anomaly events detected by the deterministic anomaly detectors.",
            input_schema={"type": "object", "properties": {
                "n": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            }},
            handler=h["get_anomaly_summary"],
        ),
        ToolSpec(
            name="get_position_state",
            description="Open positions and live account state (equity, P&L today, consecutive losses, halted flag).",
            input_schema={"type": "object", "properties": {}},
            handler=h["get_position_state"],
        ),
        ToolSpec(
            name="get_recent_fills",
            description="Last N closed trades from the last 24h (default N=10): P&L, exit reason, strategy.",
            input_schema={"type": "object", "properties": {
                "n": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            }},
            handler=h["get_recent_fills"],
        ),
        ToolSpec(
            name="get_correlation",
            description="BTC-beta for a symbol from the live correlation matrix.",
            input_schema={"type": "object", "required": ["symbol"], "properties": {
                "symbol": {"type": "string"},
            }},
            handler=h["get_correlation"],
        ),
        ToolSpec(
            name="get_hodl_benchmark",
            description="Strategy equity vs simple HODL benchmark (default BTC).",
            input_schema={"type": "object", "properties": {
                "symbol": {"type": "string", "default": "BTCUSDT"},
            }},
            handler=h["get_hodl_benchmark"],
        ),
        ToolSpec(
            name="get_recent_liquidations",
            description="Recent liquidations on a symbol. NOTE: stubbed in v1; returns a notice so you know not to rely on it.",
            input_schema={"type": "object", "required": ["symbol"], "properties": {
                "symbol": {"type": "string"},
            }},
            handler=h["get_recent_liquidations"],
        ),
        ToolSpec(
            name="get_open_interest_change",
            description="Current OI plus % change over a recent window (period='5m','15m','30m','1h','2h','4h','6h','12h','1d').",
            input_schema={"type": "object", "required": ["symbol"], "properties": {
                "symbol": {"type": "string"},
                "period": {"type": "string", "default": "5m"},
            }},
            handler=h["get_open_interest_change"],
        ),
        ToolSpec(
            name="calc",
            description="Safe arithmetic evaluator. Supports +, -, *, /, //, %, **, unary +/-. Use this for R:R math, position sizing, P&L estimation — do NOT do arithmetic in-prompt.",
            input_schema={"type": "object", "required": ["expression"], "properties": {
                "expression": {"type": "string", "description": "e.g. '(0.045 - 0.043) / (0.043 - 0.0415)' for R:R"},
            }},
            handler=h["calc"],
        ),
        ToolSpec(
            name="propose_trade",
            description=(
                "Propose a new trade. Backtest mode: executes paper-mode immediately if the "
                "risk gate accepts. Live mode: every proposal is sent to the human on Telegram "
                "for approval regardless of size. Returns {accepted, proposal_id, reason}."
            ),
            input_schema={"type": "object", "required": [
                "symbol", "side", "entry", "stop", "take_profit", "rationale",
            ], "properties": {
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["long", "short"]},
                "entry": {"type": "number"},
                "stop": {"type": "number"},
                "take_profit": {"type": "number"},
                "rationale": {"type": "string", "description": "1-3 sentences. Why this trade now."},
                "market": {"type": "string", "enum": ["spot", "perps"], "default": "spot"},
                "leverage": {"type": "integer", "minimum": 1, "maximum": 10, "default": 1},
            }},
            handler=h["propose_trade"],
        ),
        ToolSpec(
            name="propose_close",
            description=(
                "Propose closing an existing open position by symbol. Backtest: closes "
                "immediately at market. Live: Telegram approval required."
            ),
            input_schema={"type": "object", "required": ["position_symbol", "rationale"], "properties": {
                "position_symbol": {"type": "string"},
                "rationale": {"type": "string"},
            }},
            handler=h["propose_close"],
        ),
    ]
    return specs
