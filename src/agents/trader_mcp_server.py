"""Stdio MCP server exposing the TraderAgent's 16 tools to a claude-code
subprocess. Reads frozen replay state from STATE_PATH (set by the harness
each wake) and writes write-tool decisions to OUTBOX_PATH (which the
harness drains after the claude subprocess exits).

The server is launched by `claude -p --mcp-config <path>` per wake event,
so each invocation is short-lived. State and outbox files are paths
inside a per-run temp directory the harness owns.

Snapshot schema (written by TraderBacktestHarness._write_snapshot):
{
  "cursor_ts_ms": int,
  "symbol": str, "tf": str, "htf": str,
  "settings": {fee_bps, slippage_bps, ...},
  "indicators": {"5m": {...IndicatorSnapshot.model_dump()}, "1h": {...}},
  "klines_recent": [{"t","o","h","l","c","v"}, ...],
  "klines_htf_recent": [...],
  "open_positions": [{"symbol","side","qty","entry","stop","tp","leverage"}, ...],
  "closed_trades_recent": [{"symbol","side","entry","exit","pnl_usd","exit_reason"}, ...],
  "account": {"equity_usd","pnl_today_usd","consecutive_losses","halted","mode":"backtest"},
  "funding_at_cursor": {"rate_bps", "ts_ms"} | null,
  "oi_change": {"oi_oldest","oi_newest","oi_change_pct","period"} | null,
}

Outbox schema (appended by call_tool handlers for write tools):
{
  "decisions": [
    {"action": "trade"|"close", "params": {...}, "ts_ms": int},
    ...
  ]
}
"""
from __future__ import annotations

import ast
import asyncio
import json
import operator
import os
import sys
from typing import Any, Callable


# ---- Snapshot / outbox file I/O ----------------------------------------------

def _state_path() -> str:
    p = os.environ.get("TRADER_STATE_PATH")
    if not p:
        sys.stderr.write("TRADER_STATE_PATH env var not set\n")
        sys.exit(2)
    return p


def _outbox_path() -> str:
    p = os.environ.get("TRADER_OUTBOX_PATH")
    if not p:
        sys.stderr.write("TRADER_OUTBOX_PATH env var not set\n")
        sys.exit(2)
    return p


def _load_state() -> dict[str, Any]:
    with open(_state_path()) as f:
        return json.load(f)


def _append_outbox(item: dict[str, Any]) -> None:
    path = _outbox_path()
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"decisions": []}
    data.setdefault("decisions", []).append(item)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---- Safe calc evaluator (same as live trader_tools.calc) --------------------

_CALC_OPS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
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


# ---- Tool implementations ---------------------------------------------------

def tool_get_indicator_snapshot(symbol: str, tf: str) -> dict[str, Any]:
    s = _load_state()
    snaps = s.get("indicators", {})
    snap = snaps.get(tf)
    if snap is None:
        return {"error": f"no snapshot for tf={tf} (loaded: {list(snaps.keys())})"}
    return snap


def tool_get_recent_klines(symbol: str, tf: str, n: int = 20) -> dict[str, Any]:
    s = _load_state()
    key = "klines_recent" if tf == s.get("tf") else (
        "klines_htf_recent" if tf == s.get("htf") else None
    )
    if key is None:
        return {"error": f"timeframe '{tf}' not loaded for this backtest run"}
    bars = s.get(key, [])
    return {"symbol": symbol, "tf": tf, "bars": bars[-int(n):]}


def tool_get_htf_context(symbol: str) -> dict[str, Any]:
    s = _load_state()
    snaps = s.get("indicators", {})
    return {tf: snaps.get(tf) for tf in (s.get("htf"), "4h") if tf}


def tool_get_funding_basis(symbol: str) -> dict[str, Any]:
    s = _load_state()
    f = s.get("funding_at_cursor")
    if f is None:
        return {"symbol": symbol,
                "notice": "funding history not available in this backtest run"}
    return {"symbol": symbol,
            "funding_current_bps": f.get("rate_bps"),
            "funding_at_ts_ms": f.get("ts_ms"),
            "notice": "basis snapshot not available in backtest mode"}


def tool_get_orderbook_snapshot(symbol: str, market: str = "spot",
                                 limit: int = 20) -> dict[str, Any]:
    return {"symbol": symbol, "market": market,
            "notice": "orderbook snapshot not available in backtest mode "
                      "(no historical depth data); reason about liquidity "
                      "from kline volume + spread proxies if needed"}


def tool_get_news_sentiment(symbols: list[str] | None = None) -> dict[str, Any]:
    return {"sentiments": {}, "summary": "",
            "notice": "news sentiment not available in backtest mode — "
                      "no historical corpus aligned to replay cursor; "
                      "make decisions on price/indicator data alone"}


def tool_get_anomaly_summary(n: int = 5) -> dict[str, Any]:
    return {"recent_anomalies": []}


def tool_get_position_state() -> dict[str, Any]:
    s = _load_state()
    return {"positions": s.get("open_positions", []),
            "account": s.get("account", {})}


def tool_get_recent_fills(n: int = 10) -> dict[str, Any]:
    s = _load_state()
    return {"recent_closed_trades": s.get("closed_trades_recent", [])[-int(n):]}


def tool_get_correlation(symbol: str) -> dict[str, Any]:
    return {"symbol": symbol,
            "notice": "correlation matrix not loaded in this backtest run"}


def tool_get_hodl_benchmark(symbol: str = "BTCUSDT") -> dict[str, Any]:
    return {"notice": "hodl benchmark not loaded in this backtest run"}


def tool_get_recent_liquidations(symbol: str, n: int = 20) -> dict[str, Any]:
    return {"symbol": symbol, "events": [],
            "notice": "liquidation history not available in backtest mode"}


def tool_get_open_interest_change(symbol: str, period: str = "5m") -> dict[str, Any]:
    s = _load_state()
    oi = s.get("oi_change")
    if oi is None:
        return {"symbol": symbol,
                "notice": "OI history not available in backtest mode"}
    return {"symbol": symbol, **oi}


def tool_calc(expression: str) -> dict[str, Any]:
    try:
        tree = ast.parse(expression, mode="eval")
        return {"result": _safe_eval(tree)}
    except Exception as e:
        return {"error": f"calc: {e}"}


def tool_propose_trade(**kwargs) -> dict[str, Any]:
    # Validate the bare minimum; harness re-validates against risk_gate.
    required = ("symbol", "side", "entry", "stop", "take_profit", "rationale")
    missing = [k for k in required if k not in kwargs]
    if missing:
        return {"accepted": False, "reason": f"missing args: {missing}"}
    s = _load_state()
    _append_outbox({"action": "trade", "params": kwargs,
                     "ts_ms": s.get("cursor_ts_ms")})
    # We don't know if the risk gate will accept until the harness runs it.
    # Return optimistic "queued" so the agent can move on or call again.
    return {"accepted": "pending", "status": "QUEUED_FOR_HARNESS",
            "note": "Harness re-validates against risk_gate after you exit. "
                    "If rejected, you'll see it in your next wake's account state."}


def tool_propose_close(position_symbol: str, rationale: str) -> dict[str, Any]:
    s = _load_state()
    _append_outbox({
        "action": "close",
        "params": {"symbol": position_symbol, "rationale": rationale},
        "ts_ms": s.get("cursor_ts_ms"),
    })
    return {"accepted": "pending", "status": "QUEUED_FOR_HARNESS"}


# ---- Tool registry (name -> handler + schema) -------------------------------

_TOOLS: dict[str, tuple[Callable[..., dict], dict]] = {
    "get_indicator_snapshot": (
        tool_get_indicator_snapshot,
        {"type": "object", "required": ["symbol", "tf"], "properties": {
            "symbol": {"type": "string"}, "tf": {"type": "string"}}},
    ),
    "get_recent_klines": (
        tool_get_recent_klines,
        {"type": "object", "required": ["symbol", "tf"], "properties": {
            "symbol": {"type": "string"}, "tf": {"type": "string"},
            "n": {"type": "integer", "minimum": 1, "maximum": 200}}},
    ),
    "get_htf_context": (
        tool_get_htf_context,
        {"type": "object", "required": ["symbol"], "properties": {
            "symbol": {"type": "string"}}},
    ),
    "get_funding_basis": (
        tool_get_funding_basis,
        {"type": "object", "required": ["symbol"], "properties": {
            "symbol": {"type": "string"}}},
    ),
    "get_orderbook_snapshot": (
        tool_get_orderbook_snapshot,
        {"type": "object", "required": ["symbol"], "properties": {
            "symbol": {"type": "string"},
            "market": {"type": "string"}, "limit": {"type": "integer"}}},
    ),
    "get_news_sentiment": (
        tool_get_news_sentiment,
        {"type": "object", "properties": {
            "symbols": {"type": "array", "items": {"type": "string"}}}},
    ),
    "get_anomaly_summary": (
        tool_get_anomaly_summary,
        {"type": "object", "properties": {"n": {"type": "integer"}}},
    ),
    "get_position_state": (
        tool_get_position_state, {"type": "object", "properties": {}},
    ),
    "get_recent_fills": (
        tool_get_recent_fills,
        {"type": "object", "properties": {"n": {"type": "integer"}}},
    ),
    "get_correlation": (
        tool_get_correlation,
        {"type": "object", "required": ["symbol"], "properties": {
            "symbol": {"type": "string"}}},
    ),
    "get_hodl_benchmark": (
        tool_get_hodl_benchmark,
        {"type": "object", "properties": {"symbol": {"type": "string"}}},
    ),
    "get_recent_liquidations": (
        tool_get_recent_liquidations,
        {"type": "object", "required": ["symbol"], "properties": {
            "symbol": {"type": "string"}, "n": {"type": "integer"}}},
    ),
    "get_open_interest_change": (
        tool_get_open_interest_change,
        {"type": "object", "required": ["symbol"], "properties": {
            "symbol": {"type": "string"}, "period": {"type": "string"}}},
    ),
    "calc": (
        tool_calc,
        {"type": "object", "required": ["expression"], "properties": {
            "expression": {"type": "string"}}},
    ),
    "propose_trade": (
        tool_propose_trade,
        {"type": "object", "required": [
            "symbol", "side", "entry", "stop", "take_profit", "rationale",
        ], "properties": {
            "symbol": {"type": "string"},
            "side": {"type": "string", "enum": ["long", "short"]},
            "entry": {"type": "number"}, "stop": {"type": "number"},
            "take_profit": {"type": "number"},
            "rationale": {"type": "string"},
            "market": {"type": "string", "enum": ["spot", "perps"]},
            "leverage": {"type": "integer"}}},
    ),
    "propose_close": (
        tool_propose_close,
        {"type": "object", "required": ["position_symbol", "rationale"],
         "properties": {"position_symbol": {"type": "string"},
                         "rationale": {"type": "string"}}},
    ),
}


_TOOL_DESCRIPTIONS = {
    "get_indicator_snapshot": "Latest IndicatorSnapshot for (symbol, tf): RSI, StochRSI, BB, ATR, ADX, MACD, OBV, Donchian, etc.",
    "get_recent_klines": "Last N OHLCV bars (default N=20, max 200). BACKTEST: only the run's trigger TF + HTF are loaded.",
    "get_htf_context": "Higher-timeframe indicator snapshots (1h, 4h).",
    "get_funding_basis": "Funding rate (current + 21-period avg, annualized) and perp-spot basis. BACKTEST: funding only, no basis.",
    "get_orderbook_snapshot": "Top-of-book: best bid/ask, spread, top-N depth. BACKTEST: returns notice.",
    "get_news_sentiment": "Sentiment scores from the NewsAgent subagent. BACKTEST: returns notice.",
    "get_anomaly_summary": "Last N anomaly events. BACKTEST: returns empty.",
    "get_position_state": "Open positions and live account state.",
    "get_recent_fills": "Last N closed trades from the last 24h.",
    "get_correlation": "BTC-beta for a symbol. BACKTEST: not loaded.",
    "get_hodl_benchmark": "Strategy equity vs HODL benchmark. BACKTEST: not loaded.",
    "get_recent_liquidations": "Recent futures liquidations. BACKTEST: not available.",
    "get_open_interest_change": "Current OI plus % change over a recent window.",
    "calc": "Safe arithmetic evaluator. DO NOT do arithmetic in-prompt — use this.",
    "propose_trade": "Propose a new trade. BACKTEST: queued; harness re-validates against risk_gate after you exit.",
    "propose_close": "Propose closing an existing position. BACKTEST: queued; executes at next bar.",
}


# ---- MCP stdio server entrypoint --------------------------------------------

async def _main_mcp() -> None:
    """Real MCP server using the `mcp` Python SDK. Imports lazily so this
    file remains importable for unit tests even without the SDK."""
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    import mcp.types as mt

    server: Server = Server("trader-backtest")

    @server.list_tools()
    async def _list_tools() -> list[mt.Tool]:
        return [
            mt.Tool(
                name=name,
                description=_TOOL_DESCRIPTIONS.get(name, ""),
                inputSchema=schema,
            )
            for name, (_, schema) in _TOOLS.items()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None
                          ) -> list[mt.TextContent]:
        entry = _TOOLS.get(name)
        if entry is None:
            return [mt.TextContent(type="text",
                                    text=json.dumps({"error": f"unknown tool: {name}"}))]
        handler, _schema = entry
        try:
            result = handler(**(arguments or {}))
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        return [mt.TextContent(
            type="text", text=json.dumps(result, default=str)[:32_000],
        )]

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            InitializationOptions(
                server_name="trader-backtest",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    asyncio.run(_main_mcp())


if __name__ == "__main__":
    main()
