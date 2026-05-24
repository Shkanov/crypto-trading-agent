"""Backtest harness for the TraderAgent.

Replays historical klines through the IndicatorEngine and invokes a REAL
spawned TraderAgent on each wake event. Tool callbacks read only data
<= the replay cursor — no peeking forward. Tools that lack historical
fidelity (orderbook depth, news, liquidations) return an explicit
"not available in backtest" notice so the agent learns to ignore them
rather than rely on fabricated data.

Cost: every wake event is a real Opus 4.7 call (~$0.03/cycle with ~5
tool iterations). Use --max-wakes to bound spend. The TokenBudget on
the agent itself also caps daily USD.

Authority in backtest: propose_trade executes paper-mode directly (no
Telegram). Risk gate still validates. propose_close closes the matching
SimTrade at the current bar close, applying tp_slippage_bps.

Outputs (per run):
    BacktestStats: trades / win-rate / total P&L / Sharpe / max DD
    The full SimTrade ledger (for export to CSV / further analysis)
    Wake-event counts by kind
    Total tool-call count + USD spent
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from src.agents.llm_client import LLMAgent
from src.agents.trader_agent import (
    attach_tools as attach_trader_tools,
    build_trader_agent,
    run_trader_cycle,
)
from src.agents.trader_tools import TraderToolContext
from src.config.settings import Settings, get_settings
from src.models.types import FeatureVector, Kline, Position, Signal
from src.services.backtest import (
    BacktestStats,
    SimTrade,
    _stats_from_trades,
)
from src.services.trader_triggers import WakeTriggers
from src.tools.binance_client import BinanceClient
from src.tools.indicators import IndicatorEngine
from src.tools.risk_gate import AccountState, RiskGate

log = structlog.get_logger(__name__)


def _kline_from_raw(symbol: str, tf: str, r: list) -> Kline:
    return Kline(
        symbol=symbol, timeframe=tf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]),
        close=float(r[4]), volume=float(r[5]), quote_volume=float(r[7]),
        trades=int(r[8]), taker_buy_volume=float(r[9]),
        is_closed=True,
    )


@dataclass
class _SimPosition:
    """Live shape of an open SimTrade — mirrors orchestrator's Position
    so the agent's get_position_state sees the same schema."""
    sim_trade: SimTrade
    bars_held: int = 0

    def to_position(self) -> Position:
        return Position(
            symbol=self.sim_trade.symbol,
            market="spot",
            side=self.sim_trade.side,
            qty=self.sim_trade.qty,
            entry=self.sim_trade.entry_price,
            stop=self.sim_trade.stop,
            take_profit=self.sim_trade.tp,
            leverage=1,
        )


@dataclass
class TraderBacktestResult:
    stats: BacktestStats
    closed_trades: list[SimTrade]
    wake_counts: dict[str, int]
    total_tool_calls: int
    total_usd_spent: float
    wakes_invoked: int


class TraderBacktestHarness:
    """Owns the replay state. One instance per backtest run."""

    def __init__(
        self,
        symbol: str,
        tf: str = "5m",
        htf: str = "1h",
        bars: int = 200,
        max_wakes: int = 10,
        settings: Optional[Settings] = None,
    ) -> None:
        self.s = settings or get_settings()
        self.symbol = symbol
        self.tf = tf
        self.htf = htf
        self.bars = bars
        self.max_wakes = max_wakes

        self.binance = BinanceClient()
        self.indicators = IndicatorEngine()
        self.risk = RiskGate(self.s)
        self.wake_triggers = WakeTriggers(self.s)

        # Replay state
        self._klines: list[Kline] = []
        self._htf_klines: list[Kline] = []
        self._funding_hist: list[tuple[int, float]] = []  # (ts_ms, rate)
        self._oi_hist: list[tuple[int, float]] = []       # (ts_ms, sumOI)
        self.cursor_ts_ms: int = 0
        self.cursor_idx: int = 0
        self.last_price: dict[str, float] = {}

        # Account / trades
        self.equity: float = self.s.account_equity_usd
        self.starting_equity: float = self.s.account_equity_usd
        self.pnl_today: float = 0.0
        self.consecutive_losses: int = 0
        self.last_trade_ms_by_symbol: dict[str, int] = {}
        self.open_positions: list[_SimPosition] = []
        self.closed_trades: list[SimTrade] = []
        self.pending: dict = {}  # required by risk_gate path's `pid in pending` check

        # Diagnostics
        self.wake_counts: dict[str, int] = {}
        self.wakes_invoked: int = 0
        self.total_tool_calls: int = 0

        # Agent (set in prepare)
        self.agent: Optional[LLMAgent] = None

    # ------------------------------------------------------------------ data

    async def prepare(self) -> None:
        """Fetch all historical data once. Builds the agent + tools. Must
        be called inside an active asyncio loop with binance started."""
        # Fail fast before any I/O if there's no API key — avoids burning
        # Binance fetches just to error out at the agent-build step.
        if not self.s.anthropic_api_key:
            raise RuntimeError(
                "anthropic_api_key not set — backtest needs a real key to spawn the agent"
            )
        await self.binance.start()
        log.info("backtest.prepare.fetching",
                  symbol=self.symbol, tf=self.tf, htf=self.htf, bars=self.bars)
        raw = await self.binance.fetch_klines_paginated(
            self.symbol, self.tf, total=self.bars, market="spot",
        )
        self._klines = [_kline_from_raw(self.symbol, self.tf, r) for r in raw]
        htf_total = max(200, self.bars // 12)
        htf_raw = await self.binance.fetch_klines_paginated(
            self.symbol, self.htf, total=htf_total, market="spot",
        )
        self._htf_klines = [_kline_from_raw(self.symbol, self.htf, r) for r in htf_raw]

        # Funding history (perps only — may fail on spot-only assets; skip on error).
        # Paged the same way as src/services/backtest.py (Binance caps each
        # page at 1000 rows; endpoint silently caps at 200 if no startTime).
        if self._klines:
            try:
                assert self.binance.client is not None
                start_ms = self._klines[0].open_time
                end_ms = self._klines[-1].close_time
                cursor = start_ms
                rows: list[dict] = []
                while cursor < end_ms:
                    page = await self.binance.client.futures_funding_rate(
                        symbol=self.symbol, startTime=cursor,
                        endTime=end_ms, limit=1000,
                    )
                    if not page:
                        break
                    rows.extend(page)
                    last_t = int(page[-1]["fundingTime"])
                    if len(page) < 1000 or last_t <= cursor:
                        break
                    cursor = last_t + 1
                self._funding_hist = sorted(
                    {(int(r["fundingTime"]), float(r["fundingRate"])) for r in rows},
                    key=lambda x: x[0],
                )
            except Exception as e:
                log.info("backtest.funding_fetch_skip", err=str(e))

        # OI history (mainnet-only — skip on testnet)
        if not self.s.binance_testnet:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.get(
                        "https://fapi.binance.com/futures/data/openInterestHist",
                        params={"symbol": self.symbol, "period": "5m", "limit": 500},
                    )
                    r.raise_for_status()
                    for row in r.json():
                        self._oi_hist.append(
                            (int(row["timestamp"]), float(row["sumOpenInterest"]))
                        )
            except Exception as e:
                log.info("backtest.oi_fetch_skip", err=str(e))

        # Warmup indicators on the first quarter of klines (no trades happen yet)
        warmup_n = min(200, len(self._klines) // 4)
        self.indicators.warmup(self.symbol, self.tf, self._klines[:warmup_n])
        self.indicators.warmup(self.symbol, self.htf, self._htf_klines)
        self.cursor_idx = warmup_n

        # Build agent + tools
        self.agent = build_trader_agent()
        attach_trader_tools(self.agent, self._build_context())
        # Override tools that would otherwise call live Binance endpoints
        # and leak future state into the replay (orderbook, current funding,
        # current OI). Replace with backtest-aware variants that either serve
        # historical data or return an explicit "unavailable" notice.
        self._install_backtest_overrides()
        log.info("backtest.prepare.done",
                  klines=len(self._klines), htf_klines=len(self._htf_klines),
                  warmup=warmup_n, funding_pts=len(self._funding_hist),
                  oi_pts=len(self._oi_hist))

    async def close(self) -> None:
        await self.binance.close()

    # --------------------------------------------------------- tool callbacks

    def _build_context(self) -> TraderToolContext:
        return TraderToolContext(
            settings=self.s,
            binance=self.binance,
            indicator_engine=self.indicators,
            storage=None,  # type: ignore[arg-type]  # not used in backtest paths
            funding_monitor=None, basis_monitor=None,
            correlation=None, hodl=None,
            get_open_positions=lambda: [p.to_position() for p in self.open_positions],
            get_account_state=self._account_state_dict,
            get_recent_anomalies=lambda n: [],
            get_last_prices=lambda: dict(self.last_price),
            get_recent_liquidations=lambda sym, n: [],
            news_sentiment_subagent=self._stub_news,
            propose_trade_callback=self._propose_trade,
            propose_close_callback=self._propose_close,
        )

    def _account_state_dict(self) -> dict[str, Any]:
        return {
            "equity_usd": self.equity,
            "pnl_today_usd": self.pnl_today,
            "consecutive_losses": self.consecutive_losses,
            "halted": False,
            "open_position_count": len(self.open_positions),
            "mode": "backtest",
        }

    async def _stub_news(self, symbols: list[str]) -> dict[str, Any]:
        return {
            "sentiments": {},
            "summary": "",
            "notice": "news sentiment not available in backtest mode — "
                      "no historical corpus aligned to replay cursor; "
                      "make decisions on price/indicator data alone",
        }

    # ---- Backtest-mode tool overrides (replace live Binance fetches) -------

    def _install_backtest_overrides(self) -> None:
        assert self.agent is not None
        for t in self.agent.tools:
            if t.name == "get_orderbook_snapshot":
                t.handler = self._stub_orderbook
            elif t.name == "get_funding_basis":
                t.handler = self._historical_funding_basis
            elif t.name == "get_open_interest_change":
                t.handler = self._historical_oi_change
            elif t.name == "get_recent_klines":
                t.handler = self._historical_klines

    async def _stub_orderbook(self, symbol: str, market: str = "spot",
                               limit: int = 20) -> dict[str, Any]:
        return {
            "symbol": symbol, "market": market,
            "notice": "orderbook snapshot not available in backtest mode "
                      "(no historical depth data); reason about liquidity "
                      "from kline volume + spread proxies if needed",
        }

    async def _historical_funding_basis(self, symbol: str) -> dict[str, Any]:
        """Serve the funding rate at the nearest prior funding-time to the
        cursor. Returns notice when funding history is absent (spot-only
        symbol or testnet)."""
        if not self._funding_hist:
            return {
                "symbol": symbol,
                "notice": "funding history not available in backtest mode "
                          "(no perp data fetched for this symbol/run)",
            }
        # Binary search would be cleaner; linear scan is fine for ~500 rows.
        rate = None
        rate_ts = 0
        for ts, r in self._funding_hist:
            if ts > self.cursor_ts_ms:
                break
            rate = r
            rate_ts = ts
        if rate is None:
            return {"symbol": symbol,
                    "notice": "no funding data prior to replay cursor"}
        return {
            "symbol": symbol,
            "funding_current_bps": rate * 10_000,
            "funding_at_ts_ms": rate_ts,
            "notice": "basis snapshot not available in backtest mode "
                      "(no historical spot vs perp prices); funding only",
        }

    async def _historical_oi_change(self, symbol: str, period: str = "5m"
                                     ) -> dict[str, Any]:
        if not self._oi_hist:
            return {"symbol": symbol,
                    "notice": "OI history not available in backtest mode "
                              "(mainnet-only and not pre-fetched)"}
        # Pre-cursor slice
        prior = [(ts, oi) for ts, oi in self._oi_hist if ts <= self.cursor_ts_ms]
        if len(prior) < 2:
            return {"symbol": symbol,
                    "notice": "insufficient OI history before cursor"}
        oldest = prior[max(0, len(prior) - 12)][1]
        newest = prior[-1][1]
        return {
            "symbol": symbol,
            "oi_newest": newest, "oi_oldest": oldest,
            "oi_change_pct": ((newest - oldest) / oldest * 100.0)
                if oldest > 0 else None,
            "period": period, "samples": min(len(prior), 12),
        }

    async def _historical_klines(self, symbol: str, tf: str, n: int = 20
                                  ) -> dict[str, Any]:
        """Serve historical klines up to (but not past) the cursor. Without
        this override, get_recent_klines hits Binance's live endpoint and
        leaks future bars into the replay."""
        if tf == self.tf:
            klines = [k for k in self._klines if k.close_time <= self.cursor_ts_ms]
        elif tf == self.htf:
            klines = [k for k in self._htf_klines if k.close_time <= self.cursor_ts_ms]
        else:
            return {"error": f"backtest run only loaded tf={self.tf} and htf={self.htf}; "
                              f"timeframe '{tf}' not available"}
        n = min(max(int(n), 1), 200)
        recent = klines[-n:]
        return {"symbol": symbol, "tf": tf, "bars": [
            {"t": k.open_time, "o": k.open, "h": k.high, "l": k.low,
             "c": k.close, "v": k.volume} for k in recent
        ]}

    # ------------------------------------------------------- write tool impls

    async def _propose_trade(self, args: dict) -> dict:
        try:
            symbol = args["symbol"]
            side = args["side"]
            entry = float(args["entry"])
            stop = float(args["stop"])
            tp = float(args["take_profit"])
            market = args.get("market", "spot")
            leverage = int(args.get("leverage", 1))
            rationale = args.get("rationale", "")
        except (KeyError, ValueError, TypeError) as e:
            return {"accepted": False, "reason": f"bad args: {e}"}

        fee = self.s.perps_taker_fee_bps if market == "perps" else self.s.spot_taker_fee_bps
        round_trip_bps = 2 * (fee + self.s.slippage_bps)
        tp_move_bps = abs(tp - entry) / entry * 10_000 if entry else 0.0
        edge_bps = tp_move_bps - round_trip_bps

        signal = Signal(
            symbol=symbol, side=side, confidence=0.5, score=0.0,
            entry=entry, stop=stop, take_profit=tp,
            edge_bps=edge_bps, features=FeatureVector(),
            rationale=rationale[:500],
        )

        acct = AccountState(
            equity_usd=self.equity, pnl_today_usd=self.pnl_today,
            consecutive_losses=self.consecutive_losses,
            open_positions=[p.to_position() for p in self.open_positions],
            last_trade_ms_by_symbol=dict(self.last_trade_ms_by_symbol),
            halted_until_ms=0, btc_betas=None,
        )
        decision = self.risk.check(signal, acct, self.cursor_ts_ms,
                                    market, leverage)
        if not decision.ok:
            return {"accepted": False, "reason": f"risk gate: {decision.reason}"}

        # Apply entry slippage (paper-mode fill assumption: taker market)
        slip = self.s.slippage_bps / 10_000
        fill_px = entry * (1 + slip) if side == "long" else entry * (1 - slip)
        entry_fee_usd = decision.qty * fill_px * (fee / 10_000)

        sim = SimTrade(
            symbol=symbol, strategy="trader_agent_backtest", side=side,
            qty=decision.qty, entry_price=fill_px, stop=stop, tp=tp,
            entry_ts_ms=self.cursor_ts_ms,
            pnl_usd=-entry_fee_usd,  # running P&L starts with entry fee debit
        )
        self.open_positions.append(_SimPosition(sim_trade=sim))
        self.last_trade_ms_by_symbol[symbol] = self.cursor_ts_ms
        return {
            "accepted": True,
            "proposal_id": f"sim-{len(self.open_positions)}",
            "status": "EXECUTED",
            "fill_price": fill_px, "qty": decision.qty,
            "notional_usd": decision.notional_usd,
            "edge_bps": edge_bps, "rr": signal.rr,
        }

    async def _propose_close(self, args: dict) -> dict:
        symbol = args.get("symbol")
        rationale = args.get("rationale", "")
        match_idx = next(
            (i for i, p in enumerate(self.open_positions)
             if p.sim_trade.symbol == symbol),
            None,
        )
        if match_idx is None:
            return {"accepted": False,
                    "reason": f"no open position on {symbol}"}
        pos = self.open_positions.pop(match_idx)
        sim = pos.sim_trade
        # Close at current bar close, with TP-side slippage applied
        cur_px = self.last_price.get(symbol, sim.entry_price)
        slip = self.s.paper_tp_slippage_bps / 10_000
        exit_px = cur_px * (1 - slip) if sim.side == "long" else cur_px * (1 + slip)
        self._finalize(sim, exit_px, "trader_agent_close")
        return {"accepted": True, "exit_price": exit_px,
                "pnl_usd": sim.pnl_usd, "reason": rationale[:120]}

    # ----------------------------------------------- replay-time bookkeeping

    def _finalize(self, sim: SimTrade, exit_px: float, reason: str) -> None:
        """Close a SimTrade: compute final P&L, update equity, append to
        closed_trades. Stops apply stop_slippage_bps when called from the
        stop branch; we pass exit_px already-adjusted."""
        fee_bps = self.s.spot_taker_fee_bps
        exit_fee = sim.qty * exit_px * (fee_bps / 10_000)
        if sim.side == "long":
            gross = (exit_px - sim.entry_price) * sim.qty
        else:
            gross = (sim.entry_price - exit_px) * sim.qty
        sim.exit_price = exit_px
        sim.exit_reason = reason
        sim.exit_ts_ms = self.cursor_ts_ms
        # pnl_usd already has the entry-fee debit; add gross + exit fee
        sim.pnl_usd = (sim.pnl_usd or 0.0) + gross - exit_fee
        self.equity += sim.pnl_usd
        self.pnl_today += sim.pnl_usd
        if sim.pnl_usd < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self.closed_trades.append(sim)

    def _evaluate_open_positions(self, k: Kline) -> None:
        """Pessimistic stop ordering: if both stop and TP could be hit
        within a single bar's high/low, assume stop hits first. Mirrors
        the existing backtest module's approach."""
        stop_slip = self.s.paper_stop_slippage_bps / 10_000
        tp_slip = self.s.paper_tp_slippage_bps / 10_000
        still_open: list[_SimPosition] = []
        for pos in self.open_positions:
            sim = pos.sim_trade
            if sim.symbol != k.symbol:
                still_open.append(pos)
                continue
            pos.bars_held += 1
            if sim.side == "long":
                hit_stop = k.low <= sim.stop
                hit_tp = k.high >= sim.tp
            else:
                hit_stop = k.high >= sim.stop
                hit_tp = k.low <= sim.tp
            if hit_stop:
                px = sim.stop * (1 - stop_slip) if sim.side == "long" \
                    else sim.stop * (1 + stop_slip)
                self._finalize(sim, px, "stop")
                continue
            if hit_tp:
                px = sim.tp * (1 - tp_slip) if sim.side == "long" \
                    else sim.tp * (1 + tp_slip)
                self._finalize(sim, px, "tp")
                continue
            still_open.append(pos)
        self.open_positions = still_open

    # ----------------------------------------------------------------- replay

    async def run(self) -> TraderBacktestResult:
        """Walk closed bars from cursor → end. On each bar: update
        indicators, mark/manage open positions, evaluate wake triggers,
        invoke the agent (up to max_wakes), continue."""
        assert self.agent is not None, "call prepare() first"
        budget = self.agent.budget

        for k in self._klines[self.cursor_idx:]:
            self.cursor_ts_ms = k.close_time
            self.last_price[k.symbol] = k.close

            # 1) update indicator state
            st = self.indicators.get(k.symbol, k.timeframe)
            snap = st.on_closed_kline(k)

            # 2) evaluate stops/TPs on open positions BEFORE any wake
            self._evaluate_open_positions(k)

            # 3) wake-trigger evaluation
            if self.wakes_invoked >= self.max_wakes:
                # Soft-stop further wakes but keep replaying so position
                # outcomes (stop/TP) on existing trades still materialize.
                self.wakes_skipped_cap += 0  # no-op tracker
                continue

            wake = self.wake_triggers.on_closed_bar(k, snap)
            # Also check position-pressure wake (drawdown on open longs/shorts)
            if wake is None:
                wake = self.wake_triggers.on_position_pressure(
                    [p.to_position() for p in self.open_positions],
                    dict(self.last_price),
                )
            if wake is None:
                continue

            # 4) invoke the real agent on this wake event
            self.wake_counts[wake["kind"]] = self.wake_counts.get(wake["kind"], 0) + 1
            self.wakes_invoked += 1
            log.info("backtest.wake.invoke",
                      n=self.wakes_invoked, max=self.max_wakes,
                      kind=wake.get("kind"), ts_ms=k.close_time,
                      equity=self.equity,
                      open_positions=len(self.open_positions))
            try:
                cycle_result = await run_trader_cycle(self.agent, wake)
                self.total_tool_calls += cycle_result.tool_calls_made
            except Exception:
                log.exception("backtest.wake.error", kind=wake.get("kind"))

            # Budget exhausted? Stop calling the agent (replay continues so
            # open trades can still resolve).
            if budget and not budget.can_spend(0.001):
                log.warning("backtest.budget_exhausted",
                             spent=budget.spent_24h_usd(),
                             cap=budget.daily_usd)
                self.max_wakes = self.wakes_invoked  # gate further wakes

        # Flush any still-open positions at end-of-data
        for pos in list(self.open_positions):
            sim = pos.sim_trade
            last_px = self.last_price.get(sim.symbol, sim.entry_price)
            self._finalize(sim, last_px, "eod")
        self.open_positions.clear()

        span_days = (
            (self._klines[-1].close_time - self._klines[0].close_time) / 1000 / 86400
            if self._klines else 0.0
        )
        stats = _stats_from_trades(
            "trader_agent_backtest", self.closed_trades,
            self.starting_equity, span_days,
        )
        return TraderBacktestResult(
            stats=stats,
            closed_trades=self.closed_trades,
            wake_counts=self.wake_counts,
            total_tool_calls=self.total_tool_calls,
            total_usd_spent=budget.spent_24h_usd() if budget else 0.0,
            wakes_invoked=self.wakes_invoked,
        )
