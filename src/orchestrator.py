"""Async supervisor that wires the system together.

Tasks (each is an awaitable started on `run()` and supervised):

  - market_loop:        Binance WS → IndicatorEngine
  - signal_loop:        on each closed kline → generate Signal → propose
  - approval_loop:      drives Proposal state machine (auto vs Telegram)
  - expiry_loop:        expires AWAITING_USER proposals past their timeout
  - news_loop:          NewsSentimentAgent every settings.news_agent_interval_sec
  - strategy_loop:      StrategyAgent every settings.strategy_agent_interval_sec
  - telegram:           PTB Application running long-polling
  - housekeeping:       periodic P&L, halt check, daily digest

The hot path (market_loop, signal_loop, approval_loop, executor) does not
await any LLM call. LLM agents run on their own tasks and communicate via the
event bus + StrategyConfig swap.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Optional

import structlog

from src.agents.anomaly_agent import build_anomaly_agent, investigate
from src.services.anomaly_detectors import AnomalyDetectors
from src.agents.llm_client import LLMAgent
from src.agents.news_agent import build_news_agent, run_news_cycle
from src.agents.strategy_agent import build_strategy_agent, run_strategy_cycle
from src.agents.telegram_bot import TelegramBot, TelegramHandlers
from src.agents.trader_agent import (
    attach_tools as attach_trader_tools,
    build_trader_agent,
    run_trader_cycle,
)
from src.agents.trader_tools import TraderToolContext
from src.config.settings import Settings, get_settings
from src.models.types import (
    Anomaly,
    FeatureVector,
    Kline,
    Position,
    Proposal,
    ProposalStatus,
    SentimentScore,
    Signal,
    StrategyConfig,
    Trade,
    client_order_id,
    now_ms,
    stable_proposal_id,
)
from src.services.trader_triggers import WakeTriggers
from src.services.event_bus import (
    TOPIC_ANOMALY,
    TOPIC_APPROVED,
    TOPIC_FILL,
    TOPIC_NEWS,
    TOPIC_PROPOSAL,
    TOPIC_SENTIMENT,
    TOPIC_SIGNAL,
    EventBus,
)
from src.services.basis_monitor import BasisMonitor
from src.services.correlation import CorrelationMatrix
from src.services.funding_income import FundingIncomePoller
from src.services.funding_monitor import FundingMonitor
from src.services.hodl_benchmark import HodlBenchmark
from src.services.metrics import (
    basis_g,
    consecutive_losses_g,
    equity_g,
    funding_rate_g,
    halted_g,
    hodl_outperf_g,
    last_price_g,
    open_trades_g,
    pending_proposals_g,
    pnl_today_g,
    start_metrics_server,
    trade_close_c,
    trade_open_c,
)
from src.services.news import NewsService
from src.services.notional_ramp import NotionalRamp
from src.services.performance import build_report, format_report_markdown
from src.services.reconciliation import format_report as format_recon_report
from src.services.reconciliation import reconcile_on_boot
from src.services.storage import Storage
from src.services.user_data_stream import UserDataStream
from src.strategies.base import Strategy
from src.strategies.funding_harvest import FundingHarvestStrategy, HarvestParams
from src.strategies.indicator_confluence import IndicatorConfluenceStrategy
from src.strategies.level_breakout import LevelBreakoutParams, LevelBreakoutStrategy
from src.tools.binance_client import BinanceClient
from src.tools.executor import Executor
from src.tools.indicators import IndicatorEngine
from src.tools.pair_executor import PairExecutor
from src.tools.position_manager import PositionManager, TickRange
from src.tools.risk_gate import AccountState, RiskGate

log = structlog.get_logger(__name__)


def _kline_from_ws(msg: dict) -> Kline:
    k = msg["k"]
    return Kline(
        symbol=msg["s"], timeframe=k["i"],
        open_time=k["t"], close_time=k["T"],
        open=float(k["o"]), high=float(k["h"]), low=float(k["l"]), close=float(k["c"]),
        volume=float(k["v"]), quote_volume=float(k["q"]),
        trades=int(k["n"]), taker_buy_volume=float(k["V"]),
        is_closed=bool(k["x"]),
    )


class Orchestrator:
    def __init__(self, paper: bool = True) -> None:
        self.s: Settings = get_settings()
        self.paper = paper
        self.bus = EventBus()
        self.storage = Storage()
        self.binance = BinanceClient()
        self.indicators = IndicatorEngine()
        self.risk = RiskGate(self.s)
        self.executor: Optional[Executor] = None
        self.position_manager: Optional[PositionManager] = None
        self.pair_executor: Optional[PairExecutor] = None
        self.funding_monitor: Optional[FundingMonitor] = None
        self.basis_monitor: Optional[BasisMonitor] = None
        self.funding_strategy: Optional[FundingHarvestStrategy] = None
        self.hodl = HodlBenchmark()
        self.detectors = AnomalyDetectors(cooldown_sec=300)
        self.user_data_stream: Optional[UserDataStream] = None
        self.funding_income_poller: Optional[FundingIncomePoller] = None
        self.correlation: Optional[CorrelationMatrix] = None
        self.notional_ramp: Optional[NotionalRamp] = None
        self.news = NewsService()
        # Latest mid-price per symbol, used for unrealized P&L and force-flatten.
        self.last_price: dict[str, float] = {}
        # Registered strategies; orchestrator fans out kline events to each.
        self.strategies: list[Strategy] = []

        self.cfg: StrategyConfig = StrategyConfig(allowed_symbols=self.s.symbol_list)

        # Account state (paper-mode placeholder; live mode reconciles on boot)
        self.equity = self.s.account_equity_usd
        self.pnl_today = 0.0
        self.consecutive_losses = 0
        self.open_positions: list[Position] = []
        self.last_trade_ms_by_symbol: dict[str, int] = {}
        self.halted_until_ms = 0

        # In-memory pending-approval map
        self.pending: dict[str, Proposal] = {}
        # TraderAgent-originated close requests awaiting human approval.
        # close_id -> {trade_id, symbol, requested_ms, expires_at_ms, rationale}
        self.pending_closes: dict[str, dict] = {}

        # Sentiment cache from NewsSentimentAgent
        self.sentiments: dict[str, SentimentScore] = {}
        self.market_summary: str = ""

        # LLM agents
        self.news_agent: Optional[LLMAgent] = None
        self.strategy_agent: Optional[LLMAgent] = None
        self.anomaly_agent: Optional[LLMAgent] = None
        self.trader_agent: Optional[LLMAgent] = None

        # TraderAgent event-driven wake triggers (atr move, news, anomaly,
        # drawdown, heartbeat). Always present; the agent it wakes may not be.
        self.wake_triggers: WakeTriggers = WakeTriggers(self.s)
        # Coalesce trader-agent invocations — only one cycle runs at a time.
        self._trader_cycle_lock: asyncio.Lock = asyncio.Lock()
        # Bounded ring buffer of recent anomaly events for the trader-agent's
        # get_recent_anomalies tool. Kept tight — most anomalies have a short
        # half-life for trading-decision purposes.
        self._recent_anomalies: deque[dict] = deque(maxlen=50)
        # Per-symbol bounded buffers of recent futures liquidations
        # (!forceOrder@arr stream). Trader-agent reads via get_recent_liquidations.
        self._recent_liquidations: dict[str, deque[dict]] = {}

        # Telegram bot — handlers set during start()
        self.telegram: Optional[TelegramBot] = None

        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    # ---------- approval state machine ----------
    async def propose(self, signal: Signal, market: str = "spot", leverage: int = 1,
                       strategy_name: str = "indicator",
                       force_user_approval: bool = False) -> None:
        """Public hook for strategies. Routes signal through risk gate +
        approval flow + execution; emits a Trade tagged with `strategy_name`.

        `force_user_approval=True` bypasses the auto_approve_max_notional_usd
        branch and always sends to Telegram — used by the TraderAgent in live
        mode so every LLM-originated trade is human-confirmed."""
        await self._propose(signal, market=market, leverage=leverage,
                             strategy_name=strategy_name,
                             force_user_approval=force_user_approval)

    async def _propose(self, signal: Signal, market: str = "spot", leverage: int = 1,
                        strategy_name: str = "indicator",
                        force_user_approval: bool = False
                        ) -> tuple[Optional[Proposal], Optional[str]]:
        """Returns (proposal, reject_reason). On acceptance, proposal is the
        constructed Proposal (already saved + routed for execution/approval)
        and reject_reason is None. On rejection or dedupe, proposal is None
        and reject_reason is a short string. Existing strategy callers
        discard the return value — only the TraderAgent reads it."""
        betas = None
        if self.correlation:
            betas = {s: self.correlation.beta_to_btc(s) for s in self.s.symbol_list}
        # R7: pass ramp cap as override; no more mutating shared Settings.
        ramp_cap = self.notional_ramp.effective_max_notional_usd if self.notional_ramp else None
        acct = AccountState(
            equity_usd=self.equity, pnl_today_usd=self.pnl_today,
            consecutive_losses=self.consecutive_losses,
            open_positions=list(self.open_positions),
            last_trade_ms_by_symbol=dict(self.last_trade_ms_by_symbol),
            halted_until_ms=self.halted_until_ms,
            btc_betas=betas,
        )
        decision = self.risk.check(signal, acct, now_ms(), market, leverage,
                                    max_notional_override=ramp_cap)
        if not decision.ok:
            log.info("risk.rejected", symbol=signal.symbol, side=signal.side, reason=decision.reason)
            return None, f"risk gate: {decision.reason}"

        pid = stable_proposal_id(signal.symbol, signal.side, signal.id)
        if pid in self.pending:
            return None, "dedupe: a proposal for this (symbol, side, signal) is already pending"

        p = Proposal(
            id=pid, signal=signal, market=market,
            qty=decision.qty, notional_usd=decision.notional_usd,
            leverage=decision.leverage,
            expires_at_ms=now_ms() + self.s.approval_timeout_sec * 1000,
        )
        # Stash the originating strategy name in the proposal's reason so we
        # can tag the resulting Trade for per-strategy attribution.
        p.reason = f"strategy={strategy_name}"

        # Hybrid approval: small auto, large needs user. The TraderAgent
        # forces user approval regardless of size (force_user_approval=True).
        if not force_user_approval and p.notional_usd <= self.s.auto_approve_max_notional_usd:
            p.status = ProposalStatus.AUTO_APPROVED
            self.pending[pid] = p
            await self.storage.save_proposal(p)
            await self._submit(p)
            if self.telegram:
                await self.telegram.send_info(
                    f"AUTO `{pid[:8]}` {signal.side.upper()} {signal.symbol} "
                    f"@ {signal.entry:.4f} qty {p.qty:.6f} (${p.notional_usd:.2f})"
                )
            return p, None

        p.status = ProposalStatus.AWAITING_USER
        self.pending[pid] = p
        await self.storage.save_proposal(p)
        await self.storage.audit("proposal_awaiting", {"id": pid, "symbol": signal.symbol})
        if self.telegram:
            requires_2fa = p.notional_usd >= self.s.twofa_threshold_notional_usd
            await self.telegram.send_proposal(p, requires_2fa=requires_2fa)
        return p, None

    async def _submit(self, p: Proposal) -> None:
        if not self.executor or not self.position_manager:
            return
        p.status = ProposalStatus.SUBMITTED
        await self.storage.save_proposal(p)
        result = await self.executor.execute(p)
        if result.ok:
            p.status = ProposalStatus.FILLED
            self.last_trade_ms_by_symbol[p.signal.symbol] = now_ms()
            fill_px = result.fill_price or p.signal.entry
            self.open_positions.append(Position(
                symbol=p.signal.symbol, market=p.market, side=p.signal.side,
                qty=p.qty, entry=fill_px,
                stop=p.signal.stop, take_profit=p.signal.take_profit,
                leverage=p.leverage,
            ))
            # Create OPEN Trade tagged with originating strategy.
            strat_name = p.reason.replace("strategy=", "") if p.reason.startswith("strategy=") else "indicator"
            trade = Trade(
                strategy=strat_name, proposal_id=p.id, symbol=p.signal.symbol,
                market=p.market, side=p.signal.side, qty=p.qty, leverage=p.leverage,
                entry_price=fill_px,
                intended_stop=p.signal.stop, intended_tp=p.signal.take_profit,
                slippage_bps_entry=(abs(fill_px - p.signal.entry) / p.signal.entry) * 10_000
                if p.signal.entry else None,
            )
            await self.position_manager.register(trade)
            if self.telegram:
                await self.telegram.send_info(
                    f"FILLED `{p.id[:8]}` {p.signal.symbol} {p.signal.side.upper()} "
                    f"@ {fill_px:.4f} qty {p.qty:.6f}"
                )
        else:
            p.status = ProposalStatus.FAILED
            p.reason = result.error or ""
            log.warning("executor.failed", proposal_id=p.id, err=result.error)
            if self.telegram:
                await self.telegram.send_critical(f"FAILED `{p.id[:8]}`: {result.error}")
        await self.storage.save_proposal(p)
        self.pending.pop(p.id, None)

    async def _handle_anomaly(self, anomaly: Anomaly) -> None:
        """Deterministic detector fired → invoke AnomalyInvestigator (LLM) →
        enforce the recommended action. Action ∈ {continue, pause, flatten}.

        The LLM never closes positions directly. It returns an action enum;
        we apply it via the same code paths used for /pause and /flatten."""
        log.info("anomaly.fired", kind=anomaly.kind, symbol=anomaly.symbol,
                 severity=anomaly.severity, detail=anomaly.detail)
        self.bus.publish(TOPIC_ANOMALY, anomaly)
        # Persist into the ring buffer for the trader-agent's lookback tool.
        anom_dump = anomaly.model_dump()
        self._recent_anomalies.append(anom_dump)
        # TraderAgent wake on warn/critical anomalies.
        if self.trader_agent is not None:
            wake = self.wake_triggers.on_anomaly(anom_dump)
            if wake is not None:
                asyncio.create_task(self._trader_wake(wake))
        if self.telegram:
            sev = "CRITICAL" if anomaly.severity == "critical" else "ANOMALY"
            await self.telegram.send_info(
                f"*{sev}* `{anomaly.kind}` {anomaly.symbol}: {anomaly.detail}"
            )
        if not self.anomaly_agent:
            return  # No LLM → just alert and continue.
        # Build a snapshot for the LLM.
        ctx_payload = {
            "anomaly": anomaly.model_dump(),
            "last_prices": dict(self.last_price),
            "open_trades": [t.model_dump() for t in self.position_manager.open.values()] if self.position_manager else [],
            "halted": now_ms() < self.halted_until_ms,
            "pnl_today_usd": self.pnl_today,
            "consecutive_losses": self.consecutive_losses,
            "funding_now_bps": {
                s: self.funding_monitor.current_bps(s)
                for s in self.s.symbol_list
            } if self.funding_monitor else {},
        }
        try:
            action, severity, diagnosis = await investigate(self.anomaly_agent, ctx_payload)
        except Exception:
            log.exception("anomaly.llm_failed")
            return
        log.info("anomaly.action", action=action, severity=severity, diagnosis=diagnosis)
        if self.telegram:
            await self.telegram.send_info(
                f"_Anomaly verdict ({severity}): {diagnosis}_ → *{action.upper()}*"
            )
        if action == "pause":
            self.halted_until_ms = max(self.halted_until_ms, now_ms() + 2 * 3600_000)
        elif action == "flatten":
            await self._tg_flatten()

    # ---------- TraderAgent: wake helper + write-tool callbacks -------------

    async def _trader_wake(self, payload: dict) -> None:
        """Run one TraderAgent cycle. No-op if the agent isn't configured.
        Guarded by a lock so overlapping triggers don't cause concurrent
        runs (each cycle can fire 10+ tool calls — overlap = $)."""
        if self.trader_agent is None:
            return
        if self._trader_cycle_lock.locked():
            log.info("trader.wake_dropped_busy", kind=payload.get("kind"))
            return
        async with self._trader_cycle_lock:
            try:
                await run_trader_cycle(self.trader_agent, payload)
            except Exception:
                log.exception("trader.cycle.error", kind=payload.get("kind"))

    async def _trader_propose_trade(self, args: dict) -> dict:
        """Write-tool callback: convert agent args into a Signal, route
        through _propose. In live mode, force user approval regardless of
        size. In paper/backtest, auto-approve applies as usual."""
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
        # Compute edge_bps deterministically from geometry minus round-trip
        # costs. The agent doesn't get to inflate its own edge claim.
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
        force = not self.paper  # live: human approval; paper/backtest: auto
        p, reject = await self._propose(
            signal, market=market, leverage=leverage,
            strategy_name="trader_agent", force_user_approval=force,
        )
        if p is None:
            return {"accepted": False, "reason": reject or "unknown"}
        return {
            "accepted": True, "proposal_id": p.id,
            "status": p.status.value, "qty": p.qty,
            "notional_usd": p.notional_usd, "edge_bps": edge_bps,
            "rr": signal.rr,
            "awaiting_user": p.status == ProposalStatus.AWAITING_USER,
        }

    async def _trader_propose_close(self, args: dict) -> dict:
        """Write-tool callback: close an open trade by symbol. v1 closes
        only the FIRST matching trade — if multiple open positions exist
        on the same symbol, the agent should call this multiple times.

        Paper/backtest: closes immediately via PositionManager.force_close.
        Live: sends a Telegram proposal with Close/Keep buttons; the
        actual close happens in _tg_close_approve."""
        symbol = args.get("symbol")
        rationale = (args.get("rationale") or "")[:300]
        if not symbol or not self.position_manager:
            return {"accepted": False, "reason": "no position manager / no symbol"}
        match = None
        for t in self.position_manager.open.values():
            if t.symbol == symbol:
                match = t
                break
        if match is None:
            return {"accepted": False, "reason": f"no open trade on {symbol}"}

        if self.paper:
            px = self.last_price.get(symbol, match.entry_price)
            try:
                await self.position_manager.force_close(match.id, px, "trader_agent")
            except Exception as e:
                return {"accepted": False, "reason": f"force_close: {e}"}
            return {"accepted": True, "trade_id": match.id, "exit_price": px,
                    "status": "EXECUTED"}

        # --- Live: route through Telegram approval ---
        if self.telegram is None:
            return {"accepted": False,
                    "reason": "live close requires Telegram; bot not running"}
        import uuid as _uuid
        close_id = _uuid.uuid4().hex[:12]
        expires_in = self.s.approval_timeout_sec
        self.pending_closes[close_id] = {
            "trade_id": match.id, "symbol": symbol,
            "requested_ms": now_ms(),
            "expires_at_ms": now_ms() + expires_in * 1000,
            "rationale": rationale,
        }
        await self.storage.audit("trader_close_proposed",
                                  {"close_id": close_id, "symbol": symbol,
                                   "trade_id": match.id})
        current_px = self.last_price.get(symbol, match.entry_price)
        await self.telegram.send_close_proposal(
            close_id=close_id, symbol=symbol, side=match.side,
            qty=match.qty, entry=match.entry_price, current_px=current_px,
            rationale=rationale, expires_in_sec=expires_in,
        )
        return {"accepted": True, "trade_id": match.id, "close_id": close_id,
                "status": "AWAITING_USER",
                "note": "Telegram approval sent; close executes on user click."}

    async def _tg_close_approve(self, close_id: str) -> None:
        """Telegram callback: the human approved a TraderAgent-proposed
        close. Look up the request, validate not expired, force-close
        the matching trade at the current mark."""
        req = self.pending_closes.pop(close_id, None)
        if req is None:
            log.info("close.approve.unknown", close_id=close_id)
            return
        if now_ms() > req["expires_at_ms"]:
            log.info("close.approve.expired", close_id=close_id)
            if self.telegram:
                await self.telegram.send_info(
                    f"Close `{close_id[:8]}` expired before approval — ignored."
                )
            return
        if not self.position_manager:
            return
        trade = self.position_manager.open.get(req["trade_id"])
        if trade is None:
            if self.telegram:
                await self.telegram.send_info(
                    f"Close `{close_id[:8]}`: trade already closed by another path."
                )
            return
        px = self.last_price.get(req["symbol"], trade.entry_price)
        try:
            await self.position_manager.force_close(
                trade.id, px, "trader_agent_approved",
            )
        except Exception as e:
            log.exception("close.approve.force_close_failed", close_id=close_id)
            if self.telegram:
                await self.telegram.send_critical(
                    f"Close `{close_id[:8]}` execution FAILED: {e}"
                )
            return
        await self.storage.audit("trader_close_approved",
                                  {"close_id": close_id, "trade_id": trade.id})

    async def _tg_close_reject(self, close_id: str) -> None:
        """Telegram callback: the human declined to close."""
        req = self.pending_closes.pop(close_id, None)
        if req is None:
            return
        await self.storage.audit("trader_close_rejected",
                                  {"close_id": close_id,
                                   "trade_id": req.get("trade_id")})

    async def _trader_news_subagent(self, symbols: list[str]) -> dict:
        """Bridge: serve cached sentiment first (the _news_loop already runs
        the NewsAgent every news_agent_interval_sec and caches scores in
        self.sentiments). Only invoke a fresh cycle if the cache has nothing
        for any requested symbol — saves the news-agent budget."""
        cached = {s: self.sentiments[s].model_dump()
                  for s in symbols if s in self.sentiments}
        if cached:
            return {
                "sentiments": cached,
                "summary": self.market_summary,
                "cache_hit": True,
            }
        if self.news_agent is None:
            return {"error": "no cached sentiment and news agent not configured"}
        # Recompute recent returns; mirrors the _news_loop helper.
        rrs: dict[str, float] = {}
        for sym in symbols:
            snap = self.indicators.latest(sym, "5m")
            st = self.indicators.states.get((sym, "5m"))
            if st and snap and len(st.closes) >= 12:
                snap0 = list(st.closes)[-12]
                if snap0:
                    rrs[sym] = (snap.close - snap0) / snap0 * 100.0
        try:
            scores, summary = await run_news_cycle(
                self.news_agent, self.news, symbols, rrs,
            )
        except Exception as e:
            return {"error": f"news cycle: {e}"}
        for sc in scores:
            self.sentiments[sc.symbol] = sc
        if summary:
            self.market_summary = summary
        return {
            "sentiments": {s.symbol: s.model_dump() for s in scores},
            "summary": summary, "cache_hit": False,
        }

    async def _on_trade_close(self, trade: Trade) -> None:
        """Called by PositionManager when a paper trade resolves (stop/TP/manual).
        Updates in-memory account state and notifies the user."""
        # Remove the matching Position (first match — only one open per trade).
        for i, pos in enumerate(self.open_positions):
            if pos.symbol == trade.symbol and pos.side == trade.side and abs(pos.qty - trade.qty) < 1e-12:
                self.open_positions.pop(i)
                break
        # Refresh derived stats from storage (single source of truth).
        try:
            self.pnl_today = await self.storage.realized_pnl_today_usd()
            self.consecutive_losses = await self.storage.consecutive_losses()
        except Exception:
            log.exception("on_close.refresh_failed")
        pnl = trade.realized_pnl_usd or 0.0
        if self.telegram:
            emoji = "+" if pnl >= 0 else ""
            await self.telegram.send_info(
                f"CLOSED `{trade.id[:8]}` {trade.symbol} {trade.side.upper()} "
                f"@ {trade.exit_price:.4f} via {trade.exit_reason} "
                f"PnL {emoji}${pnl:.2f}"
            )

    # ---------- Telegram handlers ----------
    async def _tg_approve(self, proposal_id: str) -> None:
        p = self.pending.get(proposal_id)
        if not p or p.status != ProposalStatus.AWAITING_USER:
            return
        if now_ms() > p.expires_at_ms:
            p.status = ProposalStatus.EXPIRED
            await self.storage.save_proposal(p)
            self.pending.pop(proposal_id, None)
            return
        p.status = ProposalStatus.APPROVED
        await self.storage.save_proposal(p)
        await self._submit(p)

    async def _tg_reject(self, proposal_id: str) -> None:
        p = self.pending.get(proposal_id)
        if not p:
            return
        p.status = ProposalStatus.REJECTED
        await self.storage.save_proposal(p)
        self.pending.pop(proposal_id, None)

    async def _tg_status(self) -> str:
        current_equity = self.equity + self.pnl_today
        btc_now = self.last_price.get("BTCUSDT", 0)
        hodl_line = ""
        if btc_now > 0:
            opp = self.hodl.outperformance_usd(current_equity, btc_now)
            opp_pct = self.hodl.outperformance_pct(current_equity, btc_now)
            if opp is not None and opp_pct is not None:
                hodl_line = f"\nvs BTC HODL: `${opp:+.2f}` (`{opp_pct:+.2f}%`)"
        funding_line = ""
        if self.funding_monitor:
            parts = []
            for sym in self.s.symbol_list:
                bps = self.funding_monitor.current_bps(sym)
                if bps is not None:
                    apy = self.funding_monitor.annualized_pct(sym)
                    parts.append(f"{sym} `{bps:+.1f}bps` (≈{apy:+.0f}%APY)")
            if parts:
                funding_line = "\nFunding: " + " · ".join(parts)
        try:
            report = await build_report(self.storage)
            perf = "\n\n" + format_report_markdown(report)
        except Exception:
            perf = ""
        return (
            f"*Status* (paper={self.paper}, testnet={self.s.binance_testnet})\n"
            f"Equity: `${current_equity:.2f}` (start `${self.equity:.2f}`)\n"
            f"PnL today: `${self.pnl_today:+.2f}`\n"
            f"Positions: `{len(self.open_positions)}` · Pending: `{len(self.pending)}`\n"
            f"Halted: `{'yes' if now_ms() < self.halted_until_ms else 'no'}`\n"
            f"Cfg v{self.cfg.version} — _{self.cfg.notes or 'baseline'}_"
            f"{hodl_line}{funding_line}{perf}"
        )

    async def _tg_pause(self) -> str:
        self.halted_until_ms = now_ms() + 4 * 3600_000
        return "Paused for 4h."

    async def _tg_resume(self) -> str:
        self.halted_until_ms = 0
        return "Resumed."

    async def _tg_promote_strategy(self, prop_id: str) -> str:
        try:
            new_cfg = await self.storage.get_proposed_config(prop_id)
        except Exception as e:
            return f"Lookup failed: {e}"
        if not new_cfg:
            return f"No PENDING proposal `{prop_id}`."
        if not self._dry_run_ok(new_cfg):
            await self.storage.mark_proposed_config(prop_id, "REJECTED")
            return "Proposal rejected by dry-run (would generate too many signals on current data)."
        self.cfg = new_cfg
        await self.storage.save_strategy_config(new_cfg)
        await self.storage.mark_proposed_config(prop_id, "APPLIED")
        # Update indicator-confluence strategy's reference (it caches cfg).
        for strat in self.strategies:
            if isinstance(strat, IndicatorConfluenceStrategy):
                strat.cfg = new_cfg
        return f"Promoted `{prop_id}` → strategy v{new_cfg.version}: {new_cfg.notes}"

    async def _tg_flatten(self) -> str:
        """Close everything via the correct per-trade path (R2):
          - Pair trades route through PairExecutor.close_pair with a per-market
            price_map (spot mid + perp mark), so the perp leg uses the perp
            mark, not the spot price.
          - Single-leg indicator trades use PositionManager.force_close_all,
            which keys by symbol only (correct for single-leg).
        """
        self.halted_until_ms = now_ms() + 3600_000
        closed_count = 0
        total_pnl = 0.0
        if self.position_manager and self.pair_executor:
            # Group open trades by pair_id (proposal_id); pair trades are
            # the ones whose proposal_id has multiple legs.
            by_proposal: dict[str, list[Trade]] = {}
            for t in list(self.position_manager.open.values()):
                by_proposal.setdefault(t.proposal_id, []).append(t)
            for proposal_id, legs in by_proposal.items():
                if len(legs) > 1:
                    # It's a pair. Build the per-market price map from
                    # FundingMonitor (perp mark) + last_price (spot mid).
                    price_map: dict[tuple[str, str], float] = {}
                    for leg in legs:
                        if leg.market == "perps" and self.funding_monitor:
                            fp = self.funding_monitor.current(leg.symbol)
                            if fp and fp.mark_price > 0:
                                price_map[(leg.symbol, "perps")] = fp.mark_price
                                continue
                        # Fallback: last_price (spot mid; less accurate for perp).
                        px = self.last_price.get(leg.symbol, leg.entry_price)
                        price_map[(leg.symbol, leg.market)] = px
                    closed_pair = await self.pair_executor.close_pair(
                        legs, reason="manual", price_map=price_map,
                    )
                    for c in closed_pair:
                        self.position_manager.open.pop(c.id, None)
                        await self._on_trade_close(c)
                        closed_count += 1
                        total_pnl += c.realized_pnl_usd or 0.0
                else:
                    # Single-leg (indicator strategy). Use last_price keyed by symbol.
                    closed_singles = await self.position_manager.force_close_all(
                        {legs[0].symbol: self.last_price.get(legs[0].symbol, legs[0].entry_price)},
                        reason="manual",
                    )
                    for c in closed_singles:
                        closed_count += 1
                        total_pnl += c.realized_pnl_usd or 0.0
        self.open_positions.clear()
        self.pnl_today = await self.storage.realized_pnl_today_usd()
        if self.telegram:
            await self.telegram.send_critical(
                f"Flatten — closed {closed_count} positions, realized ${total_pnl:+.2f}. Halted 1h."
            )
        return f"Flatten requested. Closed {closed_count} positions. Halted 1h."

    # ---------- background loops ----------
    async def _warmup(self) -> None:
        # Pull recent history per (symbol, timeframe) to seed indicators.
        for sym in self.s.symbol_list:
            for tf in self.s.timeframe_list:
                try:
                    raw = await self.binance.fetch_klines(sym, tf, limit=500, market="spot")
                except Exception as e:
                    log.warning("warmup.fetch_failed", sym=sym, tf=tf, err=str(e))
                    continue
                klines = [Kline(
                    symbol=sym, timeframe=tf,
                    open_time=int(r[0]), close_time=int(r[6]),
                    open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
                    volume=float(r[5]), quote_volume=float(r[7]),
                    trades=int(r[8]), taker_buy_volume=float(r[9]),
                    is_closed=True,
                ) for r in raw]
                self.indicators.warmup(sym, tf, klines)
        log.info("warmup.done")

    async def _market_loop(self) -> None:
        # Subscribe to all (symbol, timeframe) kline streams. python-binance
        # multiplex sockets accept lowercase channel names.
        for tf in self.s.timeframe_list:
            t = asyncio.create_task(self._stream_klines(tf))
            self._tasks.append(t)

    async def _stream_klines(self, tf: str) -> None:
        # The binance client now auto-reconnects forever; loop only exits on cancel.
        try:
            async for msg in self.binance.stream_klines(self.s.symbol_list, tf):
                # Sentinel: stream reconnected, re-warm indicators for this TF.
                if msg.get("e") == "stream.reconnect":
                    for sym in self.s.symbol_list:
                        a = self.detectors.ws_gap(sym, tf)
                        if a:
                            asyncio.create_task(self._handle_anomaly(a))
                    await self._rewarm_timeframe(tf)
                    continue
                k = _kline_from_ws(msg)
                # Track last seen mid for unrealized P&L / flatten.
                self.last_price[k.symbol] = k.close
                # First BTC tick → snapshot for HODL benchmark.
                if k.symbol == "BTCUSDT" and self.hodl.snap is None:
                    self.hodl.initialize(self.equity, k.close, k.close_time)
                # Always evaluate paper position closes on each incoming bar
                # (closed or in-progress) — uses high/low so an in-progress bar
                # may already have touched a stop.
                if self.position_manager:
                    await self.position_manager.on_bar(TickRange(
                        symbol=k.symbol, high=k.high, low=k.low,
                        close=k.close, ts_ms=k.close_time,
                    ))
                if not k.is_closed:
                    continue
                st = self.indicators.get(k.symbol, k.timeframe)
                snap = st.on_closed_kline(k)
                self.bus.publish(f"market.kline:{k.symbol}:{k.timeframe}", snap)
                # Anomaly detection on every closed bar — cheap, rule-based.
                price_anom = self.detectors.price_jump(k, snap)
                if price_anom:
                    asyncio.create_task(self._handle_anomaly(price_anom))
                # TraderAgent wake on notable ATR moves. Fire-and-forget so
                # the kline stream doesn't block on agent latency.
                if self.trader_agent is not None:
                    wake = self.wake_triggers.on_closed_bar(k, snap)
                    if wake is not None:
                        asyncio.create_task(self._trader_wake(wake))
                # Fan out to all registered strategies.
                for strat in self.strategies:
                    try:
                        await strat.on_bar(k, snap)
                    except Exception:
                        log.exception("strategy.on_bar.failed", strategy=strat.name)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("market_loop.error", tf=tf)

    async def _rewarm_timeframe(self, tf: str) -> None:
        """After a WS reconnect, refetch recent history and rebuild indicator state.

        We blow away the IndicatorState for this (symbol, tf) and warmup from REST
        rather than try to merge partial state — simpler and correct."""
        for sym in self.s.symbol_list:
            try:
                raw = await self.binance.fetch_klines(sym, tf, limit=300, market="spot")
            except Exception as e:
                log.warning("rewarm.fetch_failed", sym=sym, tf=tf, err=str(e))
                continue
            klines = [Kline(
                symbol=sym, timeframe=tf,
                open_time=int(r[0]), close_time=int(r[6]),
                open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
                volume=float(r[5]), quote_volume=float(r[7]),
                trades=int(r[8]), taker_buy_volume=float(r[9]),
                is_closed=True,
            ) for r in raw]
            self.indicators.states.pop((sym, tf), None)
            self.indicators.warmup(sym, tf, klines)
        log.info("rewarm.done", tf=tf)

    async def _expiry_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(5)
            now = now_ms()
            expired = [p for p in self.pending.values()
                       if p.status == ProposalStatus.AWAITING_USER and now > p.expires_at_ms]
            for p in expired:
                p.status = ProposalStatus.EXPIRED
                await self.storage.save_proposal(p)
                self.pending.pop(p.id, None)
                if self.telegram:
                    await self.telegram.send_info(f"EXPIRED `{p.id[:8]}` {p.signal.symbol}")
            # TraderAgent-originated close requests expire too — don't want a
            # stale "Close" button executing at a price the agent never saw.
            expired_closes = [cid for cid, req in self.pending_closes.items()
                              if now > req["expires_at_ms"]]
            for cid in expired_closes:
                req = self.pending_closes.pop(cid)
                await self.storage.audit("trader_close_expired",
                                          {"close_id": cid,
                                           "trade_id": req.get("trade_id")})
                if self.telegram:
                    await self.telegram.send_info(
                        f"EXPIRED close `{cid[:8]}` on {req.get('symbol')}"
                    )

    async def _news_loop(self) -> None:
        if not self.news_agent:
            return
        while not self._stop.is_set():
            try:
                # Build recent returns for the prompt context.
                rrs: dict[str, float] = {}
                for sym in self.cfg.allowed_symbols:
                    snap = self.indicators.latest(sym, "5m")
                    snap0 = None
                    st = self.indicators.states.get((sym, "5m"))
                    if st and len(st.closes) >= 12:
                        snap0 = list(st.closes)[-12]
                    if snap and snap0:
                        rrs[sym] = (snap.close - snap0) / snap0 * 100.0
                scores, summary = await run_news_cycle(
                    self.news_agent, self.news, self.cfg.allowed_symbols, rrs,
                )
                for sc in scores:
                    self.sentiments[sc.symbol] = sc
                    self.bus.publish(TOPIC_SENTIMENT, sc)
                    # TraderAgent wake on high-magnitude sentiment moves.
                    if self.trader_agent is not None:
                        wake = self.wake_triggers.on_news({
                            "symbol": sc.symbol, "score": sc.score,
                            "summary": summary,
                        })
                        if wake is not None:
                            asyncio.create_task(self._trader_wake(wake))
                if summary:
                    self.market_summary = summary
            except Exception:
                log.exception("news_loop.error")
            await asyncio.sleep(self.s.news_agent_interval_sec)

    async def _liquidation_stream_loop(self) -> None:
        """Consume the public !forceOrder@arr futures stream and append
        per-symbol into self._recent_liquidations. Only relevant symbols
        (configured symbol_list) are retained — the all-market firehose
        carries thousands of alts we don't care about."""
        watched = set(self.s.symbol_list)
        try:
            async for ev in self.binance.stream_liquidations():
                if ev.get("e") == "stream.reconnect":
                    continue
                o = ev.get("o") or {}
                sym = o.get("s")
                if sym not in watched:
                    continue
                rec = {
                    "symbol": sym,
                    "side": o.get("S"),   # SELL = long was liquidated
                    "qty": float(o.get("z") or o.get("q") or 0.0),
                    "price": float(o.get("ap") or o.get("p") or 0.0),
                    "status": o.get("X"),
                    "ts_ms": int(o.get("T") or 0),
                }
                buf = self._recent_liquidations.get(sym)
                if buf is None:
                    buf = deque(maxlen=200)
                    self._recent_liquidations[sym] = buf
                buf.append(rec)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("liquidation_stream.error")

    def _get_recent_liquidations(self, symbol: str, n: int = 20) -> list[dict]:
        """Read-side helper for the trader-agent tool. Returns up to N most
        recent liquidations on `symbol` from the in-memory buffer."""
        buf = self._recent_liquidations.get(symbol)
        if not buf:
            return []
        return list(buf)[-n:]

    async def _trader_heartbeat_loop(self) -> None:
        """Polls the WakeTriggers for heartbeat + position-pressure events.
        Runs every 60s — the triggers themselves rate-limit by `heartbeat_sec`
        and `min_wake_gap_sec`."""
        if self.trader_agent is None:
            return
        while not self._stop.is_set():
            try:
                await asyncio.sleep(60)
                # Position-pressure: check on every tick of the heartbeat loop.
                pos_wake = self.wake_triggers.on_position_pressure(
                    list(self.open_positions), dict(self.last_price),
                )
                if pos_wake is not None:
                    asyncio.create_task(self._trader_wake(pos_wake))
                    continue  # don't fire heartbeat in the same cycle
                # Heartbeat: routine check-in regardless of activity.
                hb = self.wake_triggers.check_heartbeat()
                if hb is not None:
                    asyncio.create_task(self._trader_wake(hb))
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("trader_heartbeat.error")

    async def _strategy_loop(self) -> None:
        if not self.strategy_agent:
            return
        while not self._stop.is_set():
            await asyncio.sleep(self.s.strategy_agent_interval_sec)
            try:
                snapshot = {
                    "indicators": {
                        f"{sym}.{tf}": self.indicators.latest(sym, tf).model_dump()
                        for sym in self.cfg.allowed_symbols
                        for tf in self.s.timeframe_list
                        if self.indicators.latest(sym, tf)
                    },
                    "sentiments": {s.symbol: s.model_dump() for s in self.sentiments.values()},
                    "market_summary": self.market_summary,
                    "account": {
                        "equity_usd": self.equity, "pnl_today_usd": self.pnl_today,
                        "open_positions": [p.model_dump() for p in self.open_positions],
                        "halted": now_ms() < self.halted_until_ms,
                    },
                }
                new_cfg = await run_strategy_cycle(
                    self.strategy_agent, self.cfg, self.s.symbol_list, snapshot,
                )
                if new_cfg is None:
                    continue
                # Advisory mode: NEVER auto-swap. Store proposed config + ping
                # the operator on Telegram to /promote_strategy <id>.
                if not self.s.strategy_agent_auto_apply:
                    import uuid as _uuid
                    prop_id = _uuid.uuid4().hex[:12]
                    try:
                        await self.storage.save_proposed_config(
                            prop_id, self.cfg.version, new_cfg, new_cfg.notes,
                        )
                    except Exception:
                        log.exception("strategy.save_proposed_failed")
                        continue
                    log.info("strategy.proposed", prop_id=prop_id, notes=new_cfg.notes)
                    if self.telegram:
                        await self.telegram.send_info(
                            f"Strategy PROPOSAL `{prop_id}` from v{self.cfg.version} → v{new_cfg.version}: "
                            f"{new_cfg.notes}\nUse /promote_strategy {prop_id} to apply."
                        )
                    continue
                # Auto-apply branch (off by default).
                if self._dry_run_ok(new_cfg):
                    log.info("strategy.swap", v=new_cfg.version, notes=new_cfg.notes)
                    self.cfg = new_cfg
                    await self.storage.save_strategy_config(new_cfg)
                    if self.telegram:
                        await self.telegram.send_info(f"Strategy v{new_cfg.version}: {new_cfg.notes}")
                else:
                    log.info("strategy.swap_rejected", reason="dry_run")
            except Exception:
                log.exception("strategy_loop.error")

    def _trigger_timeframe(self) -> str:
        tfs = self.s.timeframe_list
        for cand in ("5m", "15m", "3m", "1m"):
            if cand in tfs:
                return cand
        return tfs[0]

    def _dry_run_ok(self, cfg: StrategyConfig) -> bool:
        """Real walk-forward dry-run (F7). Replays the most recent
        ~`dry_run_replay_bars` closed bars on the trigger TF through BOTH
        the current config and the proposed config, simulates trades with
        ATR-stops and RR-targets matching live logic, and only approves the
        swap if proposed P&L > current P&L on the holdout AND signal count
        is sane.

        Caveats: this uses in-memory IndicatorState (already-rolling), so
        it can't 'walk forward' past the present — it's a same-window
        head-to-head, not a true OOS test. Real OOS requires the backtest
        harness (scripts/backtest.py). Use this as a fast rejection of
        obviously-worse proposals, not as ground truth."""
        trigger_tf = self._trigger_timeframe()
        bars_to_check = 100  # bars per symbol to replay
        results: dict[str, tuple[int, float]] = {}  # cfg-name → (n_signals, pnl_proxy)
        for label, candidate in (("current", self.cfg), ("proposed", cfg)):
            n_signals = 0
            pnl_proxy = 0.0
            for sym in candidate.allowed_symbols:
                st = self.indicators.states.get((sym, trigger_tf))
                htf_st = self.indicators.states.get((sym, candidate.htf_timeframe))
                if not st or len(st.closes) < bars_to_check + 5:
                    continue
                # Re-evaluate the last `bars_to_check` snapshots using a fresh
                # rolling window. We can't perfectly replay every state, so we
                # use the rule-of-thumb: walk through cached closes and use the
                # CURRENT snapshot as the trigger, with the bar's close as
                # entry, and the next bar's high/low for stop/TP resolution.
                closes = list(st.closes)
                highs = list(st.highs) if hasattr(st, "highs") else closes
                lows = list(st.lows) if hasattr(st, "lows") else closes
                htf_snap = htf_st.last_snapshot if htf_st else None
                # We don't have historical snapshots per bar — use the latest
                # snapshot for both, varying only the close price. This is
                # imperfect but gives a directional read on threshold changes.
                snap = st.last_snapshot
                if not snap:
                    continue
                last_n_closes = closes[-bars_to_check:]
                last_n_highs = highs[-bars_to_check:]
                last_n_lows = lows[-bars_to_check:]
                for i in range(len(last_n_closes) - 2):
                    # Probe: would this candidate config emit a signal at this point?
                    probe = snap.model_copy(update={"close": last_n_closes[i]})
                    sig = generate_signal(sym, probe, htf_snap, candidate)
                    if not sig:
                        continue
                    n_signals += 1
                    # Resolve trade: did the NEXT bar hit stop or TP first?
                    nxt_high = last_n_highs[i + 1]
                    nxt_low = last_n_lows[i + 1]
                    if sig.side == "long":
                        if nxt_low <= sig.stop:
                            pnl_proxy -= abs(sig.entry - sig.stop)
                        elif nxt_high >= sig.take_profit:
                            pnl_proxy += abs(sig.take_profit - sig.entry)
                    else:
                        if nxt_high >= sig.stop:
                            pnl_proxy -= abs(sig.entry - sig.stop)
                        elif nxt_low <= sig.take_profit:
                            pnl_proxy += abs(sig.take_profit - sig.entry)
            results[label] = (n_signals, pnl_proxy)

        cur_n, cur_pnl = results.get("current", (0, 0.0))
        new_n, new_pnl = results.get("proposed", (0, 0.0))
        # Reject runaway signal generation.
        if new_n > max(20, cur_n * 3):
            log.info("dry_run.rejected", reason="signal_explosion",
                     cur_n=cur_n, new_n=new_n)
            return False
        # Promote only if proposed strictly dominates current on the holdout,
        # OR if current generated zero signals (can't compare; allow change).
        if cur_n == 0 and new_n > 0 and new_pnl > 0:
            log.info("dry_run.accepted", reason="current_idle", new_n=new_n, new_pnl=new_pnl)
            return True
        if new_pnl > cur_pnl:
            log.info("dry_run.accepted", cur_pnl=cur_pnl, new_pnl=new_pnl)
            return True
        log.info("dry_run.rejected", reason="not_dominant",
                 cur_pnl=cur_pnl, new_pnl=new_pnl)
        return False

    async def _housekeeping(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(60)
            # I6: hard kill switch. Presence of the file forces an immediate
            # 24h halt (operator must remove the file AND /resume).
            try:
                import os
                if os.path.exists(self.s.kill_switch_path):
                    if now_ms() >= self.halted_until_ms:
                        self.halted_until_ms = now_ms() + 24 * 3600_000
                        log.critical("kill_switch.tripped", path=self.s.kill_switch_path)
                        if self.telegram:
                            await self.telegram.send_critical(
                                f"KILL SWITCH file present at {self.s.kill_switch_path} — halted 24h"
                            )
            except Exception:
                log.exception("housekeeping.kill_switch_check_failed")

            # I5: continuous clock-skew check. Halt on drift > tolerance.
            try:
                import time as _t
                srv = await self.binance.client.get_server_time()
                drift_ms = abs(int(srv["serverTime"]) - int(_t.time() * 1000))
                self.binance.time_offset_ms = drift_ms
                if drift_ms > self.s.max_clock_skew_ms:
                    log.critical("clock_skew.exceeded_tolerance", drift_ms=drift_ms,
                                 tolerance=self.s.max_clock_skew_ms)
                    if self.telegram:
                        await self.telegram.send_critical(
                            f"CLOCK SKEW {drift_ms}ms exceeds tolerance "
                            f"{self.s.max_clock_skew_ms}ms — orders may fail. "
                            f"Sync NTP on the host."
                        )
                    self.halted_until_ms = max(self.halted_until_ms,
                                                now_ms() + 1 * 3600_000)
            except Exception:
                log.exception("housekeeping.clock_skew_check_failed")

            # Realized P&L since UTC midnight, computed from CLOSED trades only.
            try:
                self.pnl_today = await self.storage.realized_pnl_today_usd()
                self.consecutive_losses = await self.storage.consecutive_losses()
            except Exception:
                log.exception("housekeeping.refresh_failed")
            # J1: refresh correlation matrix if stale.
            if self.correlation and self.correlation.is_stale():
                try:
                    await self.correlation.update(self.s.symbol_list)
                except Exception:
                    log.exception("housekeeping.correlation_refresh_failed")
            # J2: drawdown-triggered ramp halve + weekly ramp review.
            if self.notional_ramp:
                try:
                    msg = await self.notional_ramp.drawdown_check(
                        self.pnl_today, self.equity, now_ms(),
                    )
                    if msg and self.telegram:
                        await self.telegram.send_critical(msg)
                    # Weekly review uses realized P&L over the last 7d.
                    since_week = now_ms() - 7 * 86_400 * 1000
                    week_pnl = await self.storage.realized_pnl_since_ms(since_week)
                    msg2 = await self.notional_ramp.weekly_review(week_pnl, now_ms())
                    if msg2 and self.telegram:
                        await self.telegram.send_info(msg2)
                except Exception:
                    log.exception("housekeeping.ramp_failed")
            # Update Prometheus gauges.
            try:
                equity_g.set(self.equity + self.pnl_today)
                pnl_today_g.set(self.pnl_today)
                pending_proposals_g.set(len(self.pending))
                halted_g.set(1 if now_ms() < self.halted_until_ms else 0)
                consecutive_losses_g.set(self.consecutive_losses)
                btc = self.last_price.get("BTCUSDT")
                if btc:
                    last_price_g.labels(symbol="BTCUSDT").set(btc)
                    opp = self.hodl.outperformance_usd(self.equity + self.pnl_today, btc)
                    if opp is not None:
                        hodl_outperf_g.set(opp)
                if self.position_manager:
                    by_strat: dict[str, int] = {}
                    for t in self.position_manager.open.values():
                        by_strat[t.strategy] = by_strat.get(t.strategy, 0) + 1
                    for strat, n in by_strat.items():
                        open_trades_g.labels(strategy=strat).set(n)
                if self.funding_monitor:
                    for sym in self.s.symbol_list:
                        bps = self.funding_monitor.current_bps(sym)
                        if bps is not None:
                            funding_rate_g.labels(symbol=sym).set(bps)
                if self.basis_monitor:
                    for sym in self.s.symbol_list:
                        b = self.basis_monitor.sample(sym)
                        if b:
                            basis_g.labels(symbol=sym).set(b.basis_bps)
            except Exception:
                log.exception("housekeeping.metrics_failed")
            if -self.pnl_today >= self.equity * self.s.max_daily_loss_pct / 100.0:
                if now_ms() >= self.halted_until_ms:
                    self.halted_until_ms = now_ms() + 24 * 3600_000
                    if self.telegram:
                        await self.telegram.send_critical(
                            f"Daily loss cap hit (${self.pnl_today:.2f}). Halted 24h."
                        )
            # Funding + basis anomaly checks (cheap, no LLM unless tripped).
            if self.funding_monitor and self.basis_monitor:
                for sym in self.s.symbol_list:
                    cur = self.funding_monitor.current_bps(sym)
                    avg = self.funding_monitor.avg_bps(sym)
                    a = self.detectors.funding_extreme(sym, cur, avg)
                    if a:
                        asyncio.create_task(self._handle_anomaly(a))
                    b = self.basis_monitor.sample(sym)
                    if b:
                        ba = self.detectors.basis_blowout(sym, b.basis_bps)
                        if ba:
                            asyncio.create_task(self._handle_anomaly(ba))

    # ---------- strategy lifecycle ----------
    def register_strategy(self, strat: Strategy) -> None:
        self.strategies.append(strat)
        log.info("orchestrator.strategy_registered", name=strat.name)

    # `equity_available_usd` & `open_trades` satisfy the StrategyContext
    # Protocol so strategies can be passed `self` and query state freely.
    def open_trades(self, strategy_name: Optional[str] = None) -> list[Trade]:
        # Synchronous shim — strategies that need fresh DB state should call
        # storage.list_open_trades directly. This returns the PositionManager's
        # in-memory snapshot, which is sufficient for entry-veto checks.
        if not self.position_manager:
            return []
        all_open = list(self.position_manager.open.values())
        if strategy_name is None:
            return all_open
        return [t for t in all_open if t.strategy == strategy_name]

    def equity_available_usd(self, strategy_name: Optional[str] = None) -> float:
        """Per-strategy equity slice. R10: uses LIVE equity (start + pnl_today)
        and tightens further by the notional ramp's current cap so a losing
        run and a drawdown-halve both shrink the per-strategy slice."""
        live_equity = max(0.0, self.equity + self.pnl_today)
        share = live_equity / max(1, len(self.strategies)) if self.strategies else live_equity
        if self.notional_ramp:
            # Don't let any single strategy's share exceed N× the current
            # max_notional cap; this caps over-allocation when ramp halves.
            share = min(share, self.notional_ramp.effective_max_notional_usd * 4)
        return share

    async def close_trade(self, trade_id: str, reason: str = "manual") -> None:
        if not self.position_manager:
            return
        # Best-effort: needs a current price. Use last_price for the symbol;
        # if absent, use the trade's entry as a degenerate fallback.
        trade = self.position_manager.open.get(trade_id)
        if not trade:
            return
        px = self.last_price.get(trade.symbol, trade.entry_price)
        await self.position_manager.force_close(trade_id, px, reason)

    async def propose_pair(self, pair) -> None:
        """Open both legs of a delta-neutral pair atomically. Routes through
        a (lightweight) risk check on combined notional + halt status, then
        PairExecutor. Pair trades skip the indicator-Proposal approval flow
        because delta-neutrality has fundamentally different risk semantics
        — net exposure is ~zero so the per-coin/correlation gates don't
        apply the same way."""
        if not self.pair_executor or not self.position_manager:
            log.warning("propose_pair.unready")
            return
        # F10: gate at proposal time, not just via housekeeping. Daily-loss
        # kill switch + halted state + notional ≤ this strategy's equity slice.
        if now_ms() < self.halted_until_ms:
            log.info("propose_pair.halted", strategy=pair.strategy)
            return
        if self.pnl_today <= -(self.equity * self.s.max_daily_loss_pct / 100.0):
            log.info("propose_pair.daily_loss_cap", strategy=pair.strategy,
                     pnl_today=self.pnl_today)
            return
        # Each strategy gets an equity slice (orchestrator splits evenly).
        share = self.equity_available_usd(pair.strategy)
        if pair.notional_usd > share:
            log.info("propose_pair.exceeds_equity_share",
                     strategy=pair.strategy, notional=pair.notional_usd, share=share)
            return
        # Auto-approve if below threshold, otherwise Telegram. Pair notional
        # is the sum of both legs' notionals.
        auto = pair.notional_usd <= self.s.auto_approve_max_notional_usd * 2
        if not auto and self.telegram:
            await self.telegram.send_info(
                f"PAIR proposal `{pair.id[:8]}` {pair.strategy} on "
                f"{pair.legs[0].symbol}: ${pair.notional_usd:.0f} notional. "
                f"{pair.rationale}. (Auto-execute disabled for pair trades > threshold; "
                f"approve via /resume_pair {pair.id[:8]} when implemented.)"
            )
        # For now, paper-mode auto-executes all pair proposals (it's the only
        # mode active until live trading is enabled). Live mode requires
        # explicit approval (TODO when going live).
        if not self.paper and not auto:
            log.warning("pair.live_above_auto_threshold_skipped", id=pair.id)
            return
        result = await self.pair_executor.open_pair(pair)
        if not result.ok:
            log.warning("pair.open_failed", err=result.error)
            if self.telegram:
                await self.telegram.send_critical(f"PAIR FAILED `{pair.id[:8]}`: {result.error}")
            return
        # Register opened legs with PositionManager (so /flatten works) and
        # with the funding strategy's internal active-pair tracker.
        for leg in result.legs:
            await self.position_manager.register(leg)
        if self.funding_strategy and pair.strategy == self.funding_strategy.name:
            await self.funding_strategy.register_active_pair(pair, result.legs)
        if self.telegram:
            await self.telegram.send_info(
                f"PAIR FILLED `{pair.id[:8]}` {pair.strategy} {pair.legs[0].symbol} "
                f"long_spot/short_perp ${pair.notional_usd:.0f} — {pair.rationale}"
            )

    # ---------- lifecycle ----------
    async def start(self) -> None:
        start_metrics_server(port=9090)
        await self.storage.init()
        await self.binance.start()
        await self.news.start()
        # Persist initial config if none yet.
        existing = await self.storage.latest_strategy_config()
        if existing:
            self.cfg = existing
        else:
            await self.storage.save_strategy_config(self.cfg)

        self.executor = Executor(self.binance, self.storage, paper=self.paper)
        self.position_manager = PositionManager(
            storage=self.storage, on_close=self._on_trade_close,
            paper=self.paper, settings=self.s,
        )
        self.pair_executor = PairExecutor(
            self.binance, self.storage, paper=self.paper, settings=self.s,
        )
        self.funding_monitor = FundingMonitor(
            symbols=self.s.symbol_list, poll_interval_s=60,
            testnet=self.s.binance_testnet,
        )
        self.basis_monitor = BasisMonitor(
            funding=self.funding_monitor, indicators=self.indicators,
        )
        self.correlation = CorrelationMatrix(binance=self.binance, days=30)
        try:
            await self.correlation.update(self.s.symbol_list)
        except Exception:
            log.exception("startup.correlation_update_failed")

        # J2: notional ramp — start tiny, scale up only on profitable weeks.
        # `starting_notional_usd` lives in settings; ceiling is the max we'd
        # ever let it grow to (separate from per-trade settings.max_notional_usd
        # which becomes the HARD upper bound the ramp respects).
        self.notional_ramp = NotionalRamp(
            storage=self.storage,
            starting_notional_usd=min(25.0, self.s.max_notional_usd),
            max_notional_usd_ceiling=self.s.max_notional_usd,
        )
        try:
            await self.notional_ramp.load()
        except Exception:
            log.exception("startup.ramp_load_failed")
        # Rebuild OPEN trades + AWAITING_USER proposals so a process restart
        # doesn't lose state.
        await self.position_manager.rehydrate()
        for p in await self.storage.list_open_proposals():
            self.pending[p.id] = p
        # Restore counters from storage.
        try:
            self.pnl_today = await self.storage.realized_pnl_today_usd()
            self.consecutive_losses = await self.storage.consecutive_losses()
        except Exception:
            log.exception("startup.pnl_refresh_failed")

        # Perp leverage/margin setup ONCE per symbol on startup, not per-trade.
        if not self.paper and self.s.market_type in ("perps", "both"):
            for sym in self.s.symbol_list:
                try:
                    await self.binance.ensure_perp_setup(sym, self.s.max_leverage)
                except Exception as e:
                    log.warning("perp_setup.failed", sym=sym, err=str(e))

        # LLM agents (skip if no key — system still runs as pure rule-based)
        if self.s.anthropic_api_key:
            self.news_agent = build_news_agent(self.news)
            self.strategy_agent = build_strategy_agent()
            self.anomaly_agent = build_anomaly_agent()
            # TraderAgent: discretionary-style operator. Opt-in via setting.
            if self.s.trader_agent_enabled:
                self.trader_agent = build_trader_agent()
                ctx = TraderToolContext(
                    settings=self.s,
                    binance=self.binance,
                    indicator_engine=self.indicators,
                    storage=self.storage,
                    funding_monitor=self.funding_monitor,
                    basis_monitor=self.basis_monitor,
                    correlation=self.correlation,
                    hodl=self.hodl,
                    get_open_positions=lambda: list(self.open_positions),
                    get_account_state=lambda: {
                        "equity_usd": self.equity,
                        "pnl_today_usd": self.pnl_today,
                        "consecutive_losses": self.consecutive_losses,
                        "halted": now_ms() < self.halted_until_ms,
                        "open_position_count": len(self.open_positions),
                    },
                    get_recent_anomalies=lambda n: list(self._recent_anomalies)[-n:],
                    get_last_prices=lambda: dict(self.last_price),
                    get_recent_liquidations=self._get_recent_liquidations,
                    news_sentiment_subagent=self._trader_news_subagent,
                    propose_trade_callback=self._trader_propose_trade,
                    propose_close_callback=self._trader_propose_close,
                )
                attach_trader_tools(self.trader_agent, ctx)
                log.info("trader_agent.attached", tools=len(self.trader_agent.tools))

        # Telegram (skip if no token)
        if self.s.telegram_bot_token and self.s.allowed_user_ids:
            handlers = TelegramHandlers(
                on_approve=self._tg_approve,
                on_reject=self._tg_reject,
                on_status=self._tg_status,
                on_pause=self._tg_pause,
                on_resume=self._tg_resume,
                on_flatten=self._tg_flatten,
                on_promote_strategy=self._tg_promote_strategy,
                on_close_approve=self._tg_close_approve,
                on_close_reject=self._tg_close_reject,
            )
            self.telegram = TelegramBot(handlers)
            await self.telegram.start()
            await self.telegram.send_info(
                f"Agent online. paper={self.paper}, testnet={self.s.binance_testnet}, "
                f"symbols={','.join(self.cfg.allowed_symbols)}"
            )

        await self._warmup()

        # Reconcile state against exchange before the hot loop starts. In
        # paper mode this is a no-op. In live mode, mismatch halts trading.
        recon = await reconcile_on_boot(self.binance, self.storage, live=not self.paper)
        if not recon.ok:
            log.critical("startup.reconciliation_failed",
                         notes=recon.notes,
                         ghost=len(recon.local_only),
                         orphan=len(recon.exchange_only))
            # Halt trading until operator resolves. Hot loop still warms +
            # streams (so /status works) but propose() is gated by halted_until_ms.
            self.halted_until_ms = now_ms() + 24 * 3600_000
            if self.telegram:
                await self.telegram.send_critical(
                    f"BOOT RECONCILIATION FAILED — trading halted 24h.\n"
                    f"```\n{format_recon_report(recon)}\n```"
                )
        else:
            log.info("startup.reconciliation_ok",
                     matched=len(recon.matched), notes=recon.notes)

        # Start FundingMonitor before strategies so they have initial data.
        if self.funding_monitor:
            await self.funding_monitor.start()

        # Live user-data-stream consumer (futures): refines fill prices,
        # surfaces account events. Funding is credited via the income poller
        # below (R1 — authoritative per-symbol attribution).
        if not self.paper:
            self.user_data_stream = UserDataStream(
                binance=self.binance, storage=self.storage,
            )
            await self.user_data_stream.start()
            self.funding_income_poller = FundingIncomePoller(
                binance=self.binance, storage=self.storage,
            )
            await self.funding_income_poller.start()

        # Register built-in strategies. Indicator-confluence runs as a
        # learning lab; funding-harvest runs as the actual edge-bearing one.
        self.register_strategy(IndicatorConfluenceStrategy(self.cfg, sentiments=self.sentiments))
        if self.funding_monitor and self.basis_monitor and self.s.funding_harvest_enabled:
            self.funding_strategy = FundingHarvestStrategy(
                funding=self.funding_monitor, basis=self.basis_monitor,
                params=HarvestParams(
                    entry_threshold_bps=self.s.funding_entry_threshold_bps,
                    entry_avg_threshold_bps=self.s.funding_entry_avg_threshold_bps,
                    exit_threshold_bps=self.s.funding_exit_threshold_bps,
                    notional_per_pair_usd=self.s.funding_notional_per_pair_usd,
                    max_concurrent_pairs=self.s.funding_max_concurrent_pairs,
                    perp_leverage=min(self.s.max_leverage, 2),
                ),
                symbols=self.s.symbol_list,
            )
            self.register_strategy(self.funding_strategy)

        # Level-breakout (inspired by the "пробой дневки" pattern from an
        # external scalping channel). Off by default — flip on only after
        # `scripts/backtest.py --strategy levelbreak` shows positive
        # deflated Sharpe on multiple symbols and windows. Requires
        # `1d` (or whatever `level_breakout_htf` is set to) in TIMEFRAMES.
        if self.s.level_breakout_enabled:
            lb_params = LevelBreakoutParams(
                htf=self.s.level_breakout_htf,
                trigger_tf=self.s.level_breakout_trigger_tf,
                atr_stop_mult=self.s.level_breakout_atr_stop_mult,
                rr_target=self.s.level_breakout_rr_target,
                cooldown_min=self.s.level_breakout_cooldown_min,
                vol_z_min=self.s.level_breakout_vol_z_min,
                rsi_long_min=self.s.level_breakout_rsi_long_min,
                rsi_short_max=self.s.level_breakout_rsi_short_max,
                max_atr_pct=self.s.level_breakout_max_atr_pct,
                trendline_enabled=self.s.level_breakout_trendline_enabled,
                trendline_tf=self.s.level_breakout_trendline_tf,
                pivot_window=self.s.level_breakout_pivot_window,
                trendline_max_age_bars=self.s.level_breakout_trendline_max_age_bars,
            )
            self.register_strategy(LevelBreakoutStrategy(
                params=lb_params, symbols=self.s.symbol_list,
            ))

        for strat in self.strategies:
            try:
                await strat.start(self)  # type: ignore[arg-type]
            except Exception:
                log.exception("strategy.start.failed", name=strat.name)

        self._tasks.append(asyncio.create_task(self._market_loop()))
        self._tasks.append(asyncio.create_task(self._expiry_loop()))
        self._tasks.append(asyncio.create_task(self._news_loop()))
        self._tasks.append(asyncio.create_task(self._strategy_loop()))
        self._tasks.append(asyncio.create_task(self._housekeeping()))
        if self.trader_agent is not None:
            self._tasks.append(asyncio.create_task(self._trader_heartbeat_loop()))
            self._tasks.append(asyncio.create_task(self._liquidation_stream_loop()))

    async def run(self) -> None:
        await self.start()
        await self._stop.wait()

    async def stop(self) -> None:
        self._stop.set()
        for strat in self.strategies:
            try:
                await strat.stop()
            except Exception:
                log.exception("strategy.stop.failed", name=strat.name)
        if self.funding_monitor:
            await self.funding_monitor.stop()
        if self.user_data_stream:
            await self.user_data_stream.stop()
        if self.funding_income_poller:
            await self.funding_income_poller.stop()
        for t in self._tasks:
            t.cancel()
        if self.telegram:
            await self.telegram.stop()
        await self.news.close()
        await self.binance.close()
