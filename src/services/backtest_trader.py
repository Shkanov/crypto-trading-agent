"""Backtest harness for the TraderAgent.

Replays historical klines through the IndicatorEngine and invokes a REAL
spawned trader on each wake event, via the Claude Code CLI (`claude -p`)
talking to our MCP server in `src/agents/trader_mcp_server.py`.

Why CLI subprocess and not the Anthropic SDK: the user has no
ANTHROPIC_API_KEY but does have Claude Code installed. The CLI handles
auth, model selection, and tool plumbing.

Per-wake flow:
    1. Harness writes snapshot.json (full state visible to the trader)
       and resets outbox.json under a per-run tempdir.
    2. Harness spawns `claude -p --mcp-config <tempdir>/mcp.json --model opus
       --strict-mcp-config --bare --output-format json ...`
    3. Claude launches our MCP server as a child stdio process, which
       reads snapshot.json (envvar TRADER_STATE_PATH) for every read tool
       and appends to outbox.json (envvar TRADER_OUTBOX_PATH) on
       propose_trade / propose_close.
    4. Claude exits; harness drains outbox.json, runs each trade decision
       through the existing risk_gate + SimTrade machinery.

Outputs (per run):
    BacktestStats: trades / win-rate / total P&L / Sharpe / max DD
    The full SimTrade ledger (for export to CSV / further analysis)
    Wake-event counts by kind
    Total claude subprocess turns + USD spent (from claude's json output)
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import structlog

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TRADER_SYSTEM_PROMPT = """\
You are an intraday crypto trader operating a small account on Binance.
This is BACKTEST MODE — historical klines are replayed; your trades and
closes are paper-executed by the harness once you exit this turn.

You have a fixed set of MCP tools (mcp__trader-backtest__*). Use them to
read state, then decide. Final action is exactly one of:
  - propose_trade: open a new position (stop and take_profit required)
  - propose_close: close an existing position
  - exit silently (no action) if nothing actionable

Decision discipline:
  - READ first. Always call get_position_state and get_indicator_snapshot
    at minimum before deciding.
  - Backtest gaps: orderbook, news, liquidations, correlation, hodl, and
    sometimes funding/OI are unavailable. The tools say so explicitly —
    do NOT invent or guess values.
  - Use the `calc` tool for arithmetic. Do not arithmetic in-prompt.
  - Every long needs stop < entry < take_profit; every short needs
    take_profit < entry < stop.
  - Hard cap: max 2 open positions. Don't pyramid the same symbol.
  - Risk gate runs after you exit. If a trade is rejected, you'll see it
    in your next wake's account state.

Be terse. Keep reasoning under 200 words before tool calls. Final text
after tools should be one sentence summarizing what you did (or "no
action — <one-line reason>").
"""


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
    total_turns: int           # sum of num_turns across claude invocations
    total_usd_spent: float     # sum of total_cost_usd across invocations
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
        per_wake_budget_usd: float = 0.50,
        settings: Optional[Settings] = None,
    ) -> None:
        self.s = settings or get_settings()
        self.symbol = symbol
        self.tf = tf
        self.htf = htf
        self.bars = bars
        self.max_wakes = max_wakes
        self.per_wake_budget_usd = per_wake_budget_usd

        self.binance = BinanceClient()
        self.indicators = IndicatorEngine()
        self.risk = RiskGate(self.s)

        # Replay state (must exist before WakeTriggers so its clock lambda
        # can read cursor_ts_ms at construction time)
        self._klines: list[Kline] = []
        self._htf_klines: list[Kline] = []
        self._funding_hist: list[tuple[int, float]] = []  # (ts_ms, rate)
        self._oi_hist: list[tuple[int, float]] = []       # (ts_ms, sumOI)
        self.cursor_ts_ms: int = 0
        self.cursor_idx: int = 0
        self.last_price: dict[str, float] = {}

        # Wake cooldowns must advance with the replay cursor, not wall-clock.
        # Without this, a backtest that finishes in seconds collapses every
        # post-first cooldown window to zero advance and suppresses all wakes.
        self.wake_triggers = WakeTriggers(
            self.s, clock_ms=lambda: self.cursor_ts_ms,
        )

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
        self.total_turns: int = 0
        self.total_usd_spent: float = 0.0

        # Per-run tempdir (filled by prepare())
        self._tempdir: Optional[str] = None
        self._state_path: Optional[Path] = None
        self._outbox_path: Optional[Path] = None
        self._mcp_config_path: Optional[Path] = None

    # ------------------------------------------------------------------ data

    async def prepare(self) -> None:
        """Fetch all historical data once. Lay out the per-run tempdir
        used for snapshot.json / outbox.json / mcp_config.json."""
        if shutil.which("claude") is None:
            raise RuntimeError(
                "`claude` CLI not found on PATH — backtest needs Claude Code "
                "installed to spawn the trader."
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

        # Per-run tempdir for snapshot/outbox/mcp-config files
        self._tempdir = tempfile.mkdtemp(prefix="trader_backtest_")
        self._state_path = Path(self._tempdir) / "snapshot.json"
        self._outbox_path = Path(self._tempdir) / "outbox.json"
        self._mcp_config_path = Path(self._tempdir) / "mcp_config.json"
        self._mcp_config_path.write_text(json.dumps({
            "mcpServers": {
                "trader-backtest": {
                    "command": sys.executable,
                    "args": ["-m", "src.agents.trader_mcp_server"],
                    "env": {
                        "TRADER_STATE_PATH": str(self._state_path),
                        "TRADER_OUTBOX_PATH": str(self._outbox_path),
                        # Inherit caller's PYTHONPATH so the editable install resolves
                        "PYTHONPATH": os.environ.get("PYTHONPATH", str(PROJECT_ROOT)),
                    },
                }
            }
        }))

        log.info("backtest.prepare.done",
                  klines=len(self._klines), htf_klines=len(self._htf_klines),
                  warmup=warmup_n, funding_pts=len(self._funding_hist),
                  oi_pts=len(self._oi_hist), tempdir=self._tempdir)

    async def close(self) -> None:
        await self.binance.close()
        # Best-effort tempdir cleanup
        if self._tempdir and os.path.isdir(self._tempdir):
            try:
                shutil.rmtree(self._tempdir)
            except OSError as e:
                log.info("backtest.tempdir.cleanup_skip", err=str(e))

    # ----------------------------------------------------- account / snapshot

    def _account_state_dict(self) -> dict[str, Any]:
        return {
            "equity_usd": self.equity,
            "pnl_today_usd": self.pnl_today,
            "consecutive_losses": self.consecutive_losses,
            "halted": False,
            "open_position_count": len(self.open_positions),
            "mode": "backtest",
        }

    def _funding_at_cursor(self) -> Optional[dict[str, Any]]:
        if not self._funding_hist:
            return None
        rate, ts = None, 0
        for ts_, r in self._funding_hist:
            if ts_ > self.cursor_ts_ms:
                break
            rate, ts = r, ts_
        if rate is None:
            return None
        return {"rate_bps": rate * 10_000, "ts_ms": ts}

    def _oi_change_at_cursor(self) -> Optional[dict[str, Any]]:
        if not self._oi_hist:
            return None
        prior = [(ts, oi) for ts, oi in self._oi_hist if ts <= self.cursor_ts_ms]
        if len(prior) < 2:
            return None
        oldest = prior[max(0, len(prior) - 12)][1]
        newest = prior[-1][1]
        return {
            "oi_oldest": oldest, "oi_newest": newest,
            "oi_change_pct": ((newest - oldest) / oldest * 100.0)
                if oldest > 0 else None,
            "period": "5m",
        }

    def _build_snapshot_payload(self) -> dict[str, Any]:
        tf_state = self.indicators.get(self.symbol, self.tf)
        htf_state = self.indicators.get(self.symbol, self.htf)

        tf_recent = [k for k in self._klines if k.close_time <= self.cursor_ts_ms][-50:]
        htf_recent = [k for k in self._htf_klines if k.close_time <= self.cursor_ts_ms][-50:]

        def _bar(k: Kline) -> dict[str, Any]:
            return {"t": k.open_time, "o": k.open, "h": k.high,
                    "l": k.low, "c": k.close, "v": k.volume}

        return {
            "cursor_ts_ms": self.cursor_ts_ms,
            "symbol": self.symbol, "tf": self.tf, "htf": self.htf,
            "settings": {
                "fee_bps_spot": self.s.spot_taker_fee_bps,
                "fee_bps_perps": self.s.perps_taker_fee_bps,
                "slippage_bps": self.s.slippage_bps,
                "max_notional_usd": self.s.max_notional_usd,
                "risk_per_trade_pct": self.s.risk_per_trade_pct,
            },
            "indicators": {
                self.tf: tf_state.last_snapshot.model_dump()
                    if tf_state.last_snapshot else None,
                self.htf: htf_state.last_snapshot.model_dump()
                    if htf_state.last_snapshot else None,
            },
            "klines_recent": [_bar(k) for k in tf_recent],
            "klines_htf_recent": [_bar(k) for k in htf_recent],
            "open_positions": [
                {"symbol": p.sim_trade.symbol, "side": p.sim_trade.side,
                 "qty": p.sim_trade.qty, "entry": p.sim_trade.entry_price,
                 "stop": p.sim_trade.stop, "tp": p.sim_trade.tp,
                 "bars_held": p.bars_held, "leverage": 1}
                for p in self.open_positions
            ],
            "closed_trades_recent": [
                {"symbol": t.symbol, "side": t.side, "entry": t.entry_price,
                 "exit": t.exit_price, "pnl_usd": t.pnl_usd,
                 "exit_reason": t.exit_reason}
                for t in self.closed_trades[-10:]
            ],
            "account": self._account_state_dict(),
            "funding_at_cursor": self._funding_at_cursor(),
            "oi_change": self._oi_change_at_cursor(),
        }

    # ------------------------------------------------------ claude subprocess

    async def _invoke_claude(self, wake_payload: dict[str, Any]) -> dict[str, Any]:
        """Write snapshot, reset outbox, spawn `claude -p`, return parsed
        json result. Outbox is drained separately by _apply_outbox()."""
        assert self._state_path and self._outbox_path and self._mcp_config_path
        self._state_path.write_text(
            json.dumps(self._build_snapshot_payload(), default=str)
        )
        self._outbox_path.write_text(json.dumps({"decisions": []}))

        user_prompt = (
            "WAKE EVENT — make a decision and exit.\n\n"
            f"Trigger:\n{json.dumps(wake_payload, default=str, indent=2)}\n\n"
            "Read state via tools, then propose_trade / propose_close / "
            "exit silently. Be terse."
        )
        cmd = [
            "claude", "-p",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--mcp-config", str(self._mcp_config_path),
            "--model", "opus",
            "--max-budget-usd", f"{self.per_wake_budget_usd:.2f}",
            "--output-format", "json",
            "--tools", "",
            "--allowedTools", "mcp__trader-backtest",
            "--dangerously-skip-permissions",
            "--system-prompt", TRADER_SYSTEM_PROMPT,
            user_prompt,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            log.warning("backtest.claude.nonzero_exit",
                         rc=proc.returncode,
                         stderr=stderr.decode(errors="replace")[:2000])
        try:
            return json.loads(stdout.decode(errors="replace"))
        except json.JSONDecodeError:
            log.warning("backtest.claude.parse_fail",
                         out=stdout.decode(errors="replace")[:500],
                         stderr=stderr.decode(errors="replace")[:500])
            return {"result": "", "total_cost_usd": 0.0, "num_turns": 0}

    async def _apply_outbox(self) -> int:
        """Drain outbox.json, run each trade/close through existing logic.
        Returns number of decisions applied."""
        assert self._outbox_path
        try:
            data = json.loads(self._outbox_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return 0
        n = 0
        for d in data.get("decisions", []):
            action = d.get("action")
            params = d.get("params", {})
            if action == "trade":
                res = await self._propose_trade(params)
                log.info("backtest.outbox.trade", **res)
            elif action == "close":
                res = await self._propose_close(params)
                log.info("backtest.outbox.close", **res)
            else:
                log.warning("backtest.outbox.unknown_action", action=action)
                continue
            n += 1
        return n

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
        invoke the trader subprocess (up to max_wakes), continue."""
        assert self._state_path is not None, "call prepare() first"

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
                continue

            wake = self.wake_triggers.on_closed_bar(k, snap)
            if wake is None:
                wake = self.wake_triggers.on_position_pressure(
                    [p.to_position() for p in self.open_positions],
                    dict(self.last_price),
                )
            if wake is None:
                continue

            # 4) spawn the trader subprocess for this wake
            self.wake_counts[wake["kind"]] = self.wake_counts.get(wake["kind"], 0) + 1
            self.wakes_invoked += 1
            log.info("backtest.wake.invoke",
                      n=self.wakes_invoked, max=self.max_wakes,
                      kind=wake.get("kind"), ts_ms=k.close_time,
                      equity=self.equity,
                      open_positions=len(self.open_positions))
            try:
                result = await self._invoke_claude(wake)
                turns = int(result.get("num_turns", 0) or 0)
                cost = float(result.get("total_cost_usd", 0.0) or 0.0)
                self.total_turns += turns
                self.total_usd_spent += cost
                applied = await self._apply_outbox()
                log.info("backtest.wake.done",
                          turns=turns, cost_usd=round(cost, 4),
                          decisions_applied=applied)
            except Exception:
                log.exception("backtest.wake.error", kind=wake.get("kind"))

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
            total_turns=self.total_turns,
            total_usd_spent=self.total_usd_spent,
            wakes_invoked=self.wakes_invoked,
        )
