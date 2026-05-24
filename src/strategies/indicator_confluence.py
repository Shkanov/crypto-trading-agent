"""Indicator-confluence strategy (the original indicator-soup approach).

Wraps the weighted-vote signal generator behind the Strategy protocol so it
can coexist with funding-harvest and other future strategies. The honest
expected-value of this strategy on majors is near zero net of costs — see
`docs/ARCHITECTURE.md`. It exists primarily as a learning lab in paper mode.

Entry: when the trigger-TF confluence score breaches threshold AND higher-TF
regime agrees, emit a single-leg Proposal with ATR-based stop and TP.
Exit: managed by PositionManager (stop/TP touch on each bar).
"""
from __future__ import annotations

import structlog

from src.models.types import IndicatorSnapshot, Kline, SentimentScore, StrategyConfig, Signal
from src.strategies.base import Strategy, StrategyContext
from src.tools.signal_generator import generate_signal

log = structlog.get_logger(__name__)


class IndicatorConfluenceStrategy(Strategy):
    name = "indicator"

    def __init__(self, cfg: StrategyConfig, *, sentiments: dict[str, SentimentScore]
                 | None = None) -> None:
        self.cfg = cfg
        # Read-through reference owned by the orchestrator (NewsSentimentAgent
        # updates it). May be None when LLM agents are disabled.
        self.sentiments = sentiments if sentiments is not None else {}
        self.ctx: StrategyContext | None = None

    async def start(self, ctx: StrategyContext) -> None:
        self.ctx = ctx
        log.info("strategy.start", name=self.name, symbols=self.cfg.allowed_symbols)

    def _trigger_timeframe(self) -> str:
        if not self.ctx:
            return "5m"
        tfs = self.ctx.settings.timeframe_list
        for cand in ("5m", "15m", "3m", "1m"):
            if cand in tfs:
                return cand
        return tfs[0]

    async def on_bar(self, k: Kline, snap: IndicatorSnapshot) -> None:
        if not self.ctx:
            return
        if k.timeframe != self._trigger_timeframe():
            return
        symbol = k.symbol
        htf = self.cfg.htf_timeframe
        htf_snap = self.ctx.indicators.latest(symbol, htf)
        sig: Signal | None = generate_signal(symbol, snap, htf_snap, self.cfg)
        if not sig:
            return

        # Sentiment veto is OFF by default (E2 demotion). The reflexivity
        # loop with NewsSentimentAgent — vetoing longs in panic news cycles
        # right at bottoms — is a known failure mode. Keep sentiment as a
        # narrative input on Telegram, not a trading filter, until you've
        # observed it correlates with bad outcomes in YOUR data.
        if self.ctx.settings.news_sentiment_veto_enabled:
            sscore = self.sentiments.get(symbol) if self.sentiments else None
            if sscore:
                if sig.side == "long" and sscore.score <= -0.6 and sscore.confidence >= 0.6:
                    log.info("indicator.sentiment_veto", symbol=symbol, side=sig.side)
                    return
                if sig.side == "short" and sscore.score >= 0.6 and sscore.confidence >= 0.6:
                    log.info("indicator.sentiment_veto", symbol=symbol, side=sig.side)
                    return

        market = "perps" if self.ctx.settings.market_type in ("perps", "both") else "spot"
        leverage = self.ctx.settings.max_leverage if market == "perps" else 1
        # F11: cap notional at this strategy's equity slice to avoid
        # over-allocating across coexisting strategies. The risk gate will
        # still apply its own max_notional_usd; whichever is tighter wins.
        # (We don't modify the Signal — risk gate sizing does its own math —
        # but we surface a log so it's visible if the slice is the binding cap.)
        share = self.ctx.equity_available_usd(self.name)
        if share > 0 and share < self.ctx.settings.max_notional_usd:
            log.debug("indicator.equity_slice_binding", share=share,
                      max_notional=self.ctx.settings.max_notional_usd)
        await self.ctx.propose(sig, market=market, leverage=leverage, strategy_name=self.name)
