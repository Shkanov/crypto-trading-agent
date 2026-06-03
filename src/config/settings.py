from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Binance
    binance_testnet: bool = True
    binance_api_key: str = ""
    binance_api_secret: str = ""

    # LLM
    anthropic_api_key: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_allowed_user_ids: str = ""

    # External data
    cryptopanic_token: str = ""
    whale_alert_api_key: str = ""
    perplexity_api_key: str = ""

    # Scope
    symbols: str = "BTCUSDT,ETHUSDT,SOLUSDT"
    timeframes: str = "1m,5m,15m,1h"
    market_type: Literal["spot", "perps", "both"] = "both"

    # Risk
    account_equity_usd: float = 1_000.0
    risk_per_trade_pct: float = 0.5
    max_notional_usd: float = 200.0
    max_daily_loss_pct: float = 3.0
    max_concurrent_positions: int = 3
    max_leverage: int = 3
    min_gap_between_trades_sec: int = 300
    max_exposure_per_coin_pct: float = 20.0
    consecutive_loss_halt_short: int = 3
    consecutive_loss_halt_long: int = 5

    # Trailing-DD / vol / daily-loss circuit breakers (sprint #12).
    # Defaults match `src/services/risk_circuits.py:CircuitConfig`; the wire-up
    # in the orchestrator pulls those defaults. The flag below gates the
    # whole subsystem so it can be turned off without removing code.
    risk_circuits_enabled: bool = True

    # Multi-strategy allocator (sprint #17). Default is equal-weight per
    # DeMiguel-Garlappi-Uppal (2009) — for ≤5 strategies, EW typically beats
    # fancier methods OOS because parameter-estimation error dominates.
    # Promote to "inverse_vol" once we have ≥6 months of clean per-strategy
    # returns, then to "hrp" once the basket exceeds ~4 strategies AND
    # cross-strategy correlation has been measured to be non-degenerate.
    allocator_enabled: bool = True
    allocator_method: Literal["equal", "inverse_vol", "hrp"] = "equal"
    allocator_fallback: Literal["equal", "inverse_vol"] = "inverse_vol"
    allocator_lookback_days: int = 90
    allocator_rebalance_days: int = 30
    allocator_hrp_turnover_threshold: float = 0.5
    # Structural constraints applied to every allocator method (min-drawdown
    # discipline): sleeves with fewer than `min_active_days` active days in the
    # lookback window are zeroed (no betting on thin track records), and no
    # single sleeve exceeds `max_weight` (bounds single-sleeve DD contribution).
    # Defaults sized to the 90d lookback; 0 / 1.0 would disable each.
    allocator_min_active_days: int = 30
    allocator_max_weight: float = 0.35
    # Edge gate + cash guard (fix for the activity-not-edge concentration bug
    # where the evidence floor zeroed event-driven sleeves and handed ~100% to
    # the always-on carry sleeve regardless of its edge). Sleeves with realized
    # window Sharpe below `allocator_min_sharpe` are zeroed; if fewer than
    # `allocator_min_surviving_sleeves` clear all gates the book holds cash
    # (AllocationResult.deployable=False) rather than concentrate. min_sharpe=0
    # = require non-negative realized edge; set very negative to disable.
    # min_surviving_sleeves=1 = deploy whatever has edge (no forced
    # diversification); raise to 2 to demand a diversified book or hold cash.
    allocator_min_sharpe: float = 0.0
    allocator_min_surviving_sleeves: int = 1

    # Approval
    auto_approve_max_notional_usd: float = 50.0
    twofa_threshold_notional_usd: float = 500.0
    approval_timeout_sec: int = 300

    # Cost model (bps)
    spot_taker_fee_bps: float = 10.0
    spot_maker_fee_bps: float = 10.0
    perps_taker_fee_bps: float = 5.0
    perps_maker_fee_bps: float = 2.0
    slippage_bps: float = 5.0
    # Paper-mode exit slippage (worse on stops, per the critic's empirical numbers)
    paper_stop_slippage_bps: float = 25.0
    paper_tp_slippage_bps: float = 5.0

    # Correlation cap as a fraction of equity (BTC-beta-weighted total exposure)
    max_correlated_exposure_pct: float = 40.0

    # LLM behavior: advisory-only by default. Set to true ONLY after 90 days
    # of paper trading shows the LLM proposals would have beaten a frozen
    # baseline. Until then, the user manually promotes via Telegram.
    strategy_agent_auto_apply: bool = False
    # NewsSentimentAgent veto power. Default off (narrative-only); set true
    # only after slipping trades you'd want vetoed actually correlate with
    # bad outcomes (which the news-sentiment/price reflexivity loop makes
    # unlikely).
    news_sentiment_veto_enabled: bool = False
    # Per-agent token budgets (USD per rolling 24h). Supervisor disables an
    # agent that exceeds its budget until the window rolls.
    llm_strategy_daily_budget_usd: float = 1.0
    llm_news_daily_budget_usd: float = 2.0
    llm_anomaly_daily_budget_usd: float = 0.5

    # TraderAgent — the "human trader at a desk" loop.
    #   Reads tools (klines, indicators, news, OI, liquidations, ...) and
    #   proposes trades. Backtest mode: direct paper execution. Live mode:
    #   every proposal requires Telegram approval (force_user_approval bypasses
    #   the auto_approve_max_notional_usd threshold for trader-agent props).
    trader_agent_enabled: bool = False
    llm_trader_daily_budget_usd: float = 5.0
    trader_agent_force_user_approval: bool = True
    trader_agent_max_tool_iters: int = 12
    # Event-driven wake triggers
    trader_agent_wake_atr_threshold: float = 1.0      # |close-open| in ATR units
    trader_agent_wake_news_sentiment_threshold: float = 0.5
    trader_agent_wake_position_drawdown_pct: float = 0.5
    trader_agent_heartbeat_sec: int = 1800            # 30 min idle check-in
    trader_agent_min_wake_gap_sec: int = 60           # per-kind dedup window

    # Level-breakout strategy (encodes the multi-TF "пробой дневки" pattern:
    # break of prior-day high/low on the trigger TF, with HTF regime + volume
    # + momentum filters, ATR-based stop, and post-stop cooldown). Inspired
    # by a scalping channel; validated only by your own backtest.
    # Requires `htf` (default "1d") to be present in `timeframes`.
    level_breakout_enabled: bool = False
    level_breakout_htf: str = "1d"
    level_breakout_trigger_tf: str = "5m"
    level_breakout_atr_stop_mult: float = 1.0
    level_breakout_rr_target: float = 2.0
    level_breakout_cooldown_min: int = 240
    level_breakout_vol_z_min: float = 1.0
    level_breakout_rsi_long_min: float = 50.0
    level_breakout_rsi_short_max: float = 50.0
    level_breakout_max_atr_pct: float = 5.0  # skip if M5 atr/close > this
    # Trendline (наклонка) variant
    level_breakout_trendline_enabled: bool = True
    level_breakout_trendline_tf: str = "15m"
    level_breakout_pivot_window: int = 3          # bars before/after for pivot confirm
    level_breakout_trendline_max_age_bars: int = 100

    # Funding-rate harvesting strategy
    funding_harvest_enabled: bool = True
    funding_entry_threshold_bps: float = 10.0      # 8h funding must exceed
    funding_entry_avg_threshold_bps: float = 5.0   # 21-period mean must exceed
    funding_exit_threshold_bps: float = 2.0        # close when funding crosses below
    funding_notional_per_pair_usd: float = 100.0   # per leg
    funding_max_concurrent_pairs: int = 2
    # Spot margin gate for the negative-funding direction (short_spot leg).
    # Must be enabled by the operator AFTER enabling margin on the Binance
    # account; otherwise live mode will refuse short-spot pairs.
    live_spot_margin_enabled: bool = False

    # Δfunding cross-sectional carry (CPCV-validated, sprint #card1)
    # Enabled by operator after reviewing cpcv_validate_dfunding results.
    dfunding_carry_enabled: bool = False
    dfunding_window_cycles: int = 21     # 21×8h=168h window; validated on top-30
    dfunding_top_n: int = 3              # legs per side (w21_tn3_bk10 config)
    dfunding_book_pct_per_side: float = 0.10   # 10% equity per leg-bundle
    dfunding_universe_size: int = 30     # top-N perps by 24h volume
    dfunding_rebalance_hours: int = 168  # weekly rebalance

    # Storage / logging
    database_url: str = "sqlite+aiosqlite:///./data/agent.db"
    log_level: str = "INFO"

    # Hard kill switch: presence of this file forces an immediate halt
    # regardless of LLM/Telegram state. Operator must explicitly /resume
    # AND delete the file before trading restarts.
    kill_switch_path: str = "./data/STOP"

    # Clock skew tolerance — halt on drift > N ms.
    max_clock_skew_ms: int = 1500

    # LLM cadences (seconds)
    strategy_agent_interval_sec: int = 600
    news_agent_interval_sec: int = 120

    # Model ids
    opus_model: str = "claude-opus-4-7"
    haiku_model: str = "claude-haiku-4-5-20251001"

    @field_validator("symbols", "timeframes")
    @classmethod
    def _strip(cls, v: str) -> str:
        return ",".join(p.strip() for p in v.split(",") if p.strip())

    @property
    def symbol_list(self) -> list[str]:
        return [s for s in self.symbols.split(",") if s]

    @property
    def timeframe_list(self) -> list[str]:
        return [t for t in self.timeframes.split(",") if t]

    @property
    def allowed_user_ids(self) -> set[int]:
        out: set[int] = set()
        for p in self.telegram_allowed_user_ids.split(","):
            p = p.strip()
            if p.isdigit():
                out.add(int(p))
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
