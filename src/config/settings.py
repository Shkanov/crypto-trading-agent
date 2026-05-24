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
