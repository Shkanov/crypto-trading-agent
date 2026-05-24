"""Live trading entry point. Requires:
  - BINANCE_TESTNET=false in .env
  - mainnet API keys with Spot+Futures enabled and IP-whitelisted
  - MAX_NOTIONAL_USD set low (start with 25-50)
"""
from __future__ import annotations

import asyncio
import logging
import signal as os_signal
import sys

import structlog
from dotenv import load_dotenv

from src.config.settings import get_settings
from src.orchestrator import Orchestrator


def _configure_logging() -> None:
    s = get_settings()
    logging.basicConfig(format="%(message)s", level=s.log_level.upper())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, s.log_level.upper())),
    )


async def amain() -> None:
    load_dotenv()
    _configure_logging()
    s = get_settings()
    if s.binance_testnet:
        print("BINANCE_TESTNET=true — use scripts/run_paper.py for testnet.", file=sys.stderr)
        sys.exit(1)
    if not (s.binance_api_key and s.binance_api_secret):
        print("Missing BINANCE_API_KEY / BINANCE_API_SECRET.", file=sys.stderr)
        sys.exit(1)
    print(f"LIVE MODE. max_notional=${s.max_notional_usd}, max_leverage={s.max_leverage}x.")
    print("Ctrl-C to abort within 5s if you didn't mean to start live.")
    await asyncio.sleep(5)

    orch = Orchestrator(paper=False)
    loop = asyncio.get_running_loop()
    for sig in (os_signal.SIGINT, os_signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(orch.stop()))
    try:
        await orch.run()
    finally:
        await orch.stop()


if __name__ == "__main__":
    asyncio.run(amain())
