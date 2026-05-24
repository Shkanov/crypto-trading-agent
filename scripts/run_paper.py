"""Paper trading entry point — runs against Binance Testnet with no real money.

  python -m scripts.run_paper

The orchestrator is identical to live mode except `paper=True`, which makes
the Executor simulate fills (no REST orders are sent). Use this to verify the
hot loop, indicators, and Telegram flow before switching to live.
"""
from __future__ import annotations

import asyncio
import logging
import signal as os_signal

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
    orch = Orchestrator(paper=True)
    loop = asyncio.get_running_loop()
    for sig in (os_signal.SIGINT, os_signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(orch.stop()))
    try:
        await orch.run()
    finally:
        await orch.stop()


if __name__ == "__main__":
    asyncio.run(amain())
