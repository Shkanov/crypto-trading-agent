"""Testnet-live entry point — REAL orders against Binance **testnet**.

  python -m scripts.run_testnet_live

Unlike run_paper (which simulates fills in-process), this runs the
orchestrator with paper=False, so the Executor sends real orders to the
testnet exchange. They appear in the Binance testnet UI
(testnet.binancefuture.com) — positions, fills, liquidation price.

Still fake money: requires BINANCE_TESTNET=true. This is the bridge
between paper mode and mainnet (run_live.py), letting you watch real
order flow in the exchange UI before risking real capital.
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
    if not s.binance_testnet:
        print("BINANCE_TESTNET=false — this entry point is testnet-only. "
              "Use scripts/run_live.py for mainnet.", file=sys.stderr)
        sys.exit(1)
    if not (s.binance_api_key and s.binance_api_secret):
        print("Missing BINANCE_API_KEY / BINANCE_API_SECRET.", file=sys.stderr)
        sys.exit(1)
    print(f"TESTNET-LIVE MODE (real orders, fake money). "
          f"max_notional=${s.max_notional_usd}, auto_approve<=${s.auto_approve_max_notional_usd}.")

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
