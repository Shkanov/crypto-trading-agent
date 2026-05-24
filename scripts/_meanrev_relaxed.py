"""One-off: run mean-rev with relaxed RSI gates to test sensitivity.

Not a permanent CLI — intentionally underscored. Edit constants below.
"""
from __future__ import annotations

import asyncio
from dotenv import load_dotenv

from src.services.backtest import backtest_mean_reversion
from src.strategies.mean_reversion import MeanReversionConfig
from src.tools.binance_client import BinanceClient


SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "APTUSDT", "SUIUSDT", "ARBUSDT", "WIFUSDT", "INJUSDT", "SEIUSDT",
    "XRPUSDT", "ADAUSDT", "LTCUSDT", "BCHUSDT",
]
TF = "5m"
HTF = "1h"
BARS = 20000


async def amain() -> None:
    load_dotenv()
    cfg_relaxed = lambda sym: MeanReversionConfig(  # noqa: E731
        allowed_symbols=[sym], htf_timeframe=HTF,
        rsi_oversold=35.0, rsi_overbought=65.0,
        stoch_oversold=25.0, stoch_overbought=75.0,
    )
    b = BinanceClient()
    await b.start()
    header = f"{'symbol':<10s} {'trades':>6s} {'win%':>5s} {'pnl':>9s} {'mdd%':>5s} {'ann%':>8s}"
    print(header)
    print("-" * len(header))
    total_pnl = 0.0
    total_trades = 0
    profitable = 0
    try:
        for sym in SYMBOLS:
            try:
                stats, _ = await backtest_mean_reversion(
                    b, symbol=sym, tf=TF, htf=HTF, bars=BARS,
                    cfg=cfg_relaxed(sym),
                )
            except Exception as e:
                print(f"{sym:<10s} ERROR {e}")
                continue
            print(f"{sym:<10s} {stats.trades:>6d} {stats.win_rate*100:>5.1f} "
                  f"${stats.total_pnl_usd:>+8.2f} {stats.max_drawdown_pct:>5.1f} "
                  f"{stats.annualized_pct:>+7.1f}%")
            total_pnl += stats.total_pnl_usd
            total_trades += stats.trades
            if stats.total_pnl_usd > 0:
                profitable += 1
    finally:
        await b.close()
    print()
    print(f"aggregate: trades={total_trades}  pnl=${total_pnl:+.2f}  "
          f"profitable={profitable}/{len(SYMBOLS)}")


if __name__ == "__main__":
    asyncio.run(amain())
