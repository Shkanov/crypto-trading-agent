"""One-off: how much of the negative P&L is the stop-slip assumption?

Reruns the 1h grid (indicator + meanrev) across 16 symbols at multiple
stop-slip levels, holding TP slip at 5 bps. The question:

  At what stop-slip do the strategies become profitable (if at all)?

Default `paper_stop_slippage_bps=25` is the dominant negative term in
every losing trade. Real Binance perp stop slip on liquid majors is
usually much smaller — measured here against a sweep.

Not a permanent CLI — intentionally underscored.
"""
from __future__ import annotations

import asyncio
from dotenv import load_dotenv

from src.config.settings import Settings
from src.services.backtest import (
    BacktestStats,
    backtest_indicator,
    backtest_mean_reversion,
)
from src.tools.binance_client import BinanceClient


SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
    "APTUSDT", "SUIUSDT", "ARBUSDT", "WIFUSDT", "INJUSDT", "SEIUSDT",
    "XRPUSDT", "ADAUSDT", "LTCUSDT", "BCHUSDT",
]
TF = "1h"
HTF = "4h"
BARS = 2000
STOP_SLIPS = [25.0, 15.0, 10.0, 5.0]
TP_SLIP = 5.0


def _settings(stop_slip: float) -> Settings:
    return Settings(
        paper_stop_slippage_bps=stop_slip,
        paper_tp_slippage_bps=TP_SLIP,
    )


async def _run_one(b: BinanceClient, fn, sym: str, settings: Settings) -> BacktestStats:
    stats, _ = await fn(b, symbol=sym, tf=TF, htf=HTF, bars=BARS, settings=settings)
    return stats


async def amain() -> None:
    load_dotenv()
    b = BinanceClient()
    await b.start()
    try:
        for stop_slip in STOP_SLIPS:
            settings = _settings(stop_slip)
            agg_ind = {"trades": 0, "pnl": 0.0, "profitable": 0, "wins": 0}
            agg_mr = {"trades": 0, "pnl": 0.0, "profitable": 0, "wins": 0}
            print(f"\n=== stop_slip={stop_slip:.0f}bps  tp_slip={TP_SLIP:.0f}bps "
                  f"({len(SYMBOLS)} symbols, {TF}, ~{BARS} bars) ===")
            print(f"{'symbol':<10s}  {'indic_pnl':>10s} {'indic_w%':>8s}  "
                  f"{'mr_pnl':>10s} {'mr_w%':>6s}")
            for sym in SYMBOLS:
                try:
                    s_ind = await _run_one(b, backtest_indicator, sym, settings)
                    s_mr = await _run_one(b, backtest_mean_reversion, sym, settings)
                except Exception as e:
                    print(f"{sym:<10s}  ERROR {e}")
                    continue
                print(f"{sym:<10s}  ${s_ind.total_pnl_usd:>+8.2f} {s_ind.win_rate*100:>6.1f}%  "
                      f"${s_mr.total_pnl_usd:>+8.2f} {s_mr.win_rate*100:>5.1f}%")
                agg_ind["trades"] += s_ind.trades; agg_ind["pnl"] += s_ind.total_pnl_usd
                agg_ind["wins"] += s_ind.wins
                if s_ind.total_pnl_usd > 0: agg_ind["profitable"] += 1
                agg_mr["trades"] += s_mr.trades; agg_mr["pnl"] += s_mr.total_pnl_usd
                agg_mr["wins"] += s_mr.wins
                if s_mr.total_pnl_usd > 0: agg_mr["profitable"] += 1
            print(f"  indicator agg: trades={agg_ind['trades']}  "
                  f"win%={agg_ind['wins']/max(1,agg_ind['trades'])*100:.1f}  "
                  f"pnl=${agg_ind['pnl']:+.2f}  "
                  f"profitable={agg_ind['profitable']}/{len(SYMBOLS)}")
            print(f"  meanrev   agg: trades={agg_mr['trades']}  "
                  f"win%={agg_mr['wins']/max(1,agg_mr['trades'])*100:.1f}  "
                  f"pnl=${agg_mr['pnl']:+.2f}  "
                  f"profitable={agg_mr['profitable']}/{len(SYMBOLS)}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
