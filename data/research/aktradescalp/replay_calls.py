"""Replay @aktradescalp's actual calls and see whether his picks have edge
that our mechanical scan misses.

Method:
  For each call (msg_id, dt_iso, symbol, side):
    1. Fetch M5 perps klines around the call timestamp (enough lookback for
       ATR, enough lookforward to resolve to stop/TP).
    2. Locate the entry bar = first M5 bar whose close_time >= dt_iso.
    3. Compute ATR(14) at the entry bar.
    4. Set entry=close, stop=close ± 1×ATR (long: -, short: +),
       TP = entry + 2R.
    5. Walk forward until stop or TP hits, or max-hold (default 24h) cuts.
    6. Apply the same fee+slippage model as backtest_level_breakout.

Output:
  Per-call ledger + summary stats. Compare to:
  - "random" baseline: same symbol, random entry time during his trading
     window, same side as his call. (Tests whether his TIMING matters.)
  - "our mechanical hits" on the same symbol over the same date range
     (tests whether his SYMBOL+TIMING combo beats our scan).
"""
from __future__ import annotations

import asyncio
import json
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.config.settings import get_settings
from src.models.types import Kline
from src.services.backtest import simulate_level_breakout
from src.strategies.level_breakout import LevelBreakoutParams
from src.tools.binance_client import BinanceClient
from src.tools.indicators import IndicatorEngine

CALLS_PATH = Path(__file__).resolve().parent / "aktradescalp_calls.json"

MAX_HOLD_HOURS = 24
ATR_STOP_MULT = 1.0
RR_TARGET = 2.0
TF = "5m"
TF_MS = 300_000


@dataclass
class Replay:
    symbol: str
    side: str
    msg_id: int
    dt_iso: str
    entry_px: float = 0.0
    stop: float = 0.0
    tp: float = 0.0
    atr: float = 0.0
    exit_px: Optional[float] = None
    exit_reason: Optional[str] = None
    bars_held: int = 0
    pnl_bps: Optional[float] = None     # signed return on entry, bps
    skipped: Optional[str] = None       # if we couldn't replay this call


def _rows_to_klines(rows: list[list], symbol: str, tf: str) -> list[Kline]:
    return [Kline(
        symbol=symbol, timeframe=tf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in rows]


async def _fetch_window(b: BinanceClient, symbol: str, center_ms: int) -> list[Kline]:
    """Fetch enough M5 perps bars to span the full channel corpus (Mar 30 →
    May 22). At 288 M5 bars/day, ~60 days = 17280 bars. Round up to 20000
    so calls near the start of the corpus still have ATR warmup."""
    raw = await b.fetch_klines_paginated(symbol, TF, total=20_000, market="perps")
    ks = _rows_to_klines(raw, symbol, TF)
    return ks


def _replay_one(call: dict, ks: list[Kline]) -> Replay:
    r = Replay(
        symbol=call["symbol"], side=call["side"],
        msg_id=call["msg_id"], dt_iso=call["dt_iso"],
    )
    if not ks:
        r.skipped = "no_klines"
        return r

    ts_ms = int(datetime.fromisoformat(call["dt_iso"]).timestamp() * 1000)
    # Find entry index: first bar whose close_time > ts_ms (we enter on the
    # NEXT closed bar after the message — no look-ahead).
    entry_ix = None
    for i, k in enumerate(ks):
        if k.close_time > ts_ms:
            entry_ix = i
            break
    if entry_ix is None or entry_ix < 20:
        r.skipped = "no_entry_bar_or_insufficient_warmup"
        return r

    # ATR(14) on closes leading up to entry. Use Wilder TR.
    eng = IndicatorEngine()
    for k in ks[:entry_ix]:
        eng.get(call["symbol"], TF).on_closed_kline(k)
    snap = eng.latest(call["symbol"], TF)
    if not snap or snap.atr14 is None:
        r.skipped = "no_atr"
        return r
    atr = snap.atr14
    if atr <= 0:
        r.skipped = "atr_zero"
        return r

    entry_bar = ks[entry_ix]
    entry_px = entry_bar.close
    r.entry_px = entry_px
    r.atr = atr
    if r.side == "long":
        r.stop = entry_px - ATR_STOP_MULT * atr
        r.tp = entry_px + RR_TARGET * ATR_STOP_MULT * atr
    else:
        r.stop = entry_px + ATR_STOP_MULT * atr
        r.tp = entry_px - RR_TARGET * ATR_STOP_MULT * atr

    max_bars = int(MAX_HOLD_HOURS * 60 / 5)
    # Walk forward
    for j in range(entry_ix + 1, min(len(ks), entry_ix + 1 + max_bars)):
        k = ks[j]
        if r.side == "long":
            hit_stop = k.low <= r.stop
            hit_tp = k.high >= r.tp
        else:
            hit_stop = k.high >= r.stop
            hit_tp = k.low <= r.tp
        if hit_stop and hit_tp:
            # Both touched in same bar — assume stop hit first (conservative).
            r.exit_px = r.stop
            r.exit_reason = "stop_first_same_bar"
            r.bars_held = j - entry_ix
            break
        if hit_stop:
            r.exit_px = r.stop
            r.exit_reason = "stop"
            r.bars_held = j - entry_ix
            break
        if hit_tp:
            r.exit_px = r.tp
            r.exit_reason = "tp"
            r.bars_held = j - entry_ix
            break
    if r.exit_px is None:
        # Time-stopped at max_bars
        end_ix = min(len(ks) - 1, entry_ix + max_bars)
        r.exit_px = ks[end_ix].close
        r.exit_reason = "time_stop"
        r.bars_held = end_ix - entry_ix

    if r.side == "long":
        ret_bps = (r.exit_px - entry_px) / entry_px * 10_000
    else:
        ret_bps = (entry_px - r.exit_px) / entry_px * 10_000

    # Costs: round-trip taker fee + entry/exit slippage
    s = get_settings()
    cost_bps = 2 * (s.perps_taker_fee_bps + s.slippage_bps)
    if r.exit_reason and r.exit_reason.startswith("stop"):
        cost_bps += s.paper_stop_slippage_bps
    elif r.exit_reason == "tp":
        cost_bps += s.paper_tp_slippage_bps
    r.pnl_bps = ret_bps - cost_bps
    return r


def _summarize(label: str, replays: list[Replay]) -> None:
    valid = [r for r in replays if r.skipped is None and r.pnl_bps is not None]
    skipped = [r for r in replays if r.skipped is not None]
    if not valid:
        print(f"\n=== {label} ===")
        print(f"  no valid replays (skipped={len(skipped)})")
        return
    pnls = [r.pnl_bps for r in valid]
    wins = sum(1 for p in pnls if p > 0)
    tps = sum(1 for r in valid if r.exit_reason == "tp")
    stops = sum(1 for r in valid if r.exit_reason and r.exit_reason.startswith("stop"))
    times = sum(1 for r in valid if r.exit_reason == "time_stop")
    long_pnls = [r.pnl_bps for r in valid if r.side == "long"]
    short_pnls = [r.pnl_bps for r in valid if r.side == "short"]
    print(f"\n=== {label} ===")
    print(f"  n={len(valid)}  skipped={len(skipped)}")
    print(f"  win rate: {wins / len(valid) * 100:.1f}%   "
          f"exits → tp={tps} stop={stops} time={times}")
    print(f"  pnl_bps:  mean={statistics.mean(pnls):+7.1f}  "
          f"median={statistics.median(pnls):+7.1f}  "
          f"sum={sum(pnls):+8.1f}  "
          f"stdev={statistics.pstdev(pnls):7.1f}")
    if long_pnls:
        print(f"    long  (n={len(long_pnls):2d}):  "
              f"mean={statistics.mean(long_pnls):+7.1f}  "
              f"wins={sum(1 for p in long_pnls if p > 0)}/{len(long_pnls)}")
    if short_pnls:
        print(f"    short (n={len(short_pnls):2d}):  "
              f"mean={statistics.mean(short_pnls):+7.1f}  "
              f"wins={sum(1 for p in short_pnls if p > 0)}/{len(short_pnls)}")
    if skipped:
        skip_reasons: dict[str, int] = {}
        for r in skipped:
            skip_reasons[r.skipped or "?"] = skip_reasons.get(r.skipped or "?", 0) + 1
        print(f"  skip reasons: {skip_reasons}")


async def amain() -> None:
    load_dotenv()
    calls = json.load(open(CALLS_PATH))
    print(f"loaded {len(calls)} calls")

    b = BinanceClient()
    await b.start()

    # Group calls by symbol to minimize fetches
    by_symbol: dict[str, list[dict]] = {}
    for c in calls:
        by_symbol.setdefault(c["symbol"], []).append(c)

    his_replays: list[Replay] = []
    random_replays: list[Replay] = []   # same symbol+side, random time in window
    skipped_symbols: list[str] = []

    try:
        for symbol, group in sorted(by_symbol.items()):
            try:
                ks = await _fetch_window(b, symbol, center_ms=0)
                if len(ks) < 50:
                    skipped_symbols.append(f"{symbol}({len(ks)}bars)")
                    for c in group:
                        r = Replay(symbol=symbol, side=c["side"],
                                   msg_id=c["msg_id"], dt_iso=c["dt_iso"])
                        r.skipped = "symbol_unavailable"
                        his_replays.append(r)
                    continue
            except Exception as e:
                msg = str(e).split("\n")[0][:60]
                skipped_symbols.append(f"{symbol}(err:{msg})")
                for c in group:
                    r = Replay(symbol=symbol, side=c["side"],
                               msg_id=c["msg_id"], dt_iso=c["dt_iso"])
                    r.skipped = "fetch_err"
                    his_replays.append(r)
                continue

            for c in group:
                his_replays.append(_replay_one(c, ks))

            # Random baseline: same symbol+side, random call time during
            # the window 07-12 UTC (his actual trading hours) on a random
            # bar in the available kline range.
            for c in group:
                rnd = dict(c)
                # Pick a random kline whose hour-of-day in UTC is in [7,12]
                eligible = [k for k in ks[100:-300]
                            if 7 <= datetime.fromtimestamp(k.close_time / 1000,
                                                            tz=timezone.utc).hour <= 12]
                if not eligible:
                    r = Replay(symbol=symbol, side=c["side"],
                               msg_id=-1, dt_iso="")
                    r.skipped = "no_random_eligible_bar"
                    random_replays.append(r)
                    continue
                k = random.choice(eligible)
                rnd["dt_iso"] = datetime.fromtimestamp(
                    (k.close_time - TF_MS) / 1000, tz=timezone.utc,
                ).isoformat()
                rnd["msg_id"] = -1
                random_replays.append(_replay_one(rnd, ks))

        if skipped_symbols:
            print(f"unavailable symbols: {len(skipped_symbols)}: "
                  f"{', '.join(skipped_symbols[:8])}"
                  + (f" ... (+{len(skipped_symbols) - 8} more)"
                     if len(skipped_symbols) > 8 else ""))
        _summarize("HIS CALLS (perps M5 replay, 1×ATR stop, 2R TP, 24h max-hold)",
                    his_replays)
        _summarize("RANDOM baseline (same sym+side, random t in 07-12 UTC)",
                    random_replays)
    finally:
        await b.close()


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(amain())
