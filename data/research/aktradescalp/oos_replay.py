"""Out-of-sample corpus replay — does @aktradescalp's selection edge persist
on data the strategy was NEVER tuned on?

The cascade system was built/validated on Apr 3 – May 26 2026 (W1/W2/W3). This
splits his calls at 2026-05-27 (the bar right after that window) into IN-SAMPLE
vs OOS and replays both with the SAME execution model as replay_calls.py
(_replay_one: enter next M5 bar, 1×ATR stop, 2R TP, 24h max-hold, real costs).

Two fixes vs replay_calls.py:
  * data source = MAINNET public perps klines (read-only) instead of testnet —
    the OOS calls are exactly the newest low-caps that testnet doesn't list, so
    testnet biased the OOS read by dropping them.
  * per-call random baseline is tagged with the SOURCE call's date so it splits
    the same way (his timing vs random timing, within each period).

This is the stability-first test the cascade research explicitly flagged as
required before any capital allocation. Read-only public data; no orders.

  uv run python data/research/aktradescalp/oos_replay.py
"""
from __future__ import annotations

import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import json  # noqa: E402

from research.ml_meta.data import load_or_fetch_klines  # noqa: E402
from src.models.types import Kline  # noqa: E402

# Reuse the EXACT replay logic so this is apples-to-apples with replay_calls.py.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "_replaymod", Path(__file__).resolve().parent / "replay_calls.py")
_rep = importlib.util.module_from_spec(_spec)
sys.modules["_replaymod"] = _rep      # dataclass needs the module registered
_spec.loader.exec_module(_rep)

CALLS_PATH = Path(__file__).resolve().parent / "aktradescalp_calls.json"
OOS_CUTOFF = "2026-05-27"          # first day after the W3 validation window
TF_MS = 300_000


def _df_to_klines(df, symbol: str) -> list[Kline]:
    """research.ml_meta.data DataFrame → Kline list (only OHLCV used downstream)."""
    out = []
    for ts, row in df.iterrows():
        out.append(Kline(
            symbol=symbol, timeframe="5m",
            open_time=int(ts.value // 1_000_000), close_time=int(row["close_time"]),
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            volume=float(row["volume"]), quote_volume=float(row["quote_volume"]),
            trades=0, taker_buy_volume=0.0, is_closed=True,
        ))
    return out


def _summ(label, replays):
    valid = [r for r in replays if r.skipped is None and r.pnl_bps is not None]
    if not valid:
        print(f"  {label:<10} n=0  (all skipped/{len(replays)})")
        return None
    pnls = [r.pnl_bps for r in valid]
    mean = statistics.mean(pnls)
    sd = statistics.pstdev(pnls) if len(pnls) > 1 else 0.0
    wins = sum(1 for p in pnls if p > 0)
    t = mean / (sd / len(pnls) ** 0.5) if sd > 0 else 0.0
    print(f"  {label:<10} n={len(valid):>2}  WR {wins/len(valid)*100:>4.0f}%  "
          f"mean {mean:+7.1f}bps  t {t:+4.2f}  sum {sum(pnls):+8.1f}  "
          f"(skip {len(replays)-len(valid)})")
    return mean


def main():
    random.seed(42)
    calls = json.load(open(CALLS_PATH))
    by_symbol: dict[str, list[dict]] = {}
    for c in calls:
        by_symbol.setdefault(c["symbol"], []).append(c)

    # Span the whole corpus (Mar 30 → Jun 5) + warmup; mainnet, cached.
    end = int(datetime(2026, 6, 7, tzinfo=timezone.utc).timestamp() * 1000)
    start = int(datetime(2026, 3, 15, tzinfo=timezone.utc).timestamp() * 1000)

    his = {"is": [], "oos": []}
    rnd = {"is": [], "oos": []}
    skipped_syms = []
    for symbol, group in sorted(by_symbol.items()):
        try:
            df = load_or_fetch_klines(symbol, "5m", start, end)
        except Exception as e:  # noqa: BLE001
            skipped_syms.append(f"{symbol}({str(e)[:30]})")
            df = None
        ks = _df_to_klines(df, symbol) if df is not None and len(df) >= 50 else []
        if not ks:
            skipped_syms.append(symbol)
        for c in group:
            bucket = "oos" if c["dt_iso"] >= OOS_CUTOFF else "is"
            his[bucket].append(_rep._replay_one(c, ks))
            # random baseline: same symbol+side, random bar in 07-12 UTC,
            # tagged to the source call's period for the split.
            if ks:
                elig = [k for k in ks[100:-300]
                        if 7 <= datetime.fromtimestamp(k.close_time/1000, tz=timezone.utc).hour <= 12]
                if elig:
                    k = random.choice(elig)
                    rc = dict(c, dt_iso=datetime.fromtimestamp(
                        (k.close_time - TF_MS)/1000, tz=timezone.utc).isoformat(), msg_id=-1)
                    rnd[bucket].append(_rep._replay_one(rc, ks))

    if skipped_syms:
        print(f"unavailable: {len(skipped_syms)}: {', '.join(skipped_syms[:12])}")
    print(f"\n=== OOS corpus replay (mainnet 5m, 1×ATR/2R/24h, cutoff {OOS_CUTOFF}) ===")
    print(f"{'IN-SAMPLE (<5-27, the tuned window)':}")
    h_is = _summ("his", his["is"]); r_is = _summ("random", rnd["is"])
    print(f"{'OUT-OF-SAMPLE (>=5-27, never tuned on)':}")
    h_oos = _summ("his", his["oos"]); r_oos = _summ("random", rnd["oos"])
    print("\ntiming delta (his − random):")
    if h_is is not None and r_is is not None:
        print(f"  in-sample : {h_is - r_is:+7.1f} bps/trade")
    if h_oos is not None and r_oos is not None:
        print(f"  OOS       : {h_oos - r_oos:+7.1f} bps/trade   "
              f"<-- does the selection edge SURVIVE out-of-sample?")


if __name__ == "__main__":
    main()
