"""Per-call stage diagnostic for the cascade detector. Shows WHICH stage
failed: cascade build, leg/PB/R² check, natorgovka, or trigger."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.models.types import Kline
from src.strategies.cascade_breakout import (
    CascadeParams,
    _atr,
    _build_cascade_chain,
    _detect_natorgovka,
    _detect_trigger,
    _find_swing_pivots,
)

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data/research/aktradescalp/scanner_cache"
CALLS = REPO / "data/research/aktradescalp/aktradescalp_calls.json"


def _row_to_kline(r, symbol):
    return Kline(
        symbol=symbol, timeframe="15m",
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    )


def _entry_idx(ks, ts):
    for i, k in enumerate(ks):
        if k.close_time > ts:
            return i
    return None


def _diagnose_at(ks, idx, side, params):
    """Run each stage at bar idx, return what failed."""
    slice_ks = ks[: idx + 1]
    if len(slice_ks) < max(params.cascade_lookback_bars, 30):
        return "too_few_bars"
    atr = _atr(slice_ks, 14)
    if atr is None:
        return "no_atr"
    highs, lows = _find_swing_pivots(
        slice_ks, k=params.swing_k, lookback=params.cascade_lookback_bars)
    if len(highs) + len(lows) < params.cascade_min_pivots:
        return f"too_few_pivots(h={len(highs)},l={len(lows)})"
    chain = _build_cascade_chain(highs, lows, side, atr, params)
    if chain is None:
        # Compute slopes inline for debug
        from src.strategies.cascade_breakout import _linreg_slope_r2
        rh = highs[-6:]; rl = lows[-6:]
        sl_h, r2_h = (_linreg_slope_r2([p.idx for p in rh], [p.price for p in rh])
                       if len(rh) >= 2 else (0.0, 0.0))
        sl_l, r2_l = (_linreg_slope_r2([p.idx for p in rl], [p.price for p in rl])
                       if len(rl) >= 2 else (0.0, 0.0))
        return (f"no_cascade(h={len(highs)},l={len(lows)},"
                f"sl_h={sl_h:+.3g},sl_l={sl_l:+.3g},r2_h={r2_h:.2f},r2_l={r2_l:.2f})")
    # Diagnostics on the chain
    if side == "long":
        level_candidates = [pp.price for pp in chain.pivots if pp.kind == "high"]
        level = max(level_candidates) if level_candidates else None
    else:
        level_candidates = [pp.price for pp in chain.pivots if pp.kind == "low"]
        level = min(level_candidates) if level_candidates else None
    if level is None:
        return f"no_level(chain_len={len(chain.pivots)})"
    nat = _detect_natorgovka(slice_ks, level, side, atr, params)
    if nat is None:
        # Why? Compute what the diagnostic would say
        bar = slice_ks[-2] if len(slice_ks) >= 2 else None
        bm = (bar.open + bar.close) / 2 if bar else None
        dist_atr = abs(bm - level) / atr if bm else None
        return f"no_natorgovka(level={level:.4g},dist_atr={dist_atr:.2f},chain={len(chain.pivots)})"
    trig = _detect_trigger(slice_ks, level, side, atr, params)
    if trig is None:
        # Why?
        bar = slice_ks[-1]
        tr = max(bar.high - bar.low, 1e-12)
        body_pct = abs(bar.close - bar.open) / tr
        prior_vol = sum(k.volume for k in slice_ks[-21:-1]) / 20
        vol_mult = bar.volume / prior_vol if prior_vol > 0 else 0
        if side == "long":
            beyond = (bar.close - level) / atr
        else:
            beyond = (level - bar.close) / atr
        return (f"no_trigger(body={body_pct:.2f},vol_mult={vol_mult:.2f},"
                f"beyond_atr={beyond:+.2f})")
    return f"OK(conf={len(chain.pivots)})"


def main():
    calls = json.load(open(CALLS))
    calls = [c for c in calls if c["side"] in ("long", "short")]
    histories = {}
    for c in calls:
        sym = c["symbol"]
        if sym in histories:
            continue
        p = CACHE / f"klines_15m_{sym}.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        histories[sym] = [_row_to_kline(r, sym) for r in rows]

    params = CascadeParams()
    print(f"=== Per-call stage diagnostic ===")
    print(f"{'msg':>4s}  {'sym':12s} {'side':5s}  off  diagnosis")
    fail_counter = {}
    for c in calls:
        ks = histories.get(c["symbol"])
        if not ks:
            continue
        ts = int(datetime.fromisoformat(c["dt_iso"]).timestamp() * 1000)
        idx = _entry_idx(ks, ts)
        if idx is None:
            continue
        for off in (-2, -1, 0, 1, 2):
            i = idx + off
            if i < 30 or i >= len(ks):
                continue
            diag = _diagnose_at(ks, i, c["side"], params)
            print(f"{c['msg_id']:>4d}  {c['symbol']:12s} {c['side']:5s}  "
                  f"{off:>+2d}   {diag}")
            head = diag.split("(")[0]
            fail_counter[head] = fail_counter.get(head, 0) + 1

    print("\n=== Failure-mode histogram (across all call×offsets) ===")
    for k, v in sorted(fail_counter.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
