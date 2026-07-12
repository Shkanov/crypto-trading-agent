"""Full 3-sleeve portfolio measurement — carry + cascade + mean-rev.

Fetches mainnet data (cached), builds each sleeve's daily-PnL series over a
common window, and runs the measurement + allocator diagnostic. Writes the
verdict to /tmp/pf_result.txt (flushed) so output survives buffering.

    uv run --extra research python -m research.portfolio.run_measure
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

from research.portfolio import sleeves as S
from research.portfolio.measure import format_report, measure

# Liquid perps all three sleeves can trade; carry needs a cross-section so we
# use a broad-ish liquid set (majors + established alts).
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "ATOMUSDT",
]
WINDOW_DAYS = 80


def _out(fh, *a):
    line = " ".join(str(x) for x in a)
    print(line); fh.write(line + "\n"); fh.flush()


def main() -> None:
    fh = open("/tmp/pf_result.txt", "w")
    end = (int(time.time() * 1000) // S.DAY_MS) * S.DAY_MS
    start = end - WINDOW_DAYS * S.DAY_MS
    _out(fh, f"window {datetime.utcfromtimestamp(start/1000).date()} .. "
             f"{datetime.utcfromtimestamp(end/1000).date()}  ({WINDOW_DAYS}d), "
             f"universe {len(UNIVERSE)}")

    _out(fh, "computing carry sleeve...")
    carry = S.carry_daily(UNIVERSE, start, end, top_n=3)
    _out(fh, f"  carry active days: {len(carry)}")

    _out(fh, "computing cascade sleeve...")
    cascade = S.cascade_daily(UNIVERSE, start, end)
    _out(fh, f"  cascade active days: {len(cascade)}")

    _out(fh, "computing mean-rev sleeve...")
    meanrev = S.meanrev_daily(UNIVERSE, start, end)
    _out(fh, f"  meanrev active days: {len(meanrev)}")

    n = WINDOW_DAYS
    returns = {
        "carry": S.to_series(carry, start, n),
        "cascade": S.to_series(cascade, start, n),
        "meanrev": S.to_series(meanrev, start, n),
    }
    # Guard: drop all-zero sleeves so the report is meaningful.
    live = {k: v for k, v in returns.items() if np.any(v != 0.0)}
    _out(fh, f"\nsleeves with activity: {list(live)}")
    if len(live) < 1:
        _out(fh, "no sleeve produced trades — check data/window"); return

    rep = measure(live, method="hrp", min_sharpe=0.0)
    _out(fh, "\n" + format_report(rep))

    # Also show the stability-gated version (min_sharpe>0 drops dead sleeves).
    rep2 = measure(live, method="hrp", min_sharpe=0.3)
    _out(fh, "\n=== with stability gate (min_sharpe=0.3) ===")
    _out(fh, "weights: " + ", ".join(f"{k}={v:.2f}" for k, v in rep2.weights.items()))
    c = rep2.combined
    _out(fh, f"combined Sharpe {c.ann_sharpe:.2f}  mean {c.mean_bps_day:.2f}bps/day "
             f"h1 {c.h1_sharpe:.2f} h2 {c.h2_sharpe:.2f}")
    _out(fh, "DONE")
    fh.close()


if __name__ == "__main__":
    main()
