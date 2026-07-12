"""Funding-gate trigger rule for the delta-neutral basis book — spec + backtest.

The gate decides, mechanically, when to run the basis vs sit in USD. The
threshold is ANCHORED TO THE ECONOMICS (not tuned): the basis's NET yield beats
the USD baseline only when the GROSS funding harvest clears

    HURDLE = USD_yield (~4.5%) + borrow+exec (~2.5%) ≈ 7%/yr annualized.

Rule (with hysteresis to avoid whipsaw near the hurdle):
  signal S_t = trailing `lookback_d`-day mean funding across the basket,
               annualized (mean of per-name annualized trailing funding).
  * turn book ON  when S_t >= ON_bps  (hurdle + margin)
  * turn book OFF when S_t <  OFF_bps (hurdle)
  * within the book, include only names whose OWN trailing annualized funding
    >= per-name hurdle (drop names gone lean/negative).
  When ON, earn that day's basket harvest; when OFF, earn USD.

Backtest over 3y: gated vs always-on-basis vs always-USD. Validates the gate
keeps you out of the 2025-26 lean regime and in during the 2023-24 fat one.
Writes /tmp/gate.txt (flushed).
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from research.portfolio import sleeves as S

OUT = open("/tmp/gate.txt", "w")
def emit(*a):
    line = " ".join(str(x) for x in a); print(line); OUT.write(line + "\n"); OUT.flush()

BASKET = ["BTCUSDT", "ETHUSDT", "LINKUSDT", "UNIUSDT", "LTCUSDT", "DOGEUSDT", "AAVEUSDT"]
USD_YIELD = 4.5          # %/yr stablecoin/T-bill baseline
BORROW_EXEC = 2.5        # %/yr basis carrying cost
HURDLE = USD_YIELD + BORROW_EXEC          # ≈7%/yr — the OFF threshold
ON_BPS = HURDLE + 2.0                     # ≈9%/yr — the ON threshold (hysteresis)
LOOKBACK_D = 21          # trailing window for the funding signal
PER_NAME_HURDLE = HURDLE  # a name must clear this to be included when ON


def daily_funding_pct(sym, start, end) -> dict[int, float]:
    """{utc_day: funding harvest that day, % of notional} (sum of 8h rates)."""
    ev = S.fetch_funding(sym, start, end)
    by = defaultdict(float)
    for t, r in ev:
        by[(t // S.DAY_MS) * S.DAY_MS] += r * 100.0
    return by


def main():
    end = (int(time.time() * 1000) // S.DAY_MS) * S.DAY_MS
    start = end - 3 * 365 * S.DAY_MS
    days = [start + i * S.DAY_MS for i in range((end - start) // S.DAY_MS)]
    # per-name daily harvest series
    series = {s: daily_funding_pct(s, start, end) for s in BASKET}
    mat = np.array([[series[s].get(d, 0.0) for d in days] for s in BASKET])  # (names, days)

    emit(f"funding-gate backtest {datetime.utcfromtimestamp(start/1000).date()} .. "
         f"{datetime.utcfromtimestamp(end/1000).date()}  ({len(days)}d)")
    emit(f"RULE: signal = trailing {LOOKBACK_D}d mean basket funding, annualized.")
    emit(f"  ON when signal >= {ON_BPS:.1f}%/yr, OFF when < {HURDLE:.1f}%/yr (hysteresis).")
    emit(f"  when ON, include only names with own trailing funding >= {PER_NAME_HURDLE:.1f}%/yr.")
    emit(f"  when OFF, hold USD at {USD_YIELD:.1f}%/yr.\n")

    usd_daily = USD_YIELD / 365
    borrow_daily = BORROW_EXEC / 365
    gated = []; always = []; onflag = False; days_on = 0
    regime_flips = 0
    for i in range(LOOKBACK_D, len(days)):
        win = mat[:, i - LOOKBACK_D:i]
        name_ann = win.mean(axis=1) * 365                 # per-name annualized
        basket_sig = name_ann.mean()
        # hysteresis
        prev = onflag
        if not onflag and basket_sig >= ON_BPS:
            onflag = True
        elif onflag and basket_sig < HURDLE:
            onflag = False
        if onflag != prev:
            regime_flips += 1
        today = mat[:, i]                                 # today's per-name harvest
        # always-on basis (equal weight all names) net of borrow
        always.append(today.mean() - borrow_daily)
        # gated
        if onflag:
            incl = name_ann >= PER_NAME_HURDLE
            day_ret = (today[incl].mean() if incl.any() else 0.0) - borrow_daily
            gated.append(day_ret); days_on += 1
        else:
            gated.append(usd_daily)
    g = np.array(gated); a = np.array(always)
    n = len(g); yrs = n / 365
    def ann(x): return x.sum() / len(x) * 365
    def sharpe(x): return x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else 0
    emit(f"time in basis: {days_on}/{n} days = {days_on/n*100:.0f}%   regime flips: {regime_flips}")
    emit(f"{'strategy':<16}{'total%':>9}{'ann%':>8}{'sharpe':>8}")
    emit(f"{'GATED (basis|USD)':<16}{g.sum():>8.1f}%{ann(g):>7.1f}%{sharpe(g):>8.2f}")
    emit(f"{'always-on basis':<16}{a.sum():>8.1f}%{ann(a):>7.1f}%{sharpe(a):>8.2f}")
    emit(f"{'always USD':<16}{USD_YIELD*yrs:>8.1f}%{USD_YIELD:>7.1f}%{'inf':>8}")
    # current state
    win = mat[:, -LOOKBACK_D:]
    cur_sig = (win.mean(axis=1) * 365).mean()
    emit(f"\nCURRENT signal (trailing {LOOKBACK_D}d): {cur_sig:.1f}%/yr → "
         f"gate {'ON' if cur_sig >= ON_BPS else 'OFF (hold USD)'}")
    emit("DONE")
    OUT.close()


if __name__ == "__main__":
    main()
