"""Long-horizon delta-neutral basis harvest — is the funding yield stable?

The 1yr BTC/ETH test showed ~3%/yr gross. This extends to ~3 years and breaks
the harvest down BY YEAR (and half-year) to expose regime dependence: the
external research said funding ran hot in late-2024 and compressed through 2026.
Funding-only (construction 1: long spot / short perp harvests the funding), so
no klines needed — cheap. Short perp RECEIVES funding when rate>0.

Writes /tmp/basis_long.txt (flushed).
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from research.portfolio import sleeves as S

OUT = open("/tmp/basis_long.txt", "w")
def emit(*a):
    line = " ".join(str(x) for x in a)
    print(line); OUT.write(line + "\n"); OUT.flush()

YEARS = 3
MAJORS = ["BTCUSDT", "ETHUSDT"]
ALTS = ["DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT",
        "ATOMUSDT", "NEARUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
        "SUIUSDT", "INJUSDT", "UNIUSDT", "AAVEUSDT"]


def annual_harvest(sym, start, end):
    """{year: (sum_funding_frac, n_events, pct_positive)} for one symbol."""
    ev = S.fetch_funding(sym, start, end)
    by = defaultdict(list)
    for t, r in ev:
        by[datetime.utcfromtimestamp(t / 1000).year].append(r)
    return {y: (float(np.sum(v)), len(v), float(np.mean(np.array(v) > 0)))
            for y, v in by.items()}


def main():
    end = (int(time.time() * 1000) // S.DAY_MS) * S.DAY_MS
    start = end - YEARS * 365 * S.DAY_MS
    emit(f"delta-neutral basis harvest (long spot/short perp), "
         f"{datetime.utcfromtimestamp(start/1000).date()} .. "
         f"{datetime.utcfromtimestamp(end/1000).date()}")

    emit("\n=== BTC & ETH, harvest by year (gross % of notional) ===")
    emit(f"{'sym':<8}{'year':>6}{'harvest%':>10}{'ann%':>8}{'fund>0':>8}{'n':>6}")
    for sym in MAJORS:
        ah = annual_harvest(sym, start, end)
        for y in sorted(ah):
            tot, n, pos = ah[y]
            ann = tot / n * (365 * 3) if n else 0     # 3 funding/day → annualize
            emit(f"{sym:<8}{y:>6}{tot*100:>9.2f}%{ann*100:>7.1f}%{pos*100:>7.0f}%{n:>6}")

    emit("\n=== liquid ALT basket, harvest by year (mean across names w/ data) ===")
    emit(f"{'year':>6}{'mean_ann%':>11}{'median_ann%':>13}{'%names>0':>10}{'n_names':>9}")
    peryear = defaultdict(list)
    for sym in ALTS:
        ah = annual_harvest(sym, start, end)
        for y, (tot, n, pos) in ah.items():
            if n > 200:                                # enough data that year
                peryear[y].append(tot / n * (365 * 3) * 100)
    for y in sorted(peryear):
        v = np.array(peryear[y])
        emit(f"{y:>6}{v.mean():>10.1f}%{np.median(v):>12.1f}%"
             f"{(v>0).mean()*100:>9.0f}%{len(v):>9}")

    # Selective book: only harvest names with positive trailing funding — proxy
    # by taking, each year, the mean of the TOP HALF of names by that year's
    # harvest (what a positive-funding selector would roughly capture).
    emit("\n=== SELECTIVE alt basis (top-half names by yearly funding) ===")
    for y in sorted(peryear):
        v = np.sort(np.array(peryear[y]))[::-1]
        top = v[: max(1, len(v) // 2)]
        emit(f"  {y}: selective mean {top.mean():>+5.1f}%/yr  (vs basket {v.mean():>+5.1f}%)")
    emit("DONE")
    OUT.close()


if __name__ == "__main__":
    main()
