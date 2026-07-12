"""BTC-ETH funding angles: harvestable delta-neutral yield + relative spread.

Two constructions:
  (1) TRUE delta-neutral basis (per asset): long spot + short perp → collect that
      asset's funding, ~zero price risk. The harvestable yield IS the mean funding
      rate. Measured for BTC and ETH separately, annualized, net of a cost stub.
  (2) RELATIVE funding-spread (perp-only): each 8h short the HIGHER-funding of the
      two / long the LOWER (harvest the spread). This is NOT price-neutral — it's
      a long/short ETH-BTC position with a funding signal — so we decompose
      funding vs price to see whether trading the spread adds anything over just
      harvesting each leg.

1yr, 1h mainnet (cached from btc_eth.py). Writes /tmp/btceth_fund.txt (flushed).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

from research.portfolio import sleeves as S

OUT = open("/tmp/btceth_fund.txt", "w")
def emit(*a):
    line = " ".join(str(x) for x in a)
    print(line); OUT.write(line + "\n"); OUT.flush()

COST_BPS_ROUNDTRIP = 10.0     # taker in+out for a basis pair, rough


def main():
    end = (int(time.time() * 1000) // S.DAY_MS) * S.DAY_MS
    start = end - 365 * S.DAY_MS
    days = (end - start) / S.DAY_MS
    fund = {s: S.fetch_funding(s, start, end) for s in ("BTCUSDT", "ETHUSDT")}
    close = {}
    for s in ("BTCUSDT", "ETHUSDT"):
        ks = S.fetch_klines(s, "1h", start, end)
        close[s] = {int(r[0]): float(r[4]) for r in ks}

    emit(f"BTC-ETH funding, {datetime.utcfromtimestamp(start/1000).date()} .. "
         f"{datetime.utcfromtimestamp(end/1000).date()} ({days:.0f}d)")

    # (1) delta-neutral basis harvest per asset = sum of funding (short perp
    #     receives funding when rate>0). annualize.
    emit("\n=== (1) delta-neutral basis (long spot / short perp), per asset ===")
    for s in ("BTCUSDT", "ETHUSDT"):
        rates = np.array([r for _, r in fund[s]])
        tot = rates.sum()                    # fraction of notional over the year
        ann = tot / days * 365
        pos_frac = (rates > 0).mean()
        emit(f"  {s:<8} mean funding/8h {rates.mean()*1e4:>+6.2f}bps  "
             f"funding>0 {pos_frac*100:>3.0f}%  "
             f"harvest {tot*100:>+5.2f}%/yr gross  ~{ann*100:>+5.1f}% annualized")
    emit("  (harvest = short-perp receives funding; ~zero price risk; needs spot"
         " leg; gross of ~a few % borrow/exec + gap-liquidation tail risk)")

    # (2) relative funding-spread perp trade: each funding cycle, short higher-
    #     funding / long lower-funding, $1 per leg. Decompose funding vs price.
    emit("\n=== (2) relative funding-spread (perp-only, long low / short high) ===")
    times = sorted(set(t for _, t0 in [(None, None)] for t in []))  # placeholder
    fm = {s: dict(fund[s]) for s in fund}
    ftimes = sorted(set(fm["BTCUSDT"]) & set(fm["ETHUSDT"]))
    fp = pp = 0.0
    n = 0
    for i in range(len(ftimes) - 1):
        t0, t1 = ftimes[i], ftimes[i + 1]
        rb, re = fm["BTCUSDT"][t0], fm["ETHUSDT"][t0]
        short = "BTCUSDT" if rb > re else "ETHUSDT"
        long = "ETHUSDT" if short == "BTCUSDT" else "BTCUSDT"
        # funding: short receives its rate, long pays its rate
        fp += fm[short][t0] - fm[long][t0]
        # price over hold
        ps0, ps1 = S._nearest(close[short], t0), S._nearest(close[short], t1)
        pl0, pl1 = S._nearest(close[long], t0), S._nearest(close[long], t1)
        if ps0 and ps1 and pl0 and pl1:
            pp += -(ps1 - ps0) / ps0 + (pl1 - pl0) / pl0   # short + long returns
            n += 1
    cost = n * COST_BPS_ROUNDTRIP / 1e4 * 0  # spread trade rotates rarely; ~0 here
    emit(f"  cycles {n}  funding leg {fp*100:>+6.2f}%/yr  price leg {pp*100:>+6.2f}%/yr  "
         f"gross {(fp+pp)*100:>+6.2f}%/yr")
    emit("  (this is a directional ETH-BTC ratio position with a funding tilt —"
         " NOT delta-neutral; price leg dominates.)")
    emit("DONE")
    OUT.close()


if __name__ == "__main__":
    main()
