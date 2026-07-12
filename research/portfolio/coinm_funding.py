"""Quantify COIN-M (coin-margined, inverse) perp funding vs USDT-M.

Coin-margined perps (BTCUSD_PERP, ETHUSD_PERP, ...) trade on Binance's dapi.
This pulls their funding history and annualizes it by year, so we can compare
the coin-margined basis yield against the USDT-M basis we already measured.

Funding rate is 8h cadence on both; the annualized % is directly comparable as
"how much funding does the short-perp basis harvest" — the difference being
COIN-M pays it IN THE COIN, USDT-M pays it in USDT.

Writes /tmp/coinm.txt (flushed).
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

CACHE = Path("/tmp/pf_cache"); CACHE.mkdir(exist_ok=True)
OUT = open("/tmp/coinm.txt", "w")
def emit(*a):
    line = " ".join(str(x) for x in a); print(line); OUT.write(line + "\n"); OUT.flush()

# COIN-M perp symbols (inverse). Try the basket majors that have coin-margined
# contracts; dapi uses <COIN>USD_PERP.
COINM = ["BTCUSD_PERP", "ETHUSD_PERP", "LINKUSD_PERP", "LTCUSD_PERP",
         "DOGEUSD_PERP", "UNIUSD_PERP", "AAVEUSD_PERP", "BNBUSD_PERP", "XRPUSD_PERP"]

# USDT-M annual harvest already measured (basis_longrun.py), for side-by-side.
USDTM = {
    "BTC": {2023: 8.6, 2024: 11.9, 2025: 5.1, 2026: 1.5},
    "ETH": {2023: 9.3, 2024: 13.0, 2025: 4.9, 2026: 0.7},
    "LINK": {2023: 11.2, 2024: 13.3, 2025: 5.1, 2026: 3.9},
    "LTC": {2023: 10.1, 2024: 14.4, 2025: 5.6, 2026: 2.0},
    "DOGE": {2023: 11.1, 2024: 14.0, 2025: 4.3, 2026: 1.5},
    "UNI": {2023: 10.9, 2024: 13.5, 2025: 6.3, 2026: 3.2},
    "AAVE": {2023: 10.2, 2024: 11.0, 2025: 5.3, 2026: 1.8},
}


def fetch_coinm_funding(sym: str, start_ms: int, end_ms: int):
    p = CACHE / f"cm_{sym}_{start_ms}_{end_ms}.json"
    if p.exists():
        return json.loads(p.read_text())
    out = []
    cur = start_ms
    with httpx.Client(base_url="https://dapi.binance.com", timeout=25.0) as cl:
        while cur < end_ms:
            r = cl.get("/dapi/v1/fundingRate", params={
                "symbol": sym, "startTime": cur, "endTime": end_ms, "limit": 1000})
            if r.status_code != 200:
                break
            pg = r.json()
            if not pg:
                break
            for row in pg:
                out.append((int(row["fundingTime"]), float(row["fundingRate"])))
            last = int(pg[-1]["fundingTime"])
            if len(pg) < 1000 or last <= cur:
                break
            cur = last + 1
    p.write_text(json.dumps(out))
    return out


def main():
    end = int(datetime(2026, 7, 12, tzinfo=timezone.utc).timestamp() * 1000)
    start = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    emit("COIN-M funding harvest by year (annualized %, short-perp basis) — "
         "paid IN THE COIN\n")
    emit(f"{'symbol':<14}{'2023':>7}{'2024':>7}{'2025':>7}{'2026':>7}"
         f"   vs USDT-M 2026")
    for sym in COINM:
        ev = fetch_coinm_funding(sym, start, end)
        if not ev:
            emit(f"{sym:<14}  (no coin-margined contract / no data)")
            continue
        by = defaultdict(list)
        for t, r in ev:
            by[datetime.utcfromtimestamp(t / 1000).year].append(r)
        cells = []
        for y in (2023, 2024, 2025, 2026):
            v = by.get(y)
            cells.append(np.mean(v) * 3 * 365 * 100 if v and len(v) > 100 else None)
        base = sym.replace("USD_PERP", "")
        um = USDTM.get(base, {}).get(2026)
        row = "".join(f"{c:>7.1f}" if c is not None else f"{'n/a':>7}" for c in cells)
        cm26 = cells[3]
        cmp = ""
        if cm26 is not None and um is not None:
            cmp = f"   COIN-M {cm26:+.1f} vs USDT-M {um:+.1f}  (Δ {cm26-um:+.1f})"
        emit(f"{sym:<14}{row}{cmp}")
    emit("\nNote: COIN-M funding accrues in the COIN (BTC/ETH), not USDT. A higher")
    emit("COIN-M rate = more coin harvested, but valued in a floating (coin) numeraire.")
    emit("DONE")
    OUT.close()


if __name__ == "__main__":
    main()
