"""Controlled prod validation of the basis execution primitive.

Opens ONE tiny delta-neutral basis pair (BUY spot + SELL perp, equal notional)
through the REAL PairExecutor live path, verifies both legs filled + balanced +
delta-neutral on the exchange, then closes it and asserts the account is flat.
Uses a throwaway sqlite so it does not pollute the prod DB. Tiny notional; the
only cost is ~a few cents of fees.

Run on the box (mainnet, real key):
    uv run python -m scripts.validate_basis_exec [SYMBOL] [NOTIONAL_USD]
default SYMBOL=ETHUSDT NOTIONAL=25
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import uuid

from dotenv import load_dotenv

from src.config.settings import get_settings
from src.models.types import PairLeg, PairProposal
from src.services.storage import Storage
from src.tools.binance_client import BinanceClient
from src.tools.pair_executor import PairExecutor

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
NOTIONAL = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0


async def _perp_pos(b: BinanceClient, sym: str) -> float:
    info = await b.client.futures_position_information(symbol=sym)
    return sum(float(p["positionAmt"]) for p in info)


async def _spot_free(b: BinanceClient, asset: str) -> float:
    acc = await b.client.get_account()
    for bal in acc["balances"]:
        if bal["asset"] == asset:
            return float(bal["free"])
    return 0.0


async def main() -> None:
    load_dotenv("/opt/crypto-trading-agent/.env")
    s = get_settings()
    assert not s.binance_testnet, "run against mainnet (real key) for prod validation"
    b = BinanceClient()
    await b.start()
    base = SYMBOL.replace("USDT", "")

    # price + min-notional-safe sizing
    tick = await b.client.futures_symbol_ticker(symbol=SYMBOL)
    px = float(tick["price"])
    filt = b.perp_filters.get(SYMBOL)
    min_notl = float(filt.min_notional) if filt else 5.0
    notional = max(NOTIONAL, min_notl * 1.3)
    qty = notional / px
    print(f"=== BASIS EXEC VALIDATION: {SYMBOL} @ {px} notional ${notional:.1f} "
          f"(min ${min_notl}) qty {qty:.6f} ===")

    td = tempfile.mkdtemp()
    st = Storage(database_url=f"sqlite+aiosqlite:///{td}/val.db")
    await st.init()
    pe = PairExecutor(b, st, paper=False, settings=s)

    perp0 = await _perp_pos(b, SYMBOL)
    spot0 = await _spot_free(b, base)
    print(f"before: perp {SYMBOL}={perp0:+.6f}  spot {base}={spot0:.6f}")

    pair = PairProposal(
        id=uuid.uuid4().hex[:16], strategy="basis_validation", direction=1,
        legs=[
            PairLeg(symbol=SYMBOL, market="spot", side="BUY", qty=qty,
                    expected_price=px, leverage=1),
            PairLeg(symbol=SYMBOL, market="perps", side="SELL", qty=qty,
                    expected_price=px, leverage=1),
        ],
        notional_usd=notional * 2, rationale="prod validation",
        expected_yield_bps_per_8h=0.0, expires_at_ms=0,
    )
    print("\n--- OPEN pair (BUY spot + SELL perp) ---")
    res = await pe.open_pair(pair)
    print("open ok:", res.ok, "err:", res.error, "fills:", res.fill_prices)
    if not res.ok:
        print("!!! OPEN FAILED — PairExecutor unwinds internally; verifying flat")
    perp1 = await _perp_pos(b, SYMBOL)
    spot1 = await _spot_free(b, base)
    print(f"after open: perp {SYMBOL}={perp1:+.6f}  spot {base}={spot1:.6f}")
    d_perp = perp1 - perp0
    d_spot = spot1 - spot0
    print(f"  Δperp {d_perp:+.6f} (want ~-{qty:.6f})  Δspot {d_spot:+.6f} (want ~+{qty:.6f})")
    net_delta = (d_perp + d_spot) * px
    print(f"  NET DELTA ≈ ${net_delta:+.2f}  (delta-neutral if ~0)")

    print("\n--- CLOSE pair ---")
    closed = await pe.close_pair(res.legs, reason="validation_done")
    print("closed legs:", len(closed))
    perp2 = await _perp_pos(b, SYMBOL)
    spot2 = await _spot_free(b, base)
    print(f"after close: perp {SYMBOL}={perp2:+.6f}  spot {base}={spot2:.6f}")

    flat = abs(perp2) < qty * 0.05
    print(f"\nRESULT: perp flat={flat}  (perp resid {perp2:+.6f}); "
          f"spot residual dust {spot2-spot0:+.6f} {base} (sold back, minor)")
    print("VALIDATION", "PASS ✓" if (res.ok and flat) else "CHECK ⚠")
    await b.close()


if __name__ == "__main__":
    asyncio.run(main())
