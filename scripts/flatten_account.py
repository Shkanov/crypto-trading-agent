"""One-off: flatten the futures account — cancel all open orders and
market-close every non-zero position (reduce-only). Honors BINANCE_TESTNET.

  python -m scripts.flatten_account            # dry-run, just lists
  python -m scripts.flatten_account --execute  # actually closes

Safe to run against testnet to clear orphaned positions.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from decimal import Decimal

from dotenv import load_dotenv

from src.tools.binance_client import BinanceClient


async def main(execute: bool) -> None:
    load_dotenv()
    client = BinanceClient()
    await client.start()
    net = "TESTNET" if client.testnet else "MAINNET"
    print(f"=== flatten account on {net} (execute={execute}) ===")

    positions = await client.futures_positions()
    open_pos = [p for p in positions if abs(float(p.get("positionAmt", 0))) > 0]

    if not open_pos:
        print("No open positions. Account already flat.")
        await client.close()
        return

    print(f"Found {len(open_pos)} open position(s):")
    for p in open_pos:
        amt = float(p["positionAmt"])
        side = "LONG" if amt > 0 else "SHORT"
        print(f"  {p['symbol']:<14} {side:<5} qty={amt:<14} entry={p.get('entryPrice')} uPnL={p.get('unRealizedProfit')}")

    if not execute:
        print("\nDry-run. Re-run with --execute to close these.")
        await client.close()
        return

    print("\nClosing...")
    for p in open_pos:
        sym = p["symbol"]
        amt = float(p["positionAmt"])
        close_side = "SELL" if amt > 0 else "BUY"
        qty = Decimal(str(abs(amt)))
        # Cancel any resting stop/TP orders first so reduceOnly close isn't rejected.
        try:
            await client.client.futures_cancel_all_open_orders(symbol=sym)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: cancel-orders warning: {e}")
        try:
            coid = f"flat-{sym[:8]}-{int(time.time()*1000) % 10_000_000}"
            res = await client.place_perp_market(sym, close_side, qty, coid, reduce_only=True)
            print(f"  {sym}: {close_side} {qty} reduceOnly → orderId={res.get('orderId')} status={res.get('status')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: CLOSE FAILED: {e}")

    # Verify
    await asyncio.sleep(2)
    positions = await client.futures_positions()
    still = [p for p in positions if abs(float(p.get("positionAmt", 0))) > 0]
    print(f"\n=== after: {len(still)} open position(s) remaining ===")
    for p in still:
        print(f"  {p['symbol']} qty={p['positionAmt']}")
    if not still:
        print("Account is FLAT. ✓")
    await client.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="actually close positions")
    args = ap.parse_args()
    asyncio.run(main(args.execute))
