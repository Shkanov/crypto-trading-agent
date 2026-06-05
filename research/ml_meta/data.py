"""PIT panel builder — cache mainnet PUBLIC market history to parquet.

Read-only public data (klines, funding) — NO account, NO API key, NO orders.
Testnet has no usable price history, so training necessarily reads mainnet
public endpoints (same as the repo's existing CPCV scripts). Fetches are
conservatively paced and honor -1003 bans so this can never disturb the live
bot (which runs on a different IP anyway).

Cache layout: research/ml_meta/_cache/{symbol}_{interval}_klines.parquet
Re-runs load from cache; pass force=True to refetch.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import httpx
import pandas as pd

FAPI = "https://fapi.binance.com"
CACHE = Path("research/ml_meta/_cache")
_PACE_S = 0.25            # sleep between paginated calls (well under rate limits)
_KLINE_LIMIT = 1500

_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def _klines_to_df(rows: list[list]) -> pd.DataFrame:
    """Parse raw Binance kline arrays into a typed, time-indexed frame. Pure."""
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume",
                                     "quote_volume", "close_time"])
    df = pd.DataFrame(rows, columns=_KLINE_COLS)
    out = pd.DataFrame({
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
        "quote_volume": df["quote_volume"].astype(float),
        "close_time": df["close_time"].astype("int64"),
    })
    out.index = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    out.index.name = "open_time"
    # Drop the still-forming last bar if its close_time is in the future.
    return out[~out.index.duplicated(keep="last")].sort_index()


def _respect_ban(state: dict) -> None:
    wait = state.get("banned_until_ms", 0) - int(time.time() * 1000)
    if wait > 0:
        time.sleep(min(wait / 1000 + 1.0, 120.0))


def _get(client: httpx.Client, path: str, params: dict, state: dict) -> list:
    """One paced GET with -1003 backoff. Returns parsed JSON list."""
    for _ in range(5):
        _respect_ban(state)
        r = client.get(path, params=params)
        if r.status_code == 200:
            time.sleep(_PACE_S)
            return r.json()
        body = r.text
        m = re.search(r"banned until (\d+)", body)
        if m:
            state["banned_until_ms"] = int(m.group(1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"giving up on {path} {params}")


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Paginated public-futures klines over [start_ms, end_ms]. Read-only."""
    state: dict = {}
    out: list[list] = []
    cursor = start_ms
    with httpx.Client(base_url=FAPI, timeout=20.0) as client:
        while cursor < end_ms:
            rows = _get(client, "/fapi/v1/klines", {
                "symbol": symbol, "interval": interval,
                "startTime": cursor, "endTime": end_ms, "limit": _KLINE_LIMIT,
            }, state)
            if not rows:
                break
            out.extend(rows)
            last_open = int(rows[-1][0])
            if len(rows) < _KLINE_LIMIT or last_open <= cursor:
                break
            cursor = last_open + 1
    return _klines_to_df(out)


def load_or_fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int,
                         *, force: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{symbol}_{interval}_klines.parquet"
    if p.exists() and not force:
        df = pd.read_parquet(p)
        if not df.empty and df.index[0] <= pd.to_datetime(start_ms, unit="ms", utc=True):
            return df.loc[pd.to_datetime(start_ms, unit="ms", utc=True):
                          pd.to_datetime(end_ms, unit="ms", utc=True)]
    df = fetch_klines(symbol, interval, start_ms, end_ms)
    if not df.empty:
        df.to_parquet(p)
    return df


def build_panel(symbols: list[str], interval: str, start_ms: int, end_ms: int,
                *, force: bool = False) -> dict[str, pd.DataFrame]:
    """Per-symbol klines, cached. Returns {symbol: df}. Symbols that fail to
    fetch are logged to stdout and skipped (don't kill the whole build)."""
    panel: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = load_or_fetch_klines(sym, interval, start_ms, end_ms, force=force)
            if df.empty:
                print(f"  {sym}: empty — skipped")
                continue
            panel[sym] = df
            print(f"  {sym}: {len(df)} bars  [{df.index[0]} .. {df.index[-1]}]")
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: FETCH FAILED — {e}")
    return panel
