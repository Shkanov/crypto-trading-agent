"""Point-in-time (PIT) universe filter for survivorship-bias-corrected backtests.

Ammann/Burdorf/Liebi/Stöckl (SSRN 4287573) measured equal-weighted survivorship
bias on crypto at **~62% per year** — a "buy top-20 by mcap monthly" backtest
on TODAY's universe can overstate returns by 4× because it implicitly drops
every name that delisted along the way.

Our cross-sectional momentum overlay (sprint #15/#16) and any equal-weighted
universe sweep need to query "which symbols were live at timestamp T?", not
"which are live today?". This module is the lookup layer over a hand-grown +
auto-built `data/research/universe/binance_delistings.json` of shape:

    {
      "BTCUSDT":  {"listed_ms": 1502942400000, "delisted_ms": null},
      "BCCUSDT":  {"listed_ms": 1500292800000, "delisted_ms": 1573257600000},
      ...
    }

The file is gitignored (lives under `data/`); the builder script
`scripts/build_pit_universe.py` populates `listed_ms` for currently-listed
symbols from the kline API. Delisted-symbol entries are manually appended
(or auto-detected if a builder run no longer sees a symbol in exchangeInfo).

This module is pure logic — no I/O, no Binance calls. `load_pit_log` reads
the JSON, every other helper takes the loaded dict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class SymbolListing:
    symbol: str
    listed_ms: int                  # first observable kline close_time
    delisted_ms: Optional[int]      # None when symbol is still trading


def load_pit_log(path: Path) -> dict[str, SymbolListing]:
    """Load `binance_delistings.json` → {symbol: SymbolListing}.

    The on-disk schema is `{symbol: {listed_ms, delisted_ms}}` with
    `delisted_ms` either an int or null. Returns an empty dict if the
    file is missing — callers should treat that as "PIT not built yet."
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, SymbolListing] = {}
    for sym, v in raw.items():
        listed = v.get("listed_ms")
        if listed is None:
            continue
        delisted = v.get("delisted_ms")
        out[sym] = SymbolListing(
            symbol=sym,
            listed_ms=int(listed),
            delisted_ms=int(delisted) if delisted is not None else None,
        )
    return out


def is_active_at(log: dict[str, SymbolListing], symbol: str, ts_ms: int) -> bool:
    """True iff `symbol` was tradeable at `ts_ms`.

    A symbol is active in `[listed_ms, delisted_ms)`. A symbol absent from
    the log is treated as never-active (so callers querying "should I
    include this in a historical universe?" get a conservative no).
    """
    entry = log.get(symbol)
    if entry is None:
        return False
    if ts_ms < entry.listed_ms:
        return False
    if entry.delisted_ms is not None and ts_ms >= entry.delisted_ms:
        return False
    return True


def eligible_universe_at(
    log: dict[str, SymbolListing],
    ts_ms: int,
    subset: Optional[Iterable[str]] = None,
) -> list[str]:
    """All symbols active at `ts_ms`, sorted. Pass `subset` to restrict the
    candidate set (e.g. only USDT pairs)."""
    candidates: Iterable[str] = subset if subset is not None else log.keys()
    return sorted(s for s in candidates if is_active_at(log, s, ts_ms))


def filter_universe_for_span(
    log: dict[str, SymbolListing],
    start_ms: int,
    end_ms: int,
    candidates: Iterable[str],
    min_coverage: float = 1.0,
) -> list[str]:
    """Symbols whose [listed, delisted) window covers at least `min_coverage`
    fraction of `[start_ms, end_ms]`.

    `min_coverage=1.0` keeps only symbols that were continuously live across
    the whole backtest window — the standard "no synthetic survivorship"
    filter. Lowering it admits symbols that were partly live, which is what
    a properly PIT-aware backtest wants (positions auto-exit at delisting).

    The caller decides how to handle non-fully-covered symbols in the
    simulator. This function just hands back the candidate list with
    coverage metadata baked into the keep/drop decision.
    """
    if end_ms <= start_ms:
        return []
    span = float(end_ms - start_ms)
    out: list[str] = []
    for s in candidates:
        entry = log.get(s)
        if entry is None:
            continue
        live_start = max(start_ms, entry.listed_ms)
        live_end = min(end_ms, entry.delisted_ms if entry.delisted_ms is not None else end_ms)
        if live_end <= live_start:
            continue
        coverage = (live_end - live_start) / span
        if coverage + 1e-9 >= min_coverage:
            out.append(s)
    return sorted(out)


def coverage_fraction(
    log: dict[str, SymbolListing],
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> float:
    """Fraction of [start_ms, end_ms] during which `symbol` was live. 0 if
    not in log or no overlap; 1 if continuously live across the window."""
    entry = log.get(symbol)
    if entry is None or end_ms <= start_ms:
        return 0.0
    live_start = max(start_ms, entry.listed_ms)
    live_end = min(end_ms, entry.delisted_ms if entry.delisted_ms is not None else end_ms)
    if live_end <= live_start:
        return 0.0
    return (live_end - live_start) / float(end_ms - start_ms)


def universe_size_over_time(
    log: dict[str, SymbolListing],
    start_ms: int,
    end_ms: int,
    step_ms: int,
    subset: Optional[Iterable[str]] = None,
) -> list[tuple[int, int]]:
    """Walk forward in `step_ms` increments and count active symbols at each
    step. Useful for plotting universe drift over time and confirming the
    PIT log is dense enough to be meaningful.

    Returns a list of `(ts_ms, n_active)`.
    """
    out: list[tuple[int, int]] = []
    if end_ms <= start_ms or step_ms <= 0:
        return out
    syms = list(subset) if subset is not None else list(log.keys())
    t = start_ms
    while t <= end_ms:
        n = sum(1 for s in syms if is_active_at(log, s, t))
        out.append((t, n))
        t += step_ms
    return out
