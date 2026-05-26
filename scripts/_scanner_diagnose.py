"""Diagnose WHY the scanner missed each call. Reuses the disk cache so it's
instant after the first validate run."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.models.types import Kline
from src.scanners.aktradescalp_scanner import (
    ScannerParams,
    SymbolFeatures,
    SymbolHistory,
    UniverseParams,
    compute_features,
    passes_universe,
    score_universe,
)

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data/research/aktradescalp/scanner_cache"
CALLS = REPO / "data/research/aktradescalp/aktradescalp_calls.json"


def _load_history(symbol: str, onboards: dict[str, int]) -> SymbolHistory | None:
    kp = CACHE / f"klines_1h_{symbol}.json"
    if not kp.exists():
        return None
    kl_rows = json.loads(kp.read_text())
    if len(kl_rows) < 30 * 24:
        return None
    klines = [Kline(
        symbol=symbol, timeframe="1h",
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in kl_rows]
    fr = json.loads((CACHE / f"funding_{symbol}.json").read_text()) if (CACHE / f"funding_{symbol}.json").exists() else []
    oi = json.loads((CACHE / f"oi_{symbol}.json").read_text()) if (CACHE / f"oi_{symbol}.json").exists() else []
    return SymbolHistory(
        symbol=symbol, klines_1h=klines,
        funding_rates=[(int(r["fundingTime"]), float(r["fundingRate"])) for r in fr],
        oi_history=[(int(r["timestamp"]), float(r["sumOpenInterest"])) for r in oi],
        listing_date_ms=onboards.get(symbol, 0),
    )


def main():
    calls = json.load(open(CALLS))
    calls = [c for c in calls if c["side"] in ("long", "short")]
    onboards = json.loads((CACHE / "exchange_info.json").read_text())
    onboards = {k: int(v) for k, v in onboards.items()}

    u = UniverseParams()
    s = ScannerParams()

    # Load all available histories
    all_syms = [p.stem.replace("klines_1h_", "") for p in CACHE.glob("klines_1h_*.json")]
    print(f"loaded {len(all_syms)} cached symbols")
    histories: dict[str, SymbolHistory] = {}
    for sym in all_syms:
        h = _load_history(sym, onboards)
        if h:
            histories[sym] = h
    print(f"valid histories: {len(histories)}")

    print(f"\n{'msg':>4s}  {'symbol':12s} {'side':5s}  "
          f"{'in_univ?':9s}  {'vol_z':>6s}  {'oi_z':>6s}  {'ret_bp':>8s}  "
          f"{'fund_pct':>9s}  {'rank/N':>10s}  {'L|S':>5s}  hist")

    in_univ = 0
    universe_n_at_call = []
    score_pairs = []

    for c in calls:
        ts = int(datetime.fromisoformat(c["dt_iso"]).timestamp() * 1000)
        sym = c["symbol"]
        side = c["side"]
        h = histories.get(sym)
        if h is None:
            print(f"{c['msg_id']:>4d}  {sym:12s} {side:5s}  NO_HISTORY")
            continue
        f = compute_features(h, ts)
        if not f.history_ok:
            print(f"{c['msg_id']:>4d}  {sym:12s} {side:5s}  HIST_TOO_SHORT  "
                  f"(n_bars≤{ts}≤={sum(1 for k in h.klines_1h if k.close_time<=ts)})")
            continue

        ok = passes_universe(f, u)
        if ok:
            in_univ += 1

        # cross-sectional rank at ts
        all_feats = {s: compute_features(hh, ts) for s, hh in histories.items()}
        eligibles = {s: ff for s, ff in all_feats.items() if passes_universe(ff, u)}
        rets = sorted([(s2, ff.ret_24h_bps) for s2, ff in eligibles.items()
                       if ff.ret_24h_bps is not None], key=lambda x: x[1])
        n = len(rets)
        bot = {s2 for s2, _ in rets[:s.rank_topn]}
        top = {s2 for s2, _ in rets[max(0, n - s.rank_topn):]}
        rank_label = "-"
        if sym in [s2 for s2, _ in rets]:
            idx = [s2 for s2, _ in rets].index(sym) + 1
            rank_label = f"{idx}/{n}"
        universe_n_at_call.append(n)

        in_top = sym in top
        in_bot = sym in bot

        vol_hit = f.vol_z_1h_sameHour_30d is not None and f.vol_z_1h_sameHour_30d >= s.vol_z_min
        oi_hit = f.oi_z_24h_30d is not None and f.oi_z_24h_30d >= s.oi_z_min
        rank_hit = in_top or in_bot
        fund = f.funding_rate_8h
        fund_hit = fund is not None and (fund > s.funding_short_bias or fund < s.funding_long_bias)
        attention = vol_hit + oi_hit + rank_hit + fund_hit
        # Also compute his rank in the scanner output
        ranked = score_universe(all_feats, ts, u, s)
        scanner_rank = next((i + 1 for i, cand in enumerate(ranked)
                             if cand.symbol == sym), None)
        score_pairs.append((c["msg_id"], side, attention, scanner_rank, ok))

        vol_z_str = f"{f.vol_z_1h_sameHour_30d:+.2f}" if f.vol_z_1h_sameHour_30d is not None else "  -  "
        oi_z_str = f"{f.oi_z_24h_30d:+.2f}" if f.oi_z_24h_30d is not None else "  -  "
        ret_str = f"{f.ret_24h_bps:+.0f}" if f.ret_24h_bps is not None else "  -  "
        fund_str = f"{f.funding_rate_8h*100:+.4f}" if f.funding_rate_8h is not None else "  -  "
        score_str = f"att={attention}"
        sc_rank_str = f"scan#{scanner_rank}" if scanner_rank else "  -  "
        vol_m = (f.quote_vol_24h_usd or 0) / 1e6
        age_d = f.days_since_listing if f.days_since_listing is not None else -1
        univ_str = "YES" if ok else f"NO(${vol_m:.0f}M,{age_d:.0f}d)"
        print(f"{c['msg_id']:>4d}  {sym:12s} {side:5s}  {univ_str:18s}  "
              f"{vol_z_str:>6s}  {oi_z_str:>6s}  {ret_str:>8s}  {fund_str:>9s}  "
              f"{rank_label:>10s}  {score_str:>6s}  {sc_rank_str}")

    print(f"\nSummary:")
    print(f"  in_universe:       {in_univ}/{len(calls)}")
    print(f"  avg universe size: {sum(universe_n_at_call)/len(universe_n_at_call):.1f}"
          if universe_n_at_call else "  no data")
    in_universe_scores = [(att, rank) for _, _, att, rank, ok in score_pairs if ok]
    if in_universe_scores:
        print(f"  attention-score histogram (in-universe, n={len(in_universe_scores)}):")
        from collections import Counter
        for k, v in sorted(Counter(att for att, _ in in_universe_scores).items()):
            print(f"    score {k}: {v}")
        print(f"  gate s.score_min: {s.score_min}")
        ranks = [r for _, r in in_universe_scores if r is not None]
        print(f"  scanner rank distribution (his symbol's rank in scanner output):")
        for cutoff in (1, 3, 5, 10):
            hit = sum(1 for r in ranks if r <= cutoff)
            print(f"    rank≤{cutoff:<3d}  {hit}/{len(in_universe_scores)}  "
                  f"({100*hit/len(in_universe_scores):.1f}%)")


if __name__ == "__main__":
    main()
