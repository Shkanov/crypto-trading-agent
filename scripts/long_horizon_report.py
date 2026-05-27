"""Render a markdown summary from a long_horizon_*.json file.

Usage:
  .venv/bin/python -m scripts.long_horizon_report data/research/long_horizon/long_horizon_*.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


STRATS = ("indicator", "meanrev", "funding")


def _fmt_strat_row(sym: str, st: dict) -> str:
    if "error" in st:
        return f"| {sym} | ERROR | — | — | — | — | — | — |"
    return (f"| {sym} | {st['trades']} | "
            f"${st['total_pnl_usd']:+.2f} | "
            f"{st['win_rate']*100:.1f}% | "
            f"{st['sharpe']:+.2f} | "
            f"{st['deflated_sharpe']:+.2f} | "
            f"{st['max_drawdown_pct']:.2f}% | "
            f"{st['annualized_pct']:+.1f}% |")


def _aggregate(strat: str, results: list[dict]) -> dict:
    ok = [r[strat] for r in results if strat in r and "error" not in r[strat]]
    if not ok:
        return {"strat": strat, "n": 0}
    n = len(ok)
    total = sum(s["total_pnl_usd"] for s in ok)
    n_pos = sum(1 for s in ok if s["total_pnl_usd"] > 0)
    n_trades = sum(s["trades"] for s in ok)
    avg_sh = sum(s["sharpe"] for s in ok) / n
    avg_dsh = sum(s["deflated_sharpe"] for s in ok) / n
    avg_ann = sum(s["annualized_pct"] for s in ok) / n
    best = max(ok, key=lambda s: s["total_pnl_usd"])
    worst = min(ok, key=lambda s: s["total_pnl_usd"])
    return {
        "strat": strat,
        "n": n,
        "n_pos": n_pos,
        "total_pnl": total,
        "n_trades": n_trades,
        "avg_sharpe": avg_sh,
        "avg_defl_sharpe": avg_dsh,
        "avg_annualized_pct": avg_ann,
        "best_symbol": best["symbol"] if "symbol" in best else "?",
        "best_pnl": best["total_pnl_usd"],
        "worst_symbol": worst["symbol"] if "symbol" in worst else "?",
        "worst_pnl": worst["total_pnl_usd"],
    }


def render(data: dict) -> str:
    results = data["results"]
    # Attach symbol into each strat dict for "best/worst" lookups
    for r in results:
        for st in STRATS:
            if st in r and "error" not in r[st]:
                r[st]["symbol"] = r["symbol"]

    days_approx = data.get("bars", 0) * 15 / 60 / 24
    out: list[str] = []
    out.append(f"# Long-horizon backtest")
    out.append("")
    out.append(f"- Generated: `{data.get('generated_at', '?')}`")
    out.append(f"- 15m bars: `{data.get('bars')}` (~{days_approx:.0f} days)")
    out.append(f"- Funding days: `{data.get('funding_days')}`")
    out.append(f"- Universe: `{len(data.get('universe', []))}` symbols")
    out.append("")
    out.append("## Universe aggregate")
    out.append("")
    out.append("| strategy | symbols | +pnl/n | total_pnl | trades | avg_sharpe | avg_defl | avg_annual | best | worst |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for st in STRATS:
        a = _aggregate(st, results)
        if a["n"] == 0:
            out.append(f"| {st} | 0 | — | — | — | — | — | — | — | — |")
            continue
        out.append(
            f"| {st} | {a['n']} | {a['n_pos']}/{a['n']} | "
            f"${a['total_pnl']:+,.2f} | {a['n_trades']} | "
            f"{a['avg_sharpe']:+.2f} | {a['avg_defl_sharpe']:+.2f} | "
            f"{a['avg_annualized_pct']:+.1f}% | "
            f"{a['best_symbol']} (${a['best_pnl']:+.2f}) | "
            f"{a['worst_symbol']} (${a['worst_pnl']:+.2f}) |"
        )

    for st in STRATS:
        out.append("")
        out.append(f"## {st}")
        out.append("")
        out.append("| symbol | trades | pnl | win_rate | sharpe | defl_sharpe | max_dd% | annualized% |")
        out.append("|---|---|---|---|---|---|---|---|")
        # Sort by pnl desc, errors at bottom
        sortable = [
            (r["symbol"], r.get(st, {})) for r in results
        ]
        sortable.sort(
            key=lambda x: x[1].get("total_pnl_usd", float("-inf"))
            if "error" not in x[1] else float("-inf") - 1
        )
        sortable.reverse()
        for sym, stats in sortable:
            out.append(_fmt_strat_row(sym, stats))

    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to long_horizon_*.json (glob supported)")
    ap.add_argument("--out", default=None,
                    help="Write markdown here (default: alongside JSON with .md)")
    args = ap.parse_args()

    matches = sorted(glob.glob(args.path))
    if not matches:
        raise SystemExit(f"no matches for {args.path}")
    json_path = Path(matches[-1])
    data = json.loads(json_path.read_text())
    md = render(data)
    out_path = Path(args.out) if args.out else json_path.with_suffix(".md")
    out_path.write_text(md)
    print(md)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
