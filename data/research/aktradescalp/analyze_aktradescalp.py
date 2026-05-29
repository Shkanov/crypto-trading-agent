"""Aggregate the parsed @aktradescalp corpus and probe discretion hypotheses.

Output: text report to stdout + a derived JSONL of "trade calls" for
downstream replay-backtesting.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN_PATH = HERE / "aktradescalp_messages.json"
CALLS_OUT = HERE / "aktradescalp_calls.json"

# Message-class heuristics. The denominator we care about is "entries" —
# messages where the author actually announces a trade (ticker + direction +
# (TF or setup-tag)). Recap / commentary / scratch / meta messages don't
# count as setups.
TICKER_HINT = re.compile(r"[A-Z0-9]{2,15}USDT")
DIR_HINT = re.compile(r"\b(лонг|шорт|long|short)\b", re.IGNORECASE)
SCRATCH_HINT = re.compile(
    r"\b(скидыва\w+|перемудри\w+|без меня|не успева\w+|сложн\w+ рынок|не понимаю|"
    r"пляшем дальше|без позиций|отлучился)\b",
    re.IGNORECASE,
)
RECAP_HINT = re.compile(
    r"\b(удалось|пособирать|собирать|итог\w*|неплох\w*\s+начал\w*|давал\s+\d|"
    r"был\s+перебор|конец\s+дня|рыноч\w+\s+сегодня|без меня)\b",
    re.IGNORECASE,
)


def classify(text: str) -> str:
    has_ticker = bool(TICKER_HINT.search(text))
    has_dir = bool(DIR_HINT.search(text))
    has_recap = bool(RECAP_HINT.search(text))
    has_scratch = bool(SCRATCH_HINT.search(text))

    if has_recap and not has_dir:
        return "recap"
    if has_scratch and not has_dir:
        return "scratch"
    if has_ticker and has_dir:
        return "entry"
    if has_ticker and not has_dir:
        # "стоим перед дневкой" / "монета в пике" — pre-trade anchoring or
        # observation. Count separately.
        return "anchor"
    if not has_ticker and not has_dir:
        return "meta"
    return "other"


def main() -> None:
    data: list[dict] = json.load(open(IN_PATH))
    # Annotate class
    for m in data:
        m["msg_class"] = classify(m["text"])

    n = len(data)
    classes = Counter(m["msg_class"] for m in data)
    print(f"=== corpus: {n} messages over "
          f"{data[0]['dt_iso'][:10]} → {data[-1]['dt_iso'][:10]} ===\n")
    print("message classes:")
    for k, v in classes.most_common():
        print(f"  {k:8s} {v:3d}  ({v / n * 100:4.1f}%)")
    print()

    # ── Entries: the real signal set ──────────────────────────────────────
    entries = [m for m in data if m["msg_class"] == "entry"]
    print(f"=== ENTRY MESSAGES: {len(entries)} ===\n")

    # Side distribution
    sides = Counter(m["side"] for m in entries)
    print("side distribution (long/short bias):")
    for k, v in sides.most_common():
        if not k:
            continue
        print(f"  {k:6s} {v:3d}  ({v / len(entries) * 100:4.1f}%)")
    print()

    # TF tag distribution
    tf_setups: Counter[str] = Counter()
    for m in entries:
        key = "+".join(m["tf_tags"]) if m["tf_tags"] else "(no_tf)"
        tf_setups[key] += 1
    print("TF tags on entries:")
    for k, v in tf_setups.most_common():
        print(f"  {k:10s} {v:3d}  ({v / len(entries) * 100:4.1f}%)")
    print()

    # Setup (breakout / trendline / mixed)
    setups = Counter(m["setup"] or "(none)" for m in entries)
    print("setup tag on entries:")
    for k, v in setups.most_common():
        print(f"  {k:20s} {v:3d}  ({v / len(entries) * 100:4.1f}%)")
    print()

    # Symbol concentration
    sym_counter: Counter[str] = Counter()
    for m in entries:
        for t in m["tickers"]:
            sym_counter[t] += 1
    unique_syms = len(sym_counter)
    print(f"symbol universe: {unique_syms} unique symbols across "
          f"{sum(sym_counter.values())} ticker-mentions in entries")
    print("top 15 symbols by mention count:")
    for sym, cnt in sym_counter.most_common(15):
        print(f"  {sym:14s} {cnt:3d}")
    print()

    # Time-of-day distribution (UTC; Moscow = UTC+3, so add 3 for local feel)
    hour_bins_utc: Counter[int] = Counter()
    hour_bins_msk: Counter[int] = Counter()
    weekday_counter: Counter[str] = Counter()
    for m in entries:
        dt = datetime.fromisoformat(m["dt_iso"])
        hour_bins_utc[dt.hour] += 1
        hour_bins_msk[(dt + timedelta(hours=3)).hour] += 1
        weekday_counter[dt.strftime("%a")] += 1
    print("entries by hour of day (Moscow time, UTC+3):")
    for h in sorted(hour_bins_msk):
        bar = "█" * hour_bins_msk[h]
        print(f"  {h:02d}h  {hour_bins_msk[h]:3d}  {bar}")
    print()
    print("entries by weekday:")
    for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        if d in weekday_counter:
            print(f"  {d}  {weekday_counter[d]:3d}")
    print()

    # ── Activity per day: how many calls/day, gap days, scratch rate ──────
    per_day: dict[str, dict] = defaultdict(
        lambda: {"entry": 0, "scratch": 0, "recap": 0, "anchor": 0, "meta": 0,
                 "other": 0})
    for m in data:
        d = m["dt_iso"][:10]
        per_day[d][m["msg_class"]] += 1
    n_days_with_entry = sum(1 for d in per_day.values() if d["entry"] > 0)
    n_days_with_scratch = sum(1 for d in per_day.values() if d["scratch"] > 0)
    n_days_total = (datetime.fromisoformat(data[-1]["dt_iso"]).date()
                    - datetime.fromisoformat(data[0]["dt_iso"]).date()).days + 1
    print(f"calendar span: {n_days_total} days")
    print(f"  days with ≥1 entry:    {n_days_with_entry}  "
          f"({n_days_with_entry / n_days_total * 100:.0f}%)")
    print(f"  days with ≥1 scratch:  {n_days_with_scratch}  "
          f"({n_days_with_scratch / n_days_total * 100:.0f}%)")
    print(f"  scratch:entry msg ratio: "
          f"{classes.get('scratch', 0)}/{classes.get('entry', 0)} = "
          f"{classes.get('scratch', 0) / max(1, classes.get('entry', 0)):.2f}")
    print()

    # Distribution of entries per active day
    counts_per_active_day = [d["entry"] for d in per_day.values() if d["entry"] > 0]
    if counts_per_active_day:
        cpa = sorted(counts_per_active_day)
        mean_cpa = sum(cpa) / len(cpa)
        median_cpa = cpa[len(cpa) // 2]
        print(f"calls per active day:  mean={mean_cpa:.1f}  median={median_cpa}  "
              f"max={cpa[-1]}  (n={len(cpa)} active days)")
        bucket = Counter(c for c in cpa)
        for k in sorted(bucket):
            print(f"  {k} calls/day: {bucket[k]:3d} days")
    print()

    # ── Recap selection bias ─────────────────────────────────────────────
    # For each recap, count how many tickers in it were called THAT day.
    # And how many of that day's tickers are NOT mentioned in the recap.
    by_date = defaultdict(list)
    for m in data:
        by_date[m["dt_iso"][:10]].append(m)
    recap_days = [d for d, msgs in by_date.items() if any(x["msg_class"] == "recap" for x in msgs)]
    print(f"recap days: {len(recap_days)}")
    mentioned_winners = 0
    mentioned_total = 0
    unmentioned_calls = 0
    for d in recap_days:
        msgs = by_date[d]
        calls = [m for m in msgs if m["msg_class"] == "entry"]
        recap_text = " ".join(m["text"] for m in msgs if m["msg_class"] == "recap")
        recap_syms = set(TICKER_HINT.findall(recap_text))
        call_syms = set()
        for m in calls:
            call_syms.update(m["tickers"])
        mentioned_in_recap = call_syms & recap_syms
        not_mentioned = call_syms - recap_syms
        mentioned_total += len(call_syms)
        mentioned_winners += len(mentioned_in_recap)
        unmentioned_calls += len(not_mentioned)
    print(f"  on recap days, {mentioned_winners}/{mentioned_total} "
          f"({mentioned_winners / max(1, mentioned_total) * 100:.0f}%) of "
          f"day's call-symbols are named in the recap")
    print(f"  → if recaps name winners more than losers, the user-perceived "
          f"hit-rate is biased upward")
    print()

    # ── Persist a tradeable-calls JSON for downstream backtest ────────────
    out_calls = []
    for m in entries:
        # Only keep calls with exactly one ticker — bundled multi-ticker
        # entries ("EDEN long, BSB short") need manual handling; rare anyway.
        if len(m["tickers"]) != 1 or not m["side"] or m["side"] == "both":
            continue
        out_calls.append({
            "msg_id": m["msg_id"],
            "dt_iso": m["dt_iso"],
            "symbol": m["tickers"][0],
            "side": m["side"],
            "tf_tags": m["tf_tags"],
            "setup": m["setup"],
            "text": m["text"],
        })
    CALLS_OUT.write_text(json.dumps(out_calls, ensure_ascii=False, indent=2))
    print(f"wrote {len(out_calls)} replayable calls -> {CALLS_OUT}")


if __name__ == "__main__":
    main()
