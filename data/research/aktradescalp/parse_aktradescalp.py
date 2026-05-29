"""Parse t.me/s/aktradescalp HTML pages into structured records.

Output: aktradescalp_messages.json (next to this script) — list of dicts
with msg_id, dt_iso, text, plus light extracted features.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGES_DIR = HERE / "pages"
OUT_PATH = HERE / "aktradescalp_messages.json"

# Split a page into per-message HTML blocks. Each top-level message has a
# wrapper div with class containing "tgme_widget_message_wrap". We split on
# the wrap delimiter and keep everything until the *next* wrap (or end of
# file) — the per-message footer (which contains <time>) sits inside this
# range, so a naive cut at "footer" would lose the timestamp.
MSG_SPLIT_RE = re.compile(
    r'<div class="tgme_widget_message_wrap[^"]*"[^>]*>',
)
DATA_POST_RE = re.compile(r'data-post="aktradescalp/(\d+)"')
TIME_RE = re.compile(r'<time datetime="([^"]+)"')
TEXT_BLOCK_RE = re.compile(
    r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
    re.DOTALL,
)
TAG_STRIP_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

# Feature extraction patterns
TICKER_RE = re.compile(r"#?([A-Z0-9]{2,15}USDT|[A-Z0-9]{2,15}USD)")
LONG_RU = re.compile(r"\bлонг\b", re.IGNORECASE)
SHORT_RU = re.compile(r"\bшорт\b", re.IGNORECASE)
PROBOI_RU = re.compile(r"\bпробо[йяиве]\w*\b", re.IGNORECASE)
NAKLONKA_RU = re.compile(r"\bнаклонк\w*\b", re.IGNORECASE)
DNEVKA_RU = re.compile(r"\bдневк\w*\b", re.IGNORECASE)
SCRATCH_RU = re.compile(
    r"\b(скидыва\w+|перемудри\w+|без меня|не успева\w+|сложн\w+ рынок|пляшем дальше)\b",
    re.IGNORECASE,
)
TF_DAY_RE = re.compile(r"\b(1D|D1|дневк)", re.IGNORECASE)
TF_15M_RE = re.compile(r"\b(15M|M15|5M\s*[+]\s*1D|D1\s*[+]\s*5M|M5\s*[+]\s*D1|1D\s*\+\s*5M)\b", re.IGNORECASE)
TF_5M_RE = re.compile(r"\b(5M|M5)\b", re.IGNORECASE)
HOUR_LEVELS = re.compile(r"\b(стоим|перед|под|над|внутри)\b", re.IGNORECASE)


def parse_html(html: str) -> list[dict]:
    out: list[dict] = []
    # Split on the wrap-div delimiter; each chunk after the first is one
    # message (the first chunk is the page header / pre-list content).
    chunks = MSG_SPLIT_RE.split(html)
    for block in chunks[1:]:
        m_id = DATA_POST_RE.search(block)
        m_time = TIME_RE.search(block)
        if not m_id or not m_time:
            continue
        text_block = TEXT_BLOCK_RE.search(block)
        raw_text = text_block.group(1) if text_block else ""
        # Strip HTML tags but keep their text content; collapse whitespace.
        # Preserve "#TICKER" tokens that come from <a> inside the text — by
        # the time we strip tags they're already in the text body.
        clean = TAG_STRIP_RE.sub(" ", raw_text)
        clean = clean.replace("&nbsp;", " ").replace("&amp;", "&")
        clean = WS_RE.sub(" ", clean).strip()
        out.append({
            "msg_id": int(m_id.group(1)),
            "dt_iso": m_time.group(1),
            "text": clean,
        })
    return out


def extract_features(text: str) -> dict:
    has_long = bool(LONG_RU.search(text))
    has_short = bool(SHORT_RU.search(text))
    has_proboi = bool(PROBOI_RU.search(text))
    has_naklonka = bool(NAKLONKA_RU.search(text))
    has_dnevka = bool(DNEVKA_RU.search(text))
    has_scratch = bool(SCRATCH_RU.search(text))

    # Side classification: an entry-message style is "<TICKER> <direction> [tf]"
    # A "no-trade" recap message tends to have no #ticker.
    if has_long and not has_short:
        side = "long"
    elif has_short and not has_long:
        side = "short"
    elif has_long and has_short:
        side = "both"  # rare, e.g. "EDENUSDT long, BSB short"
    else:
        side = ""

    tickers = sorted(set(t for t in TICKER_RE.findall(text)))

    # TF tag — coarse. "1D+5M" / "D1+M5" patterns mean multi-TF break.
    tf_tags: list[str] = []
    if re.search(r"\b(1D|D1|дневк)", text, re.IGNORECASE):
        tf_tags.append("D1")
    if re.search(r"\b(15M|M15)\b", text, re.IGNORECASE):
        tf_tags.append("M15")
    if re.search(r"\b(5M|M5)\b", text, re.IGNORECASE):
        tf_tags.append("M5")

    setup = ""
    if has_proboi:
        setup = "breakout"
    if has_naklonka:
        setup = "trendline" if not setup else f"{setup}+trendline"

    # Conviction language: "🔥" / "ну ок" / "мощн", emojis tend to mark
    # post-hoc satisfaction; "стоим перед" / "стоим под" mark pre-trade
    # anchoring near a level.
    has_emoji_fire = "🔥" in text
    has_anchor = bool(re.search(r"\bстоим\b", text, re.IGNORECASE))
    has_recap = bool(re.search(r"\b(удалось|собирать|итог|неплох\w*\s+начал\w*|давал)\b", text, re.IGNORECASE))

    return {
        "tickers": tickers,
        "side": side,
        "tf_tags": tf_tags,
        "setup": setup,
        "scratch": has_scratch,
        "anchor": has_anchor,
        "recap_like": has_recap,
        "fire_emoji": has_emoji_fire,
    }


def main() -> None:
    records: dict[int, dict] = {}
    for fn in sorted(PAGES_DIR.glob("page_*.html")):
        html = fn.read_text(encoding="utf-8", errors="ignore")
        for r in parse_html(html):
            r.update(extract_features(r["text"]))
            records[r["msg_id"]] = r
    ordered = sorted(records.values(), key=lambda r: r["dt_iso"])
    OUT_PATH.write_text(json.dumps(ordered, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"wrote {len(ordered)} messages -> {OUT_PATH}")
    if ordered:
        print(f"date range: {ordered[0]['dt_iso']} -> {ordered[-1]['dt_iso']}")


if __name__ == "__main__":
    main()
