#!/usr/bin/env bash
# Walk back through https://t.me/s/aktradescalp to collect all history.
# Each page contains ~12 messages and supports ?before=<msg_id> pagination.
set -euo pipefail

OUT_DIR="$(cd "$(dirname "$0")" && pwd)/pages"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.html

URL="https://t.me/s/aktradescalp"
page=1
before=""

while true; do
  fn="$OUT_DIR/page_${page}.html"
  if [ -z "$before" ]; then
    curl -sS "$URL" -o "$fn"
  else
    curl -sS "${URL}?before=${before}" -o "$fn"
  fi
  count=$(grep -c 'tgme_widget_message ' "$fn" || true)
  first_id=$(grep -oE 'data-post="aktradescalp/[0-9]+"' "$fn" | head -1 | grep -oE '[0-9]+' || echo "")
  last_id=$(grep -oE 'data-post="aktradescalp/[0-9]+"' "$fn" | tail -1 | grep -oE '[0-9]+' || echo "")
  echo "page $page: $count messages, ids $first_id..$last_id"
  if [ "$count" -eq 0 ] || [ -z "$first_id" ]; then break; fi
  if [ "$first_id" = "1" ] || [ "$first_id" -le 1 ]; then break; fi
  before="$first_id"
  page=$((page + 1))
  sleep 0.5
  if [ "$page" -gt 30 ]; then
    echo "safety cap hit"; break
  fi
done
