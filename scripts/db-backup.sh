#!/bin/sh
# Daily Postgres backup → S3-compatible storage (S3, Backblaze B2, R2).
# Designed to run as a sidecar container or systemd timer on the host.
#
# Required env:
#   POSTGRES_HOST     (default: postgres)
#   POSTGRES_USER     (default: agent)
#   POSTGRES_DB       (default: agent)
#   PGPASSWORD        (from .env, NOT logged)
#   BACKUP_BUCKET     (e.g. s3://my-bucket/crypto-agent-backups)
#   AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY  (or B2/R2 equivalents)
#   BACKUP_RETENTION_DAYS (default: 30)
#
# Restore: see DEPLOY.md "Disaster recovery" section.

set -eu

HOST="${POSTGRES_HOST:-postgres}"
USER="${POSTGRES_USER:-agent}"
DB="${POSTGRES_DB:-agent}"
BUCKET="${BACKUP_BUCKET:-}"
RETENTION="${BACKUP_RETENTION_DAYS:-30}"

if [ -z "$BUCKET" ]; then
  echo "BACKUP_BUCKET not set; skipping upload (running dump only)"
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
out="/tmp/agent-${stamp}.sql.gz"

echo "==> pg_dump → $out"
pg_dump -h "$HOST" -U "$USER" -d "$DB" --no-owner --no-acl \
  | gzip -9 > "$out"

size=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out")
echo "    size: ${size} bytes"

if [ -n "$BUCKET" ]; then
  if command -v aws >/dev/null 2>&1; then
    echo "==> upload to $BUCKET"
    aws s3 cp "$out" "$BUCKET/$(basename "$out")"
    # Best-effort prune of old backups.
    echo "==> prune > ${RETENTION} days"
    aws s3 ls "$BUCKET/" | awk '{print $4}' \
      | while read f; do
          # filename pattern: agent-YYYYMMDDTHHMMSSZ.sql.gz
          ts=$(echo "$f" | sed -nE 's/agent-([0-9]{8})T.*/\1/p')
          if [ -n "$ts" ]; then
            age_days=$(( ( $(date -u +%s) - $(date -u -d "${ts:0:4}-${ts:4:2}-${ts:6:2}" +%s 2>/dev/null \
              || date -u -j -f "%Y%m%d" "$ts" +%s) ) / 86400 ))
            if [ "$age_days" -gt "$RETENTION" ]; then
              echo "    deleting $f (age ${age_days}d)"
              aws s3 rm "$BUCKET/$f"
            fi
          fi
        done
  else
    echo "    aws CLI not available — backup left at $out"
  fi
fi

# Verify integrity locally.
gunzip -t "$out" && echo "==> dump integrity OK"
