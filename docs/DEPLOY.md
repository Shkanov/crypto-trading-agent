# Deployment

Single-host deployment on a small VPS (Hetzner CX22 €4.51/mo or AWS t4g.small).
The stack: agent container + Postgres + Redis, behind a systemd unit, with
Prometheus metrics exposed on loopback for local scraping.

## 1. VPS prerequisites

```bash
# As root on a fresh Ubuntu 24.04 / Debian 12 VPS
apt-get update && apt-get install -y \
  docker.io docker-compose-plugin \
  fail2ban ufw chrony curl

# Firewall: only SSH inbound. Everything else loopback.
ufw default deny incoming && ufw default allow outgoing
ufw allow OpenSSH
ufw enable

# NTP — clock skew is the #1 source of -1021 errors on Binance.
systemctl enable --now chrony
chronyc tracking      # confirm offset is < 10ms

# Non-root user for the agent.
useradd -m -s /bin/bash -G docker agent
```

## 2. First-time setup

```bash
sudo -iu agent
git clone https://github.com/YOU/crypto-trading-agent.git /opt/crypto-trading-agent
cd /opt/crypto-trading-agent

# Secrets — NEVER commit this file.
cp .env.example .env
$EDITOR .env
#   - BINANCE_TESTNET=false
#   - BINANCE_API_KEY=...           (mainnet; withdrawal DISABLED; IP whitelisted)
#   - BINANCE_API_SECRET=...
#   - ANTHROPIC_API_KEY=...
#   - TELEGRAM_BOT_TOKEN=...
#   - TELEGRAM_ALLOWED_USER_IDS=YOUR_TG_ID
#   - POSTGRES_PASSWORD=<strong-random-string>
#   - MAX_NOTIONAL_USD=25           (notional ramp will scale up from here)
#   - DATABASE_URL=postgresql+asyncpg://agent:${POSTGRES_PASSWORD}@postgres:5432/agent

chmod 600 .env
mkdir -p data logs

# First boot.
docker compose up -d
docker compose logs -f agent
```

## 3. Binance API key safety (do this BEFORE the agent runs live)

In Binance → API Management:

- [ ] Enable Spot & Margin Trading
- [ ] Enable Futures
- [ ] **DISABLE withdrawals** (most-overlooked safety — without this, a key
      compromise drains the account; with it, attacker can only trade)
- [ ] IP-whitelist to the VPS's static public IP
- [ ] Rotate the key quarterly (set a calendar reminder)

If you can't set a static IP on the VPS, use a tunnel through a static IP
proxy you control. Never operate live without IP whitelist.

## 4. Persist as a systemd service

```bash
sudo cp deploy/systemd/crypto-trading-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now crypto-trading-agent
sudo systemctl status crypto-trading-agent
```

After a reboot, the stack comes back automatically. `docker compose ps` will
show the agent + postgres + redis healthy.

## 5. Metrics & dashboards

Metrics are at `http://127.0.0.1:9090/metrics` inside the VPS (loopback only).

To scrape from a personal Grafana Cloud free tier, install the Grafana Alloy
agent on the VPS, point it at `localhost:9090`, and forward to your Cloud
endpoint. Or run Prometheus on the same VPS:

```yaml
# /etc/prometheus/prometheus.yml
scrape_configs:
  - job_name: agent
    static_configs:
      - targets: ["127.0.0.1:9090"]
```

Useful alerts:

- `agent_halted == 1 for 1h` — agent stuck in halted state, needs attention.
- `rate(agent_ws_reconnects_total[5m]) > 1` — WS connection unstable.
- `agent_pnl_today_usd < -(0.03 * agent_equity_usd)` — daily loss approaching cap.
- `agent_llm_budget_remaining_usd{agent="StrategyAgent"} < 0.1` — LLM hitting budget.

## 6. Backups

The DB sidecar runs `pg_dump` daily and uploads to S3/B2/R2 (configure in `.env`):

```ini
BACKUP_BUCKET=s3://my-bucket/crypto-agent
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
BACKUP_RETENTION_DAYS=30
```

Uncomment the `backup` service in `docker-compose.yml` to enable.

**Test the restore path** before relying on backups. Quarterly:

```bash
# On a scratch VPS:
docker compose down
docker volume rm crypto-trading-agent_pgdata
docker compose up -d postgres
aws s3 cp s3://my-bucket/crypto-agent/agent-YYYYMMDDTHHMMSSZ.sql.gz - \
  | gunzip | docker compose exec -T postgres psql -U agent -d agent
```

## 7. The hard kill switch

If anything looks wrong:

```bash
# Halt all new trading IMMEDIATELY (existing positions stay open).
ssh agent@vps touch /opt/crypto-trading-agent/data/STOP

# Resume requires (a) deleting the file AND (b) sending /resume in Telegram.
```

Or from Telegram:
- `/flatten` — close all open positions and halt 1h.
- `/pause` — halt 4h, no closes.
- `/status` — current state.

## 8. Going live — ramp policy

The notional ramp (`src/services/notional_ramp.py`) starts at $25 max notional
and:
- Increases 10% per profitable week.
- Halves on any 3%-of-equity drawdown day.
- Floor: $10. Ceiling: `MAX_NOTIONAL_USD` from `.env`.

This means a fresh agent on a $1k account starts with $25 trades. After 12
profitable weeks (~3 months), max trade size reaches ~$78. After a year of
clean trading, ~$220. The math is intentionally cautious. Resist the
temptation to bypass.

## 9. Upgrading the agent

```bash
ssh agent@vps
cd /opt/crypto-trading-agent
git fetch && git checkout vX.Y.Z
sudo systemctl restart crypto-trading-agent
# Watch logs for the boot reconciliation:
docker compose logs -f agent | grep -E "startup.reconciliation|startup.pnl"
```

If `startup.reconciliation_failed` appears, the agent has halted itself. Do
NOT bypass — investigate the mismatch, resolve manually on Binance, then
`touch data/RESUME` (we don't auto-resume from reconciliation failures).

## 10. Disaster recovery checklist

Quarterly drill:

1. SSH to a scratch VPS.
2. `git clone` and `cp .env.example .env`; fill in **testnet** keys.
3. Restore the most recent backup with the `psql ... < dump` command above.
4. `docker compose up -d`. Watch the boot reconciliation report.
5. Confirm `/status` returns the expected positions/equity.
6. `docker compose down` and document the restore time. Target: < 30 min.
