#!/usr/bin/env bash
# One-shot VPS setup for crypto-trading-agent (paper mode, SQLite, no Docker needed).
# Run as root on a fresh Ubuntu 22.04 / 24.04 server:
#   curl -sSL https://raw.githubusercontent.com/Shkanov/crypto-trading-agent/main/deploy/setup_vps.sh | bash
set -euo pipefail

REPO_URL="https://github.com/Shkanov/crypto-trading-agent.git"
INSTALL_DIR="/opt/crypto-trading-agent"
SERVICE_USER="agent"

echo "=== 1. System packages ==="
apt-get update -q
apt-get install -y --no-install-recommends git curl build-essential tmux

echo "=== 2. Install uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

echo "=== 3. Clone repo ==="
if [ -d "$INSTALL_DIR" ]; then
    git -C "$INSTALL_DIR" pull
else
    git clone "$REPO_URL" "$INSTALL_DIR"
fi
cd "$INSTALL_DIR"

echo "=== 4. Install Python deps ==="
uv sync --frozen

echo "=== 5. Create data dir ==="
mkdir -p data logs

echo ""
echo "=== DONE ==="
echo ""
echo "Next steps:"
echo "  1. Copy your .env file:  scp .env root@<server>:$INSTALL_DIR/.env"
echo "  2. Add Telegram token + user ID to .env"
echo "  3. Start the agent:"
echo "       cd $INSTALL_DIR && tmux new -s agent"
echo "       uv run python -m scripts.run_paper"
echo "       (Ctrl+B, D to detach)"
echo ""
echo "To reconnect: tmux attach -t agent"
echo "Logs will print to stdout in the tmux session."
