#!/usr/bin/env bash
# start_server.sh — Phase 2: Start the TextRemover FastAPI backend
#
# Usage:
#   cd poc_textremover/poc
#   bash python-backend/start_server.sh
#
# The server will:
#   - Listen on 0.0.0.0:8000 (all interfaces) so phones on the same Wi-Fi can reach it
#   - Advertise itself via mDNS as _textremover._tcp.local. (if zeroconf is installed)
#   - Load the PaddleOCR + LaMa pipeline on the first request
#
# On Android emulator, the app reaches this server at http://10.0.2.2:8000
# On a physical phone on the same Wi-Fi, it auto-discovers via mDNS or you can
# set the IP manually in the app's Settings panel.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================"
echo "  TextRemover API — Phase 2"
echo "  Root: $PROJECT_ROOT"
echo "================================================"

# Activate virtual environment if present
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
  echo "[start_server] Activated .venv"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
  source "$PROJECT_ROOT/venv/bin/activate"
  echo "[start_server] Activated venv"
fi

# Run from the project root so that python-backend.api.main is resolvable
cd "$PROJECT_ROOT"

echo "[start_server] Starting uvicorn on 0.0.0.0:8000 ..."
echo "[start_server] Health check: http://localhost:8000/api/health"
echo "[start_server] Press Ctrl+C to stop."
echo ""

python -m uvicorn "python-backend.api.main:app" \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level info
