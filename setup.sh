#!/usr/bin/env bash
# Run on the server from inside this folder: bash setup.sh
set -e
command -v python3 >/dev/null || { echo "python3 missing"; exit 1; }
python3 -m venv .venv 2>/dev/null || { apt-get update && apt-get install -y python3-venv && python3 -m venv .venv; }
. .venv/bin/activate
pip install -q -r requirements.txt
python -m unittest test_ai_bust_live
python ai_bust_live.py refresh --recorded fixtures/snapshot_recorded_2026-10-05.json
echo "Setup OK. Start dashboard with: bash start.sh"
