#!/usr/bin/env bash
# Starts the dashboard on 127.0.0.1:8000. It has no login, so it is NOT exposed to the network.
# View it over an SSH tunnel:  ssh -L 8000:127.0.0.1:8000 user@your-server   then open http://localhost:8000/
cd "$(dirname "$0")"
export AI_BUST_UA="${AI_BUST_UA:?set AI_BUST_UA to \"Your Name your@email\" (SEC requires a descriptive User-Agent)}"
AI_BUST_REFRESH_MIN="${AI_BUST_REFRESH_MIN:-60}" nohup .venv/bin/uvicorn ai_bust_live:app --host 127.0.0.1 --port 8000 > live.log 2>&1 &
echo "started, pid $!  (log: live.log)"
