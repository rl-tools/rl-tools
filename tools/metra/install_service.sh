#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE=user
IP=0.0.0.0
PORT=13340
while [[ $# -gt 0 ]]; do
    case "$1" in
        --system) MODE=system; shift;;
        --ip) IP="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        *) echo "Usage: $0 [--system] [--ip <ip>] [--port <port>]" >&2; exit 1;;
    esac
done

DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/metra"
mkdir -p "$DATA_DIR"

UNIT="[Unit]
Description=metra metrics tracking server
After=network.target

[Service]
Environment=METRA_DB=$DATA_DIR/metra.sqlite
ExecStart=/usr/bin/env python3 $SCRIPT_DIR/metra/server.py --ip $IP --port $PORT
Restart=on-failure
RestartSec=5"

if [[ "$MODE" == "user" ]]; then
    UNIT_DIR="$HOME/.config/systemd/user"
    mkdir -p "$UNIT_DIR"
    printf '%s\n\n[Install]\nWantedBy=default.target\n' "$UNIT" > "$UNIT_DIR/metra.service"
    systemctl --user daemon-reload
    systemctl --user enable --now metra
    systemctl --user --no-pager status metra || true
    echo "metra: serving on http://$IP:$PORT (db: $DATA_DIR/metra.sqlite)"
    echo "metra: smoke test: curl http://127.0.0.1:$PORT/api/names"
    echo "metra: note: user services stop at logout unless lingering is enabled: loginctl enable-linger $USER (may require an admin once)"
else
    printf '%s\nUser=%s\n\n[Install]\nWantedBy=multi-user.target\n' "$UNIT" "$USER" | sudo tee /etc/systemd/system/metra.service > /dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable --now metra
    sudo systemctl --no-pager status metra || true
    echo "metra: serving on http://$IP:$PORT (db: $DATA_DIR/metra.sqlite)"
    echo "metra: smoke test: curl http://127.0.0.1:$PORT/api/names"
fi
