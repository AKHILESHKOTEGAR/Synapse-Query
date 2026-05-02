#!/bin/bash
set -e

BACKEND_DIR="$(cd "$(dirname "$0")/backend" && pwd)"
VENV="$BACKEND_DIR/.venv"
PYTHON="/opt/homebrew/bin/python3.12"

# Create venv if missing
if [ ! -d "$VENV" ]; then
    echo "Creating venv with Python 3.12..."
    "$PYTHON" -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -r "$BACKEND_DIR/requirements.txt"
fi

echo "Starting backend on http://localhost:8000"
cd "$BACKEND_DIR"
"$VENV/bin/python" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
