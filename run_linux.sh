#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_app.sh — Linux / macOS launcher for the Chart Analysis GUI
#
# Usage:  ./run_app.sh
#
# The script:
#   1. Changes to the project root (wherever this file lives).
#   2. Activates the local virtual environment if it is not already active.
#   3. Launches  python src/main_modern.py
# ---------------------------------------------------------------------------

set -euo pipefail

# ── Resolve the project root reliably even when called from another dir ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"

# ── Activate the venv only when it isn't already active ──────────────────
# VIRTUAL_ENV is set by the activate script; absent in a bare shell.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
        echo "ERROR: Virtual environment not found at '$VENV_DIR'."
        echo "Run the installer first:  bash install_linux.sh"
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    echo "» Activated virtual environment: $VENV_DIR"
else
    echo "» Virtual environment already active: $VIRTUAL_ENV"
fi

# ── Launch the application ────────────────────────────────────────────────
echo "» Starting Chart Analysis GUI…"
exec python src/main_modern.py "$@"
