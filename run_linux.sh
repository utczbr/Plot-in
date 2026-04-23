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

VENV_DIR=""
for candidate in "$SCRIPT_DIR/.venv" "$SCRIPT_DIR/venv"; do
    if [[ -f "$candidate/bin/activate" ]]; then
        VENV_DIR="$candidate"
        break
    fi
done

# ── Activate the venv only when it isn't already active ──────────────────
# VIRTUAL_ENV is set by the activate script; absent in a bare shell.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -z "$VENV_DIR" ]]; then
        echo "ERROR: Virtual environment not found at '$SCRIPT_DIR/.venv' or '$SCRIPT_DIR/venv'."
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
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
    echo "ERROR: Could not find a Python interpreter (python3 or python)."
    exit 1
fi
exec "$PYTHON_BIN" src/main_modern.py "$@"
