#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_app.command — macOS double-click launcher for the Chart Analysis GUI
#
# macOS opens .command files in Terminal when double-clicked (as long as
# Terminal is allowed in System Preferences → Privacy → Full Disk Access).
#
# The script:
#   1. Changes to the project root (wherever this file lives).
#   2. Removes the quarantine flag that Gatekeeper may add.
#   3. Activates the local virtual environment if it is not already active.
#   4. Launches  python src/main_modern.py
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Remove quarantine attribute so macOS doesn't block the script on first run
xattr -d com.apple.quarantine "$0" 2>/dev/null || true

VENV_DIR=""
for candidate in "$SCRIPT_DIR/.venv" "$SCRIPT_DIR/venv"; do
    if [[ -f "$candidate/bin/activate" ]]; then
        VENV_DIR="$candidate"
        break
    fi
done

# ── Activate venv if not already active ──────────────────────────────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -z "$VENV_DIR" ]]; then
        echo "ERROR: Virtual environment not found at '$SCRIPT_DIR/.venv' or '$SCRIPT_DIR/venv'."
        echo "Run the installer first:  ./install_macos.command"
        echo ""
        echo "Press Enter to close."
        read -r
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    echo "» Activated virtual environment: $VENV_DIR"
else
    echo "» Virtual environment already active: $VIRTUAL_ENV"
fi

# ── Verify python3 is present ─────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found."
    echo "Install Python from https://www.python.org or via Homebrew."
    echo ""
    echo "Press Enter to close."
    read -r
    exit 1
fi

# ── Launch the application ────────────────────────────────────────────────
echo "» Starting Chart Analysis GUI…"
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
exec "$PYTHON_BIN" src/main_modern.py "$@"
