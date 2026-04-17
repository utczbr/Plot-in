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

VENV_DIR="$SCRIPT_DIR/venv"

# ── Activate venv if not already active ──────────────────────────────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
        echo "ERROR: Virtual environment not found at '$VENV_DIR'."
        echo "Run the installer first:  bash install_macos.command"
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
exec python src/main_modern.py "$@"
