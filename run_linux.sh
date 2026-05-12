#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_linux.sh — Linux launcher for the Chart Analysis GUI
#
# Usage:  ./run_linux.sh          (from terminal or file manager)
#
# The script:
#   1. Changes to the project root (wherever this file lives).
#   2. Activates the local virtual environment created by install_linux.sh.
#   3. Launches src/main_modern.py using the venv python directly.
#
# NOTE: Do NOT use 'set -e' here — we want to print error messages and pause
# before the terminal closes, so users who double-click can read the output.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Helper: pause so the terminal stays open when double-clicked ──────────
_pause() {
    echo ""
    echo "Press Enter to close..."
    read -r _
}

# ── Locate the virtual environment ───────────────────────────────────────
VENV_DIR=""
for candidate in "$SCRIPT_DIR/.venv" "$SCRIPT_DIR/venv"; do
    if [[ -f "$candidate/bin/activate" ]]; then
        VENV_DIR="$candidate"
        break
    fi
done

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -z "$VENV_DIR" ]]; then
        echo ""
        echo "  ERROR: Virtual environment not found."
        echo ""
        echo "  Expected location: $SCRIPT_DIR/.venv"
        echo ""
        echo "  Please run the installer first:"
        echo "    bash install_linux.sh"
        echo ""
        echo "  If you already ran the installer and still see this message,"
        echo "  make sure you are running this file from the Plot-in folder."
        _pause
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    echo "» Activated: $VENV_DIR"
else
    echo "» Virtual environment already active: $VIRTUAL_ENV"
fi

# ── Always use the venv python by absolute path ───────────────────────────
# Do NOT fall back to system python3 — the system interpreter won't have
# the project's dependencies installed.
PYTHON_BIN="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    # VIRTUAL_ENV may have been set externally; derive from it
    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        PYTHON_BIN="$VIRTUAL_ENV/bin/python"
    else
        echo ""
        echo "  ERROR: python not found inside the virtual environment."
        echo "  Expected: $VENV_DIR/bin/python"
        echo ""
        echo "  The virtual environment may be corrupted."
        echo "  Delete the .venv folder and run install_linux.sh again."
        _pause
        exit 1
    fi
fi

# ── Launch the application ────────────────────────────────────────────────
echo "» Starting Chart Analysis GUI..."
"$PYTHON_BIN" src/main_modern.py "$@"
APP_EXIT=$?

if [[ $APP_EXIT -ne 0 ]]; then
    echo ""
    echo "  The application exited with error code $APP_EXIT."
    echo "  Check the messages above for details."
    _pause
fi
exit $APP_EXIT
