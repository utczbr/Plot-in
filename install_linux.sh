#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# install_linux.sh — Linux installer launcher for the Chart Analysis app
#
# Usage:  bash install_linux.sh        (from terminal or file manager)
#
# NOTE: Do NOT use 'set -e' — we want to print errors and pause before the
# terminal closes so users who double-click from a file manager can read them.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Helper: pause so the terminal stays open when double-clicked ──────────
_pause() {
    echo ""
    echo "Press Enter to close..."
    read -r _
}

# ── Use a pre-built installer binary if one is present ───────────────────
for candidate in \
    chart-analysis-installer-linux \
    chart-analysis-installer-linux-x86_64 \
    chart-analysis-installer-linux-amd64 \
    chart-analysis-installer-linux-arm64 \
    chart-analysis-installer; do
    if [[ -x "$SCRIPT_DIR/$candidate" ]]; then
        exec "$SCRIPT_DIR/$candidate" "$@"
    fi
done

# ── Find python3 ──────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo ""
    echo "  ERROR: python3 is not installed or not in PATH."
    echo ""
    echo "  Install it with:"
    echo "    sudo apt install python3 python3-venv    # Debian / Ubuntu"
    echo "    sudo dnf install python3                 # Fedora / RHEL"
    echo ""
    _pause
    exit 1
fi

echo "Using Python: $(python3 --version)"
echo ""
python3 install.py "$@"
STATUS=$?

if [[ $STATUS -ne 0 ]]; then
    echo ""
    echo "  Installer exited with error code $STATUS."
    echo "  Check the messages above for details."
    _pause
fi
exit $STATUS
