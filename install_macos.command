#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "=== Chart Analysis Installer ==="
echo ""

for app_bundle in \
    chart-analysis-installer-macos.app \
    chart-analysis-installer-darwin-arm64.app \
    chart-analysis-installer-darwin-x86_64.app \
    chart-analysis-installer.app; do
    app_root="$SCRIPT_DIR/$app_bundle"
    app_binary="$app_root/Contents/MacOS/${app_bundle%.app}"
    if [[ -x "$app_binary" ]]; then
        exec "$app_binary" "$@"
    fi
done

for candidate in \
    chart-analysis-installer-macos \
    chart-analysis-installer-darwin-arm64 \
    chart-analysis-installer-darwin-x86_64 \
    chart-analysis-installer; do
    if [[ -x "$SCRIPT_DIR/$candidate" ]]; then
        exec "$SCRIPT_DIR/$candidate" "$@"
    fi
done

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 is not installed."
    echo "Install it from https://www.python.org or run: xcode-select --install"
    echo ""
    echo "Press Enter to close."
    read -r
    exit 1
fi

echo "Starting installer with Python..."
echo ""
cd "$SCRIPT_DIR"
python3 install.py "$@"
status=$?

if [[ $status -ne 0 ]]; then
    echo ""
    echo "Installer exited with an error (code $status)."
    echo "Press Enter to close."
    read -r
fi
exit $status
