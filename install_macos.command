#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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

cd "$SCRIPT_DIR"
python3 install.py "$@"
