#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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

cd "$SCRIPT_DIR"
python3 install.py "$@"
