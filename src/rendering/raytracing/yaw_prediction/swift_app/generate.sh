#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

xcodegen generate --spec "$SCRIPT_DIR/project.yml" --project "$SCRIPT_DIR"
