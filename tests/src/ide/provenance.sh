#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec cmake -DBUNDLE_DIR="${1:-$SCRIPT_DIR/../../../static/ide/build}" -P "$SCRIPT_DIR/provenance.cmake"
