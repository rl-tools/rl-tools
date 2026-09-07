#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec cmake -DBUNDLE_DIR="${1:?usage: tools/ide/bundle.sh <candidate bundle directory>}" -P "$SCRIPT_DIR/toolchain/cmake/bundle.cmake"
