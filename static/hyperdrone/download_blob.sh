#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLOB_DIR="${SCRIPT_DIR}/blob"
BLOB_REVISION="40b11c1c666a016a688d78ee91d72414c52e4dcb"

if [[ -e "${BLOB_DIR}/.git" ]]; then
    if [[ "$(git -C "${BLOB_DIR}" rev-parse HEAD)" != "${BLOB_REVISION}" ]]; then
        echo "Blob checkout is not at ${BLOB_REVISION}; leaving it unchanged." >&2
        exit 1
    fi
    echo "Blob repository already exists at ${BLOB_DIR}; leaving it unchanged."
    exit 0
fi

git clone --depth=1 --no-checkout https://github.com/rl-tools/hyperdrone.dev-blob "${BLOB_DIR}"
git -C "${BLOB_DIR}" fetch --depth=1 origin "${BLOB_REVISION}"
git -C "${BLOB_DIR}" checkout --detach "${BLOB_REVISION}"
