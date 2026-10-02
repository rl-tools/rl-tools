#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
DATA_SOURCE="${RL_TOOLS_TEST_DATA_SOURCE:-https://huggingface.co/datasets/rl-tools/test-data}"
DATA_REVISION="${RL_TOOLS_TEST_DATA_REVISION:-cbff0561f763c84119cd0b1cc56b127039b32462}"

if ! git lfs version >/dev/null 2>&1; then
    echo "git-lfs is required to download RLtools test data" >&2
    exit 1
fi

if [[ ! -e "${DATA_DIR}" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --branch master --no-checkout "${DATA_SOURCE}" "${DATA_DIR}"
elif [[ -d "${DATA_DIR}/.git" ]]; then
    # fetch all heads: published revisions may live on branches other than master
    GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_DIR}" fetch origin
else
    echo "Test data path exists but is not a Git checkout: ${DATA_DIR}" >&2
    exit 1
fi

GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_DIR}" checkout --detach "${DATA_REVISION}"
git -C "${DATA_DIR}" lfs pull
