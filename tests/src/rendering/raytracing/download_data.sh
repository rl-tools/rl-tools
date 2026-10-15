#!/usr/bin/env bash
# Fetches the golden corpora into tests/data. The scene is pinned by conta reference in
# golden_manifest.h; the renderer resolves it (CONTA_ROOT selects a local conta checkout).
# DATA_REVISION is the test-data commit of the corpus versions pinned in golden_manifest.h
# (GOLDEN_LEDGER.md); bump both together when publishing a regenerated corpus.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../../../data"
DATA_SOURCE="${RL_TOOLS_TEST_DATA_SOURCE:-https://huggingface.co/datasets/rl-tools/test-data}"
DATA_REVISION="${RL_TOOLS_RAYTRACING_GOLDEN_DATA_REVISION:-}"
LFS_INCLUDE="rendering/raytracing/goldens/**"

if [[ "${1:-}" == "--revision" ]]; then
    echo "${DATA_REVISION}"
    exit 0
fi
if [[ -z "${DATA_REVISION}" ]]; then
    echo "No golden corpus revision is pinned; set RL_TOOLS_RAYTRACING_GOLDEN_DATA_REVISION to its reviewed test-data commit, then pin it here and in GOLDEN_LEDGER.md." >&2
    exit 1
fi

if ! git lfs version >/dev/null 2>&1; then
    echo "git-lfs is required to download the raytracing golden data" >&2
    exit 1
fi

if [[ ! -e "${DATA_DIR}" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --branch master --no-checkout "${DATA_SOURCE}" "${DATA_DIR}"
elif [[ -d "${DATA_DIR}/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_DIR}" fetch origin
else
    echo "Test data path exists but is not a Git checkout: ${DATA_DIR}" >&2
    exit 1
fi

GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_DIR}" checkout --detach "${DATA_REVISION}"
git -C "${DATA_DIR}" lfs pull --include="${LFS_INCLUDE}" --exclude=''
