#!/usr/bin/env bash
# Fetches only what the raytracing golden tests read from the test-data repository: the golden
# corpora under rendering/raytracing/goldens/ and the ProcTHOR scene (75 MB instead of the full
# 1.6 GB that tests/download_data.sh pulls). The checkout lands in tests/data like the full
# download; every other LFS file stays a pointer until tests/download_data.sh materializes it.
# DATA_REVISION is the test-data commit of the corpus versions pinned in golden_manifest.h
# (GOLDEN_LEDGER.md); bump both together when publishing a regenerated corpus.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../../../data"
DATA_SOURCE="${RL_TOOLS_TEST_DATA_SOURCE:-https://huggingface.co/datasets/rl-tools/test-data}"
DATA_REVISION="${RL_TOOLS_RAYTRACING_GOLDEN_DATA_REVISION:-cb62bfcfd9bd67da742953c136dd55c054936faf}"
LFS_INCLUDE="rendering/raytracing/goldens/**,ProcTHOR-Train-1.glb"

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
