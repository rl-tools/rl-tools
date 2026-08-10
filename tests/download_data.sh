#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
DATA_SOURCE="${RL_TOOLS_TEST_DATA_SOURCE:-https://huggingface.co/datasets/rl-tools/test-data}"
DATA_REVISION="${RL_TOOLS_TEST_DATA_REVISION:-cb62bfcfd9bd67da742953c136dd55c054936faf}"
DOWNLOAD_SET="${1:-"--all"}"

case "${DOWNLOAD_SET}" in
    --all|--raytracing-goldens)
        ;;
    *)
        echo "Usage: $0 [--all|--raytracing-goldens]" >&2
        exit 2
        ;;
esac

if ! git lfs version >/dev/null 2>&1; then
    echo "git-lfs is required to download RL-Tools test data" >&2
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

if [[ "${DOWNLOAD_SET}" == "--raytracing-goldens" ]]; then
    git -C "${DATA_DIR}" sparse-checkout init --no-cone
    GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_DIR}" sparse-checkout set '/.gitattributes' '/rendering_raytracing_golden/'
elif [[ "$(git -C "${DATA_DIR}" config --bool core.sparseCheckout || true)" == "true" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_DIR}" sparse-checkout disable
fi

GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_DIR}" checkout --detach "${DATA_REVISION}"

if [[ "${DOWNLOAD_SET}" == "--raytracing-goldens" ]]; then
    git -C "${DATA_DIR}" lfs pull --include='rendering_raytracing_golden/**' --exclude=''
else
    git -C "${DATA_DIR}" lfs pull
fi
