#!/usr/bin/env bash
# Self-containment rot guard: build both sdists, install exactly those two artifacts into a
# fresh venv outside the repository, and run scripts/smoke.py against the vendored rl-tools tree
# on the GENERIC backend. Intended for CI (clean container) and local use.
set -euo pipefail

PYTHON_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCENE="${PYTHON_DIR}/../../tests/data/ProcTHOR-Train-1.glb"
if [[ ! -f "${SCENE}" ]]; then
    echo "test scene missing: ${SCENE} (run tests/download_data.sh)" >&2
    exit 1
fi
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

python3 -m venv "${WORK_DIR}/venv"
"${WORK_DIR}/venv/bin/pip" install -q build
for package in hyperlux hyperdrone; do
    (cd "${PYTHON_DIR}/${package}" && "${WORK_DIR}/venv/bin/python" -m build --sdist -o "${WORK_DIR}/dist")
done
"${WORK_DIR}/venv/bin/pip" install -q "${WORK_DIR}"/dist/hyperlux-*.tar.gz "${WORK_DIR}"/dist/hyperdrone-*.tar.gz

cd /
env -u HYPERLUX_RLTOOLS_ROOT \
    HYPERLUX_BACKEND=GENERIC \
    HYPERLUX_CACHE_DIR="${WORK_DIR}/cache" \
    "${WORK_DIR}/venv/bin/python" "${PYTHON_DIR}/scripts/smoke.py" --vendored --scene "${SCENE}"
