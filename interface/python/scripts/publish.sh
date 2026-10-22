#!/usr/bin/env bash
# Build, check, rehearse and upload the hyperlux and hyperdrone release pair.
#   publish.sh                 build both, rehearse the install, upload hyperlux then hyperdrone
#   publish.sh --build-only    keep the artifacts in interface/python/dist/release.* without uploading
#   publish.sh --hyperdrone-only   a hyperdrone post-release against the published hyperlux
set -euo pipefail

usage() {
    sed -n '2,5p' "$0" | sed 's/^# //; s/^#//'
}

MODE=""
case "${1:-}" in
    ""|--build-only|--hyperdrone-only) MODE="${1:-}" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

PYTHON_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPOSITORY_DIR="$(cd "${PYTHON_DIR}/../.." && pwd)"
PYTHON="${REPOSITORY_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    python3 -m venv "${REPOSITORY_DIR}/.venv"
fi
"${PYTHON}" -m pip install --upgrade build twine

VERSION="$("${PYTHON}" "${PYTHON_DIR}/scripts/versions.py")"
HYPERLUX_VERSION="${VERSION%%.post*}"
if TAG="$(git -C "${REPOSITORY_DIR}" describe --tags --exact-match 2>/dev/null)" && [[ "${TAG}" != "v${VERSION}" ]]; then
    echo "HEAD is tagged ${TAG} but the packages carry ${VERSION}" >&2
    exit 1
fi

mkdir -p "${PYTHON_DIR}/dist"
RELEASE_DIR="$(mktemp -d "${PYTHON_DIR}/dist/release.XXXXXX")"
if [[ "${MODE}" != "--hyperdrone-only" ]]; then
    "${PYTHON}" -m build "${PYTHON_DIR}/hyperlux" --outdir "${RELEASE_DIR}"
fi
"${PYTHON}" -m build "${PYTHON_DIR}/hyperdrone" --outdir "${RELEASE_DIR}"
"${PYTHON}" -m twine check "${RELEASE_DIR}"/*

# rehearsal: the pin must resolve against the pair just built (or the published hyperlux)
SCRATCH="$(mktemp -d)"
trap 'rm -rf "${SCRATCH}"' EXIT
python3 -m venv "${SCRATCH}/venv"
"${SCRATCH}/venv/bin/pip" install -q --find-links "${RELEASE_DIR}" "hyperdrone==${VERSION}"
"${SCRATCH}/venv/bin/python" -c "import hyperdrone, hyperlux; assert hyperlux.__version__ == '${HYPERLUX_VERSION}', hyperlux.__version__"
echo "Release artifacts: ${RELEASE_DIR}"

if [[ "${MODE}" == "--build-only" ]]; then
    exit 0
fi
if [[ "${MODE}" != "--hyperdrone-only" ]]; then
    "${PYTHON}" -m twine upload --repository pypi "${RELEASE_DIR}"/hyperlux-*
    until "${SCRATCH}/venv/bin/pip" download -q --no-deps --no-cache-dir -d "${SCRATCH}/index" "hyperlux==${HYPERLUX_VERSION}"; do
        echo "waiting for the index to serve hyperlux ${HYPERLUX_VERSION}"
        sleep 15
    done
fi
"${PYTHON}" -m twine upload --repository pypi "${RELEASE_DIR}"/hyperdrone-*
