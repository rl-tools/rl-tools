#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--build-only]"
    echo "Build and check the bundled sdist and wheel, then upload both to PyPI."
    echo "Use --build-only to keep the artifacts without uploading."
}

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi
case "${1:-}" in
    ""|--build-only) ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

PACKAGE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPOSITORY_DIR="$(cd "${PACKAGE_DIR}/../../.." && pwd)"
PYTHON="${REPOSITORY_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    python3 -m venv "${REPOSITORY_DIR}/.venv"
fi
"${PYTHON}" -m pip install --upgrade build twine

mkdir -p "${PACKAGE_DIR}/dist"
RELEASE_DIR="$(mktemp -d "${PACKAGE_DIR}/dist/release.XXXXXX")"

# The default build creates the wheel from the sdist, which contains the vendored sources.
"${PYTHON}" -m build "${PACKAGE_DIR}" --outdir "${RELEASE_DIR}"
"${PYTHON}" -m twine check "${RELEASE_DIR}"/*
echo "Release artifacts: ${RELEASE_DIR}"

if [[ "${1:-}" != "--build-only" ]]; then
    "${PYTHON}" -m twine upload --repository pypi "${RELEASE_DIR}"/*
fi
