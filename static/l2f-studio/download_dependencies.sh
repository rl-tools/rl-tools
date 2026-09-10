#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/external"
cd "${SCRIPT_DIR}/external"

checkout_dependency() {
    local source="$1" directory="$2" revision="$3"
    if ! git init --initial-branch=main "${directory}" ||
       ! git -C "${directory}" fetch --depth=1 "${source}" "${revision}" ||
       ! git -C "${directory}" checkout --detach "${revision}"; then
        echo "Failed to check out ${directory} at ${revision}" >&2
        return 1
    fi
}

status=0
checkout_dependency https://github.com/nlohmann/json.git json bfb07786cd7eb841a6d9030bcd822d3b1c4d3b56 || status=1
checkout_dependency https://github.com/rl-tools/l2f-studio-blob.git blob bc1cc324ae298549a7f14b97564beac97c5cf439 || status=1
checkout_dependency https://github.com/rl-tools/conta-data.git conta 9784708a5eebdd0016467476045a2deffdfce3c5 || status=1
exit "${status}"
