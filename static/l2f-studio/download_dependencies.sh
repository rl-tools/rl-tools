#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/external"

checkout_dependency() {
    local source="$1" directory="$2" revision="$3"
    git init --initial-branch=main "${directory}"
    git -C "${directory}" fetch --depth=1 "${source}" "${revision}"
    git -C "${directory}" checkout --detach "${revision}"
}

checkout_dependency https://github.com/nlohmann/json.git json bfb07786cd7eb841a6d9030bcd822d3b1c4d3b56
checkout_dependency https://github.com/rl-tools/l2f-studio-blob.git blob bc1cc324ae298549a7f14b97564beac97c5cf439
checkout_dependency https://github.com/rl-tools/conta-data.git conta 9784708a5eebdd0016467476045a2deffdfce3c5
