#!/usr/bin/env bash
# Packs the RLtools headers into a tar the browser IDE mounts read-only into the compiler's filesystem,
# copies the example program into the page's build directory, and records the commit in a manifest.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/static/ide/build"

COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
COMMIT_TIME="$(git -C "$REPO_ROOT" show -s --format=%ct HEAD 2>/dev/null || date +%s)"

mkdir -p "$BUILD_DIR/examples"
tar --format=ustar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@${COMMIT_TIME}" -C "$REPO_ROOT/include" -cf "$BUILD_DIR/rl_tools_include.tar" rl_tools
cp "$REPO_ROOT/src/rl/environments/pendulum/sac/wasi/training.cpp" "$BUILD_DIR/examples/pendulum_sac.cpp"
printf '{"rl_tools_commit": "%s", "generated": "%s"}\n' "$COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BUILD_DIR/manifest.json"
echo "wrote $BUILD_DIR/rl_tools_include.tar ($(du -h "$BUILD_DIR/rl_tools_include.tar" | cut -f1)), examples/pendulum_sac.cpp, manifest.json"
