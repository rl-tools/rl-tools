#!/usr/bin/env bash
# Checks that the served toolchain package was built from the pins in tools/ide/toolchain/CMakeLists.txt and that its
# artifacts are the ones the manifest hashes: a stale or foreign package in static/ide/build fails here. Needs only sha256sum.
# usage: tests/src/ide/provenance.sh [<toolchain dir>]      default: static/ide/build/toolchain
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TOOLCHAIN_DIR="${1:-$REPO_ROOT/static/ide/build/toolchain}"
SUPERBUILD="$REPO_ROOT/tools/ide/toolchain/CMakeLists.txt"
MANIFEST="$TOOLCHAIN_DIR/toolchain.json"
failures=0

pin(){
    sed -n "s/^set($1 \([^ )]*\).*/\1/p" "$SUPERBUILD" | head -n 1
}
field(){
    sed -n "s/^ *\"$1\": \"\{0,1\}\([^\",]*\)\"\{0,1\},\{0,1\}\$/\1/p" "$MANIFEST" | head -n 1
}
expect(){
    local description="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        echo "ok   $description: $actual"
    else
        echo "FAIL $description: expected '$expected', found '$actual'"
        failures=$((failures + 1))
    fi
}

for artifact in llvm.wasm sysroot.tar toolchain.json SHA256SUMS; do
    if [ ! -f "$TOOLCHAIN_DIR/$artifact" ]; then
        echo "FAIL missing $TOOLCHAIN_DIR/$artifact (build it with the tools/ide/toolchain superbuild)"
        exit 1
    fi
done

expect "manifest format" "1" "$(field format)"
expect "llvm tag" "$(pin RL_TOOLS_IDE_LLVM_TAG)" "$(field llvm_tag)"
expect "llvm repository" "$(pin RL_TOOLS_IDE_LLVM_REPOSITORY)" "$(field llvm_repository)"
expect "wasi-libc tag" "$(pin RL_TOOLS_IDE_WASI_LIBC_TAG)" "$(field wasi_libc_tag)"
expect "wasi-libc repository" "$(pin RL_TOOLS_IDE_WASI_LIBC_REPOSITORY)" "$(field wasi_libc_repository)"
expect "target" "$(pin RL_TOOLS_IDE_TARGET)" "$(field target)"
expected_patches=""
for patch in "$REPO_ROOT"/tools/ide/toolchain/patches/*.patch; do
    [ -f "$patch" ] || continue
    entry="{\"file\": \"$(basename "$patch")\", \"sha256\": \"$(sha256sum "$patch" | cut -d' ' -f1)\"}"
    expected_patches="${expected_patches:+$expected_patches, }$entry"
done
expect "patch series" "[$expected_patches]" "$(sed -n 's/^ *"patches": \(\[.*\]\),$/\1/p' "$MANIFEST" | head -n 1)"
expect "llvm.wasm sha256" "$(sha256sum "$TOOLCHAIN_DIR/llvm.wasm" | cut -d' ' -f1)" "$(field llvm_wasm_sha256)"
expect "sysroot.tar sha256" "$(sha256sum "$TOOLCHAIN_DIR/sysroot.tar" | cut -d' ' -f1)" "$(field sysroot_tar_sha256)"
expect "llvm.wasm bytes" "$(stat -c %s "$TOOLCHAIN_DIR/llvm.wasm")" "$(field llvm_wasm_bytes)"
expect "sysroot.tar bytes" "$(stat -c %s "$TOOLCHAIN_DIR/sysroot.tar")" "$(field sysroot_tar_bytes)"
if [ -z "$(field llvm_commit)" ] || [ -z "$(field llvm_version)" ]; then
    echo "FAIL manifest does not record the LLVM commit and version"
    failures=$((failures + 1))
fi
if (cd "$TOOLCHAIN_DIR" && sha256sum --check --quiet SHA256SUMS); then
    echo "ok   SHA256SUMS verifies"
else
    echo "FAIL SHA256SUMS does not verify"
    failures=$((failures + 1))
fi

if [ "$failures" -eq 0 ]; then
    echo "PASS: $TOOLCHAIN_DIR matches the pins ($(field llvm_version) @ $(field llvm_commit))"
else
    echo "FAIL: $failures provenance check(s) failed for $TOOLCHAIN_DIR"
    exit 1
fi
