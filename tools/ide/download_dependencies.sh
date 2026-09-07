#!/usr/bin/env bash
# Soft submodules of the browser IDE toolchain: pinned checkouts under tools/ide/external (ignored by git), each fetched at
# depth 1 by commit, so the repository carries only these pins and no submodule metadata. Re-running is instant when the
# checkouts are at their pins; a changed pin moves them with one more depth-1 fetch.
#
# usage: tools/ide/download_dependencies.sh
# RL_TOOLS_IDE_EXTERNAL_DIR places the checkouts elsewhere (a local disk is much faster than NFS for the 150,000 files of
# llvm-project); RL_TOOLS_IDE_LLVM_PROJECT_REPOSITORY and RL_TOOLS_IDE_WASI_LIBC_REPOSITORY point at mirrors.
set -euo pipefail

LLVM_PROJECT_REPOSITORY=https://github.com/rl-tools/llvm-project.git
LLVM_PROJECT_COMMIT=4371bcb74f3737e5f5e0ec6c5d8ddf2839724890
LLVM_PROJECT_UPSTREAM=llvmorg-22.1.8
WASI_LIBC_REPOSITORY=https://github.com/WebAssembly/wasi-libc.git
WASI_LIBC_COMMIT=2e6fb9d8ee0cdf9e431fbcabe8af3115de000a13
WASI_LIBC_UPSTREAM=wasi-sdk-34

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="${RL_TOOLS_IDE_EXTERNAL_DIR:-$SCRIPT_DIR/external}"

checkout(){
    local name="$1" repository="$2" commit="$3" directory="$EXTERNAL_DIR/$1"
    if [ -d "$directory/.git" ]; then
        if [ "$(git -C "$directory" rev-parse HEAD)" = "$commit" ]; then
            echo "$name: at $commit"
            return
        fi
        echo "$name: moving to $commit"
        git -C "$directory" fetch --depth 1 origin "$commit"
        git -c advice.detachedHead=false -C "$directory" checkout --force --detach "$commit"
    else
        if [ -e "$directory" ] && [ -n "$(find "$directory" -type f -print -quit 2>/dev/null)" ]; then
            echo "$directory exists, is not a git checkout and is not empty; move it away first" >&2
            exit 1
        fi
        echo "$name: cloning $repository at $commit (depth 1) into $directory"
        mkdir -p "$directory"
        git init -q "$directory"
        git -C "$directory" remote add origin "$repository"
        git -C "$directory" fetch --depth 1 origin "$commit"
        git -c advice.detachedHead=false -C "$directory" checkout -q --detach FETCH_HEAD
    fi
    echo "$name: $(git -C "$directory" rev-parse HEAD)"
}

checkout llvm-project "${RL_TOOLS_IDE_LLVM_PROJECT_REPOSITORY:-$LLVM_PROJECT_REPOSITORY}" "$LLVM_PROJECT_COMMIT"
checkout wasi-libc "${RL_TOOLS_IDE_WASI_LIBC_REPOSITORY:-$WASI_LIBC_REPOSITORY}" "$WASI_LIBC_COMMIT"
