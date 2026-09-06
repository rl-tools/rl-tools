#!/usr/bin/env bash
# Rebases the WASI patch series (patches/*.patch) onto another LLVM tag or commit and re-exports it.
# The series is maintained as commits: each patch becomes a commit on a scratch worktree of the current pin, the commits
# are rebased onto the new pin, and `git format-patch` writes the series back with stable headers. Conflicts stop in the
# worktree for manual resolution (git rebase --continue), after which `--export` finishes the job.
#
# usage: tools/ide/toolchain/rebase_patches.sh <new tag or commit> [--checkout <llvm_project-src>]
#        tools/ide/toolchain/rebase_patches.sh --export <new tag or commit> [--checkout <llvm_project-src>]
# The checkout defaults to the FetchContent tree of the newest build directory under .dependencies. The new pin is fetched
# at depth 1 into that checkout if it is not present yet. RL_TOOLS_IDE_PATCHWORK_DIR places the scratch worktree (a local
# disk is much faster than NFS for the checkout). Afterwards set RL_TOOLS_IDE_LLVM_TAG in CMakeLists.txt.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PATCH_DIR="$SCRIPT_DIR/patches"
TOOLCHAIN_DIR="${RL_TOOLS_IDE_TOOLCHAIN_DIR:-/vm/data/rl-tools/ide-toolchain}"
WORKTREE="${RL_TOOLS_IDE_PATCHWORK_DIR:-$TOOLCHAIN_DIR/patchwork}"

export_only=0
checkout=""
new_pin=""
while [ $# -gt 0 ]; do
    case "$1" in
        --export) export_only=1 ;;
        --checkout) checkout="$2"; shift ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) new_pin="$1" ;;
    esac
    shift
done
if [ -z "$new_pin" ]; then
    echo "usage: $0 [--export] <new tag or commit> [--checkout <llvm_project-src>]" >&2
    exit 2
fi
if [ -z "$checkout" ]; then
    checkout="$(ls -td "$REPO_ROOT"/.dependencies/*/llvm_project-src 2>/dev/null | head -n 1 || true)"
fi
if [ ! -d "$checkout/.git" ]; then
    echo "no llvm-project checkout found (configure the superbuild first or pass --checkout)" >&2
    exit 1
fi
current_pin="$(sed -n 's/^set(RL_TOOLS_IDE_LLVM_TAG \([^ )]*\).*/\1/p' "$SCRIPT_DIR/CMakeLists.txt" | head -n 1)"

fetch_pin(){
    local pin="$1"
    if git -C "$checkout" rev-parse --verify --quiet "$pin^{commit}" >/dev/null; then
        return
    fi
    if [ "${#pin}" -eq 40 ]; then
        git -C "$checkout" fetch --depth 1 origin "$pin"
    else
        git -C "$checkout" fetch --depth 1 origin tag "$pin" --no-tags
    fi
}

if [ "$export_only" -eq 0 ]; then
    fetch_pin "$current_pin"
    fetch_pin "$new_pin"
    base="$(git -C "$checkout" rev-parse "$current_pin^{commit}")"
    target="$(git -C "$checkout" rev-parse "$new_pin^{commit}")"
    if [ -e "$WORKTREE" ]; then
        echo "$WORKTREE exists; finish or remove it first (git -C $checkout worktree remove --force $WORKTREE)" >&2
        exit 1
    fi
    git -C "$checkout" worktree prune
    git -c advice.detachedHead=false -C "$checkout" worktree add --detach "$WORKTREE" "$base"
    git -C "$WORKTREE" -c user.name=rl-tools -c user.email=rl-tools@localhost am --3way "$PATCH_DIR"/*.patch
    echo "series applied on $current_pin, rebasing onto $new_pin ($target)"
    if ! git -C "$WORKTREE" -c user.name=rl-tools -c user.email=rl-tools@localhost rebase --onto "$target" "$base"; then
        echo "conflicts: resolve them in $WORKTREE (git rebase --continue), then run: $0 --export $new_pin" >&2
        exit 1
    fi
fi

target="$(git -C "$checkout" rev-parse "$new_pin^{commit}")"
if [ "$(git -C "$WORKTREE" merge-base "$target" HEAD)" != "$target" ]; then
    echo "$WORKTREE is not rebased onto $new_pin yet" >&2
    exit 1
fi
rm -f "$PATCH_DIR"/*.patch
git -C "$WORKTREE" format-patch --zero-commit --no-signature --output-directory "$PATCH_DIR" "$target"..HEAD >/dev/null
git -C "$checkout" worktree remove --force "$WORKTREE"
echo "exported $(ls "$PATCH_DIR"/*.patch | wc -l) patches for $new_pin into $PATCH_DIR"
echo "next: set RL_TOOLS_IDE_LLVM_TAG to $new_pin in $SCRIPT_DIR/CMakeLists.txt and rebuild"
