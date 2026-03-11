#!/bin/bash
# Example: ./tools/bisect.sh d28a708e771470f9969f90cc7296972daf84ba19 9d7be9914c521eebf1c81b32bd5e3af80c9c203d -- rl_environments_mujoco_ant_ppo_blas
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="rl-tools-bisect"

usage() {
    echo "Usage: $0 [--patch <patch_file>] <commit1> <commit2> [commit3 ...] [-- <target>]"
    echo ""
    echo "Clones the repo at each commit into /tmp/rl-tools-<hash>,"
    echo "runs ./tools/init.sh in each, and optionally builds+runs <target>."
    echo "Each commit runs in its own tmux window."
    echo ""
    echo "Options:"
    echo "  --patch <file>  Apply a patch file to each checked-out repo"
    exit 1
}

if [ $# -lt 2 ]; then
    usage
fi

COMMITS=()
TARGET=""
PATCH_FILE=""

while [ $# -gt 0 ]; do
    if [ "$1" = "--patch" ]; then
        shift
        PATCH_FILE="$(realpath "$1")"
        shift
    elif [ "$1" = "--" ]; then
        shift
        TARGET="$1"
        shift
        break
    else
        COMMITS+=("$1")
        shift
    fi
done

if [ ${#COMMITS[@]} -lt 1 ]; then
    usage
fi

for COMMIT in "${COMMITS[@]}"; do
    DIR="/tmp/rl-tools-${COMMIT}"
    if [ -d "$DIR" ]; then
        echo "Removing existing $DIR"
        rm -rf "$DIR"
    fi
    echo "Cloning $COMMIT into $DIR"
    git clone "$REPO_ROOT" "$DIR"
    git -C "$DIR" checkout "$COMMIT"
    if [ -n "$PATCH_FILE" ]; then
        git -C "$DIR" apply "$PATCH_FILE"
    fi
done

tmux kill-session -t "$SESSION" 2>/dev/null || true

for i in "${!COMMITS[@]}"; do
    COMMIT="${COMMITS[$i]}"
    DIR="/tmp/rl-tools-${COMMIT}"
    SHORT="${COMMIT:0:8}"

    if [ -n "$TARGET" ]; then
        CMD="cd $DIR && export RL_TOOLS_TEST_DATA_SOURCE=$REPO_ROOT/tests/data && cp $REPO_ROOT/tools/init.sh tools/init.sh && cp $REPO_ROOT/tests/download_data.sh ./tests/download_data.sh && bash ./tools/init.sh && cmake --build build --target $TARGET -j\$(nproc); exec bash"
    else
        CMD="cd $DIR && export RL_TOOLS_TEST_DATA_SOURCE=$REPO_ROOT/tests/data && cp $REPO_ROOT/tools/init.sh tools/init.sh && cp $REPO_ROOT/tests/download_data.sh ./tests/download_data.sh && bash ./tools/init.sh; exec bash"
    fi

    if [ "$i" -eq 0 ]; then
        tmux new-session -d -s "$SESSION" -n "$SHORT" "bash -c '$CMD'"
    else
        tmux new-window -t "$SESSION" -n "$SHORT" "bash -c '$CMD'"
    fi
done

exec tmux attach -t "$SESSION"
