#!/usr/bin/env bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
$SCRIPT_DIR/init.sh

export RL_TOOLS_EXTRACK_EXPERIMENT="$(date '+%Y-%m-%d_%H-%M-%S')"
unset OMP_NUM_THREADS
N_PROC="${N_PROC:-$(nproc)}"
echo "N_PROC: $N_PROC"
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mapfile -t TARGETS < build/rl_zoo_nightly_targets.txt
echo "Nightly targets:"
printf '  %s\n' "${TARGETS[@]}"
echo "Total: ${#TARGETS[@]} targets"

cmake --build build -j $N_PROC --target rl_zoo_nightly

run() {
  echo "tmux send-keys -t \$TMUX_PANE '$* && sleep 10 && exit' C-m; exec bash -i"
}

map() {
  local base_cmd="$1"
  local options="$2"

  echo "$options" | while IFS= read -r opt; do
    if [[ -n "$opt" ]]; then
      run "$base_cmd $opt"
    fi
  done
}

N_SEEDS=10

TMPF=$(mktemp)
{
  for t in "${TARGETS[@]}"; do
    for s in $(seq 0 $((N_SEEDS - 1))); do
      run "./build/src/rl/zoo/$t" -s "$s"
    done
  done
#  map "./build/src/rl/zoo/rl_zoo_ant_v4_ppo" "$(seq 0 9 | xargs -I{} echo -s {})"
#  map "./build/src/rl/zoo/rl_zoo_pendulum_v1_sac" "$(seq 0 5 | xargs -I{} echo -s {})"
#  run "./build/src/rl/zoo/rl_zoo_pendulum_v1_sac -s 1337"
} | tee /dev/stderr | parallel --tmux -j $N_PROC