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

TARGETS=(
  rl_zoo_acrobot_swingup_v0_sac
  rl_zoo_ant_v4_ppo
  rl_zoo_ant_v4_td3
  rl_zoo_bottleneck_v0_ppo
  rl_zoo_flag_ppo
  rl_zoo_flag_ppo_gru
  rl_zoo_flag_ppo_gru_asymmetric
  rl_zoo_flag_sac
  rl_zoo_flag_td3
  rl_zoo_l2f_ppo
  rl_zoo_l2f_sac
  rl_zoo_l2f_td3
  rl_zoo_pendulum_v1_ppo
  rl_zoo_pendulum_v1_sac
  rl_zoo_pendulum_v1_td3
  rl_zoo_reacher_v0_ppo
  rl_zoo_reacher_visual_v0_ppo
)

cmake --build build -j $N_PROC --target "${TARGETS[@]}"

run() {
  echo "tmux send-keys -t \$TMUX_PANE '$*' C-m; exec bash -i"
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

TMPF=$(mktemp)
{
  for t in "${TARGETS[@]}"; do
    for s in $(seq 0 9); do
      run "./build/src/rl/zoo/$t" -s "$s"
    done
  done
#  map "./build/src/rl/zoo/rl_zoo_ant_v4_ppo" "$(seq 0 9 | xargs -I{} echo -s {})"
#  map "./build/src/rl/zoo/rl_zoo_pendulum_v1_sac" "$(seq 0 5 | xargs -I{} echo -s {})"
#  run "./build/src/rl/zoo/rl_zoo_pendulum_v1_sac -s 1337"
} | tee /dev/stderr | parallel --tmux -j $N_PROC
# 2>"$TMPF" 
# &
# PARALLEL_PID=$!
# sleep 0.5


# ATTACH_CMD=$(grep -oP 'tmux -S \S+ attach' "$TMPF" | head -n 1)
# SOCKET=$(echo "$ATTACH_CMD" | grep -oP '(?<=-S )\S+')
# rm "$TMPF"

# trap "tmux -S \"$SOCKET\" kill-session 2>/dev/null; kill $PARALLEL_PID 2>/dev/null" EXIT INT TERM

# tmux -S "$SOCKET" attach
# wait $PARALLEL_PID