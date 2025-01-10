export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=$MKL_NUM_THREADS
export RL_TOOLS_EXTRACK_EXPERIMENT=$(date '+%Y-%m-%d_%H-%M-%S')
timeout 2s /home/jonas/rl-tools/cmake-build-release/src/rl/zoo/rl_zoo_flag_sac --extrack-experiment $RL_TOOLS_EXTRACK_EXPERIMENT || true
echo "Real run -------------------------------------------"
cat experiment.sh | xargs -P "$1" -I{} bash -c '{} --extrack-experiment "$RL_TOOLS_EXTRACK_EXPERIMENT"'

