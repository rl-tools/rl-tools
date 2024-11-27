# note this should be run from the project dir
# usage: ./tools/run.sh /home/jonas/rl-tools/cmake-build-release/src/rl/zoo/rl_zoo_l2f_sac 20
set -e
export MKL_NUM_THREADS=1
export RL_TOOLS_EXTRACK_EXPERIMENT=$(date '+%Y-%m-%d_%H-%M-%S')
echo "RL_TOOLS_EXTRACK_EXPERIMENT=$RL_TOOLS_EXTRACK_EXPERIMENT"
$1 -s 0 --extrack-experiment $RL_TOOLS_EXTRACK_EXPERIMENT
seq 1 $2 | xargs -n 1 -P 16 -I {} $1 -s {} --extrack-experiment $RL_TOOLS_EXTRACK_EXPERIMENT

