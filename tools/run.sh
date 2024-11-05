# note this should be run from the project dir
export MKL_NUM_THREADS=1
export RL_TOOLS_EXTRACK_EXPERIMENT=$(date '+%Y-%m-%d_%H-%M-%S')
echo "RL_TOOLS_EXTRACK_EXPERIMENT=$RL_TOOLS_EXTRACK_EXPERIMENT"
#seq 0 99 | xargs -n 1 -P 4 -I {} ./cmake-build-wsl-release/src/rl/environments/l2f/dr_sac/rl_environments_l2f_dr_sac {}
seq 0 299 | xargs -n 1 -P 4 -I {} ./cmake-build-wsl-release/src/rl/environments/l2f/dr_sac/rl_environments_l2f_dr_sac {} true

