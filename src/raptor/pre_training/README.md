```
find src/raptor/dynamics_parameters | sort | grep -v '/\.[^/]*$' | grep 'json$' | MKL_NUM_THREADS=1 RL_TOOLS_EXTRACK_EXPERIMENT=$(date '+%Y-%m-%d_%H-%M-%S') xargs -I{} -P 16 /home/jonas/rl-tools/cmake-build-release/src/raptor/foundation_policy_pre_training {}
```