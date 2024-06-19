Redistributable libraries included:
NVIDIA CUBLAS: 12.4
Intel oneMKL: 2023.1

These binaries should be mostly OS independent. They were built using Ubuntu 20.04 hence all distributions that have a libstdc++ that corresponds to gcc 9.3 should be able to run them. We have tested them successfully on Ubuntu 22.04 and a recent Arch Linux setup.

The CUDA examples can e.g. also be run from docker (assuming the relese dir is the cwd):
docker run --gpus all -it --mount type=bind,source=$(pwd),target=/rl_tools nvidia/cuda:12.4.0-base-ubuntu22.04 /rl_tools/bin/rl_environments_mujoco_ant_ppo_cuda_full

I also found this helpful for keeping my laptop in performance mode:
while true; do sleep 1; echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; done

Finally, you can give the process the highest priority:
docker run --gpus all -it --cap-add=SYS_NICE --mount type=bind,source=$(pwd),target=/rl_tools nvidia/cuda:12.4.0-base-ubuntu22.04 nice -n-20 /rl_tools/bin/rl_environments_mujoco_ant_ppo_cuda_full