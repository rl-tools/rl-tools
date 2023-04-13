#!/usr/bin/env bash
cd ../../../../../../..
pwd
while true
do
  sudo time nice -n -20 cmake-build-release-clang-16/tests/src/rl/environments/mujoco/ant/cuda/test_rl_environments_mujoco_ant_training_ppo_cuda
done