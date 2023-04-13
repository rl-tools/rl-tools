```
./build.sh
docker run --rm -it -p 6006:6006 --mount type=bind,source=$(cd ../../..; pwd),target=/backprop_tools,readonly --cap-add=sys_nice --runtime=nvidia --gpus all backprop_tools:cuda
CC=clang CXX=clang++ cmake /backprop_tools/ -DLAYER_IN_C_ENABLE_TESTS:BOOL=ON -DLAYER_IN_C_BACKEND_ENABLE_MKL:BOOL=ON -DLAYER_IN_C_BACKEND_ENABLE_CUDA:BOOL=ON -DLAYER_IN_C_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=ON
cmake --build . --target test_rl_environments_mujoco_ant_training_ppo_cuda -j$(nproc)
```
Moving to the src folder (so that the Ant-v4 env can find the xml file, if it can't be found MuJoCo will segfault)
```
cd /backprop_tools/; nice -n -20 /build/tests/src/rl/environments/mujoco/ant/cuda/test_rl_environments_mujoco_ant_training_ppo_cuda
tensorboard --logdir logs --bind_all
```
