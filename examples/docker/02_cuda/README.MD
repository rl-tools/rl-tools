### Go to the current directory
```
cd backprop_tools/examples/docker/02_cuda
```
### Building the environment
Build the docker container (based on an NVIDIA cuda ubuntu image) installing Intel MKL as well as a recent version of CMake. The image is tagged as `backprop_tools:cuda`
```
./build.sh
```
### Running the container and configuring BackpropTools
Run the container in interactive mode, mounting the source code from the host in readonly mode. We expose port 6006 for tensorboard and allow the sys_nice capability to be able to run the tests with a high priority
```
mkdir /tmp/backprop_tools_checkpoints
docker run --rm -it -p 6006:6006 --mount type=bind,source=$(cd ../../..; pwd),target=/backprop_tools,readonly --cap-add=sys_nice --mount type=bind,source=/tmp/backprop_tools_checkpoints,target=/build/checkpoints --runtime=nvidia --gpus all backprop_tools:cuda
```
Configure the BackpropTools using CMake
```
CC=clang CXX=clang++ cmake /backprop_tools/ -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DBACKPROP_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON -DBACKPROP_TOOLS_ENABLE_CLI11:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON
```
To benchmark the training speed you can disable checkpoints, deterministic evaluation and tensorboard logging using the flag `BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_BENCHMARK`:
```
CC=clang CXX=clang++ cmake /backprop_tools/ -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DBACKPROP_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON -DBACKPROP_TOOLS_ENABLE_CLI11:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_BENCHMARK:BOOL=ON
```
And build the test_rl_environments_mujoco_ant_training_ppo_cuda target which trains the MuJoCo Ant-v4 environment using PPO on the GPU
```
cmake --build . --target rl_environments_mujoco_ant_training_ppo_cuda -j$(nproc)
```
Optional: Start tensorboard in the background
```
tensorboard --logdir logs --bind_all 2> /dev/null > /dev/null &
```
Execute the built binary with a high priority
```
nice -n -20 /build/src/rl/environments/mujoco/ant/ppo/cuda/rl_environments_mujoco_ant_training_ppo_cuda 
```
You can observe the training using the browser on your host machine to access Tensorboard at `http://localhost:6006/
