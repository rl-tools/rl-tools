### Go to the current directory
```
cd examples/docker/03_cuda
```
### Building the environment
Build the docker container (based on an NVIDIA cuda ubuntu image) installing Intel MKL as well as a recent version of CMake. The image is tagged as `backprop_tools:cuda`
```
docker build -f Dockerfile -t backprop_tools:cuda . --no-cache
```
### Running the container and configuring BackpropTools
Run the container in interactive mode, mounting the source code from the host in readonly mode. We expose port 6006 for tensorboard and allow the sys_nice capability to be able to run the tests with a high priority
```
docker run --rm -it -p 6006:6006 --mount type=bind,source=$(cd ../../..; pwd),target=/backprop_tools,readonly --cap-add=sys_nice --mount type=bind,source=$(pwd)/../checkpoints,target=/build/checkpoints --runtime=nvidia --gpus all backprop_tools:cuda
```
Configure the BackpropTools using CMake
```
CC=clang CXX=clang++ cmake /backprop_tools/ -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DBACKPROP_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON -DBACKPROP_TOOLS_ENABLE_CLI11:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON
```
And build the targets
```
cmake --build . -j$(nproc)
```
Execute PPO based Ant-v4 training in the benchmark configuration (without checkpoints and tensorboard logging) with a high priority
```
nice -n -20 /build/src/rl/environments/mujoco/ant/ppo/cuda/rl_environments_mujoco_ant_training_ppo_cuda_benchmark
```
If you want to save checkpoints (e.g. to replay the behavior in the MuJoCo UI as described in [03_mujoco_ui](../03_mujoco_ui/README.MD)) and Tensorboard logs run the following
Optional: Start tensorboard in the background
```
tensorboard --logdir logs --bind_all 2> /dev/null > /dev/null &
```
Run the executable with checkpointing and Tensorboard logging enabled:
```
nice -n -20 /build/src/rl/environments/mujoco/ant/ppo/cuda/rl_environments_mujoco_ant_training_ppo_cuda_full
```
If you ran Tensorboard in the background before, you can observe the training using the browser on your host machine to access Tensorboard at `http://localhost:6006/`.
