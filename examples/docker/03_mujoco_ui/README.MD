### Go to the current directory
```
cd backprop_tools/examples/docker/03_cuda
```
### Building the environment
This docker image builds on top of the `backprop_tools:cuda` which is built in [02_cuda](../02_cuda/README.MD) and tagged as `backprop_tools:cuda-ui`
```
./build.sh
```
### Running the container and configuring BackpropTools
```
xhost +local:docker
```
```
docker run --rm -it -e DISPLAY=$DISPLAY --device=/dev/dri -v /tmp/.X11-unix:/tmp/.X11-unix --mount type=bind,source=$(cd ../../..; pwd),target=/backprop_tools,readonly --mount type=bind,source=/tmp/backprop_tools_checkpoints,target=/checkpoints,readonly backprop_tools:cuda-ui
```
Configure BackpropTools using CMake (note the additional `-DBACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON` compared to `02_cuda` which enables the Mujoco UI targets)
```
cmake /backprop_tools/ -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_ENABLE_CLI11:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON
```
```
cmake --build . --target rl_environments_mujoco_ant_evaluation_ppo -j$(nproc)
```
```
cd /
/build/src/rl/environments/mujoco/ant/
```
