### Go to the current directory
```
cd examples/docker/04_mujoco_ui
```
### Building the environment
Build the docker image and tag it as `rl_tools:mujoco_ui`
```
docker build -f Dockerfile -t rl_tools:mujoco_ui . --no-cache
```
### Running the container and configuring RLtools
```
xhost +local:docker
```
```
docker run --rm -it -e DISPLAY=$DISPLAY --device=/dev/dri -v /tmp/.X11-unix:/tmp/.X11-unix --mount type=bind,source=$(cd ../../..; pwd),target=/rl_tools,readonly --mount type=bind,source=$(pwd)/../checkpoints,target=/checkpoints,readonly rl_tools:mujoco_ui
```
Configure RLtools using CMake (note the additional `-DRL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON` compared to `02_cuda` which enables the Mujoco UI targets)
```
cmake /rl_tools/ -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_ENABLE_TARGETS:BOOL=ON -DRL_TOOLS_ENABLE_CLI11:BOOL=ON -DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DRL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON -DRL_TOOLS_ENABLE_HDF5:BOOL=ON
```
```
cmake --build . -j$(nproc)
```
```
cd /
/build/src/rl/environments/mujoco/ant/rl_environments_mujoco_ant_evaluation_ppo
```
