### Recommended: Using docker to create the release
```
docker build -t backprop_tools:tests .
```
```
docker run --rm -it --mount type=bind,source=$(cd ..; pwd),target=/backprop_tools,readonly backprop_tools:tests
```
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON -DBACKPROP_TOOLS_ENABLE_CLI11:BOOL=ON
```
