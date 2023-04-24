![BackpropTools: Mujoco Ant-v4 PPO](https://backprop.tools/backprop_tools_mujoco_ant_ppo.gif)

## Naming Convention
We use `snake_case` for variables/instances, functions as well as namespaces and `PascalCase` for structs/classes. Furthermore, we use upper case `CAMEL_CASE` for compile-time constants. 

## Building
```
mkdir build
cd build
CUDA_PATH=/opt/cuda-12.1 cmake .. \
-DBACKPROP_TOOLS_ENABLE_TESTS:BOOL=ON \
-DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON \
-DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON \
-DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON \
-DBACKPROP_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON \
-DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON \
-DBACKPROP_TOOLS_TEST_MACHINE_LENOVO_P1:BOOL=ON \
-DBACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON \
-DBACKPROP_TOOLS_TESTS_ENABLE_EIGEN:BOOL=ON
cmake --build . -j$(nproc)
```