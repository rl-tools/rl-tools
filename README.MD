<div align="center">
  <center><h1><span style="color:#7DB9B6">BackpropTools</span>: A Fast, Portable Deep Reinforcement Learning Library for Continuous Control</h1></center>
</div>

<div align="center">
  <img src="https://github.com/BackpropTools/media/blob/master/backprop_tools_mujoco_ant_ppo.gif" alt="animated" width='900'/>
</div>

<div align="center">
    Trained on a 2020 MacBook Pro (M1) using <span style="color:#7DB9B6">BackpropTools</span> PPO
</div>

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
