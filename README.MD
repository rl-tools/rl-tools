<div align="center">
  <center><h1><span style="color:#7DB9B6">BackpropTools</span>: A Fast, Portable Deep Reinforcement Learning Library for Continuous Control</h1></center>
</div>

<div align="center">
  <img src="https://github.com/BackpropTools/media/blob/master/backprop_tools_mujoco_ant_ppo.gif" alt="animated" width='600'/>
</div>

<div align="center">
    Trained on a 2020 MacBook Pro (M1) using <span style="color:#7DB9B6">BackpropTools</span> PPO
</div>

## Naming Convention
We use `snake_case` for variables/instances, functions as well as namespaces and `PascalCase` for structs/classes. Furthermore, we use upper case `CAMEL_CASE` for compile-time constants. 

## Getting Started
The most deterministic way to get started using BackpropTools is to use Docker. Mandatory prerequesites for this is:
1. Docker is installed ([official installation instructions](https://docs.docker.com/engine/install/))
 
Furthermore, if you want to use GPU acceleration you need the NVIDIA Docker runtime to be able to acceess the GPU from within the container
2. NVIDIA Container Toolkit ([official installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
3. An up to date NVIDIA driver (in the examples we are using CUDA 12.x hence a driver >=525.60.13 is required)


## CLion
#### Examplary settings for a Intel-based machine with a CUDA-compatible GPU
CMake options:
```
-DBACKPROP_TOOLS_ENABLE_TESTS:BOOL=ON
-DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON
-DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON
-DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON
-DBACKPROP_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON
-DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON
-DBACKPROP_TOOLS_TEST_MACHINE_LENOVO_P1:BOOL=ON
-DBACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON
-DBACKPROP_TOOLS_TESTS_ENABLE_EIGEN:BOOL=ON
```
Environment:
```
CUDA_PATH=/opt/cuda-12.1; MKL_ROOT=/opt/intel/oneapi/mkl/2023.1.0/
```

## Building
```
mkdir build
cd build
CUDA_PATH=/opt/cuda-12.1 MKL_ROOT=/opt/intel/oneapi/mkl/2023.1.0 cmake .. \
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
