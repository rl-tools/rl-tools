<div align="center">
  <center><h1><span style="color:#7DB9B6">BackpropTools</span>: A Fast, Portable Deep Reinforcement Learning Library for Continuous Control</h1></center>
</div>

<div align="center">
<img src="https://github.com/BackpropTools/media/blob/master/pendulum_v1_inference.gif" alt="animated" height='200'/>
</div>
<div align="center">
    Trained on a 2020 MacBook Pro (M1) using <span style="color:#7DB9B6">BackpropTools</span> TD3
</div>
</br>
<div align="center">
<img src="https://github.com/BackpropTools/media/blob/master/backprop_tools_mujoco_ant_ppo.gif" alt="animated" height='300'/>  
</div>

<div align="center">
    Trained on a 2020 MacBook Pro (M1) using <span style="color:#7DB9B6">BackpropTools</span> PPO
</div>





## Getting Started
### Content
1. [Cloning the repository](#cloning-the-repository)
2. [Docker requirements](#docker)
    1. [Mandatory](#mandatory)
    2. [Optional](#optional)
3. [Unix native](#unix)
4. 
### Cloning the repository
```
git clone --recursive https://github.com/BackpropTools/BackpropTools.git
```
### Docker
The most deterministic way to get started using BackpropTools is to use Docker. In our experiments on Linux using the NVIDIA container runtime we were able to achieve close to native performance.
[Docker instructions & examples](examples/docker/README.MD)

### Unix
For maximum performance and malleability for research and development we recommend to run BackpropTools natively on e.g. Linux or macOS. Since BackpropTools itself is dependency free the most basic examples don't need any platform setup. However, for an improved experience, we support HDF5 checkpointing and Tensorboard logging as well as optimized BLAS libraries which comes with some system-dependent requirements. 
#### Linux
In the case of Linux the dependencies are well-defined in the Dockerfiles of the [Docker examples](examples/docker/README.MD).
#### macOS

## CLion
#### Examplary settings for a Intel-based machine with a CUDA-compatible GPU
CMake options:
```
-DBACKPROP_TOOLS_ENABLE_TESTS:BOOL=ON
-DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON
-DBACKPROP_TOOLS_ENABLE_CLI11:BOOL=ON
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

## Naming Convention
We use `snake_case` for variables/instances, functions as well as namespaces and `PascalCase` for structs/classes. Furthermore, we use upper case `CAMEL_CASE` for compile-time constants. 
