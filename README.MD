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





## Content
1. [Getting Started](#getting-started)
   1. [Cloning the repository](#cloning-the-repository)
   2. [Docker](#docker)
   3. [Unix](#unix)
      1. [Linux](#linux)
      2. [macOS](#macos)
   4. [Windows](#windows)
2. [Naming Convention](#naming-convention)
## Getting Started
### Cloning the repository
```
git clone --recursive https://github.com/BackpropTools/BackpropTools.git
```
### Docker
The most deterministic way to get started using BackpropTools is to use Docker. In our experiments on Linux using the NVIDIA container runtime we were able to achieve close to native performance.
[Docker instructions & examples](examples/docker/README.MD)

### Unix
For maximum performance and malleability for research and development we recommend to run BackpropTools natively on e.g. Linux or macOS. Since BackpropTools itself is dependency free the most basic examples don't need any platform setup. However, for an improved experience, we support HDF5 checkpointing and Tensorboard logging as well as optimized BLAS libraries which comes with some system-dependent requirements. 
[Unix instructions & examples](examples/unix/README.MD)

### Windows
tbd

## Naming Convention
We use `snake_case` for variables/instances, functions as well as namespaces and `PascalCase` for structs/classes. Furthermore, we use upper case `CAMEL_CASE` for compile-time constants. 
