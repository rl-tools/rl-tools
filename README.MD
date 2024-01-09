<div align="center">
  <center><h1><span style="color:#7DB9B6">RLtools</span>: A Fast, Portable Deep Reinforcement Learning Library for Continuous Control</h1></center>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2306.03530">Paper on arXiv</a> | <a href="https://rl.tools">Live demo (browser)</a> 
  </br>
</br>
  <a href="https://mybinder.org/v2/gh/rl-tools/documentation/binder?labpath=01-Containers.ipynb">
  <img src="https://mybinder.org/badge_logo.svg" alt="Run tutorials on Binder">
  </a>
  <a href="https://docs.rl.tools">
  <img src="https://img.shields.io/badge/Documentation-Read%20the%20Docs-blue.svg" alt="Documentation">
  </a>
</p>



<div align="center">
<img src="https://github.com/rl-tools/media/blob/master/pendulum_v1_inference.gif" alt="animated" height='200'/>
<img src="https://github.com/rl-tools/media/blob/master/racing_car.gif" alt="animated" height='200'/>
</div>
<div align="center">
    Trained on a 2020 MacBook Pro (M1) using <span style="color:#7DB9B6">RLtools</span> TD3
</div>
</br>

<div align="center">
<img src="https://github.com/rl-tools/media/blob/master/rl_tools_mujoco_ant_ppo.gif" alt="animated" width='350'/>  
</div>

<div align="center">
    Trained on a 2020 MacBook Pro (M1) using <span style="color:#7DB9B6">RLtools</span> PPO
</div>
</br>

## Benchmarks

<div align="center">
<img src="https://github.com/rl-tools/media/blob/master/benchmark.png"/ width=600>  
</div>
<div align="center">
    Benchmarks of training the Pendulum swing-up using different RL libraries and across different devices (RLtools)
</div>

</br>
<div align="center">
<img src="https://github.com/rl-tools/media/blob/master/microcontroller_inference.png"/ width=600>  
</div>
<div align="center">
    Benchmarks of the inference frequency for a two-layer [64, 64] fully-connected neural network across different microcontrollers (types and architectures).
</div>





## Content
- [Getting Started](#getting-started)
  - [Cloning the Repository](#cloning-the-repository)
  - [Docker](#docker)
  - [Native](#native)
    - [Unix (Linux and macOS)](#unix-linux-and-macos)
    - [Windows](#windows)
- [Embedded Platforms](#embedded-platforms)
- [Naming Convention](#naming-convention)
- [Citing](#citing)

## Algorithms
| Algorithm | Example                                                                                                                                                                                                                             |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **TD3**   | [Pendulum](./src/rl/environments/pendulum/td3/cpu/training.h), [Car](./src/rl/environments/car/car.cpp), [MuJoCo Ant-v4](./src/rl/environments/mujoco/ant/td3/training.h), [Acrobot](./src/rl/environments/acrobot/td3/acrobot.cpp) |
| **PPO**   | [MuJoCo Ant-v4 (CPU)](./src/rl/environments/mujoco/ant/ppo/cpu/training.h), [MuJoCo Ant-v4 (CUDA)](./src/rl/environments/mujoco/ant/ppo/cuda/training.h)                                                                            |
| **SAC**   | [Pendulum](./src/rl/environments/pendulum/sac/cpu/training.cpp), [Acrobot](./src/rl/environments/acrobot/sac/acrobot.cpp)                                                                                                           |
## Getting Started
The getting started documentation is divided in two parts: a tutorial on how RLtools works internally and replication instructions for the results from the paper. 
### Tutorial on RLtools internals
| Chapter |  Documentation                                                                                | Interactive Notebook                                                                                                                                    |
|---------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0       | [Overview                    ](https://docs.rl.tools/overview.html)                           | -                                                                                                                                                       |
| 1       | [Containers                  ](https://docs.rl.tools/01-Containers.html)                      | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rl-tools/documentation/binder?labpath=01-Containers.ipynb)                  | 
| 2       | [Multiple Dispatch           ](https://docs.rl.tools/02-Multiple%20Dispatch.html)             | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rl-tools/documentation/binder?labpath=02-Multiple%20Dispatch.ipynb)         | 
| 3       | [Deep Learning               ](https://docs.rl.tools/03-Deep%20Learning.html)                 | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rl-tools/documentation/binder?labpath=03-Deep%20Learning.ipynb)             | 
| 4       | [CPU Acceleration            ](https://docs.rl.tools/04-CPU%20Acceleration.html)              | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rl-tools/documentation/binder?labpath=04-CPU%20Acceleration.ipynb)          | 
| 5       | [MNIST Classification        ](https://docs.rl.tools/05-MNIST%20Classification.html)          | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rl-tools/documentation/binder?labpath=05-MNIST%20Classification.ipynb)      | 
| 6       | [Deep Reinforcement Learning ](https://docs.rl.tools/06-Deep%20Reinforcement%20Learning.html) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rl-tools/documentation/binder?labpath=06-Deep%20Reinforcement%20Learning.ipynb) | 

Note: you can also run the tutorial (Jupyter Notebooks) locally using a single command:
```
docker run -p 8888:8888 rltools/documentation
```
After running the Docker container, open the link that is displayed in the CLI (http://127.0.0.1:8888/...) in your browser and enjoy tinkering with the tutorial!

### Cloning the repository
To build the examples from source (either in Docker or natively), first the repository should be cloned.
Instead of cloning all submodules using `git clone --recursive` which takes a lot of space and bandwidth we recommend cloning the main repo containing all the standalone code for `RLtools` and then cloning the required sets of submodules later:
```
git clone https://github.com/rl-tools/rl-tools.git rl_tools
```
#### Cloning submodules
There are three classes of submodules:
1. External dependencies (in `external/`)
   * E.g. HDF5 for checkpointing, Tensorboard for logging, or MuJoCo for the simulation of contact dynamics
2. Examples/Code for embedded platforms (in `embedded_platforms/`)
3. Redistributable dependencies (in `redistributable/`)
4. Test dependencies (in `tests/lib`)
4. Test data (in `tests/data`)

These sets of submodules can be cloned additively/independent of eachother.
For most use-cases (like e.g. most of the Docker examples) you should clone the submodules for external dependencies:
```
cd RLtools
```
```
git submodule update --init --recursive -- external
```

The submodules for the embedded platforms, the redistributable binaries and test dependencies/data can be cloned in the same fashion (by replacing `external` with the appropriate folder from the enumeration above). 
Note: Make sure that for the redistributable dependencies and test data `git-lfs` is installed (e.g. `sudo apt install git-lfs` on Ubuntu) and activated (`git lfs install`) otherwise only the metadata of the blobs is downloaded.

### Docker
The most deterministic way to get started using <span style="color:#7DB9B6">RLtools</span> not only for replication of the results but for modifying the code is using Docker. In our experiments on Linux using the NVIDIA container runtime we were able to achieve close to native performance.
[Docker instructions & examples](examples/docker/README.MD)

### Native
In comparison to running the release binaries or building from source in Docker, the native setup heavily depends on the configuration of the machine it is run on (installed packages, overwritten defaults etc.). Hence we provide guidelines on how to setup the environment for research and development of <span style="color:#7DB9B6">RLtools</span> that should run on the default configuration of the particular platform but might not work out of the box if it has been customized.  
#### Unix (Linux and macOS)
For maximum performance and malleability for research and development we recommend to run <span style="color:#7DB9B6">RLtools</span> natively on e.g. Linux or macOS. Since <span style="color:#7DB9B6">RLtools</span> itself is dependency free the most basic examples don't need any platform setup. However, for an improved experience, we support HDF5 checkpointing and Tensorboard logging as well as optimized BLAS libraries which comes with some system-dependent requirements. 
[Unix instructions & examples](examples/unix/README.MD)

#### Windows
[Windows instructions & examples](examples/windows/README.MD)

## Embedded Platforms
### Inference & Training
- [iOS](https://github.com/RLtools/iOS)
- [teensy](https://github.com/RLtools/teensy)
### Inference
- [Crazyflie](embedded_platforms/crazyflie)
- [ESP32](embedded_platforms/esp32)
- [PX4](https://github.com/RLtools/rl_tools_px4)

## Naming Convention
We use `snake_case` for variables/instances, functions as well as namespaces and `PascalCase` for structs/classes. Furthermore, we use upper case `SNAKE_CASE` for compile-time constants. 

## Citing
When using RLtools in an academic work please cite our publication using the following Bibtex citation:
```
@misc{eschmann2023rltools,
      title={RLtools: A Fast, Portable Deep Reinforcement Learning Library for Continuous Control}, 
      author={Jonas Eschmann and Dario Albani and Giuseppe Loianno},
      year={2023},
      eprint={2306.03530},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
