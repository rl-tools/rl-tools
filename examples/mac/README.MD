## macOS (native) Examples
### Basic Requirements
To compile the examples on macOS, a C++17 compiler is required. On macOS, the tools required for building code (git, clang, etc.) can be installed with a single command using the command line:
```
xcode-select --install
```
Furthermore `cmake` is required to configure the build. On macOS, this can be installed by either:
* Following the instructions on the [CMake website](https://cmake.org/install/)
* Or installing it using [homebrew](https://brew.sh/): `brew install cmake`

With this setup you can run the basic rl example:

[__01 - Basic RL__](01_basic_rl/README.MD): Dependency-free RL

### Extended Requirements
Note: Tested on macOS Monterey using the specified versions (it is highly likely that it also works with the most recent (minor) versions)
#### HDF5: Persisting models to disk
To conveniently persist trained models and other numeric data to disk BackpropTools uses the popular `HDF5` format hence for the more advanced examples, the `HDF5` library is required. On macOS, this can be installed using `brew`:
```
brew install hdf5@1.12
```
#### Tensorboard: Logging and visualizing the training progress
BackpropTools supports Tensorboard logging, hence it is convenient to have Tensorboard installed. Assuming `python3` and `pip` have been already installed:
```
pip install tensorboard
```
The [Tensorboard C++ interface](https://github.com/RustingSword/tensorboard_logger) depends on Protocol buffers which can be installed using `brew`:
```
brew install protobuf@21.12
```
#### MuJoCo UI: Visualizing trained policies
The MuJoCo UI requires `glfw3` which can be installed on macOS using `brew`:
```
brew install glfw@3.3.8
```

From here you can continue with the instructions for the following specific examples:
* [__02 - Accelerate__](02_blas/README.MD): Examples running on CPU but accelerated using Intel MKL
* [__03 - CUDA__](03_cuda/README.MD): Examples running on the GPU
* [__04 - MuJoCo UI__](04_mujoco_ui/README.MD): Using checkpoints saved from the previous example to visualize the trained policy in the MuJoCo Simulator
* [__05 - WASM__](05_wasm/README.MD): Examples running in the browser using WebAssembly
