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

[__00 - Basic RL__: Dependency-free RL](../examples/00_BASIC_RL.MD)

### Extended Requirements
Note: Tested on macOS Monterey using the specified versions (it is highly likely that it also works with the most recent (minor) versions)
#### HDF5: Persisting models to disk
To conveniently persist trained models and other numeric data to disk RLtools uses the popular `HDF5` format hence for the more advanced examples, the `HDF5` library is required. On macOS, this can be installed using `brew`:
```
brew install hdf5
```
#### Tensorboard: Logging and visualizing the training progress
RLtools supports Tensorboard logging, hence it is convenient to have Tensorboard installed. Assuming `python3` and `pip` have been already installed:
```
pip install tensorboard
```
The [Tensorboard C++ interface](https://github.com/RustingSword/tensorboard_logger) depends on Protocol buffers which can be installed using `brew`:
```
brew install protobuf
```
#### MuJoCo UI: Visualizing trained policies
The MuJoCo UI requires `glfw3` which can be installed on macOS using `brew`:
```
brew install glfw
```

From here you can continue with the instructions for the accelerated examples:

[__01 - BLAS__: Accelerated RL](../examples/01_BLAS.MD)
