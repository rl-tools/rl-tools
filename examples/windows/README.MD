## Windows Examples
### Basic Requirements
Install the following software:
* [git](https://git-scm.com/download/win)
  * You can use the default settings for all options
* [CMake](https://cmake.org/install/)
* [Visual Studio](https://visualstudio.microsoft.com/)
  * Select the component for Desktop C++ development
  * This installs the required C++ compiler

Using the MinGW based `Git Bash` terminal, clone the repository:
```
git clone --recursive https://github.com/BackpropTools/BackpropTools.git
```
```
cd BackpropTools
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --target rl_environments_pendulum_td3_training_standalone
./src/rl/environments/penduolum/td3/cpu/Release/rl_environments_pendulum_td3_training_standalone.exe
```


### Extended Requirements
The extended requirements can be mixed and matched as needed they are not necessarily needed for every example (depending on the CMake configuration flags used)

#### [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html) (tested with oneMKL 2023.1.0)
BLAS library that accelerates matrix multiplications on Intel CPUs
After installing MKL use `CMD.exe` to source the `setvars.bat` which sets the search paths for the MKL libraries (don't forget to include the quotes):
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```
Alternatively if you are using PowerShell:
```
cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
```
Now, after moving to the previously created `build` directory, configure the BackpropTools to use MKL:
Sourcing MKL is importang because even if CMake can find the libraries during the build, the execution will fail because it can not find the libraries at runtime.
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON
```
After that build and run the target as before:
```
cmake --build . --config Release --target rl_environments_pendulum_td3_training_standalone
```
And execute the built binary:
```
src\rl\environments\pendulum\td3\cpu\Release\rl_environments_pendulum_td3_training_standalone.exe
```
#### __Protocol Buffers__
Data exchange format used by Tensorboard
Clone from: https://github.com/protocolbuffers/protobuf and build from source (tested useing v22.3). After building install it using `cmake --install .` (admin shell required)

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON
```

```
cmake --build . --config Release --target rl_environments_mujoco_ant_training_ppo_benchmark
```

```
bin\Release\rl_environments_mujoco_ant_training_ppo_benchmark.exe
```

#### __HDF5__
Download from e.g. https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.14/hdf5-1.14.0/bin/windows/hdf5-1.14.0-Std-win10_64-vs17.zip
and install using  the installer.
Add the HDF5 installation location to the CMake search path (in the environment variables)
```
CMAKE_PREFIX_PATH=C:\Program Files\HDF_Group\HDF5\1.14.0
```

#### __Boost__
Download from e.g. https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.zip

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON
```

```
cmake --build . --config Release --target rl_environments_mujoco_ant_training_ppo_standalone
```

```
bin\Release\rl_environments_mujoco_ant_training_ppo_benchmark.exe
```

#### [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) (we tested this on CUDA 12.1)



## Alternative using VCPKG

```
 -DCMAKE_TOOLCHAIN_FILE=../../vcpkg/scripts/buildsystems/vcpkg.cmake
```

## Alternative: using conan

```
conan profile detect --force
```
In the repository directory:
```
conan install . --output-folder=conan_debug --build=missing --settings=build_type=Debug
conan install . --output-folder=conan_release --build=missing --settings=build_type=Release
```
### VS Code
* Install the CMake Tools extension
* `ctrl-shift p` => `CMake: Select Configure Preset` => select the conan debug or release preset
* `ctrl-shift p` => `CMake: Set Build Preset` => select the build preset in accordance to the configuration preset 
* `ctrl-shift p` => `CMake: Set Debug Target` => select the desired target e.g. `rl_environments_mujoco_ant_training_ppo_standalone`
* `ctro-shift p` => `CMake: Debug`

### CLI
```
cmake .. -G "Visual Studio 15 2017" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake .. -DCMAKE_TOOLCHAIN_FILE:STRING=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON
```
```
cmake --build . --config Release --target rl_environments_mujoco_ant_training_ppo_standalone
```

Warning: Don't forget to source the MKL location (like described beforehand) before running it

```
.\bin\Release\rl_environments_mujoco_ant_training_ppo_standalone.exe
```