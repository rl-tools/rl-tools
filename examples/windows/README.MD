## Windows Examples
### Content
* Basic requirements (git, compiler)
* Extended requirements for accelerated training (Intel MKL, NVIDIA CUDA)
* Additional requirements for convenience (HDF5 checkpointing, Tensorboard logging), either:
  * Installation using the conan package manager

Tested on Windows 10 using Visual Studio 2022
### Basic Requirements
Install the following software:
* [git](https://git-scm.com/download/win)
  * You can use the default settings for all options
* [Visual Studio](https://visualstudio.microsoft.com/)
  * Select the component for Desktop C++ development
  * This installs the required C++ compiler

Using the `Developer Command Prompt for VS XXXX` (which is `CMD.exe` but including the paths to developer tools like CMake)
```
git clone --recursive https://github.com/BackpropTools/BackpropTools.git
```
```
cd BackpropTools
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --target rl_environments_pendulum_td3_training_standalone
src\rl\environments\pendulum\td3\cpu\Release\rl_environments_pendulum_td3_training_standalone.exe
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
Note: This step needs to be re-done after every restart of the terminal.
Now, after moving to the previously created `build` directory, configure the BackpropTools to use MKL:
Sourcing MKL is important because even if CMake can find the libraries during the build, the execution will fail because it can not find the libraries at runtime.
For VS code you can alter the link in the start menu entry to contain the sourcing adding `setvars.bat` to the Target field in the same manner as e.g. in the following:
```
Target: "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && "C:\Users\Jonas.Eschmann\AppData\Local\Programs\Microsoft VS Code\Code.exe"
```
Instead of using VS code after sourcing the `setvars.bat` in the terminal like described before you can also build it from the cli:
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
#### [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) (we tested this on CUDA 11.8 and 12.1)

### Alternative: using conan
It is convenient to let conan manage the exact dependency versions. We are using the recently released conan `2.0` to manage all dependencies except Intel MKL and NVIDIA CUDA which should be installed apriori.
```
conan profile detect --force
```
In the repository directory:
```
conan install . --output-folder=build_conan --build=missing --settings=build_type=Debug
conan install . --output-folder=build_conan --build=missing --settings=build_type=Release
```
### VS Code
* Install the CMake Tools extension
* `ctrl-shift p` => `CMake: Select Configure Preset` => select the conan debug or release preset
* `ctrl-shift p` => `CMake: Set Build Preset` => select the build preset in accordance to the configuration preset 
* `ctrl-shift p` => `CMake: Set Debug Target` => select the desired target e.g. `rl_environments_mujoco_ant_training_ppo_standalone`
* `ctro-shift p` => `CMake: Debug`
* Note: Make sure not to build while the target is running (running the executable seems to prevent the file to be overwritten by the linker)

### CLI
```
cmake .. -DCMAKE_TOOLCHAIN_FILE:STRING=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON
```
```
cmake --build . --config Release --target rl_environments_mujoco_ant_training_ppo_standalone
```

Warning: Don't forget to source the MKL location (like described beforehand) before running it

```
.\bin\Release\rl_environments_mujoco_ant_training_ppo_standalone.exe
```

### Packaging Release
The command to create the release binaries is the following (in the previously set up conan_build directory)
```
cmake .. -DCMAKE_TOOLCHAIN_FILE:STRING=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DBACKPROP_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON -DBACKPROP_TOOLS_ENABLE_CLI11:BOOL=ON -DBACKPROP_TOOLS_BUILD_TYPE:STRING=Release -DBACKPROP_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES:BOOL=ON -DCMAKE_INSTALL_PREFIX:STRING=install
```
The build can take a long time because it compiles and packages the CUDA device code for every CUDA compute capability not only for the one supported by the GPU in the current machine:
```
cmake --build . --config Release
```
This will install the binaries and all dependencies into an `install` subdirectory in the current binary directory e.g. `build_conan` where they can be found under `install/backprop_tools/bin`. This folder can be distributed and should run on any Windows 10 computer independent of any additional dependencies.
```
cmake --install .
```

### Testing the dependencielessness of the release binaries using Windows Docker containers
Replace the path with the absolute path to where you unzipped the release archive
```
docker run -it --rm -v C:\Users\Jonas.Eschmann\git\BackpropTools\build_conan\install\backprop_tools\bin:C:\app -w C:\app mcr.microsoft.com/windows/nanoserver:ltsc2019 rl_environments_pendulum_td3_training_blas.exe
```