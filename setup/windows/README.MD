# Example Setup: Windows


Like described in [https://github.com/rl-tools/example](https://github.com/rl-tools/example) `RLtools` should work out of the box. For hardware acceleration and other conveniences additional setup is required:

The following is tested with the Visual Studio 2022 C++ compiler on Windows 11

## Hardware Acceleration (MKL)
Install Intel MKL/oneMKL (tested with `2024.0`). You should find a `setvars.bat` to activate the MKL environment. This has to be done in the Command Prompt (not in PowerShell!)
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

After that you can start e.g. CLion:
```
"C:\Program Files\JetBrains\CLion 2023.3.3\bin\clion64.exe"
```

## CUDA
If you have CUDA installed and the `CUDA_PATH` environment variable is set you should be able to compile the CUDA targets using `RL_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON` in the CMake configuration step. Tested with CUDA `12.3` and `11.8`.

## Ameneities (HDF5 checkpointing, Tensorboard logging)
On linux we can use existing package managers to install the required libraries. On Windows we use `conan`. The dependencies are configured in the `conanfile.txt`.
If you are using CLion, you can just use the `conan` plugin which integrates it with your CMake profiles.

Initialize the desired submodules:
```
git submodule update --init -- external/cli11
git submodule update --init -- external/glfw
git submodule update --init -- external/highfive
git submodule update --init -- external/json
git submodule update --init -- external/libattopng
git submodule update --init -- external/mujoco
git submodule update --init -- external/tensorboard
```

From the command line:
```
conan install . --output-folder=build_conan --build=missing --settings=build_type=Release
```
The following command assumes MKL and CUDA (in its options):
```
cmake .. -DCMAKE_TOOLCHAIN_FILE:STRING=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_ENABLE_TARGETS=ON -DRL_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DRL_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON -DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DRL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON -DRL_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DRL_TOOLS_ENABLE_HDF5:BOOL=ON -DRL_TOOLS_ENABLE_CLI11:BOOL=ON -DRL_TOOLS_BUILD_TYPE:STRING=Release -DRL_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES:BOOL=ON -DCMAKE_INSTALL_PREFIX:STRING=install
```