### Packaging Release using conan
```
conan profile detect --force
```
Inside the root of the RLtools repo:
```
conan install . --output-folder=build_conan --build=missing --settings=build_type=Release
```
The command to create the release binaries is the following (in the previously set up conan_build directory)
```
cd build_conan
```
```
cmake .. -DCMAKE_TOOLCHAIN_FILE:STRING=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_ENABLE_TARGETS=ON -DRL_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DRL_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON -DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DRL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON -DRL_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DRL_TOOLS_ENABLE_HDF5:BOOL=ON -DRL_TOOLS_ENABLE_CLI11:BOOL=ON -DRL_TOOLS_ENABLE_BOOST_BEAST:BOOL=ON -DRL_TOOLS_ENABLE_JSON:BOOL=ON -DRL_TOOLS_BUILD_TYPE:STRING=Release -DRL_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES:BOOL=ON -DCMAKE_INSTALL_PREFIX:STRING=install
```
The build can take a long time because it compiles and packages the CUDA device code for every CUDA compute capability not only for the one supported by the GPU in the current machine:
```
cmake --build . --config Release
```
This will install the binaries and all dependencies into an `install` subdirectory in the current binary directory e.g. `build_conan` where they can be found under `install/rl_tools/bin`. This folder can be distributed and should run on any Windows 10 computer independent of any additional dependencies.
```
cmake --install .
```

### Testing the dependencielessness of the release binaries using Windows Docker containers
Replace the path with the absolute path to where you unzipped the release archive
```
docker run -it --rm -v C:\Users\Jonas.Eschmann\rl_tools_example\external\rl_tools\build_conan\install\bin:C:\app -w C:\app mcr.microsoft.com/windows/nanoserver:ltsc2019 rl_environments_mujoco_ant_ppo_blas.exe
```
