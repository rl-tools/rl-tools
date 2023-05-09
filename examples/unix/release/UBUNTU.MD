Optional: Set up using clang (which speeds up MuJoCo a bit)
```
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-16 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-16 100
CC=clang CXX=clang++ conan profile detect --name clang --force
```
Otherwise, use the default compiler for your os:
```
conan profile detect --force
```
In the repository directory:
```
conan install . --output-folder=build_conan --build=missing --settings=build_type=Debug && conan install . --output-folder=build_conan --build=missing --settings=build_type=Release
```

```
cd build_conan
```

```
CC=clang-16 CXX=clang++-16 CUDAToolkit_ROOT=/opt/cuda-11.8 cmake .. -DCMAKE_TOOLCHAIN_FILE:STRING=conan_toolchain.cmake -DCMAKE_CUDA_COMPILER=/opt/cuda-11.8/bin/nvcc -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DBACKPROP_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON -DBACKPROP_TOOLS_HDF5_HIGHFIVE_FROM_CONAN:BOOL=ON -DBACKPROP_TOOLS_ENABLE_CLI11:BOOL=ON -DBACKPROP_TOOLS_BUILD_TYPE:STRING=Release -DBACKPROP_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES:BOOL=ON -DCMAKE_INSTALL_PREFIX:STRING=install
```
```
cmake --build . -j$(nproc)
```
```
cmake --install .
```

```
(cd install/backprop_tools && tar -czvf backprop_tools-0.1.0-linux-x64_64.tar.gz *)
```

```
docker run -it --rm -w /bpt/bin -v $(pwd)/install/backprop_tools:/bpt archlinux 
````
```
docker run --rm -it -w /bpt/bin -v $(pwd)/install/backprop_tools:/bpt --cap-add=sys_nice --runtime=nvidia --gpus all archlinux nice -n -20 ./rl_environments_mujoco_ant_training_ppo_cuda_full 
````