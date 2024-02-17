### Recommended: Using docker to create the release
```
cd release/linux
```
```
rm -rf install/*
```
```
docker build -t rl_tools:release_linux . --no-cache
```
```
docker run --rm -it --mount type=bind,source=$(cd ../..; pwd),target=/rl_tools,readonly --mount type=bind,source=$(pwd)/install,target=/install -v $(pwd)/build.sh:/build.sh:ro rl_tools:release_linux
/build.sh /install
````

#### Testing the release
#### Headless
```
docker run --rm -it -w / --mount type=bind,source=$(pwd)/checkpoints,target=/checkpoints --mount type=bind,source=$(pwd)/install,target=/install --cap-add=sys_nice --runtime=nvidia --gpus all archlinux nice -n -20 /bin/bash /install/bin/rl_environments_mujoco_ant_ppo_cuda_full
```
#### With UI
```
docker build -f Dockerfile_ui -t rl_tools:release_linux_ui . --no-cache
```
```
xhost +local:docker
```
```
docker run --rm -it -e DISPLAY=$DISPLAY --device=/dev/dri -v /tmp/.X11-unix:/tmp/.X11-unix --mount type=bind,source=$(pwd)/install,target=/install --mount type=bind,source=$(pwd)/checkpoints,target=/checkpoints,readonly rl_tools:release_linux_ui /install/rl_tools/bin/rl_environments_mujoco_ant_evaluation_ppo
```

### For debugging the release process: Using a native machine to create the release
Optional: Set up using clang (which speeds up MuJoCo a bit)
```
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-16 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-16 100
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
In the following remove the CC and CXX definitions if you want to use your default compiler and adjust the path to the CUDAToolkit_ROOT and the path to nvcc (CMAKE_CUDA_COMPILER)
```
CC=clang CXX=clang++ CUDAToolkit_ROOT=/opt/cuda-11.8 cmake .. -DCMAKE_CUDA_COMPILER=/opt/cuda-11.8/bin/nvcc -DCMAKE_TOOLCHAIN_FILE:STRING=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DRL_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON -DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DRL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON -DRL_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DRL_TOOLS_ENABLE_HDF5:BOOL=ON -DRL_TOOLS_HDF5_HIGHFIVE_FROM_CONAN:BOOL=ON -DRL_TOOLS_ENABLE_CLI11:BOOL=ON -DRL_TOOLS_BUILD_TYPE:STRING=Release -DRL_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES:BOOL=ON -DCMAKE_INSTALL_PREFIX:STRING=install
```
```
cmake --build . -j$(nproc)
```
```
cmake --install .
```

```
(cd install/rl_tools && tar -czvf rl_tools-0.1.0-linux-x64_64.tar.gz *)
```

#### Testing the binaries
```
docker run -it --rm -w /rlt/bin -v $(pwd)/install/rl_tools:/rlt archlinux 
````
```
docker run --rm -it -w /rlt/bin -v $(pwd)/install/rl_tools:/rlt --cap-add=sys_nice --runtime=nvidia --gpus all archlinux nice -n -20 ./rl_environments_mujoco_ant_training_ppo_cuda_full 
````


