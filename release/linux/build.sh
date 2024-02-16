#!/usr/bin/env bash
# Takes the install path as the first argument
if [ -z "$1" ]
then
  echo "No install path specified"
  exit 1
fi
set -e
conan profile detect --force
conan install /rl_tools --output-folder=conan_build --build=missing -s build_type=Release
CC=clang CXX=clang++ cmake /rl_tools -DCMAKE_TOOLCHAIN_FILE:STRING=conan_build/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_ENABLE_TARGETS=ON -DRL_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DRL_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON -DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DRL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON -DRL_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DRL_TOOLS_ENABLE_HDF5:BOOL=ON -DRL_TOOLS_ENABLE_CLI11:BOOL=ON -DRL_TOOLS_ENABLE_BOOST_BEAST:BOOL=ON -DRL_TOOLS_ENABLE_JSON:BOOL=ON -DRL_TOOLS_BUILD_TYPE:STRING=Release -DRL_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES:BOOL=ON -DCMAKE_INSTALL_PREFIX:STRING=$1
cmake --build . -j$(nproc)
cmake --install .
(cd $1 && tar -czvf rl_tools-1.0.0-linux-x64_64.tar.gz *)
