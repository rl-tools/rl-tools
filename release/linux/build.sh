#!/usr/bin/env bash
# Takes the install path as the first argument
if [ -z "$1" ]
then
  echo "No install path specified"
  exit 1
fi
set -e
conan profile detect --force
conan install /backprop_tools --output-folder=. --build=missing --settings=build_type=Release
CC=clang CXX=clang++ cmake /backprop_tools -DCMAKE_TOOLCHAIN_FILE:STRING=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DBACKPROP_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON -DBACKPROP_TOOLS_ENABLE_CLI11:BOOL=ON -DBACKPROP_TOOLS_BUILD_TYPE:STRING=Release -DBACKPROP_TOOLS_INSTALL_INCLUDE_REDISTRIBUTABLES:BOOL=ON -DCMAKE_INSTALL_PREFIX:STRING=$1
cmake --build . -j$(nproc)
cmake --install .
(cd $1/backprop_tools && tar -czvf backprop_tools-0.1.0-linux-x64_64.tar.gz *)
