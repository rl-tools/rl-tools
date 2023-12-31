## CLion
#### Examplary settings for a Intel-based machine with a CUDA-compatible GPU
CMake options:
```
-DRL_TOOLS_ENABLE_HDF5:BOOL=ON
-DRL_TOOLS_ENABLE_CLI11:BOOL=ON
-DRL_TOOLS_ENABLE_TENSORBOARD:BOOL=ON
-DRL_TOOLS_ENABLE_LIBATTOPNG:BOOL=ON
-DRL_TOOLS_ENABLE_GTK:BOOL=ON
-DRL_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON
-DRL_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON
-DRL_TOOLS_ENABLE_JSON:BOOL=ON
-DRL_TOOLS_ENABLE_BOOST_BEAST:BOOL=ON
-DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON
-DRL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON
-DRL_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_ENABLE_UI:BOOL=ON
-DRL_TOOLS_ENABLE_TESTS:BOOL=ON
-DRL_TOOLS_DISABLE_LTO:BOOL=OFF
```
Environment:
```
CUDA_PATH=/opt/cuda-12.1; MKL_ROOT=/opt/intel/oneapi/mkl/2023.1.0/
```

## Building manually
Building manually with everything enabled (on a Intel-based machine with a CUDA-compatible GPU)
```
mkdir build
cd build
CUDA_PATH=/opt/cuda-12.1 MKL_ROOT=/opt/intel/oneapi/mkl/2023.1.0 cmake .. \
-DRL_TOOLS_ENABLE_TESTS:BOOL=ON \
-DRL_TOOLS_ENABLE_HDF5:BOOL=ON \
-DRL_TOOLS_ENABLE_TENSORBOARD:BOOL=ON \
-DRL_TOOLS_ENABLE_JSON:BOOL=ON \
-DRL_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON \
-DRL_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON \
-DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON \
-DRL_TOOLS_TEST_MACHINE_LENOVO_P1:BOOL=ON \
-DRL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON \
-DRL_TOOLS_TESTS_ENABLE_EIGEN:BOOL=ON
cmake --build . -j$(nproc)
```
