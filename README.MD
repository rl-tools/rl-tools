## Naming Convention
We use `snake_case` for variables/instances, functions as well as namespaces and `PascalCase` for structs/classes. Furthermore, we use upper case `CAMEL_CASE` for compile-time constants. 

## Building
```
mkdir build
cd build
CUDA_PATH=/opt/cuda-12.1 cmake .. \
-DLAYER_IN_C_ENABLE_TESTS:BOOL=ON \
-DLAYER_IN_C_ENABLE_HDF5:BOOL=ON \
-DLAYER_IN_C_ENABLE_TENSORBOARD:BOOL=ON \
-DLAYER_IN_C_BACKEND_ENABLE_MKL:BOOL=ON \
-DLAYER_IN_C_BACKEND_ENABLE_CUDA:BOOL=ON \
-DLAYER_IN_C_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON \
-DLAYER_IN_C_TEST_MACHINE_LENOVO_P1:BOOL=ON \
-DLAYER_IN_C_TESTS_ENABLE_UI:BOOL=ON \
-DLAYER_IN_C_TESTS_ENABLE_EIGEN:BOOL=ON
cmake --build . -j$(nproc)
```