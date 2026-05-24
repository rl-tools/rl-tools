Detect CUDA arch and optimize build
```
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j --target rl_environments_l2f_visual_yaw_cuda
```
