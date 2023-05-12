Redistributable libraries included:
NVIDIA CUBLAS: 11.8
Intel oneMKL: 2023.1
Visual Studio C++: 2022

Note: In our experience the CUDA training of the MuJoCo Ant-v4 is about 25% slower than under Linux using the same Laptop (probably due to the compiler/compiler options). In our tests running the linux binaries under Windows using WSL actually was faster when using the GPU (make sure that your GPU is available in WSL in this case using e.g. the nvidia-smi command).