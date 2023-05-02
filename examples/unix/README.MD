## Unix Examples
On unix systems the usage of the examples is very similar. The major difference is the usage of different BLAS backends (e.g. Apple Accelerate vs. Intel MKL) which manifests in using either the `-DBACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE:BOOL=ON` or the `-DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON` option when configuring the project using CMake. For the setup of the dependencies refer to:

* [macOS](setup/MAC.MD)
* [ubuntu](setup/UBUNTU.MD)
