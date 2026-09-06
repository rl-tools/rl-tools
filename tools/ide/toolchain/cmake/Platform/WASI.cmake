# CMake has no WASI platform; the toolchain file puts this directory on the module path so CMAKE_SYSTEM_NAME=WASI resolves,
# exactly as the toolchain files shipped by wasi-sdk do. LLVM's option handling (patched) branches on this variable.
set(WASI 1)
