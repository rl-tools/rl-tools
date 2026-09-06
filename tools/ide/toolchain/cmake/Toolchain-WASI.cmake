# Cross toolchain for wasm32-wasip1 driven by a host clang and lld. The superbuild (tools/ide/toolchain) hands every input in
# as a cache entry; try_compile re-reads this file in a fresh project, so those entries are forwarded explicitly.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
    RL_TOOLS_IDE_HOST_LLVM_BIN RL_TOOLS_IDE_TARGET RL_TOOLS_IDE_WASI_SYSROOT RL_TOOLS_IDE_WASI_RESOURCE_DIR RL_TOOLS_IDE_WASI_CPU
    RL_TOOLS_IDE_PREFIX_MAP RL_TOOLS_IDE_WASI_LLVM_FLAGS RL_TOOLS_IDE_LTO RL_TOOLS_IDE_LTO_JOBS)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

set(CMAKE_SYSTEM_NAME WASI)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR wasm32)

set(CMAKE_C_COMPILER "${RL_TOOLS_IDE_HOST_LLVM_BIN}/clang")
set(CMAKE_CXX_COMPILER "${RL_TOOLS_IDE_HOST_LLVM_BIN}/clang++")
set(CMAKE_ASM_COMPILER "${RL_TOOLS_IDE_HOST_LLVM_BIN}/clang")
set(CMAKE_AR "${RL_TOOLS_IDE_HOST_LLVM_BIN}/llvm-ar")
set(CMAKE_RANLIB "${RL_TOOLS_IDE_HOST_LLVM_BIN}/llvm-ranlib")
set(CMAKE_NM "${RL_TOOLS_IDE_HOST_LLVM_BIN}/llvm-nm")
set(CMAKE_C_COMPILER_TARGET ${RL_TOOLS_IDE_TARGET})
set(CMAKE_CXX_COMPILER_TARGET ${RL_TOOLS_IDE_TARGET})
set(CMAKE_ASM_COMPILER_TARGET ${RL_TOOLS_IDE_TARGET})
set(CMAKE_SYSROOT "${RL_TOOLS_IDE_WASI_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_C_LINKER_DEPFILE_SUPPORTED OFF)
set(CMAKE_CXX_LINKER_DEPFILE_SUPPORTED OFF)

set(rl_tools_ide_flags "-mcpu=${RL_TOOLS_IDE_WASI_CPU}")
if(RL_TOOLS_IDE_WASI_RESOURCE_DIR)
    string(APPEND rl_tools_ide_flags " -resource-dir=${RL_TOOLS_IDE_WASI_RESOURCE_DIR}")
endif()
if(RL_TOOLS_IDE_PREFIX_MAP)
    string(APPEND rl_tools_ide_flags " -ffile-prefix-map=${RL_TOOLS_IDE_PREFIX_MAP}")
endif()
set(rl_tools_ide_link_flags "${rl_tools_ide_flags}")
# The module needs what the plain sysroot builds do not: wasi-libc's opt-in emulation of mmap, getpid and signal()/raise()
# (each an #error in the corresponding header without the define), so that the POSIX surface LLVM includes compiles and
# links unchanged where wasi-libc can stand in (process clocks are left out on purpose: their header would also declare
# the rlimit functions wasi-libc does not implement); a memory ceiling clang and lld can grow into; a stack that C++
# compilation does not overflow (first in memory, so an overflow traps instead of corrupting the heap); and LTO so host
# APIs that are compiled in but never called leave no imports
if(RL_TOOLS_IDE_WASI_LLVM_FLAGS)
    string(APPEND rl_tools_ide_flags " -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_GETPID -D_WASI_EMULATED_SIGNAL")
    string(APPEND rl_tools_ide_link_flags " -lwasi-emulated-mman -lwasi-emulated-getpid -lwasi-emulated-signal -Wl,--strip-all -Wl,--max-memory=4294967296 -Wl,-z,stack-size=8388608,--stack-first")
    if(RL_TOOLS_IDE_LTO STREQUAL "full")
        set(rl_tools_ide_lto_flag " -flto")
    elseif(RL_TOOLS_IDE_LTO STREQUAL "thin")
        set(rl_tools_ide_lto_flag " -flto=thin")
        if(RL_TOOLS_IDE_LTO_JOBS)
            string(APPEND rl_tools_ide_link_flags " -Wl,--thinlto-jobs=${RL_TOOLS_IDE_LTO_JOBS}")
        endif()
    else()
        set(rl_tools_ide_lto_flag "")
    endif()
    string(APPEND rl_tools_ide_flags "${rl_tools_ide_lto_flag}")
    string(APPEND rl_tools_ide_link_flags "${rl_tools_ide_lto_flag}")
endif()
set(CMAKE_C_FLAGS_INIT "${rl_tools_ide_flags}")
set(CMAKE_CXX_FLAGS_INIT "${rl_tools_ide_flags}")
set(CMAKE_ASM_FLAGS_INIT "${rl_tools_ide_flags}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${rl_tools_ide_link_flags}")
