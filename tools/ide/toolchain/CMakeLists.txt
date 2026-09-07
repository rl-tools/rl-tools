cmake_minimum_required(VERSION 3.24)

# Superbuild for the browser IDE's compiler: clang and lld for wasm32-wasip1 as one WASI module (llvm.wasm) plus the sysroot
# that both the module and the programs it compiles link against (sysroot.tar). It is a standalone project, not part of the
# RLtools configure; every stage is an ExternalProject and all build trees and outputs live under RL_TOOLS_IDE_TOOLCHAIN_DIR,
# outside the repository; the sources are FetchContent checkouts in the repository's .dependencies tree like every other
# RLtools dependency: RLtools' fork of llvm-project at the branch that carries the WASI commit, and stock wasi-libc. See
# README.md next to this file.

set(RL_TOOLS_IDE_LLVM_REPOSITORY https://github.com/rl-tools/llvm-project.git)
set(RL_TOOLS_IDE_LLVM_BRANCH rltools-wasi-22.1.8)
set(RL_TOOLS_IDE_LLVM_UPSTREAM llvmorg-22.1.8)
set(RL_TOOLS_IDE_WASI_LIBC_REPOSITORY https://github.com/WebAssembly/wasi-libc.git)
set(RL_TOOLS_IDE_WASI_LIBC_TAG wasi-sdk-34)

set(RL_TOOLS_IDE_HOST_LLVM 22)
set(RL_TOOLS_IDE_TARGET wasm32-wasip1)
set(RL_TOOLS_IDE_WASI_CPU lime1)
set(RL_TOOLS_IDE_LLVM_VERSION_SUFFIX -rltools)

project(rl_tools_ide_toolchain NONE)
include(ExternalProject)

get_filename_component(RL_TOOLS_REPOSITORY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
if(DEFINED ENV{RL_TOOLS_IDE_TOOLCHAIN_DIR})
    set(rl_tools_ide_default_toolchain_dir "$ENV{RL_TOOLS_IDE_TOOLCHAIN_DIR}")
else()
    set(rl_tools_ide_default_toolchain_dir /vm/data/rl-tools/ide-toolchain)
endif()
set(RL_TOOLS_IDE_TOOLCHAIN_DIR "${rl_tools_ide_default_toolchain_dir}" CACHE PATH "Build trees and outputs of the superbuild (outside the repository); sources go to .dependencies")
set(RL_TOOLS_IDE_JOBS 5 CACHE STRING "Parallel jobs per sub-build, also the ThinLTO backend threads of the module link")
set(RL_TOOLS_IDE_LTO thin CACHE STRING "Link-time optimization of the wasm module: thin, full or off (a full LTO link of the module peaks above 16 GB)")
option(RL_TOOLS_IDE_ASSERTIONS "LLVM assertions in the wasm module" ON)
option(RL_TOOLS_IDE_BUILD_HOST_CLANG "Build the cross compiler from the pinned source with the system compiler instead of using the distro clang" OFF)
set(RL_TOOLS_IDE_HOST_LLVM_BIN "" CACHE PATH "Directory with clang, clang++, wasm-ld, llvm-ar, llvm-ranlib and llvm-nm (default: /usr/lib/llvm-${RL_TOOLS_IDE_HOST_LLVM}/bin)")
set(RL_TOOLS_IDE_OUTPUT_DIR ${RL_TOOLS_REPOSITORY_DIR}/static/ide/build/toolchain CACHE PATH "Where the served package is copied")

set_property(CACHE RL_TOOLS_IDE_LTO PROPERTY STRINGS full thin off)
if(NOT RL_TOOLS_IDE_LTO MATCHES "^(full|thin|off)$")
    message(FATAL_ERROR "RL_TOOLS_IDE_LTO must be full, thin or off (got '${RL_TOOLS_IDE_LTO}')")
endif()

set(RL_TOOLS_IDE_HOST_DIR ${RL_TOOLS_IDE_TOOLCHAIN_DIR}/host)
set(RL_TOOLS_IDE_SYSROOT_BUILD_DIR ${RL_TOOLS_IDE_TOOLCHAIN_DIR}/sysroot)
set(RL_TOOLS_IDE_SYSROOT_DIR ${RL_TOOLS_IDE_TOOLCHAIN_DIR}/prefix/usr)
set(RL_TOOLS_IDE_RESOURCE_DIR ${RL_TOOLS_IDE_TOOLCHAIN_DIR}/resource)
set(RL_TOOLS_IDE_WASM_DIR ${RL_TOOLS_IDE_TOOLCHAIN_DIR}/wasm)
set(RL_TOOLS_IDE_PACKAGE_DIR ${RL_TOOLS_IDE_TOOLCHAIN_DIR}/out)
set(RL_TOOLS_IDE_PARITY_DIR ${RL_TOOLS_IDE_TOOLCHAIN_DIR}/parity)
set(RL_TOOLS_IDE_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(RL_TOOLS_IDE_TESTS_DIR ${RL_TOOLS_REPOSITORY_DIR}/tests/src/ide)

# ---------------------------------------------------------------------------------------------------------------------- host tools
find_program(RL_TOOLS_IDE_GIT git REQUIRED)
find_program(RL_TOOLS_IDE_TAR tar REQUIRED)
execute_process(COMMAND ${RL_TOOLS_IDE_TAR} --version OUTPUT_VARIABLE rl_tools_ide_tar_version ERROR_QUIET)
if(NOT rl_tools_ide_tar_version MATCHES "GNU tar")
    message(FATAL_ERROR "GNU tar is required for the deterministic sysroot archive (found: ${RL_TOOLS_IDE_TAR})")
endif()
find_program(RL_TOOLS_IDE_NINJA ninja)
if(RL_TOOLS_IDE_NINJA)
    set(RL_TOOLS_IDE_GENERATOR Ninja)
else()
    set(RL_TOOLS_IDE_GENERATOR "Unix Makefiles")
endif()
find_program(RL_TOOLS_IDE_NODE node)

if(RL_TOOLS_IDE_BUILD_HOST_CLANG)
    set(RL_TOOLS_IDE_HOST_LLVM_BIN_DIR ${RL_TOOLS_IDE_HOST_DIR}/bin)
    set(rl_tools_ide_host_compiler "from source (${RL_TOOLS_IDE_LLVM_BRANCH}), built with the system compiler")
else()
    if(RL_TOOLS_IDE_HOST_LLVM_BIN)
        set(RL_TOOLS_IDE_HOST_LLVM_BIN_DIR ${RL_TOOLS_IDE_HOST_LLVM_BIN})
    else()
        set(RL_TOOLS_IDE_HOST_LLVM_BIN_DIR /usr/lib/llvm-${RL_TOOLS_IDE_HOST_LLVM}/bin)
    endif()
    set(rl_tools_ide_missing_tools)
    foreach(tool clang clang++ wasm-ld llvm-ar llvm-ranlib llvm-nm)
        if(NOT EXISTS ${RL_TOOLS_IDE_HOST_LLVM_BIN_DIR}/${tool})
            list(APPEND rl_tools_ide_missing_tools ${tool})
        endif()
    endforeach()
    if(rl_tools_ide_missing_tools)
        message(FATAL_ERROR "Missing in ${RL_TOOLS_IDE_HOST_LLVM_BIN_DIR}: ${rl_tools_ide_missing_tools}\n"
            "The cross compiler is the distro clang/lld of major version ${RL_TOOLS_IDE_HOST_LLVM}. On Ubuntu:\n"
            "    sudo apt install clang-${RL_TOOLS_IDE_HOST_LLVM} lld-${RL_TOOLS_IDE_HOST_LLVM}\n"
            "or point RL_TOOLS_IDE_HOST_LLVM_BIN at a directory holding these tools, or set RL_TOOLS_IDE_BUILD_HOST_CLANG=ON to build them from the pinned source.")
    endif()
    execute_process(COMMAND ${RL_TOOLS_IDE_HOST_LLVM_BIN_DIR}/clang --version OUTPUT_VARIABLE rl_tools_ide_host_compiler ERROR_QUIET)
    string(REGEX REPLACE "\n.*" "" rl_tools_ide_host_compiler "${rl_tools_ide_host_compiler}")
endif()

# ------------------------------------------------------------------------------------------------------------------------ stage 0
# The pins above are the only place the sources are named. FetchContent checks them out at configure time into the
# repository's per-build-directory .dependencies tree, exactly as the main RLtools configure does, so checkouts survive a
# deleted build directory; the stages below read from those directories. llvm-project follows its branch: a reconfigure
# with network access moves the checkout to the branch tip and the stages rebuild what changed, without network the
# checkout is kept with a warning, and RL_TOOLS_OFFLINE_BUILD never asks. SOURCE_SUBDIR names a directory that does not
# exist so neither tree is added as a subproject (wasi-libc has a top-level CMakeLists.txt).
option(RL_TOOLS_OFFLINE_BUILD "Use the sources in .dependencies without checking for updates" OFF)
get_filename_component(CMAKE_BINARY_DIR_BASENAME ${CMAKE_BINARY_DIR} NAME)
if(NOT DEFINED FETCHCONTENT_BASE_DIR)
    set(rl_tools_ide_write_test "${RL_TOOLS_REPOSITORY_DIR}/.cmake_write_test")
    execute_process(COMMAND ${CMAKE_COMMAND} -E touch "${rl_tools_ide_write_test}" RESULT_VARIABLE rl_tools_ide_write_test_result ERROR_QUIET OUTPUT_QUIET)
    if(rl_tools_ide_write_test_result EQUAL 0)
        file(REMOVE "${rl_tools_ide_write_test}")
        set(rl_tools_ide_fetchcontent_root "${RL_TOOLS_REPOSITORY_DIR}")
    else()
        set(rl_tools_ide_fetchcontent_root "${CMAKE_BINARY_DIR}")
    endif()
    set(FETCHCONTENT_BASE_DIR "${rl_tools_ide_fetchcontent_root}/.dependencies/${CMAKE_BINARY_DIR_BASENAME}" CACHE PATH "Base dir for FetchContent downloads" FORCE)
else()
    set(FETCHCONTENT_BASE_DIR "${FETCHCONTENT_BASE_DIR}/${CMAKE_BINARY_DIR_BASENAME}" CACHE PATH "Base dir for FetchContent downloads")
endif()
if(RL_TOOLS_OFFLINE_BUILD)
    set(FETCHCONTENT_FULLY_DISCONNECTED ON)
endif()
message(STATUS "RLtools IDE toolchain sources: ${FETCHCONTENT_BASE_DIR}")
include(FetchContent)
# CMake's GIT_SHALLOW clones every branch tip (--no-single-branch); the command below fetches exactly one ref at depth 1
# and no tags, which is the smallest transfer possible, and moves an existing checkout to the ref's tip on reconfigure
function(rl_tools_ide_declare_source name repository ref)
    set(command ${CMAKE_COMMAND}
        -DGIT=${RL_TOOLS_IDE_GIT}
        -DREPOSITORY=${repository}
        -DREF=${ref}
        -DDIRECTORY=<SOURCE_DIR>
        -P ${RL_TOOLS_IDE_CMAKE_DIR}/fetch_source.cmake
    )
    FetchContent_Declare(${name}
        DOWNLOAD_COMMAND ${command}
        UPDATE_COMMAND ${command}
        SOURCE_SUBDIR rl-tools-not-a-subproject
    )
endfunction()
rl_tools_ide_declare_source(llvm_project ${RL_TOOLS_IDE_LLVM_REPOSITORY} ${RL_TOOLS_IDE_LLVM_BRANCH})
rl_tools_ide_declare_source(wasi_libc ${RL_TOOLS_IDE_WASI_LIBC_REPOSITORY} ${RL_TOOLS_IDE_WASI_LIBC_TAG})
foreach(name llvm_project wasi_libc)
    string(TOUPPER ${name} name_upper)
    if(NOT EXISTS "${FETCHCONTENT_BASE_DIR}/${name}-src" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_${name_upper})
        message(STATUS "Fetching ${name} (this can take a while)")
    endif()
endforeach()
FetchContent_MakeAvailable(llvm_project wasi_libc)
set(RL_TOOLS_IDE_LLVM_SOURCE_DIR ${llvm_project_SOURCE_DIR})
set(RL_TOOLS_IDE_WASI_LIBC_SOURCE_DIR ${wasi_libc_SOURCE_DIR})
foreach(pair "RL_TOOLS_IDE_LLVM_SOURCE_DIR;RL_TOOLS_IDE_LLVM_COMMIT" "RL_TOOLS_IDE_WASI_LIBC_SOURCE_DIR;RL_TOOLS_IDE_WASI_LIBC_COMMIT")
    list(GET pair 0 directory_variable)
    list(GET pair 1 commit_variable)
    execute_process(COMMAND ${RL_TOOLS_IDE_GIT} -C ${${directory_variable}} rev-parse --verify --quiet HEAD RESULT_VARIABLE result OUTPUT_VARIABLE ${commit_variable} OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${${directory_variable}} is not a git checkout")
    endif()
endforeach()
if(DEFINED CACHE{RL_TOOLS_IDE_LLVM_LAST_COMMIT} AND NOT RL_TOOLS_IDE_LLVM_LAST_COMMIT STREQUAL RL_TOOLS_IDE_LLVM_COMMIT)
    message(STATUS "llvm-project moved from ${RL_TOOLS_IDE_LLVM_LAST_COMMIT} to ${RL_TOOLS_IDE_LLVM_COMMIT}; the stages rebuild what changed")
endif()
set(RL_TOOLS_IDE_LLVM_LAST_COMMIT ${RL_TOOLS_IDE_LLVM_COMMIT} CACHE INTERNAL "llvm-project commit of the previous configure")
string(REGEX REPLACE "\\.git$" "" RL_TOOLS_IDE_LLVM_REPOSITORY_URL "${RL_TOOLS_IDE_LLVM_REPOSITORY}")

# ------------------------------------------------------------------------------------------------------------------------ stage 1
# Native tablegen binaries of the pinned version; the cross build cannot run the ones it builds itself. With
# RL_TOOLS_IDE_BUILD_HOST_CLANG the same tree also yields the cross compiler.
set(rl_tools_ide_host_targets llvm-tblgen llvm-min-tblgen clang-tblgen)
set(rl_tools_ide_host_projects clang)
set(rl_tools_ide_host_args
    -DCMAKE_BUILD_TYPE=Release
    -DLLVM_TARGETS_TO_BUILD=WebAssembly
    -DLLVM_DEFAULT_TARGET_TRIPLE=${RL_TOOLS_IDE_TARGET}
    -DLLVM_BUILD_TOOLS=OFF
    -DLLVM_BUILD_RUNTIME=OFF
    -DLLVM_INCLUDE_UTILS=OFF
    -DLLVM_INCLUDE_RUNTIMES=OFF
    -DLLVM_INCLUDE_TESTS=OFF
    -DLLVM_INCLUDE_EXAMPLES=OFF
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_INCLUDE_DOCS=OFF
    -DLLVM_ENABLE_ZLIB=OFF
    -DLLVM_ENABLE_ZSTD=OFF
    -DLLVM_ENABLE_LIBXML2=OFF
    -DCLANG_BUILD_TOOLS=OFF
    -DCLANG_BUILD_EXAMPLES=OFF
    -DCLANG_INCLUDE_TESTS=OFF
    -DCLANG_INCLUDE_DOCS=OFF
)
if(RL_TOOLS_IDE_BUILD_HOST_CLANG)
    list(APPEND rl_tools_ide_host_targets clang clang-resource-headers lld llvm-ar llvm-ranlib llvm-nm)
    set(rl_tools_ide_host_projects "clang;lld")
else()
    list(APPEND rl_tools_ide_host_args
        -DCMAKE_C_COMPILER=${RL_TOOLS_IDE_HOST_LLVM_BIN_DIR}/clang
        -DCMAKE_CXX_COMPILER=${RL_TOOLS_IDE_HOST_LLVM_BIN_DIR}/clang++
    )
endif()
ExternalProject_Add(host
    PREFIX ${CMAKE_BINARY_DIR}/host
    DOWNLOAD_COMMAND ""
    SOURCE_DIR ${RL_TOOLS_IDE_LLVM_SOURCE_DIR}/llvm
    BINARY_DIR ${RL_TOOLS_IDE_HOST_DIR}
    CMAKE_GENERATOR ${RL_TOOLS_IDE_GENERATOR}
    CMAKE_ARGS ${rl_tools_ide_host_args}
    CMAKE_CACHE_ARGS -DLLVM_ENABLE_PROJECTS:STRING=${rl_tools_ide_host_projects}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> -j ${RL_TOOLS_IDE_JOBS} --target ${rl_tools_ide_host_targets}
    INSTALL_COMMAND ""
    USES_TERMINAL_CONFIGURE ON
    USES_TERMINAL_BUILD ON
)

# ------------------------------------------------------------------------------------------------------------------------ stage 2
# The sysroot, in the order wasi-sdk uses: compiler-rt builtins, wasi-libc, then libc++abi and libc++ (the fork carries the
# one libc++abi commit the module needs). One cross toolchain file serves every sub-build; the superbuild hands it its inputs
# as cache entries.
set(rl_tools_ide_toolchain_args
    -DCMAKE_TOOLCHAIN_FILE=${RL_TOOLS_IDE_CMAKE_DIR}/Toolchain-WASI.cmake
    -DRL_TOOLS_IDE_HOST_LLVM_BIN=${RL_TOOLS_IDE_HOST_LLVM_BIN_DIR}
    -DRL_TOOLS_IDE_TARGET=${RL_TOOLS_IDE_TARGET}
    -DRL_TOOLS_IDE_WASI_SYSROOT=${RL_TOOLS_IDE_SYSROOT_DIR}
    -DRL_TOOLS_IDE_WASI_CPU=${RL_TOOLS_IDE_WASI_CPU}
    -DRL_TOOLS_IDE_PREFIX_MAP=${RL_TOOLS_IDE_TOOLCHAIN_DIR}=/rl-tools-ide-toolchain
)
set(rl_tools_ide_resource_args -DRL_TOOLS_IDE_WASI_RESOURCE_DIR=${RL_TOOLS_IDE_RESOURCE_DIR})
set(rl_tools_ide_builtins_dir ${RL_TOOLS_IDE_SYSROOT_DIR}/lib/wasm32-unknown-wasip1)

set(rl_tools_ide_compiler_rt_args
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${RL_TOOLS_IDE_SYSROOT_DIR}
    -DCMAKE_C_COMPILER_WORKS=ON
    -DCMAKE_CXX_COMPILER_WORKS=ON
    -DCMAKE_ASM_COMPILER_WORKS=ON
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY
    -DCOMPILER_RT_BAREMETAL_BUILD=ON
    -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON
    -DCOMPILER_RT_BUILD_BUILTINS=ON
    -DCOMPILER_RT_BUILD_XRAY=OFF
    -DCOMPILER_RT_BUILD_SANITIZERS=OFF
    -DCOMPILER_RT_BUILD_LIBFUZZER=OFF
    -DCOMPILER_RT_BUILD_PROFILE=OFF
    -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF
    -DCOMPILER_RT_BUILD_MEMPROF=OFF
    -DCOMPILER_RT_BUILD_ORC=OFF
    -DCOMPILER_RT_BUILD_GWP_ASAN=OFF
    -DCOMPILER_RT_INCLUDE_TESTS=OFF
    -DCOMPILER_RT_HAS_FPIC_FLAG=OFF
    -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON
)
ExternalProject_Add(compiler-rt
    PREFIX ${CMAKE_BINARY_DIR}/compiler-rt
    DOWNLOAD_COMMAND ""
    SOURCE_DIR ${RL_TOOLS_IDE_LLVM_SOURCE_DIR}/compiler-rt
    BINARY_DIR ${RL_TOOLS_IDE_SYSROOT_BUILD_DIR}/compiler-rt
    CMAKE_GENERATOR ${RL_TOOLS_IDE_GENERATOR}
    CMAKE_ARGS ${rl_tools_ide_toolchain_args} ${rl_tools_ide_compiler_rt_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> -j ${RL_TOOLS_IDE_JOBS}
    DEPENDS host
    USES_TERMINAL_CONFIGURE ON
    USES_TERMINAL_BUILD ON
    USES_TERMINAL_INSTALL ON
)
# The host clang finds its own headers and the wasm32 builtins through one resource directory, the way wasi-sdk builds
ExternalProject_Add_Step(compiler-rt resource-dir
    COMMAND ${CMAKE_COMMAND}
        -DCLANG=${RL_TOOLS_IDE_HOST_LLVM_BIN_DIR}/clang
        -DBUILTINS=${rl_tools_ide_builtins_dir}
        -DRESOURCE_DIR=${RL_TOOLS_IDE_RESOURCE_DIR}
        -P ${RL_TOOLS_IDE_CMAKE_DIR}/resource_dir.cmake
    DEPENDEES install
    COMMENT "Assembling the host resource directory"
)

set(rl_tools_ide_wasi_libc_args
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${RL_TOOLS_IDE_SYSROOT_DIR}
    -DTARGET_TRIPLE=${RL_TOOLS_IDE_TARGET}
    -DBUILTINS_LIB=${rl_tools_ide_builtins_dir}/libclang_rt.builtins.a
    -DUSE_WASM_COMPONENT_LD=OFF
    -DBUILD_SHARED=OFF
    -DCHECK_SYMBOLS=OFF
)
ExternalProject_Add(wasi-libc
    PREFIX ${CMAKE_BINARY_DIR}/wasi-libc
    DOWNLOAD_COMMAND ""
    SOURCE_DIR ${RL_TOOLS_IDE_WASI_LIBC_SOURCE_DIR}
    BINARY_DIR ${RL_TOOLS_IDE_SYSROOT_BUILD_DIR}/wasi-libc
    CMAKE_GENERATOR ${RL_TOOLS_IDE_GENERATOR}
    CMAKE_ARGS ${rl_tools_ide_toolchain_args} ${rl_tools_ide_resource_args} ${rl_tools_ide_wasi_libc_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> -j ${RL_TOOLS_IDE_JOBS}
    DEPENDS compiler-rt
    USES_TERMINAL_CONFIGURE ON
    USES_TERMINAL_BUILD ON
    USES_TERMINAL_INSTALL ON
)

set(rl_tools_ide_libcxx_runtimes "libcxx;libcxxabi")
set(rl_tools_ide_libcxx_args
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${RL_TOOLS_IDE_SYSROOT_DIR}
    -DCMAKE_C_COMPILER_WORKS=ON
    -DCMAKE_CXX_COMPILER_WORKS=ON
    -DCMAKE_POSITION_INDEPENDENT_CODE=OFF
    -DLLVM_COMPILER_CHECKED=ON
    -DUNIX=ON
    -DLIBCXX_ENABLE_THREADS=ON
    -DLIBCXX_HAS_PTHREAD_API=ON
    -DLIBCXX_HAS_EXTERNAL_THREAD_API=OFF
    -DLIBCXX_HAS_WIN32_THREAD_API=OFF
    -DLIBCXX_ENABLE_SHARED=OFF
    -DLIBCXX_ENABLE_EXCEPTIONS=OFF
    -DLIBCXX_ENABLE_FILESYSTEM=ON
    -DLIBCXX_ENABLE_ABI_LINKER_SCRIPT=OFF
    -DLIBCXX_CXX_ABI=libcxxabi
    -DLIBCXX_HAS_MUSL_LIBC=OFF
    -DLIBCXX_ABI_VERSION=2
    -DLIBCXX_INSTALL_MODULES=OFF
    -DLIBCXX_LIBDIR_SUFFIX=/${RL_TOOLS_IDE_TARGET}
    -DLIBCXX_INCLUDE_TESTS=OFF
    -DLIBCXX_INCLUDE_BENCHMARKS=OFF
    -DLIBCXXABI_ENABLE_EXCEPTIONS=OFF
    -DLIBCXXABI_ENABLE_SHARED=OFF
    -DLIBCXXABI_SILENT_TERMINATE=ON
    -DLIBCXXABI_ENABLE_THREADS=ON
    -DLIBCXXABI_HAS_PTHREAD_API=ON
    -DLIBCXXABI_HAS_EXTERNAL_THREAD_API=OFF
    -DLIBCXXABI_HAS_WIN32_THREAD_API=OFF
    -DLIBCXXABI_USE_LLVM_UNWINDER=OFF
    -DLIBCXXABI_LIBDIR_SUFFIX=/${RL_TOOLS_IDE_TARGET}
)
ExternalProject_Add(libcxx
    PREFIX ${CMAKE_BINARY_DIR}/libcxx
    DOWNLOAD_COMMAND ""
    SOURCE_DIR ${RL_TOOLS_IDE_LLVM_SOURCE_DIR}/runtimes
    BINARY_DIR ${RL_TOOLS_IDE_SYSROOT_BUILD_DIR}/libcxx
    CMAKE_GENERATOR ${RL_TOOLS_IDE_GENERATOR}
    CMAKE_ARGS ${rl_tools_ide_toolchain_args} ${rl_tools_ide_resource_args} ${rl_tools_ide_libcxx_args}
    CMAKE_CACHE_ARGS -DLLVM_ENABLE_RUNTIMES:STRING=${rl_tools_ide_libcxx_runtimes}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> -j ${RL_TOOLS_IDE_JOBS}
    DEPENDS wasi-libc
    USES_TERMINAL_CONFIGURE ON
    USES_TERMINAL_BUILD ON
    USES_TERMINAL_INSTALL ON
)
add_custom_target(sysroot DEPENDS libcxx)

# ------------------------------------------------------------------------------------------------------------------------ stage 3
# clang, lld and llvm-ar as one llvm-driver module from the fork. Every LLVM tool that could join the driver is switched off explicitly:
# the driver absorbs every configured GENERATE_DRIVER tool and LLVM_BUILD_TOOLS=OFF alone does not stop that.
set(rl_tools_ide_llvm_tools_off
    bugpoint bugpoint-passes dsymutil dxil-dis gold llc lli llvm-as llvm-as-fuzzer llvm-bcanalyzer llvm-c-test llvm-cat
    llvm-cfi-verify llvm-cgdata llvm-config llvm-cov llvm-ctxprof-util llvm-cvtres llvm-cxxdump llvm-cxxfilt llvm-cxxmap
    llvm-debuginfo-analyzer llvm-debuginfod llvm-debuginfod-find llvm-diff llvm-dis llvm-dis-fuzzer llvm-dlang-demangle-fuzzer
    llvm-dwarfdump llvm-dwarfutil llvm-dwp llvm-exegesis llvm-extract llvm-gpu-loader llvm-gsymutil llvm-ifs llvm-isel-fuzzer
    llvm-itanium-demangle-fuzzer llvm-jitlink llvm-jitlistener llvm-libtool-darwin llvm-link llvm-lipo llvm-lto llvm-lto2
    llvm-mc llvm-mc-assemble-fuzzer llvm-mc-disassemble-fuzzer llvm-mca llvm-microsoft-demangle-fuzzer llvm-ml llvm-modextract
    llvm-mt llvm-nm llvm-objcopy llvm-objdump llvm-opt-fuzzer llvm-opt-report llvm-pdbutil llvm-profdata llvm-profgen llvm-rc
    llvm-readobj llvm-readtapi llvm-reduce llvm-remarkutil llvm-rtdyld llvm-rust-demangle-fuzzer llvm-shlib llvm-sim llvm-size
    llvm-special-case-list-fuzzer llvm-split llvm-stress llvm-strings llvm-symbolizer llvm-tli-checker llvm-undname llvm-xray
    llvm-yaml-numeric-parser-fuzzer llvm-yaml-parser-fuzzer lto obj2yaml opt opt-viewer reduce-chunk-list remarks-shlib sancov
    sanstats spirv-tools verify-uselistorder vfabi-demangle-fuzzer xcode-toolchain yaml2obj
)
if(RL_TOOLS_IDE_ASSERTIONS)
    set(rl_tools_ide_assertions ON)
else()
    set(rl_tools_ide_assertions OFF)
endif()
set(rl_tools_ide_llvm_wasm_projects "clang;lld")
set(rl_tools_ide_llvm_wasm_links "clang;clang++")
set(rl_tools_ide_llvm_wasm_args
    -DRL_TOOLS_IDE_WASI_LLVM_FLAGS=ON
    -DRL_TOOLS_IDE_LTO=${RL_TOOLS_IDE_LTO}
    -DRL_TOOLS_IDE_LTO_JOBS=${RL_TOOLS_IDE_JOBS}
    -DLLVM_NATIVE_TOOL_DIR=${RL_TOOLS_IDE_HOST_DIR}/bin
    -DCMAKE_BUILD_TYPE=MinSizeRel
    -DLLVM_ENABLE_ASSERTIONS=${rl_tools_ide_assertions}
    -DLLVM_HOST_TRIPLE=${RL_TOOLS_IDE_TARGET}
    -DLLVM_DEFAULT_TARGET_TRIPLE=${RL_TOOLS_IDE_TARGET}
    -DLLVM_TARGETS_TO_BUILD=WebAssembly
    -DLLVM_ENABLE_THREADS=OFF
    -DLLVM_ENABLE_PIC=OFF
    -DLLVM_BUILD_STATIC=ON
    -DLLVM_BUILD_SHARED_LIBS=OFF
    -DLLVM_BUILD_TOOLS=OFF
    -DLLVM_BUILD_UTILS=OFF
    -DLLVM_BUILD_RUNTIME=OFF
    -DLLVM_INCLUDE_UTILS=OFF
    -DLLVM_INCLUDE_RUNTIMES=OFF
    -DLLVM_INCLUDE_TESTS=OFF
    -DLLVM_INCLUDE_EXAMPLES=OFF
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_INCLUDE_DOCS=OFF
    -DLLVM_ENABLE_ZLIB=OFF
    -DLLVM_ENABLE_ZSTD=OFF
    -DLLVM_ENABLE_LIBXML2=OFF
    -DLLVM_TOOL_LLVM_DRIVER_BUILD=ON
    -DLLVM_TOOL_LLVM_AR_BUILD=ON
    -DCLANG_BUILD_TOOLS=OFF
    -DCLANG_BUILD_EXAMPLES=OFF
    -DCLANG_ENABLE_ARCMT=OFF
    -DCLANG_ENABLE_STATIC_ANALYZER=OFF
    -DCLANG_INCLUDE_TESTS=OFF
    -DCLANG_INCLUDE_DOCS=OFF
    -DCLANG_TOOL_CLANG_SCAN_DEPS_BUILD=OFF
    -DCLANG_TOOL_CLANG_INSTALLAPI_BUILD=OFF
    -DLLD_BUILD_TOOLS=OFF
    -DDEFAULT_SYSROOT=/usr
    -DCLANG_RESOURCE_DIR=/usr
    -DLLVM_APPEND_VC_REV=ON
    -DLLVM_VERSION_SUFFIX=${RL_TOOLS_IDE_LLVM_VERSION_SUFFIX}
    -DLLVM_FORCE_VC_REPOSITORY=${RL_TOOLS_IDE_LLVM_REPOSITORY_URL}
    -DLLVM_FORCE_VC_REVISION=${RL_TOOLS_IDE_LLVM_COMMIT}
)
foreach(tool IN LISTS rl_tools_ide_llvm_tools_off)
    string(TOUPPER ${tool} tool_upper)
    string(REPLACE "-" "_" tool_upper ${tool_upper})
    list(APPEND rl_tools_ide_llvm_wasm_args -DLLVM_TOOL_${tool_upper}_BUILD=OFF)
endforeach()
ExternalProject_Add(llvm-wasm
    PREFIX ${CMAKE_BINARY_DIR}/llvm-wasm
    DOWNLOAD_COMMAND ""
    SOURCE_DIR ${RL_TOOLS_IDE_LLVM_SOURCE_DIR}/llvm
    BINARY_DIR ${RL_TOOLS_IDE_WASM_DIR}
    CMAKE_GENERATOR ${RL_TOOLS_IDE_GENERATOR}
    CMAKE_ARGS ${rl_tools_ide_toolchain_args} ${rl_tools_ide_resource_args} ${rl_tools_ide_llvm_wasm_args}
    CMAKE_CACHE_ARGS
        -DLLVM_ENABLE_PROJECTS:STRING=${rl_tools_ide_llvm_wasm_projects}
        -DCLANG_LINKS_TO_CREATE:STRING=${rl_tools_ide_llvm_wasm_links}
    BUILD_COMMAND ${CMAKE_COMMAND}
        -DGIT=${RL_TOOLS_IDE_GIT}
        -DSOURCE_DIR=${RL_TOOLS_IDE_LLVM_SOURCE_DIR}
        -DBUILD_DIR=<BINARY_DIR>
        -DJOBS=${RL_TOOLS_IDE_JOBS}
        -DTARGETS=llvm-driver,clang-resource-headers
        -P ${RL_TOOLS_IDE_CMAKE_DIR}/build_with_source_date.cmake
    INSTALL_COMMAND ""
    DEPENDS host libcxx
    USES_TERMINAL_CONFIGURE ON
    USES_TERMINAL_BUILD ON
)

# ------------------------------------------------------------------------------------------------------------------------ stage 4
# The manifest is assembled from a configure-time part (pins, flags, host facts) and a build-time part (commits, hashes)
function(rl_tools_ide_json_escape output value)
    string(REPLACE "\\" "\\\\" value "${value}")
    string(REPLACE "\"" "\\\"" value "${value}")
    set(${output} "${value}" PARENT_SCOPE)
endfunction()
function(rl_tools_ide_json_array output)
    set(items)
    foreach(item IN LISTS ARGN)
        rl_tools_ide_json_escape(item "${item}")
        list(APPEND items "\"${item}\"")
    endforeach()
    list(JOIN items ", " joined)
    set(${output} "[${joined}]" PARENT_SCOPE)
endfunction()
set(rl_tools_ide_host_os "${CMAKE_HOST_SYSTEM_NAME}")
if(EXISTS /etc/os-release)
    file(STRINGS /etc/os-release rl_tools_ide_pretty_name REGEX "^PRETTY_NAME=")
    string(REGEX REPLACE "^PRETTY_NAME=\"?([^\"]*)\"?$" "\\1" rl_tools_ide_host_os "${rl_tools_ide_pretty_name}")
endif()
rl_tools_ide_json_escape(rl_tools_ide_host_os_json "${rl_tools_ide_host_os}")
rl_tools_ide_json_escape(rl_tools_ide_host_compiler_json "${rl_tools_ide_host_compiler}")
rl_tools_ide_json_array(rl_tools_ide_json_llvm_args ${rl_tools_ide_toolchain_args} ${rl_tools_ide_resource_args} ${rl_tools_ide_llvm_wasm_args} "-DLLVM_ENABLE_PROJECTS:STRING=${rl_tools_ide_llvm_wasm_projects}" "-DCLANG_LINKS_TO_CREATE:STRING=${rl_tools_ide_llvm_wasm_links}")
rl_tools_ide_json_array(rl_tools_ide_json_compiler_rt_args ${rl_tools_ide_toolchain_args} ${rl_tools_ide_compiler_rt_args})
rl_tools_ide_json_array(rl_tools_ide_json_wasi_libc_args ${rl_tools_ide_toolchain_args} ${rl_tools_ide_resource_args} ${rl_tools_ide_wasi_libc_args})
rl_tools_ide_json_array(rl_tools_ide_json_libcxx_args ${rl_tools_ide_toolchain_args} ${rl_tools_ide_resource_args} ${rl_tools_ide_libcxx_args} "-DLLVM_ENABLE_RUNTIMES:STRING=${rl_tools_ide_libcxx_runtimes}")
rl_tools_ide_json_array(rl_tools_ide_json_host_args ${rl_tools_ide_host_args} "-DLLVM_ENABLE_PROJECTS:STRING=${rl_tools_ide_host_projects}")
string(TOLOWER "${rl_tools_ide_assertions}" rl_tools_ide_assertions_json)
if(rl_tools_ide_assertions_json STREQUAL "on")
    set(rl_tools_ide_assertions_json true)
else()
    set(rl_tools_ide_assertions_json false)
endif()
set(rl_tools_ide_static_manifest ${CMAKE_BINARY_DIR}/manifest_static.json)
file(WRITE ${rl_tools_ide_static_manifest}
"  \"format\": 1,
  \"target\": \"${RL_TOOLS_IDE_TARGET}\",
  \"cpu\": \"${RL_TOOLS_IDE_WASI_CPU}\",
  \"llvm_repository\": \"${RL_TOOLS_IDE_LLVM_REPOSITORY}\",
  \"llvm_branch\": \"${RL_TOOLS_IDE_LLVM_BRANCH}\",
  \"llvm_upstream\": \"${RL_TOOLS_IDE_LLVM_UPSTREAM}\",
  \"llvm_version_suffix\": \"${RL_TOOLS_IDE_LLVM_VERSION_SUFFIX}\",
  \"wasi_libc_repository\": \"${RL_TOOLS_IDE_WASI_LIBC_REPOSITORY}\",
  \"wasi_libc_tag\": \"${RL_TOOLS_IDE_WASI_LIBC_TAG}\",
  \"host_os\": \"${rl_tools_ide_host_os_json}\",
  \"host_cmake\": \"${CMAKE_VERSION}\",
  \"host_generator\": \"${RL_TOOLS_IDE_GENERATOR}\",
  \"host_compiler_from_source\": ${RL_TOOLS_IDE_BUILD_HOST_CLANG},
  \"lto\": \"${RL_TOOLS_IDE_LTO}\",
  \"assertions\": ${rl_tools_ide_assertions_json},
  \"cmake_args_host\": ${rl_tools_ide_json_host_args},
  \"cmake_args_compiler_rt\": ${rl_tools_ide_json_compiler_rt_args},
  \"cmake_args_wasi_libc\": ${rl_tools_ide_json_wasi_libc_args},
  \"cmake_args_libcxx\": ${rl_tools_ide_json_libcxx_args},
  \"cmake_args_llvm\": ${rl_tools_ide_json_llvm_args},
")
string(REPLACE "host_compiler_from_source\": ON" "host_compiler_from_source\": true" rl_tools_ide_static_content "")
file(READ ${rl_tools_ide_static_manifest} rl_tools_ide_static_content)
string(REPLACE "\"host_compiler_from_source\": ON" "\"host_compiler_from_source\": true" rl_tools_ide_static_content "${rl_tools_ide_static_content}")
string(REPLACE "\"host_compiler_from_source\": OFF" "\"host_compiler_from_source\": false" rl_tools_ide_static_content "${rl_tools_ide_static_content}")
file(WRITE ${rl_tools_ide_static_manifest} "${rl_tools_ide_static_content}")

# Ninja needs a declared producer for every file dependency; an ExternalProject's outputs have none, its build stamp does
ExternalProject_Get_Property(llvm-wasm STAMP_DIR)
set(rl_tools_ide_llvm_wasm_stamp ${STAMP_DIR}/llvm-wasm-build)
set(rl_tools_ide_package_outputs
    ${RL_TOOLS_IDE_PACKAGE_DIR}/llvm.wasm
    ${RL_TOOLS_IDE_PACKAGE_DIR}/sysroot.tar
    ${RL_TOOLS_IDE_PACKAGE_DIR}/toolchain.json
    ${RL_TOOLS_IDE_PACKAGE_DIR}/SHA256SUMS
)
add_custom_command(OUTPUT ${rl_tools_ide_package_outputs}
    COMMAND ${CMAKE_COMMAND}
        -DGIT=${RL_TOOLS_IDE_GIT}
        -DTAR=${RL_TOOLS_IDE_TAR}
        -DHOST_LLVM_BIN=${RL_TOOLS_IDE_HOST_LLVM_BIN_DIR}
        -DLLVM_SOURCE_DIR=${RL_TOOLS_IDE_LLVM_SOURCE_DIR}
        -DWASI_LIBC_SOURCE_DIR=${RL_TOOLS_IDE_WASI_LIBC_SOURCE_DIR}
        -DSYSROOT_DIR=${RL_TOOLS_IDE_SYSROOT_DIR}
        -DRESOURCE_HEADERS=${RL_TOOLS_IDE_WASM_DIR}/usr/include
        -DMODULE=${RL_TOOLS_IDE_WASM_DIR}/bin/llvm
        -DEPOCH_FILE=${RL_TOOLS_IDE_WASM_DIR}/SOURCE_DATE_EPOCH
        -DSTATIC_MANIFEST=${rl_tools_ide_static_manifest}
        -DPACKAGE_DIR=${RL_TOOLS_IDE_PACKAGE_DIR}
        -DOUTPUT_DIR=${RL_TOOLS_IDE_OUTPUT_DIR}
        -P ${RL_TOOLS_IDE_CMAKE_DIR}/package.cmake
    DEPENDS llvm-wasm ${rl_tools_ide_llvm_wasm_stamp} ${RL_TOOLS_IDE_CMAKE_DIR}/package.cmake ${rl_tools_ide_static_manifest}
    COMMENT "Packaging llvm.wasm, sysroot.tar, toolchain.json into ${RL_TOOLS_IDE_OUTPUT_DIR}"
    VERBATIM
)
add_custom_target(package ALL DEPENDS ${rl_tools_ide_package_outputs})

# ------------------------------------------------------------------------------------------------------------------------ stage 5
# Verification re-runs only when the package or the checks change: bundle the headers, build the parity program with the
# host clang against the same sysroot, then module shape, identity, stack probe, pipeline with parity, and provenance.
set(rl_tools_ide_verify_stamp ${RL_TOOLS_IDE_PACKAGE_DIR}/VERIFIED)
set(rl_tools_ide_verify_commands
    COMMAND ${RL_TOOLS_REPOSITORY_DIR}/tools/ide/bundle.sh
    COMMAND ${CMAKE_COMMAND} -E env
        RL_TOOLS_IDE_TOOLCHAIN_DIR=${RL_TOOLS_IDE_TOOLCHAIN_DIR}
        RL_TOOLS_IDE_HOST_LLVM_BIN=${RL_TOOLS_IDE_HOST_LLVM_BIN_DIR}
        OUTPUT_DIR=${RL_TOOLS_IDE_PARITY_DIR}
        ${RL_TOOLS_REPOSITORY_DIR}/src/rl/environments/pendulum/sac/wasi/build.sh
)
if(RL_TOOLS_IDE_NODE)
    list(APPEND rl_tools_ide_verify_commands
        COMMAND ${RL_TOOLS_IDE_NODE} --no-warnings ${RL_TOOLS_IDE_TESTS_DIR}/toolchain.mjs --toolchain ${RL_TOOLS_IDE_OUTPUT_DIR}
        COMMAND ${RL_TOOLS_IDE_NODE} --no-warnings ${RL_TOOLS_IDE_TESTS_DIR}/pipeline.mjs --reference ${RL_TOOLS_IDE_PARITY_DIR}/training.wasm
    )
else()
    message(WARNING "node not found: the verify stage skips the module checks and the pipeline test (Ubuntu: sudo apt install nodejs)")
endif()
list(APPEND rl_tools_ide_verify_commands
    COMMAND ${RL_TOOLS_IDE_TESTS_DIR}/provenance.sh ${RL_TOOLS_IDE_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E touch ${rl_tools_ide_verify_stamp}
)
add_custom_command(OUTPUT ${rl_tools_ide_verify_stamp}
    ${rl_tools_ide_verify_commands}
    DEPENDS ${rl_tools_ide_package_outputs}
        ${RL_TOOLS_IDE_TESTS_DIR}/toolchain.mjs ${RL_TOOLS_IDE_TESTS_DIR}/pipeline.mjs ${RL_TOOLS_IDE_TESTS_DIR}/provenance.sh
        ${RL_TOOLS_REPOSITORY_DIR}/src/rl/environments/pendulum/sac/wasi/build.sh ${RL_TOOLS_REPOSITORY_DIR}/src/rl/environments/pendulum/sac/wasi/training.cpp
    WORKING_DIRECTORY ${RL_TOOLS_REPOSITORY_DIR}
    COMMENT "Verifying the toolchain package"
    USES_TERMINAL
    VERBATIM
)
add_custom_target(verify ALL DEPENDS ${rl_tools_ide_verify_stamp})

message(STATUS "RLtools IDE toolchain configure summary:")
message(STATUS "  llvm-project      ${RL_TOOLS_IDE_LLVM_BRANCH} @ ${RL_TOOLS_IDE_LLVM_COMMIT} = ${RL_TOOLS_IDE_LLVM_UPSTREAM} + WASI commit (${RL_TOOLS_IDE_LLVM_REPOSITORY})")
message(STATUS "  wasi-libc         ${RL_TOOLS_IDE_WASI_LIBC_TAG} @ ${RL_TOOLS_IDE_WASI_LIBC_COMMIT} (${RL_TOOLS_IDE_WASI_LIBC_REPOSITORY})")
message(STATUS "  cross compiler    ${RL_TOOLS_IDE_HOST_LLVM_BIN_DIR} (${rl_tools_ide_host_compiler})")
message(STATUS "  sources           ${FETCHCONTENT_BASE_DIR}")
message(STATUS "  toolchain dir     ${RL_TOOLS_IDE_TOOLCHAIN_DIR}")
message(STATUS "  output            ${RL_TOOLS_IDE_OUTPUT_DIR}")
message(STATUS "  generator         ${RL_TOOLS_IDE_GENERATOR}, ${RL_TOOLS_IDE_JOBS} jobs")
message(STATUS "  lto/assertions    ${RL_TOOLS_IDE_LTO}/${rl_tools_ide_assertions}")
if(RL_TOOLS_IDE_NODE)
    message(STATUS "  verify            node ${RL_TOOLS_IDE_NODE}, parity build with the cross compiler")
else()
    message(STATUS "  verify            provenance and parity build only (no node)")
endif()
