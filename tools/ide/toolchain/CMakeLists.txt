cmake_minimum_required(VERSION 3.26)
project(rl_tools_ide_toolchain NONE)
include(FetchContent)
include(ExternalProject)

set(llvm_project_repository https://github.com/rl-tools/llvm-project.git)
set(llvm_project_ref rltools-wasi-22.1.8)
set(llvm_project_upstream llvmorg-22.1.8)
set(wasi_libc_repository https://github.com/WebAssembly/wasi-libc.git)
set(wasi_libc_ref wasi-sdk-34)
set(wasi_libc_upstream "${wasi_libc_ref}")
set(RL_TOOLS_IDE_HOST_LLVM 22)
set(RL_TOOLS_IDE_TARGET wasm32-wasip1)
set(RL_TOOLS_IDE_WASI_CPU lime1)
set(IDE_VERSION_SUFFIX -rltools)

get_filename_component(RL_TOOLS_REPOSITORY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(IDE_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(IDE_TESTS_DIR "${RL_TOOLS_REPOSITORY_DIR}/tests/src/ide")
set(IDE_EXAMPLES_FILE "${RL_TOOLS_REPOSITORY_DIR}/static/ide/examples.json")
include("${IDE_CMAKE_DIR}/metadata.cmake")

set(RL_TOOLS_IDE_JOBS 5 CACHE STRING "Parallel jobs per stage, at most 5")
set(RL_TOOLS_IDE_LTO thin CACHE STRING "WASM compiler LTO: thin, full or off")
set_property(CACHE RL_TOOLS_IDE_LTO PROPERTY STRINGS thin full off)
option(RL_TOOLS_IDE_ASSERTIONS "LLVM assertions in the WASM compiler" ON)
option(RL_TOOLS_IDE_BOOTSTRAP_HOST "Build only a native clang/lld for a subsequent toolchain build" OFF)
option(RL_TOOLS_OFFLINE_BUILD "Use existing dependency checkouts without checking for updates" OFF)
set(RL_TOOLS_IDE_HOST_LLVM_BIN "/usr/lib/llvm-${RL_TOOLS_IDE_HOST_LLVM}/bin" CACHE PATH "Native clang/lld tool directory")
set(RL_TOOLS_IDE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bundle" CACHE PATH "Candidate bundle root")
set(RL_TOOLS_IDE_STAGE_DIR "${RL_TOOLS_REPOSITORY_DIR}/static/ide/build" CACHE PATH "Verified bundle served by the IDE")
get_filename_component(RL_TOOLS_IDE_OUTPUT_DIR "${RL_TOOLS_IDE_OUTPUT_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
get_filename_component(RL_TOOLS_IDE_STAGE_DIR "${RL_TOOLS_IDE_STAGE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
if(NOT RL_TOOLS_IDE_LTO MATCHES "^(thin|full|off)$" OR NOT RL_TOOLS_IDE_JOBS MATCHES "^[1-5]$")
    message(FATAL_ERROR "RL_TOOLS_IDE_LTO must be thin, full or off; RL_TOOLS_IDE_JOBS must be 1 through 5.")
endif()
if(CMAKE_CONFIGURATION_TYPES)
    message(FATAL_ERROR "The IDE superbuild requires a single-configuration generator, such as Ninja or Unix Makefiles.")
endif()
find_program(IDE_GIT git REQUIRED)
set(IDE_HOST_DIR "${CMAKE_BINARY_DIR}/host")
set(IDE_SYSROOT_BUILD_DIR "${CMAKE_BINARY_DIR}/sysroot")
set(IDE_SYSROOT_DIR "${CMAKE_BINARY_DIR}/prefix/usr")
set(IDE_RESOURCE_DIR "${CMAKE_BINARY_DIR}/resource")
set(IDE_WASM_DIR "${CMAKE_BINARY_DIR}/wasm")
set(IDE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_FILE}" "${IDE_CMAKE_DIR}/Toolchain-WASI.cmake"
    "${IDE_CMAKE_DIR}/fetch_source.cmake")
if(RL_TOOLS_OFFLINE_BUILD)
    set(FETCHCONTENT_FULLY_DISCONNECTED ON)
endif()
set(sources llvm_project)
if(NOT RL_TOOLS_IDE_BOOTSTRAP_HOST)
    list(APPEND sources wasi_libc)
endif()
set(source_checks)
foreach(source IN LISTS sources)
    set(fetch "${CMAKE_COMMAND}" "-DGIT=${IDE_GIT}" "-DREPOSITORY=${${source}_repository}" "-DREF=${${source}_ref}"
        "-DDIRECTORY=<SOURCE_DIR>" -P "${IDE_CMAKE_DIR}/fetch_source.cmake")
    FetchContent_Declare(${source} DOWNLOAD_COMMAND ${fetch} UPDATE_COMMAND ${fetch} SOURCE_SUBDIR rl-tools-not-a-subproject)
    FetchContent_MakeAvailable(${source})
    ide_run(${source}_commit "${IDE_GIT}" -C "${${source}_SOURCE_DIR}" rev-parse HEAD)
    ide_run(git_dir "${IDE_GIT}" -C "${${source}_SOURCE_DIR}" rev-parse --absolute-git-dir)
    list(APPEND IDE_CONFIGURE_DEPENDS "${git_dir}/HEAD")
    set(revision "${${source}_SOURCE_DIR};${${source}_commit};${${source}_repository};${${source}_ref}")
    if(DEFINED ${source}_built_revision AND NOT "${${source}_built_revision}" STREQUAL revision)
        if(source STREQUAL "llvm_project")
            file(REMOVE_RECURSE "${IDE_HOST_DIR}")
        endif()
        file(REMOVE_RECURSE "${IDE_SYSROOT_BUILD_DIR}" "${IDE_SYSROOT_DIR}" "${IDE_RESOURCE_DIR}" "${IDE_WASM_DIR}")
    endif()
    set(${source}_built_revision "${revision}" CACHE INTERNAL "Configured dependency revision" FORCE)
    set(check "${CMAKE_COMMAND}" "-DGIT=${IDE_GIT}" "-DDIRECTORY=${${source}_SOURCE_DIR}"
        "-DEXPECTED_COMMIT=${${source}_commit}" -P "${IDE_CMAKE_DIR}/fetch_source.cmake")
    ide_run(ignored ${check})
    list(APPEND source_checks COMMAND ${check})
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${IDE_CONFIGURE_DEPENDS})
add_custom_target(check-sources ${source_checks} VERBATIM)
ide_run(IDE_SOURCE_DATE_EPOCH "${IDE_GIT}" -C "${llvm_project_SOURCE_DIR}" show -s --format=%ct HEAD)

function(ide_external name)
    cmake_parse_arguments(PARSE_ARGV 1 stage "" "SOURCE;BINARY" "ARGS;TARGETS;DEPENDS;BYPRODUCTS")
    ExternalProject_Add(${name}
        PREFIX "${CMAKE_BINARY_DIR}/stamps/${name}"
        SOURCE_DIR "${stage_SOURCE}"
        BINARY_DIR "${stage_BINARY}"
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        CMAKE_GENERATOR "${CMAKE_GENERATOR}"
        LIST_SEPARATOR "|"
        CMAKE_ARGS ${stage_ARGS}
        CONFIGURE_HANDLED_BY_BUILD ON
        BUILD_COMMAND "${CMAKE_COMMAND}" -E env "SOURCE_DATE_EPOCH=${IDE_SOURCE_DATE_EPOCH}"
            "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel ${RL_TOOLS_IDE_JOBS} --target ${stage_TARGETS}
        BUILD_ALWAYS ON
        BUILD_BYPRODUCTS ${stage_BYPRODUCTS}
        INSTALL_COMMAND ""
        DEPENDS ${stage_DEPENDS}
        USES_TERMINAL_CONFIGURE ON
        USES_TERMINAL_BUILD ON
    )
    ExternalProject_Add_StepDependencies(${name} configure ${IDE_CONFIGURE_DEPENDS})
endfunction()

set(IDE_LLVM_COMMON_ARGS
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

if(RL_TOOLS_IDE_BOOTSTRAP_HOST)
    ide_external(host-compiler SOURCE "${llvm_project_SOURCE_DIR}/llvm" BINARY "${IDE_HOST_DIR}"
        ARGS ${IDE_LLVM_COMMON_ARGS} -DCMAKE_BUILD_TYPE=Release "-DLLVM_ENABLE_PROJECTS=clang|lld"
        TARGETS clang clang-resource-headers lld llvm-ar llvm-ranlib llvm-nm DEPENDS check-sources
        BYPRODUCTS "${IDE_HOST_DIR}/bin/clang" "${IDE_HOST_DIR}/bin/clang++" "${IDE_HOST_DIR}/bin/wasm-ld"
            "${IDE_HOST_DIR}/bin/llvm-ar" "${IDE_HOST_DIR}/bin/llvm-ranlib" "${IDE_HOST_DIR}/bin/llvm-nm")
    message(STATUS "Use -DRL_TOOLS_IDE_HOST_LLVM_BIN=${IDE_HOST_DIR}/bin in the IDE toolchain configure.")
    return()
endif()

foreach(tool clang clang++ wasm-ld llvm-ar llvm-ranlib llvm-nm)
    if(NOT EXISTS "${RL_TOOLS_IDE_HOST_LLVM_BIN}/${tool}")
        message(FATAL_ERROR "Missing ${tool} in ${RL_TOOLS_IDE_HOST_LLVM_BIN}. Install clang/lld ${RL_TOOLS_IDE_HOST_LLVM}, or configure a separate build with RL_TOOLS_IDE_BOOTSTRAP_HOST=ON.")
    endif()
endforeach()
ide_run(host_version "${RL_TOOLS_IDE_HOST_LLVM_BIN}/clang" -dumpversion)
if(NOT host_version MATCHES "^${RL_TOOLS_IDE_HOST_LLVM}([.]|$)")
    message(FATAL_ERROR "The host clang must have major version ${RL_TOOLS_IDE_HOST_LLVM}; found ${host_version}.")
endif()
find_program(IDE_TAR tar REQUIRED)
ide_run(tar_version "${IDE_TAR}" --version)
if(NOT tar_version MATCHES "GNU tar")
    message(FATAL_ERROR "GNU tar is required for deterministic ustar packaging.")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${IDE_EXAMPLES_FILE}")
set(IDE_BUILTINS "${IDE_SYSROOT_DIR}/lib/wasm32-unknown-wasip1/libclang_rt.builtins.a")
set(IDE_LIBC "${IDE_SYSROOT_DIR}/lib/${RL_TOOLS_IDE_TARGET}/libc.a")
set(IDE_LIBCXX "${IDE_SYSROOT_DIR}/lib/${RL_TOOLS_IDE_TARGET}/libc++.a")
set(IDE_LIBCXXABI "${IDE_SYSROOT_DIR}/lib/${RL_TOOLS_IDE_TARGET}/libc++abi.a")

set(ide_host_args ${IDE_LLVM_COMMON_ARGS}
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_C_COMPILER=${RL_TOOLS_IDE_HOST_LLVM_BIN}/clang"
    "-DCMAKE_CXX_COMPILER=${RL_TOOLS_IDE_HOST_LLVM_BIN}/clang++"
    -DLLVM_ENABLE_PROJECTS=clang)
ide_external(host SOURCE "${llvm_project_SOURCE_DIR}/llvm" BINARY "${IDE_HOST_DIR}"
    ARGS ${ide_host_args} TARGETS llvm-tblgen llvm-min-tblgen clang-tblgen DEPENDS check-sources
    BYPRODUCTS "${IDE_HOST_DIR}/bin/llvm-tblgen" "${IDE_HOST_DIR}/bin/llvm-min-tblgen" "${IDE_HOST_DIR}/bin/clang-tblgen")

set(ide_toolchain_args
    "-DCMAKE_TOOLCHAIN_FILE=${IDE_CMAKE_DIR}/Toolchain-WASI.cmake"
    "-DRL_TOOLS_IDE_HOST_LLVM_BIN=${RL_TOOLS_IDE_HOST_LLVM_BIN}"
    "-DRL_TOOLS_IDE_TARGET=${RL_TOOLS_IDE_TARGET}"
    "-DRL_TOOLS_IDE_WASI_SYSROOT=${IDE_SYSROOT_DIR}")
set(bootstrap_flags "-mcpu=${RL_TOOLS_IDE_WASI_CPU}")
string(APPEND bootstrap_flags " \"-ffile-prefix-map=${CMAKE_BINARY_DIR}=/rl-tools-ide-build\"")
string(APPEND bootstrap_flags " \"-ffile-prefix-map=${llvm_project_SOURCE_DIR}=/rl-tools-ide-sources/llvm-project\"")
string(APPEND bootstrap_flags " \"-ffile-prefix-map=${wasi_libc_SOURCE_DIR}=/rl-tools-ide-sources/wasi-libc\"")
set(bootstrap_link "${bootstrap_flags}")
set(runtime_flags "${bootstrap_flags} \"-resource-dir=${IDE_RESOURCE_DIR}\"")
set(runtime_link "${runtime_flags}")
set(module_flags "${runtime_flags} -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_GETPID -D_WASI_EMULATED_SIGNAL")
set(module_link "${runtime_link} -lwasi-emulated-mman -lwasi-emulated-getpid -lwasi-emulated-signal -Wl,--strip-all -Wl,--max-memory=4294967296 -Wl,-z,stack-size=8388608,--stack-first")
if(RL_TOOLS_IDE_LTO STREQUAL "thin")
    string(APPEND module_flags " -flto=thin")
    string(APPEND module_link " -flto=thin -Wl,--thinlto-jobs=${RL_TOOLS_IDE_JOBS}")
elseif(RL_TOOLS_IDE_LTO STREQUAL "full")
    string(APPEND module_flags " -flto")
    string(APPEND module_link " -flto")
endif()
foreach(stage bootstrap runtime module)
    set(ide_${stage}_args ${ide_toolchain_args}
        "-DCMAKE_C_FLAGS:STRING=${${stage}_flags}" "-DCMAKE_CXX_FLAGS:STRING=${${stage}_flags}"
        "-DCMAKE_ASM_FLAGS:STRING=${${stage}_flags}" "-DCMAKE_EXE_LINKER_FLAGS:STRING=${${stage}_link}")
endforeach()
set(ide_compiler_rt_args ${ide_bootstrap_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${IDE_SYSROOT_DIR}
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

ide_external(compiler-rt SOURCE "${llvm_project_SOURCE_DIR}/compiler-rt" BINARY "${IDE_SYSROOT_BUILD_DIR}/compiler-rt"
    ARGS ${ide_compiler_rt_args} TARGETS install DEPENDS host BYPRODUCTS "${IDE_BUILTINS}")

ide_run(host_resource "${RL_TOOLS_IDE_HOST_LLVM_BIN}/clang" -print-resource-dir)
file(GLOB_RECURSE host_headers CONFIGURE_DEPENDS "${host_resource}/include/*")
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/resource-inputs.txt" CONTENT "${host_headers}\n" @ONLY)
list(APPEND host_headers "${CMAKE_BINARY_DIR}/resource-inputs.txt")
set(ide_resource_outputs "${IDE_RESOURCE_DIR}/include/stddef.h" "${IDE_RESOURCE_DIR}/lib/wasm32-unknown-wasip1/libclang_rt.builtins.a")
add_custom_command(OUTPUT ${ide_resource_outputs}
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${IDE_RESOURCE_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${host_resource}/include" "${IDE_RESOURCE_DIR}/include"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${IDE_SYSROOT_DIR}/lib/wasm32-unknown-wasip1" "${IDE_RESOURCE_DIR}/lib/wasm32-unknown-wasip1"
    COMMAND "${CMAKE_COMMAND}" -E touch ${ide_resource_outputs}
    DEPENDS compiler-rt "${IDE_BUILTINS}" ${host_headers} VERBATIM)
add_custom_target(resource-dir DEPENDS ${ide_resource_outputs})
set(ide_wasi_libc_args ${ide_runtime_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${IDE_SYSROOT_DIR}
    -DTARGET_TRIPLE=${RL_TOOLS_IDE_TARGET}
    -DBUILTINS_LIB=${IDE_BUILTINS}
    -DBUILD_SHARED=OFF
    -DCHECK_SYMBOLS=OFF
)

ide_external(wasi-libc SOURCE "${wasi_libc_SOURCE_DIR}" BINARY "${IDE_SYSROOT_BUILD_DIR}/wasi-libc"
    ARGS ${ide_wasi_libc_args} TARGETS install DEPENDS resource-dir BYPRODUCTS "${IDE_LIBC}")
set(ide_libcxx_args ${ide_runtime_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${IDE_SYSROOT_DIR}
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
    "-DLLVM_ENABLE_RUNTIMES=libcxx|libcxxabi"
)

ide_external(libcxx SOURCE "${llvm_project_SOURCE_DIR}/runtimes" BINARY "${IDE_SYSROOT_BUILD_DIR}/libcxx"
    ARGS ${ide_libcxx_args} TARGETS install DEPENDS wasi-libc BYPRODUCTS "${IDE_LIBCXX}" "${IDE_LIBCXXABI}")
add_custom_target(sysroot DEPENDS libcxx)

foreach(project llvm clang)
    set(ide_${project}_tools)
    string(TOUPPER "${project}" prefix)
    set(directory "${llvm_project_SOURCE_DIR}/${project}/tools")
    file(GLOB children CONFIGURE_DEPENDS RELATIVE "${directory}" "${directory}/*")
    foreach(child IN LISTS children)
        if(EXISTS "${directory}/${child}/CMakeLists.txt")
            string(TOUPPER "${child}" option)
            string(REPLACE "-" "_" option "${option}")
            set(enabled OFF)
            if((project STREQUAL "llvm" AND child STREQUAL "llvm-driver") OR (project STREQUAL "clang" AND child STREQUAL "driver"))
                set(enabled ON)
            endif()
            list(APPEND ide_${project}_tools "-D${prefix}_TOOL_${option}_BUILD=${enabled}")
        endif()
    endforeach()
endforeach()
string(REGEX REPLACE "[.]git$" "" IDE_LLVM_REPOSITORY_URL "${llvm_project_repository}")

set(ide_llvm_wasm_args ${ide_module_args} ${IDE_LLVM_COMMON_ARGS} ${ide_llvm_tools} ${ide_clang_tools}
    "-DLLVM_NATIVE_TOOL_DIR=${IDE_HOST_DIR}/bin"
    # LLVM's cached default otherwise switches header generation from llvm-min-tblgen on the second configure.
    "-DLLVM_TABLEGEN=${IDE_HOST_DIR}/bin/llvm-tblgen"
    "-DCLANG_TABLEGEN=${IDE_HOST_DIR}/bin/clang-tblgen"
    -DCMAKE_BUILD_TYPE=MinSizeRel
    "-DLLVM_ENABLE_ASSERTIONS=${RL_TOOLS_IDE_ASSERTIONS}"
    "-DLLVM_HOST_TRIPLE=${RL_TOOLS_IDE_TARGET}"
    "-DLLVM_ENABLE_PROJECTS=clang|lld"
    -DCLANG_LINKS_TO_CREATE=clang++
    -DLLVM_ENABLE_THREADS=OFF -DLLVM_ENABLE_PIC=OFF -DLLVM_BUILD_STATIC=ON
    -DLLVM_BUILD_SHARED_LIBS=OFF -DLLVM_BUILD_UTILS=OFF
    -DCLANG_ENABLE_OBJC_REWRITER=OFF -DCLANG_ENABLE_STATIC_ANALYZER=OFF -DLLD_BUILD_TOOLS=OFF
    -DDEFAULT_SYSROOT=/usr -DCLANG_RESOURCE_DIR=/usr -DLLVM_APPEND_VC_REV=ON
    "-DLLVM_VERSION_SUFFIX=${IDE_VERSION_SUFFIX}"
    "-DLLVM_FORCE_VC_REPOSITORY=${IDE_LLVM_REPOSITORY_URL}" "-DLLVM_FORCE_VC_REVISION=${llvm_project_commit}")
ide_external(llvm-wasm SOURCE "${llvm_project_SOURCE_DIR}/llvm" BINARY "${IDE_WASM_DIR}"
    ARGS ${ide_llvm_wasm_args} TARGETS llvm-driver clang-resource-headers DEPENDS libcxx
    BYPRODUCTS "${IDE_WASM_DIR}/bin/llvm" "${IDE_WASM_DIR}/usr/include/stddef.h")

set(configuration "{}")
ide_json_field(configuration target "${RL_TOOLS_IDE_TARGET}")
ide_json_field(configuration cpu "${RL_TOOLS_IDE_WASI_CPU}")
ide_json_field(configuration llvm_version_suffix "${IDE_VERSION_SUFFIX}")
foreach(source IN LISTS sources)
    string(REPLACE "llvm_project" "llvm" prefix "${source}")
    foreach(field repository ref commit upstream)
        ide_json_field(configuration "${prefix}_${field}" "${${source}_${field}}")
    endforeach()
endforeach()
string(JSON configuration SET "${configuration}" format 1)
string(JSON configuration SET "${configuration}" source_date_epoch ${IDE_SOURCE_DATE_EPOCH})
ide_json_field(configuration lto "${RL_TOOLS_IDE_LTO}")
if(RL_TOOLS_IDE_ASSERTIONS)
    string(JSON configuration SET "${configuration}" assertions true)
else()
    string(JSON configuration SET "${configuration}" assertions false)
endif()
ide_json_field(configuration host_os "${CMAKE_HOST_SYSTEM_NAME}")
ide_json_field(configuration host_cmake "${CMAKE_VERSION}")
ide_json_field(configuration host_generator "${CMAKE_GENERATOR}")
foreach(stage host compiler_rt wasi_libc libcxx llvm_wasm)
    set(arguments "[]")
    foreach(argument IN LISTS ide_${stage}_args)
        string(REPLACE "|" ";" argument "${argument}")
        string(JSON index LENGTH "${arguments}")
        ide_json_field(arguments "${index}" "${argument}")
    endforeach()
    string(JSON configuration SET "${configuration}" "cmake_args_${stage}" "${arguments}")
endforeach()
set(IDE_CONFIGURATION "${CMAKE_BINARY_DIR}/toolchain-config.json")
file(CONFIGURE OUTPUT "${IDE_CONFIGURATION}" CONTENT "${configuration}\n" @ONLY)

file(GLOB_RECURSE sysroot_inputs CONFIGURE_DEPENDS "${IDE_SYSROOT_DIR}/*" "${IDE_WASM_DIR}/usr/include/*")
set(package_files toolchain/llvm.wasm toolchain/sysroot.tar toolchain/toolchain.json toolchain/SHA256SUMS)
set(package_outputs ${package_files})
list(TRANSFORM package_outputs PREPEND "${RL_TOOLS_IDE_OUTPUT_DIR}/")
add_custom_command(OUTPUT ${package_outputs}
    COMMAND "${CMAKE_COMMAND}" "-DCONFIG=${IDE_CONFIGURATION}" "-DHOST_LLVM_BIN=${RL_TOOLS_IDE_HOST_LLVM_BIN}"
        "-DLLVM_SOURCE_DIR=${llvm_project_SOURCE_DIR}" "-DSYSROOT_DIR=${IDE_SYSROOT_DIR}"
        "-DRESOURCE_HEADERS=${IDE_WASM_DIR}/usr/include" "-DMODULE=${IDE_WASM_DIR}/bin/llvm"
        "-DPACKAGE_DIR=${RL_TOOLS_IDE_OUTPUT_DIR}/toolchain" -P "${IDE_CMAKE_DIR}/package.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch ${package_outputs}
    DEPENDS llvm-wasm "${IDE_WASM_DIR}/bin/llvm" "${IDE_WASM_DIR}/usr/include/stddef.h"
        "${IDE_BUILTINS}" "${IDE_LIBC}" "${IDE_LIBCXX}" "${IDE_LIBCXXABI}" ${sysroot_inputs}
        "${IDE_CONFIGURATION}" "${IDE_CMAKE_DIR}/package.cmake" "${IDE_CMAKE_DIR}/metadata.cmake"
    VERBATIM)
add_custom_target(package DEPENDS ${package_outputs})

file(READ "${IDE_EXAMPLES_FILE}" examples)
set(bundle_files rl_tools_include.tar manifest.json examples.json)
file(GLOB_RECURSE bundle_inputs CONFIGURE_DEPENDS "${RL_TOOLS_REPOSITORY_DIR}/include/rl_tools/*")
string(JSON count LENGTH "${examples}" examples)
math(EXPR last "${count} - 1")
foreach(index RANGE ${last})
    string(JSON name MEMBER "${examples}" examples ${index})
    string(JSON source GET "${examples}" examples "${name}" source)
    list(APPEND bundle_files "examples/${name}.cpp")
    list(APPEND bundle_inputs "${RL_TOOLS_REPOSITORY_DIR}/${source}")
endforeach()
ide_run(repository_git_dir "${IDE_GIT}" -C "${RL_TOOLS_REPOSITORY_DIR}" rev-parse --absolute-git-dir)
list(APPEND bundle_inputs "${repository_git_dir}/HEAD")
execute_process(COMMAND "${IDE_GIT}" -C "${RL_TOOLS_REPOSITORY_DIR}" symbolic-ref -q HEAD OUTPUT_VARIABLE repository_ref OUTPUT_STRIP_TRAILING_WHITESPACE)
foreach(ref "${repository_git_dir}/${repository_ref}" "${repository_git_dir}/packed-refs")
    if(EXISTS "${ref}" AND NOT IS_DIRECTORY "${ref}")
        list(APPEND bundle_inputs "${ref}")
    endif()
endforeach()
set(bundle_outputs ${bundle_files})
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/bundle-inputs.txt" CONTENT "${bundle_inputs}\n" @ONLY)
list(APPEND bundle_inputs "${CMAKE_BINARY_DIR}/bundle-inputs.txt")
list(TRANSFORM bundle_outputs PREPEND "${RL_TOOLS_IDE_OUTPUT_DIR}/")
add_custom_command(OUTPUT ${bundle_outputs}
    COMMAND "${CMAKE_COMMAND}" "-DBUNDLE_DIR=${RL_TOOLS_IDE_OUTPUT_DIR}" -P "${IDE_CMAKE_DIR}/bundle.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch ${bundle_outputs}
    DEPENDS ${bundle_inputs} "${IDE_EXAMPLES_FILE}" "${IDE_CMAKE_DIR}/bundle.cmake" "${IDE_CMAKE_DIR}/metadata.cmake"
    VERBATIM)
add_custom_target(bundle DEPENDS ${bundle_outputs})

set(example_flags)
string(JSON count LENGTH "${examples}" flags)
if(count GREATER 0)
    math(EXPR last "${count} - 1")
    foreach(index RANGE ${last})
        string(JSON flag GET "${examples}" flags ${index})
        string(REPLACE ";" "\\;" flag "${flag}")
        list(APPEND example_flags "${flag}")
    endforeach()
endif()
string(JSON example_source GET "${examples}" examples pendulum_sac source)
set(IDE_REFERENCE "${CMAKE_BINARY_DIR}/parity/training.wasm")
add_custom_command(OUTPUT "${IDE_REFERENCE}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/parity"
    COMMAND "${RL_TOOLS_IDE_HOST_LLVM_BIN}/clang++" "--target=${RL_TOOLS_IDE_TARGET}"
        "--sysroot=${IDE_SYSROOT_DIR}" "-resource-dir=${IDE_RESOURCE_DIR}" ${example_flags}
        "${RL_TOOLS_REPOSITORY_DIR}/${example_source}" -o "${IDE_REFERENCE}" -MMD -MF "${IDE_REFERENCE}.d"
    DEPENDS package "${RL_TOOLS_REPOSITORY_DIR}/${example_source}" "${IDE_EXAMPLES_FILE}"
        "${IDE_LIBC}" "${IDE_LIBCXX}" "${IDE_LIBCXXABI}" ${ide_resource_outputs}
    DEPFILE "${IDE_REFERENCE}.d"
    WORKING_DIRECTORY "${RL_TOOLS_REPOSITORY_DIR}"
    VERBATIM)
add_custom_target(parity DEPENDS "${IDE_REFERENCE}")

enable_testing()
set(IDE_REQUIRE_TESTS ON)
add_subdirectory("${IDE_TESTS_DIR}" tests)
file(GLOB test_inputs CONFIGURE_DEPENDS "${IDE_TESTS_DIR}/*.mjs" "${IDE_TESTS_DIR}/*.cmake"
    "${IDE_TESTS_DIR}/*.py" "${IDE_TESTS_DIR}/CMakeLists.txt" "${RL_TOOLS_REPOSITORY_DIR}/static/ide/*.js" "${RL_TOOLS_REPOSITORY_DIR}/static/ide/*.html")
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/verification-inputs.txt" CONTENT "${test_inputs}\n" @ONLY)
list(APPEND test_inputs "${CMAKE_BINARY_DIR}/verification-inputs.txt")
if(NOT IDE_NODE_EXECUTABLE)
    set(IDE_NODE_EXECUTABLE node)
endif()
set(IDE_VERIFIED "${CMAKE_BINARY_DIR}/VERIFIED")
add_custom_command(OUTPUT "${IDE_VERIFIED}"
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${IDE_VERIFIED}"
    COMMAND "${IDE_NODE_EXECUTABLE}" -e "if (Number(process.versions.node.split('.')[0]) < 22) throw new Error('Verification requires Node.js 22+')"
    COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${CMAKE_BINARY_DIR}" --output-on-failure --no-tests=error -L "^ide-verify$" --timeout 300 -j1
    COMMAND "${CMAKE_COMMAND}" -E touch "${IDE_VERIFIED}"
    DEPENDS ${package_outputs} ${bundle_outputs} ${test_inputs} "${CMAKE_CURRENT_LIST_FILE}" "${RL_TOOLS_REPOSITORY_DIR}/static/ide/package.json"
    USES_TERMINAL VERBATIM)
add_custom_target(verify DEPENDS "${IDE_VERIFIED}")
add_custom_target(verify-training
    COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${CMAKE_BINARY_DIR}" --output-on-failure --no-tests=error -L "^ide-training$" --timeout 900 -j1
    DEPENDS verify parity USES_TERMINAL VERBATIM)

if(RL_TOOLS_IDE_STAGE_DIR STREQUAL RL_TOOLS_IDE_OUTPUT_DIR)
    message(FATAL_ERROR "Candidate and served bundle directories must be different so verification precedes staging.")
endif()
set(stage_outputs)
foreach(file IN LISTS package_files bundle_files)
    set(destination "${RL_TOOLS_IDE_STAGE_DIR}/${file}")
    get_filename_component(directory "${destination}" DIRECTORY)
    add_custom_command(OUTPUT "${destination}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${directory}"
        COMMAND "${CMAKE_COMMAND}" -E copy "${RL_TOOLS_IDE_OUTPUT_DIR}/${file}" "${destination}"
        DEPENDS "${IDE_VERIFIED}" "${RL_TOOLS_IDE_OUTPUT_DIR}/${file}" "${CMAKE_CURRENT_LIST_FILE}" VERBATIM)
    list(APPEND stage_outputs "${destination}")
endforeach()
add_custom_target(stage ALL DEPENDS ${stage_outputs})

message(STATUS "RLtools IDE toolchain configure summary:")
message(STATUS "  llvm sources  ${llvm_project_SOURCE_DIR}")
message(STATUS "  libc sources  ${wasi_libc_SOURCE_DIR}")
message(STATUS "  cross compiler ${RL_TOOLS_IDE_HOST_LLVM_BIN} (clang ${host_version})")
message(STATUS "  build root    ${CMAKE_BINARY_DIR}")
message(STATUS "  candidate     ${RL_TOOLS_IDE_OUTPUT_DIR}")
message(STATUS "  served bundle ${RL_TOOLS_IDE_STAGE_DIR}")
message(STATUS "  generator/jobs ${CMAKE_GENERATOR}/${RL_TOOLS_IDE_JOBS}")
message(STATUS "  lto/assertions ${RL_TOOLS_IDE_LTO}/${RL_TOOLS_IDE_ASSERTIONS}")
