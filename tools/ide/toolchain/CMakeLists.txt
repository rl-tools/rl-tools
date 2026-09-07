cmake_minimum_required(VERSION 3.26)
project(rl_tools_ide_toolchain NONE)

get_filename_component(RL_TOOLS_REPOSITORY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(IDE_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(IDE_TESTS_DIR "${RL_TOOLS_REPOSITORY_DIR}/tests/src/ide")
set(IDE_LOCK_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../dependencies.json")
set(IDE_EXAMPLES_FILE "${RL_TOOLS_REPOSITORY_DIR}/static/ide/examples.json")
include("${IDE_CMAKE_DIR}/metadata.cmake")
include("${IDE_CMAKE_DIR}/sources.cmake")
include("${IDE_CMAKE_DIR}/external.cmake")
include("${IDE_CMAKE_DIR}/flags.cmake")

set(RL_TOOLS_IDE_JOBS 5 CACHE STRING "Parallel jobs per stage, at most 5")
set(RL_TOOLS_IDE_LTO thin CACHE STRING "WASM compiler LTO: thin, full or off")
set_property(CACHE RL_TOOLS_IDE_LTO PROPERTY STRINGS thin full off)
option(RL_TOOLS_IDE_ASSERTIONS "LLVM assertions in the WASM compiler" ON)
set(RL_TOOLS_IDE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bundle" CACHE PATH "Candidate bundle root")
set(RL_TOOLS_IDE_STAGE_DIR "${RL_TOOLS_REPOSITORY_DIR}/static/ide/build" CACHE PATH "Verified bundle served by the IDE")
get_filename_component(RL_TOOLS_IDE_OUTPUT_DIR "${RL_TOOLS_IDE_OUTPUT_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
get_filename_component(RL_TOOLS_IDE_STAGE_DIR "${RL_TOOLS_IDE_STAGE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
if(DEFINED RL_TOOLS_IDE_TOOLCHAIN_DIR OR RL_TOOLS_IDE_BUILD_HOST_CLANG)
    message(FATAL_ERROR "Use -B to choose the build root. Build a host compiler separately with tools/ide/bootstrap.")
endif()
if(NOT RL_TOOLS_IDE_LTO MATCHES "^(thin|full|off)$" OR NOT RL_TOOLS_IDE_JOBS MATCHES "^[1-5]$")
    message(FATAL_ERROR "RL_TOOLS_IDE_LTO must be thin, full or off; RL_TOOLS_IDE_JOBS must be 1 through 5.")
endif()
if(CMAKE_CONFIGURATION_TYPES)
    message(FATAL_ERROR "The IDE superbuild requires a single-configuration generator, such as Ninja or Unix Makefiles.")
endif()
file(READ "${IDE_LOCK_FILE}" lock)
string(JSON RL_TOOLS_IDE_TARGET GET "${lock}" target)
string(JSON RL_TOOLS_IDE_WASI_CPU GET "${lock}" cpu)
string(JSON RL_TOOLS_IDE_HOST_LLVM GET "${lock}" host_llvm_major)
string(JSON IDE_VERSION_SUFFIX GET "${lock}" llvm_version_suffix)
string(JSON IDE_LLVM_REPOSITORY GET "${lock}" sources llvm-project repository)
set(RL_TOOLS_IDE_HOST_LLVM_BIN "/usr/lib/llvm-${RL_TOOLS_IDE_HOST_LLVM}/bin" CACHE PATH "Native clang/lld tool directory")
foreach(tool clang clang++ wasm-ld llvm-ar llvm-ranlib llvm-nm)
    if(NOT EXISTS "${RL_TOOLS_IDE_HOST_LLVM_BIN}/${tool}")
        message(FATAL_ERROR "Missing ${tool} in ${RL_TOOLS_IDE_HOST_LLVM_BIN}. Install clang/lld ${RL_TOOLS_IDE_HOST_LLVM}, or use tools/ide/bootstrap.")
    endif()
endforeach()
find_program(IDE_GIT git REQUIRED)
find_program(IDE_TAR tar REQUIRED)
ide_run(tar_version "${IDE_TAR}" --version)
if(NOT tar_version MATCHES "GNU tar")
    message(FATAL_ERROR "GNU tar is required for deterministic ustar packaging.")
endif()
ide_fetch_sources(source_revisions "${IDE_LOCK_FILE}")
string(JSON lock SET "${lock}" sources "${source_revisions}")
string(JSON IDE_LLVM_SOURCE GET "${source_revisions}" llvm-project directory)
string(JSON IDE_LIBC_SOURCE GET "${source_revisions}" wasi-libc directory)
string(JSON IDE_LLVM_COMMIT GET "${source_revisions}" llvm-project commit)
ide_run(IDE_SOURCE_DATE_EPOCH "${IDE_GIT}" -C "${IDE_LLVM_SOURCE}" show -s --format=%ct HEAD)
ide_run(host_version "${RL_TOOLS_IDE_HOST_LLVM_BIN}/clang" -dumpversion)
if(NOT host_version MATCHES "^${RL_TOOLS_IDE_HOST_LLVM}([.]|$)")
    message(FATAL_ERROR "The host clang must have major version ${RL_TOOLS_IDE_HOST_LLVM}; found ${host_version}.")
endif()
set(IDE_CONFIGURE_DEPENDS "${IDE_LOCK_FILE}" "${IDE_EXAMPLES_FILE}" "${IDE_CMAKE_DIR}/Toolchain-WASI.cmake"
    "${IDE_CMAKE_DIR}/flags.cmake" "${IDE_CMAKE_DIR}/external.cmake" "${IDE_CMAKE_DIR}/sources.cmake"
    "${IDE_CMAKE_DIR}/fetch_source.cmake")
foreach(source "${IDE_LLVM_SOURCE}" "${IDE_LIBC_SOURCE}")
    ide_run(git_dir "${IDE_GIT}" -C "${source}" rev-parse --absolute-git-dir)
    list(APPEND IDE_CONFIGURE_DEPENDS "${git_dir}/HEAD")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${IDE_CONFIGURE_DEPENDS})
set(IDE_HOST_DIR "${CMAKE_BINARY_DIR}/host")
set(IDE_SYSROOT_BUILD_DIR "${CMAKE_BINARY_DIR}/sysroot")
set(IDE_SYSROOT_DIR "${CMAKE_BINARY_DIR}/prefix/usr")
set(IDE_RESOURCE_DIR "${CMAKE_BINARY_DIR}/resource")
set(IDE_WASM_DIR "${CMAKE_BINARY_DIR}/wasm")
set(revision_file "${CMAKE_BINARY_DIR}/source-revisions.json")
if(EXISTS "${revision_file}")
    file(READ "${revision_file}" previous_revisions)
    if(NOT previous_revisions STREQUAL source_revisions)
        string(JSON previous_llvm GET "${previous_revisions}" llvm-project commit)
        if(NOT previous_llvm STREQUAL IDE_LLVM_COMMIT)
            file(REMOVE_RECURSE "${IDE_HOST_DIR}")
        endif()
        file(REMOVE_RECURSE "${IDE_SYSROOT_BUILD_DIR}" "${IDE_SYSROOT_DIR}" "${IDE_RESOURCE_DIR}" "${IDE_WASM_DIR}")
    endif()
endif()
file(CONFIGURE OUTPUT "${revision_file}" CONTENT "${source_revisions}" @ONLY)
ide_check_sources("${revision_file}")
add_custom_target(check-sources
    COMMAND "${CMAKE_COMMAND}" "-DREVISION_FILE=${revision_file}" -P "${IDE_CMAKE_DIR}/sources.cmake"
    VERBATIM)
set(IDE_BUILTINS "${IDE_SYSROOT_DIR}/lib/wasm32-unknown-wasip1/libclang_rt.builtins.a")
set(IDE_LIBC "${IDE_SYSROOT_DIR}/lib/${RL_TOOLS_IDE_TARGET}/libc.a")
set(IDE_LIBCXX "${IDE_SYSROOT_DIR}/lib/${RL_TOOLS_IDE_TARGET}/libc++.a")
set(IDE_LIBCXXABI "${IDE_SYSROOT_DIR}/lib/${RL_TOOLS_IDE_TARGET}/libc++abi.a")
include("${IDE_CMAKE_DIR}/llvm_options.cmake")

set(ide_host_args ${IDE_LLVM_COMMON_ARGS}
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_C_COMPILER=${RL_TOOLS_IDE_HOST_LLVM_BIN}/clang"
    "-DCMAKE_CXX_COMPILER=${RL_TOOLS_IDE_HOST_LLVM_BIN}/clang++"
    -DLLVM_ENABLE_PROJECTS=clang)
ide_external(host SOURCE "${IDE_LLVM_SOURCE}/llvm" BINARY "${IDE_HOST_DIR}"
    ARGS ${ide_host_args} TARGETS llvm-tblgen llvm-min-tblgen clang-tblgen DEPENDS check-sources
    BYPRODUCTS "${IDE_HOST_DIR}/bin/llvm-tblgen" "${IDE_HOST_DIR}/bin/llvm-min-tblgen" "${IDE_HOST_DIR}/bin/clang-tblgen")

set(ide_toolchain_args
    "-DCMAKE_TOOLCHAIN_FILE=${IDE_CMAKE_DIR}/Toolchain-WASI.cmake"
    "-DRL_TOOLS_IDE_HOST_LLVM_BIN=${RL_TOOLS_IDE_HOST_LLVM_BIN}"
    "-DRL_TOOLS_IDE_TARGET=${RL_TOOLS_IDE_TARGET}"
    "-DRL_TOOLS_IDE_WASI_SYSROOT=${IDE_SYSROOT_DIR}")
ide_wasi_flags(bootstrap_flags bootstrap_link "" FALSE)
ide_wasi_flags(runtime_flags runtime_link "${IDE_RESOURCE_DIR}" FALSE)
ide_wasi_flags(module_flags module_link "${IDE_RESOURCE_DIR}" TRUE)
foreach(stage bootstrap runtime module)
    set(ide_${stage}_args ${ide_toolchain_args}
        "-DCMAKE_C_FLAGS:STRING=${${stage}_flags}" "-DCMAKE_CXX_FLAGS:STRING=${${stage}_flags}"
        "-DCMAKE_ASM_FLAGS:STRING=${${stage}_flags}" "-DCMAKE_EXE_LINKER_FLAGS:STRING=${${stage}_link}")
endforeach()
set(ide_compiler_rt_args
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

set(ide_compiler_rt_args ${ide_bootstrap_args} ${ide_compiler_rt_args})
ide_external(compiler-rt SOURCE "${IDE_LLVM_SOURCE}/compiler-rt" BINARY "${IDE_SYSROOT_BUILD_DIR}/compiler-rt"
    ARGS ${ide_compiler_rt_args} TARGETS install DEPENDS host BYPRODUCTS "${IDE_BUILTINS}")

ide_run(host_resource "${RL_TOOLS_IDE_HOST_LLVM_BIN}/clang" -print-resource-dir)
file(GLOB_RECURSE host_headers CONFIGURE_DEPENDS "${host_resource}/include/*")
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/resource-inputs.txt" CONTENT "${host_headers}\n" @ONLY)
list(APPEND host_headers "${CMAKE_BINARY_DIR}/resource-inputs.txt")
set(ide_resource_outputs "${IDE_RESOURCE_DIR}/include/stddef.h" "${IDE_RESOURCE_DIR}/lib/wasm32-unknown-wasip1/libclang_rt.builtins.a")
add_custom_command(OUTPUT ${ide_resource_outputs}
    COMMAND "${CMAKE_COMMAND}" "-DCLANG=${RL_TOOLS_IDE_HOST_LLVM_BIN}/clang"
        "-DBUILTINS=${IDE_SYSROOT_DIR}/lib/wasm32-unknown-wasip1" "-DRESOURCE_DIR=${IDE_RESOURCE_DIR}" -P "${IDE_CMAKE_DIR}/resource_dir.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch ${ide_resource_outputs}
    DEPENDS compiler-rt "${IDE_BUILTINS}" ${host_headers} "${IDE_CMAKE_DIR}/resource_dir.cmake" VERBATIM)
add_custom_target(resource-dir DEPENDS ${ide_resource_outputs})
set(ide_wasi_libc_args
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${IDE_SYSROOT_DIR}
    -DTARGET_TRIPLE=${RL_TOOLS_IDE_TARGET}
    -DBUILTINS_LIB=${IDE_BUILTINS}
    -DBUILD_SHARED=OFF
    -DCHECK_SYMBOLS=OFF
)

set(ide_wasi_libc_args ${ide_runtime_args} ${ide_wasi_libc_args})
ide_external(wasi-libc SOURCE "${IDE_LIBC_SOURCE}" BINARY "${IDE_SYSROOT_BUILD_DIR}/wasi-libc"
    ARGS ${ide_wasi_libc_args} TARGETS install DEPENDS resource-dir BYPRODUCTS "${IDE_LIBC}")
set(ide_libcxx_args
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
)

set(ide_libcxx_args ${ide_runtime_args} ${ide_libcxx_args} "-DLLVM_ENABLE_RUNTIMES=libcxx|libcxxabi")
ide_external(libcxx SOURCE "${IDE_LLVM_SOURCE}/runtimes" BINARY "${IDE_SYSROOT_BUILD_DIR}/libcxx"
    ARGS ${ide_libcxx_args} TARGETS install DEPENDS wasi-libc BYPRODUCTS "${IDE_LIBCXX}" "${IDE_LIBCXXABI}")
add_custom_target(sysroot DEPENDS libcxx)

ide_select_tools(ide_llvm_tools "${IDE_LLVM_SOURCE}/llvm/tools" LLVM llvm-driver)
ide_select_tools(ide_clang_tools "${IDE_LLVM_SOURCE}/clang/tools" CLANG driver)
string(REGEX REPLACE "[.]git$" "" IDE_LLVM_REPOSITORY_URL "${IDE_LLVM_REPOSITORY}")

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
    "-DLLVM_FORCE_VC_REPOSITORY=${IDE_LLVM_REPOSITORY_URL}" "-DLLVM_FORCE_VC_REVISION=${IDE_LLVM_COMMIT}")
ide_external(llvm-wasm SOURCE "${IDE_LLVM_SOURCE}/llvm" BINARY "${IDE_WASM_DIR}"
    ARGS ${ide_llvm_wasm_args} TARGETS llvm-driver clang-resource-headers DEPENDS libcxx
    BYPRODUCTS "${IDE_WASM_DIR}/bin/llvm" "${IDE_WASM_DIR}/usr/include/stddef.h")

set(configuration "{}")
foreach(field target cpu llvm_version_suffix)
    string(JSON value GET "${lock}" "${field}")
    ide_json_field(configuration "${field}" "${value}")
endforeach()
foreach(pair "llvm-project|llvm" "wasi-libc|wasi_libc")
    string(REPLACE "|" ";" pair "${pair}")
    list(GET pair 0 source)
    list(GET pair 1 prefix)
    foreach(field repository ref commit upstream)
        string(JSON value GET "${lock}" sources "${source}" "${field}")
        ide_json_field(configuration "${prefix}_${field}" "${value}")
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
    ide_json_array(arguments ${ide_${stage}_args})
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
        "-DLLVM_SOURCE_DIR=${IDE_LLVM_SOURCE}" "-DSYSROOT_DIR=${IDE_SYSROOT_DIR}"
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

ide_read_array(example_flags "${examples}" flags)
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
include("${IDE_TESTS_DIR}/tests.cmake")
ide_add_tests(BUNDLE "${RL_TOOLS_IDE_OUTPUT_DIR}" REFERENCE "${IDE_REFERENCE}" REQUIRED)
file(GLOB test_inputs CONFIGURE_DEPENDS "${IDE_TESTS_DIR}/*.mjs" "${IDE_TESTS_DIR}/*.cmake"
    "${IDE_TESTS_DIR}/*.py" "${RL_TOOLS_REPOSITORY_DIR}/static/ide/*.js" "${RL_TOOLS_REPOSITORY_DIR}/static/ide/*.html")
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/verification-inputs.txt" CONTENT "${test_inputs}\n" @ONLY)
list(APPEND test_inputs "${CMAKE_BINARY_DIR}/verification-inputs.txt")
set(IDE_VERIFIED "${CMAKE_BINARY_DIR}/VERIFIED")
add_custom_command(OUTPUT "${IDE_VERIFIED}"
    COMMAND "${CMAKE_COMMAND}" "-DBUILD_DIR=${CMAKE_BINARY_DIR}" "-DNODE=${IDE_NODE_EXECUTABLE}"
        "-DSTAMP=${IDE_VERIFIED}" -DLABEL=ide-verify -P "${IDE_CMAKE_DIR}/verify.cmake"
    DEPENDS ${package_outputs} ${bundle_outputs} ${test_inputs} "${IDE_CMAKE_DIR}/verify.cmake"
        "${RL_TOOLS_REPOSITORY_DIR}/static/ide/package.json" "${IDE_LOCK_FILE}"
    USES_TERMINAL VERBATIM)
add_custom_target(verify DEPENDS "${IDE_VERIFIED}")
add_custom_target(verify-training
    COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${CMAKE_BINARY_DIR}" --output-on-failure --no-tests=error -L "^ide-training$" --timeout 900 -j1
    DEPENDS verify parity USES_TERMINAL VERBATIM)

set(stage_files ${package_files} ${bundle_files})
set(stage_outputs ${stage_files})
list(TRANSFORM stage_outputs PREPEND "${RL_TOOLS_IDE_STAGE_DIR}/")
string(REPLACE ";" "\n" stage_list "${stage_files}")
set(IDE_STAGE_FILES "${CMAKE_BINARY_DIR}/stage-files.txt")
file(CONFIGURE OUTPUT "${IDE_STAGE_FILES}" CONTENT "${stage_list}\n" @ONLY)
if(RL_TOOLS_IDE_STAGE_DIR STREQUAL RL_TOOLS_IDE_OUTPUT_DIR)
    message(FATAL_ERROR "Candidate and served bundle directories must be different so verification precedes staging.")
endif()
add_custom_command(OUTPUT ${stage_outputs}
    COMMAND "${CMAKE_COMMAND}" "-DSOURCE=${RL_TOOLS_IDE_OUTPUT_DIR}" "-DDESTINATION=${RL_TOOLS_IDE_STAGE_DIR}"
        "-DFILES=${IDE_STAGE_FILES}" -P "${IDE_CMAKE_DIR}/stage.cmake"
    DEPENDS "${IDE_VERIFIED}" ${package_outputs} ${bundle_outputs} "${IDE_STAGE_FILES}" "${IDE_CMAKE_DIR}/stage.cmake"
    VERBATIM)
add_custom_target(stage ALL DEPENDS ${stage_outputs})

message(STATUS "RLtools IDE toolchain configure summary:")
message(STATUS "  llvm sources  ${IDE_LLVM_SOURCE}")
message(STATUS "  libc sources  ${IDE_LIBC_SOURCE}")
message(STATUS "  cross compiler ${RL_TOOLS_IDE_HOST_LLVM_BIN} (clang ${host_version})")
message(STATUS "  build root    ${CMAKE_BINARY_DIR}")
message(STATUS "  candidate     ${RL_TOOLS_IDE_OUTPUT_DIR}")
message(STATUS "  served bundle ${RL_TOOLS_IDE_STAGE_DIR}")
message(STATUS "  generator/jobs ${CMAKE_GENERATOR}/${RL_TOOLS_IDE_JOBS}")
message(STATUS "  lto/assertions ${RL_TOOLS_IDE_LTO}/${RL_TOOLS_IDE_ASSERTIONS}")
