# Stage 4 of the superbuild: assembles the served package from the stage 2 sysroot and the stage 3 module, hashes it, and
# writes the manifest (configure-time facts from STATIC_MANIFEST plus what is only known after the build).
# Arguments: -DGIT= -DTAR= -DHOST_LLVM_BIN= -DLLVM_SOURCE_DIR= -DWASI_LIBC_SOURCE_DIR= -DSYSROOT_DIR=
#            -DRESOURCE_HEADERS= -DMODULE= -DEPOCH_FILE= -DSTATIC_MANIFEST= -DPACKAGE_DIR= -DOUTPUT_DIR=
foreach(required GIT TAR HOST_LLVM_BIN LLVM_SOURCE_DIR WASI_LIBC_SOURCE_DIR SYSROOT_DIR RESOURCE_HEADERS MODULE EPOCH_FILE STATIC_MANIFEST PACKAGE_DIR OUTPUT_DIR)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "package.cmake: -D${required} is required")
    endif()
endforeach()
foreach(input MODULE EPOCH_FILE STATIC_MANIFEST)
    if(NOT EXISTS ${${input}})
        message(FATAL_ERROR "package.cmake: ${input} ${${input}} does not exist")
    endif()
endforeach()
foreach(input SYSROOT_DIR RESOURCE_HEADERS)
    if(NOT IS_DIRECTORY ${${input}})
        message(FATAL_ERROR "package.cmake: ${input} ${${input}} is not a directory")
    endif()
endforeach()

function(rl_tools_ide_run output)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${ARGN} failed:\n${stderr}")
    endif()
    set(${output} "${stdout}" PARENT_SCOPE)
endfunction()
function(rl_tools_ide_json_escape output value)
    string(REPLACE "\\" "\\\\" value "${value}")
    string(REPLACE "\"" "\\\"" value "${value}")
    set(${output} "${value}" PARENT_SCOPE)
endfunction()

file(READ ${EPOCH_FILE} epoch)
string(STRIP "${epoch}" epoch)
if(NOT epoch MATCHES "^[0-9]+$")
    message(FATAL_ERROR "package.cmake: ${EPOCH_FILE} does not hold a timestamp")
endif()

file(MAKE_DIRECTORY ${PACKAGE_DIR})
set(staging ${PACKAGE_DIR}/sysroot)
file(REMOVE_RECURSE ${staging})
file(MAKE_DIRECTORY ${staging})
file(GLOB sysroot_children RELATIVE ${SYSROOT_DIR} ${SYSROOT_DIR}/*)
foreach(child IN LISTS sysroot_children)
    file(COPY ${SYSROOT_DIR}/${child} DESTINATION ${staging})
endforeach()
file(GLOB header_children RELATIVE ${RESOURCE_HEADERS} ${RESOURCE_HEADERS}/*)
foreach(child IN LISTS header_children)
    file(COPY ${RESOURCE_HEADERS}/${child} DESTINATION ${staging}/include)
endforeach()
file(GLOB top_level RELATIVE ${staging} ${staging}/*)
list(SORT top_level)
rl_tools_ide_run(ignored ${TAR} --format=ustar --sort=name --owner=0 --group=0 --numeric-owner --mtime=@${epoch} -C ${staging} -cf ${PACKAGE_DIR}/sysroot.tar ${top_level})
file(REMOVE_RECURSE ${staging})
file(COPY_FILE ${MODULE} ${PACKAGE_DIR}/llvm.wasm)

file(SHA256 ${PACKAGE_DIR}/llvm.wasm module_sha256)
file(SIZE ${PACKAGE_DIR}/llvm.wasm module_bytes)
file(SHA256 ${PACKAGE_DIR}/sysroot.tar sysroot_sha256)
file(SIZE ${PACKAGE_DIR}/sysroot.tar sysroot_bytes)
rl_tools_ide_run(llvm_commit ${GIT} -C ${LLVM_SOURCE_DIR} rev-parse HEAD)
rl_tools_ide_run(wasi_libc_commit ${GIT} -C ${WASI_LIBC_SOURCE_DIR} rev-parse HEAD)
rl_tools_ide_run(host_compiler ${HOST_LLVM_BIN}/clang --version)
string(REGEX REPLACE "\n.*" "" host_compiler "${host_compiler}")
rl_tools_ide_run(host_linker ${HOST_LLVM_BIN}/wasm-ld --version)
string(REGEX REPLACE "\n.*" "" host_linker "${host_linker}")
rl_tools_ide_json_escape(host_compiler "${host_compiler}")
rl_tools_ide_json_escape(host_linker "${host_linker}")

file(STRINGS ${LLVM_SOURCE_DIR}/cmake/Modules/LLVMVersion.cmake version_lines REGEX "set\\(LLVM_VERSION_(MAJOR|MINOR|PATCH) ")
foreach(line IN LISTS version_lines)
    if(line MATCHES "LLVM_VERSION_([A-Z]+) ([0-9]+)")
        set(version_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
    endif()
endforeach()
file(READ ${STATIC_MANIFEST} static_part)
string(REGEX MATCH "\"llvm_version_suffix\": \"([^\"]*)\"" ignored "${static_part}")
set(llvm_version "${version_MAJOR}.${version_MINOR}.${version_PATCH}${CMAKE_MATCH_1}")
string(TIMESTAMP built "%Y-%m-%dT%H:%M:%SZ" UTC)

file(WRITE ${PACKAGE_DIR}/toolchain.json "{
${static_part}  \"llvm_commit\": \"${llvm_commit}\",
  \"llvm_version\": \"${llvm_version}\",
  \"wasi_libc_commit\": \"${wasi_libc_commit}\",
  \"host_compiler\": \"${host_compiler}\",
  \"host_compiler_path\": \"${HOST_LLVM_BIN}/clang\",
  \"host_linker\": \"${host_linker}\",
  \"source_date_epoch\": ${epoch},
  \"llvm_wasm_sha256\": \"${module_sha256}\",
  \"llvm_wasm_bytes\": ${module_bytes},
  \"sysroot_tar_sha256\": \"${sysroot_sha256}\",
  \"sysroot_tar_bytes\": ${sysroot_bytes},
  \"built\": \"${built}\"
}
")
file(WRITE ${PACKAGE_DIR}/SHA256SUMS "${module_sha256}  llvm.wasm\n${sysroot_sha256}  sysroot.tar\n")

file(MAKE_DIRECTORY ${OUTPUT_DIR})
foreach(artifact llvm.wasm sysroot.tar toolchain.json SHA256SUMS)
    file(COPY_FILE ${PACKAGE_DIR}/${artifact} ${OUTPUT_DIR}/${artifact})
endforeach()
message(STATUS "Packaged ${llvm_version} (${llvm_commit}): llvm.wasm ${module_bytes} bytes, sysroot.tar ${sysroot_bytes} bytes -> ${OUTPUT_DIR}")
