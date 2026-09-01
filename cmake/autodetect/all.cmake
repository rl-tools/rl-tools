# tiers of optional dependencies
# tier 0: compiler capabilities (bf16)
# tier 1: backend (MKL, OpenBLAS, Accelerate)
# tier 2: json, hdf5
# tier 3: tensorboard
# tier 4: cuda
# tier 5: cli11

# The RL_TOOLS_*_ENABLE_* variables are detection outputs (plain variables + compile
# definitions on rl_tools_full), not inputs: forcing one from the command line used to skip
# detection without adding the link libraries. Scrub and warn; the inputs are the
# RL_TOOLS_*_DISABLE_* options.
foreach(RL_TOOLS_DETECTION_OUTPUT
        RL_TOOLS_BACKEND_ENABLE_MKL RL_TOOLS_BACKEND_ENABLE_BLAS RL_TOOLS_BACKEND_ENABLE_DNNL
        RL_TOOLS_BACKEND_ENABLE_OPENBLAS RL_TOOLS_BACKEND_ENABLE_ACCELERATE
        RL_TOOLS_BACKEND_ENABLE_CUDA RL_TOOLS_BACKEND_ENABLE_CUDNN
        RL_TOOLS_ENABLE_JSON RL_TOOLS_ENABLE_HDF5 RL_TOOLS_ENABLE_ZLIB
        RL_TOOLS_ENABLE_TENSORBOARD RL_TOOLS_ENABLE_CLI11 RL_TOOLS_ENABLE_GTEST)
    if(DEFINED CACHE{${RL_TOOLS_DETECTION_OUTPUT}})
        message(WARNING "${RL_TOOLS_DETECTION_OUTPUT} is a detection output, not an input - ignoring it (use the RL_TOOLS_*_DISABLE_* options to opt out)")
        unset(${RL_TOOLS_DETECTION_OUTPUT} CACHE)
    endif()
endforeach()

find_package(Git QUIET)

include(cmake/autodetect/tier0-compiler.cmake)
include(cmake/autodetect/tier1-blas.cmake)
include(cmake/autodetect/tier2-json-hdf5-zlib.cmake)
include(cmake/autodetect/tier3-tensorboard.cmake)
include(cmake/autodetect/tier4-cuda.cmake)
include(cmake/autodetect/tier5-cli11.cmake)
include(cmake/autodetect/git-hash.cmake)
include(cmake/autodetect/git-diff.cmake)
