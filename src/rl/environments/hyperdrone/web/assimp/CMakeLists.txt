cmake_minimum_required(VERSION 3.22)
project(rl_tools_wasm_assimp LANGUAGES C CXX)

include("${RLTOOLS_ROOT}/cmake/dependencies/assimp.cmake")
if(RL_TOOLS_OFFLINE_BUILD)
    set(FETCHCONTENT_FULLY_DISCONNECTED OFF)
endif()
FetchContent_MakeAvailable(assimp)
