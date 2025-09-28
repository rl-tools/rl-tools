if(NOT RL_TOOLS_DISABLE_JSON)
    find_package(nlohmann_json QUIET)
    if(nlohmann_json_FOUND)
        get_target_property(nlohmann_json_INCLUDE_DIRS nlohmann_json::nlohmann_json INCLUDE_DIRECTORIES)
        message(STATUS "Found existing/system nlohmann_json ${nlohmann_json_VERSION} at ${nlohmann_json_DIR}")
        target_link_libraries(rl_tools_full INTERFACE nlohmann_json::nlohmann_json)
    else()
        message(STATUS "nlohmann_json is required but not found. Using FetchContent.")
        FetchContent_Declare(nlohmann_json
                GIT_REPOSITORY https://github.com/nlohmann/json.git
                GIT_TAG        a0c1318830519eac027a31edec1a99ce1ae5670e
        )
        FetchContent_MakeAvailable(nlohmann_json)
        target_include_directories(rl_tools_full INTERFACE ${nlohmann_json_SOURCE_DIR}/include)
    endif()
    target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_ENABLE_JSON)
endif()

if(NOT RL_TOOLS_DISABLE_HDF5)
    find_package(HDF5 QUIET)
    if(HDF5_FOUND)
        message(STATUS "Found existing/system HDF5 ${HDF5_VERSION} at ${HDF5_INCLUDE_DIRS}")
        target_link_libraries(rl_tools_full INTERFACE HDF5::HDF5)
        message(STATUS "Using FetchContent to get HighFive")
        FetchContent_Declare(highfive
                GIT_REPOSITORY https://github.com/rl-tools/highfive.git
                GIT_TAG   be68bd0efcef338a016fba448d6444089fd196d5
        )
        set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "Disable Boost usage in HighFive" FORCE)
        set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "Disable HighFive examples" FORCE)
        set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "Disable HighFive tests" FORCE)
        FetchContent_MakeAvailable(highfive)
        target_include_directories(rl_tools_full INTERFACE ${highfive_SOURCE_DIR}/include)
        target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_ENABLE_HDF5)
    else()
        message(WARNING "HDF5 not found. HDF5 is disabled. Suppress this warning by setting RL_TOOLS_DISABLE_HDF5=ON")
    endif()
endif()


if(NOT RL_TOOLS_DISABLE_ZLIB)
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        message(STATUS "Found existing/system ZLIB ${ZLIB_VERSION} at ${ZLIB_INCLUDE_DIRS}")
        target_link_libraries(rl_tools_full INTERFACE ZLIB::ZLIB)
        target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_ENABLE_ZLIB)
    else()
        set(ZLIB_BUILD_EXAMPLES OFF)
        set(ZLIB_ENABLE_TESTS OFF)
        set(ZLIB_COMPAT ON)
        FetchContent_Declarezl(zlib
                GIT_REPOSITORY https://github.com/zlib-ng/zlib-ng.git
                GIT_TAG   8d10c309d7d468b9b2ecfd8d5f1b063d6396af17
        )
        target_link_libraries(rl_tools_full INTERFACE zlibstatic)
        target_compile_definitions(rl_tools_zlib INTERFACE RL_TOOLS_ENABLE_ZLIB)
    endif()
endif()

