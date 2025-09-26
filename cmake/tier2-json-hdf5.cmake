if(NOT RL_TOOLS_DISABLE_JSON)
    find_package(nlohmann_json QUIET)
    if(nlohmann_json_FOUND)
        get_target_property(nlohmann_json_INCLUDE_DIRS nlohmann_json::nlohmann_json INCLUDE_DIRECTORIES)
        message(STATUS "Found existing/system nlohmann_json ${nlohmann_json_VERSION} at ${nlohmann_json_DIR}")
    else()
        message(STATUS "nlohmann_json is required but not found. Using FetchContent.")
        FetchContent_Declare(nlohmann_json
                GIT_REPOSITORY https://github.com/nlohmann/json.git
                GIT_TAG        v3.11.3
                GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(nlohmann_json)
    endif()
    target_link_libraries(rl_tools_full INTERFACE nlohmann_json::nlohmann_json)
    target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_ENABLE_JSON)
endif()

if(NOT RL_TOOLS_DISABLE_HDF5)
    find_package(HDF5 QUIET)
    if(HDF5_FOUND)
        message(STATUS "Found existing/system HDF5 ${HDF5_VERSION} at ${HDF5_INCLUDE_DIRS}")
        target_link_libraries(rl_tools_full INTERFACE HDF5::HDF5)
    else()
        FetchContent_Declare(hdf5
                GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
                GIT_TAG   7bf340440909d468dbb3cf41f0ea0d87f5050cea
                GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(hdf5)
        target_link_libraries(rl_tools_full INTERFACE hdf5-static)
    endif()
    message(STATUS "Using FetchContent to get HighFive")
    FetchContent_Declare(highfive
            GIT_REPOSITORY https://github.com/rl-tools/highfive.git
            GIT_TAG   be68bd0efcef338a016fba448d6444089fd196d5
            GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(highfive)
    target_include_directories(rl_tools_full INTERFACE ${highfive_SOURCE_DIR}/include)
    target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_ENABLE_HDF5)
endif()
