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
                GIT_TAG        v3.11.3
                GIT_SHALLOW    TRUE
        )
        target_include_directories(rl_tools_full INTERFACE ${nlohmann_json_SOURCE_DIR}/include)
    endif()
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
                GIT_TAG   hdf5_1.14.6
                GIT_SHALLOW    TRUE
        )
        set(HDF5_ENABLE_Z_LIB_SUPPORT OFF CACHE BOOL "Disable Zlib for HDF5" FORCE)
        set(HDF5_ENABLE_Z_LIB_SUPPORT   OFF CACHE BOOL "Disable Zlib support" FORCE)
        set(BUILD_SHARED_LIBS           OFF CACHE BOOL "Build only static HDF5 libraries" FORCE)
        set(HDF5_BUILD_TOOLS            OFF CACHE BOOL "Do not build HDF5 command-line tools" FORCE)
        set(HDF5_BUILD_EXAMPLES         OFF CACHE BOOL "Do not build HDF5 examples" FORCE)
        set(HDF5_BUILD_TESTING          OFF CACHE BOOL "Do not build HDF5 tests" FORCE)
        set(HDF5_BUILD_CPP_LIB          OFF CACHE BOOL "Do not build the HDF5 C++ library" FORCE)
        set(HDF5_BUILD_FORTRAN_LIB      OFF CACHE BOOL "Do not build the HDF5 Fortran library" FORCE)
        set(HDF5_ENABLE_MIRROR_VFD      OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(hdf5)
        target_link_libraries(rl_tools_full INTERFACE hdf5-static)
    endif()
    message(STATUS "Using FetchContent to get HighFive")
    FetchContent_Declare(highfive
            GIT_REPOSITORY https://github.com/rl-tools/highfive.git
            GIT_TAG   be68bd0efcef338a016fba448d6444089fd196d5
    )
    FetchContent_MakeAvailable(highfive)
    target_include_directories(rl_tools_full INTERFACE ${highfive_SOURCE_DIR}/include)
    target_compile_definitions(rl_tools_full INTERFACE RL_TOOLS_ENABLE_HDF5)
endif()
