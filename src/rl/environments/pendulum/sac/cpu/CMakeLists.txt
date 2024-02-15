
find_package(Threads)

if(Threads_FOUND)
add_executable(
        rl_environments_pendulum_sac_standalone
        training.cpp
)
target_link_libraries(
        rl_environments_pendulum_sac_standalone
        PRIVATE
        rl_tools
        Threads::Threads
)
target_compile_definitions(rl_environments_pendulum_sac_standalone PRIVATE RL_TOOLS_BACKEND_DISABLE_BLAS)
target_compile_definitions(rl_environments_pendulum_sac_standalone PRIVATE RL_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_pendulum_sac_standalone PRIVATE RL_TOOLS_DISABLE_TENSORBOARD)
RL_TOOLS_INSTALL(rl_environments_pendulum_sac_standalone)

# RL Algorithm sac test full training blas
if(RL_TOOLS_BACKEND_ENABLE_BLAS)
add_executable(
        rl_environments_pendulum_sac_blas
        training.cpp
)
target_link_libraries(
        rl_environments_pendulum_sac_blas
        PRIVATE
        rl_tools
)
target_compile_definitions(rl_environments_pendulum_sac_blas PRIVATE RL_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_pendulum_sac_blas PRIVATE RL_TOOLS_DISABLE_TENSORBOARD)
#target_compile_options(rl_environments_pendulum_sac_blas PRIVATE -Ofast -march=native -mtune=native)
RL_TOOLS_INSTALL(rl_environments_pendulum_sac_blas)
endif()

if(RL_TOOLS_BACKEND_ENABLE_BLAS)
    add_executable(
            rl_environments_pendulum_sac_benchmark
            training.cpp
    )
    target_link_libraries(
            rl_environments_pendulum_sac_benchmark
            PRIVATE
            RLtools::RLtools
            Threads::Threads
    )
    target_compile_definitions(rl_environments_pendulum_sac_benchmark PRIVATE RL_TOOLS_DISABLE_HDF5)
    target_compile_definitions(rl_environments_pendulum_sac_benchmark PRIVATE RL_TOOLS_DISABLE_TENSORBOARD)
    target_compile_definitions(rl_environments_pendulum_sac_benchmark PRIVATE BENCHMARK)
    target_compile_options(rl_environments_pendulum_sac_benchmark PRIVATE -Ofast)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_options(rl_environments_pendulum_sac_benchmark PRIVATE -march=native)
    endif()
    RL_TOOLS_INSTALL(rl_environments_pendulum_sac_benchmark)
endif()
endif()
