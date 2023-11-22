# RL Algorithm td3 test full training standalone
add_executable(
        rl_environments_pendulum_td3_training_bare
        standalone.cpp
)
target_link_libraries(
        rl_environments_pendulum_td3_training_bare
        PRIVATE
        rl_tools
)
target_compile_definitions(rl_environments_pendulum_td3_training_bare PRIVATE RL_TOOLS_TEST_BARE)
target_compile_definitions(rl_environments_pendulum_td3_training_bare PRIVATE RL_TOOLS_BACKEND_DISABLE_BLAS)
target_compile_definitions(rl_environments_pendulum_td3_training_bare PRIVATE RL_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_pendulum_td3_training_bare PRIVATE RL_TOOLS_DISABLE_TENSORBOARD)
target_compile_options(rl_environments_pendulum_td3_training_bare PRIVATE -nostdlib)
RL_TOOLS_INSTALL(rl_environments_pendulum_td3_training_bare)

# RL Algorithm td3 test full training standalone
add_executable(
        rl_environments_pendulum_td3_training_standalone
        standalone.cpp
)
target_link_libraries(
        rl_environments_pendulum_td3_training_standalone
        PRIVATE
        rl_tools
)
target_compile_definitions(rl_environments_pendulum_td3_training_standalone PRIVATE RL_TOOLS_BACKEND_DISABLE_BLAS)
target_compile_definitions(rl_environments_pendulum_td3_training_standalone PRIVATE RL_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_pendulum_td3_training_standalone PRIVATE RL_TOOLS_DISABLE_TENSORBOARD)
RL_TOOLS_INSTALL(rl_environments_pendulum_td3_training_standalone)

# RL Algorithm td3 test full training blas
if(RL_TOOLS_BACKEND_ENABLE_BLAS)
add_executable(
        rl_environments_pendulum_td3_training_blas
        standalone.cpp
)
target_link_libraries(
        rl_environments_pendulum_td3_training_blas
        PRIVATE
        rl_tools
)
target_compile_definitions(rl_environments_pendulum_td3_training_blas PRIVATE RL_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_pendulum_td3_training_blas PRIVATE RL_TOOLS_DISABLE_TENSORBOARD)
RL_TOOLS_INSTALL(rl_environments_pendulum_td3_training_blas)
endif()

# RL Algorithm td3 test full training benchmark (disabling evaluation)
if(RL_TOOLS_BACKEND_ENABLE_BLAS)
add_executable(
        rl_environments_pendulum_td3_training_blas_benchmark
        standalone.cpp
)
target_link_libraries(
        rl_environments_pendulum_td3_training_blas_benchmark
        PRIVATE
        rl_tools
)
target_compile_definitions(rl_environments_pendulum_td3_training_blas_benchmark PRIVATE RL_TOOLS_DISABLE_HDF5)
target_compile_definitions(rl_environments_pendulum_td3_training_blas_benchmark PRIVATE RL_TOOLS_DISABLE_TENSORBOARD)
target_compile_definitions(rl_environments_pendulum_td3_training_blas_benchmark PRIVATE RL_TOOLS_RL_ENVIRONMENTS_PENDULUM_DISABLE_EVALUATION)
target_compile_options(rl_environments_pendulum_td3_training_blas_benchmark PRIVATE --fast-math)
RL_TOOLS_INSTALL(rl_environments_pendulum_td3_training_blas_benchmark)
endif()

# RL Algorithm td3 test full training tensorboard
if(RL_TOOLS_ENABLE_TENSORBOARD AND RL_TOOLS_BACKEND_ENABLE_BLAS)
add_executable(
        rl_environments_pendulum_td3_training_blas_tensorboard
        standalone.cpp
)
target_link_libraries(
        rl_environments_pendulum_td3_training_blas_tensorboard
        PRIVATE
        rl_tools
)
RL_TOOLS_INSTALL(rl_environments_pendulum_td3_training_blas_tensorboard)
endif()
