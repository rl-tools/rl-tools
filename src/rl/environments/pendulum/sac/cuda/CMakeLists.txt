project(rl_environments_pendulum_cuda_sac CUDA)

if(RL_TOOLS_BACKEND_ENABLE_CUDA AND RL_TOOLS_BACKEND_ENABLE_MKL)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CXX_STANDARD  17)
    add_executable(
            rl_environments_pendulum_sac_cuda
            sac.cu
    )
    target_link_libraries(
            rl_environments_pendulum_sac_cuda
            PRIVATE
            RLtools::Minimal
    )
    #target_compile_options(rl_environments_pendulum_sac_cuda PRIVATE -gG)
endif()
