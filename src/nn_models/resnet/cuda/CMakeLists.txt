project(nn_models_resnet_inference_cuda CUDA)

if(RL_TOOLS_BACKEND_ENABLE_CUDA AND RL_TOOLS_BACKEND_ENABLE_CUDNN AND RL_TOOLS_ENABLE_HDF5)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CXX_STANDARD  17)

    add_executable(nn_models_resnet_inference_cuda ../resnet_inference_cuda.cu)
    target_link_libraries(nn_models_resnet_inference_cuda PRIVATE rl_tools_full)
    target_include_directories(nn_models_resnet_inference_cuda PRIVATE ${stb_SOURCE_DIR})
endif()
