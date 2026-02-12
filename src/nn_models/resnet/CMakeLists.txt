if(RL_TOOLS_ENABLE_HDF5)
    include(FetchContent)
    FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(stb)

    add_executable(nn_models_resnet_inference resnet_inference.cpp)
    target_link_libraries(nn_models_resnet_inference PRIVATE RLtools::RLtools)
    target_include_directories(nn_models_resnet_inference PRIVATE ${stb_SOURCE_DIR})
endif()
