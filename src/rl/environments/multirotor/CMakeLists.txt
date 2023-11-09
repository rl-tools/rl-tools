add_subdirectory(td3)

if(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS AND BACKPROP_TOOLS_ENABLE_HDF5 AND BACKPROP_TOOLS_ENABLE_TENSORBOARD AND BACKPROP_TOOLS_ENABLE_JSON AND BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_ENABLE_UI)
    add_executable(
            rl_environments_multirotor_evaluation_td3
            evaluate_actor.cpp
    )
    target_link_libraries(
            rl_environments_multirotor_evaluation_td3
            PRIVATE
            backprop_tools
            nlohmann_json::nlohmann_json
    )
    BACKPROP_TOOLS_INSTALL(rl_environments_multirotor_evaluation_td3)
    add_executable(
            rl_environments_multirotor_assessment_td3
            assessment/assessment.cpp
    )
    target_link_libraries(
            rl_environments_multirotor_assessment_td3
            PRIVATE
            backprop_tools
            nlohmann_json::nlohmann_json
    )
    BACKPROP_TOOLS_INSTALL(rl_environments_multirotor_assessment_td3)
endif()


if(BACKPROP_TOOLS_ENABLE_JSON AND BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_ENABLE_UI)
find_package(Threads REQUIRED)
find_package(Boost REQUIRED)
add_executable(
        rl_environments_multirotor_comparison
        comparison.cpp
)
target_link_libraries(
        rl_environments_multirotor_comparison
        PRIVATE
        backprop_tools
        nlohmann_json::nlohmann_json
        Threads::Threads
        Boost::boost
)
endif()


if(BACKPROP_TOOLS_BACKEND_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()



add_subdirectory(ui)

if(BACKPROP_TOOLS_ENABLE_JSON)
add_executable(
        rl_environments_multirotor_validation
        validation.cpp
)
target_link_libraries(
        rl_environments_multirotor_validation
        PRIVATE
        backprop_tools
        nlohmann_json::nlohmann_json
        Threads::Threads
        Boost::boost
)
endif()