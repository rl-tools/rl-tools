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
            HighFive
            glfw
            CLI11::CLI11
            nlohmann_json::nlohmann_json
    )
    BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_multirotor_evaluation_td3)
    BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_multirotor_evaluation_td3)
    BACKPROP_TOOLS_INSTALL(rl_environments_multirotor_evaluation_td3)
endif()
if(BACKPROP_TOOLS_BACKEND_ENABLE_BLAS AND BACKPROP_TOOLS_ENABLE_HDF5 AND BACKPROP_TOOLS_ENABLE_TENSORBOARD AND BACKPROP_TOOLS_ENABLE_JSON AND BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_ENABLE_UI)
    add_executable(
            rl_environments_multirotor_assessment_td3
            assessment/assessment.cpp
    )
    target_link_libraries(
            rl_environments_multirotor_assessment_td3
            PRIVATE
            backprop_tools
            HighFive
            glfw
            CLI11::CLI11
            nlohmann_json::nlohmann_json
    )
    BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_multirotor_assessment_td3)
    BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_multirotor_assessment_td3)
    BACKPROP_TOOLS_INSTALL(rl_environments_multirotor_assessment_td3)
endif()


if(BACKPROP_TOOLS_ENABLE_JSON AND BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_ENABLE_UI)
find_package(Threads)
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
)
endif()

