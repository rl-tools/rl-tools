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

find_package(Boost REQUIRED)

message(STATUS "Boost version: ${Boost_VERSION}")
add_executable(rl_environments_multirotor_ui ui.cpp)
target_link_libraries(
        rl_environments_multirotor_ui
        PRIVATE
        backprop_tools
        HighFive
        tensorboard_logger
        Boost::boost
        nlohmann_json::nlohmann_json
)
#target_compile_definitions(rl_environments_multirotor_ui PRIVATE BOOST_BEAST_USE_STD_STRING_VIEW)
