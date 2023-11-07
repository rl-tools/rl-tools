
if(BACKPROP_TOOLS_ENABLE_JSON AND BACKPROP_TOOLS_ENABLE_BOOST_BEAST)
find_package(Boost REQUIRED)
message(STATUS "Boost version: ${Boost_VERSION}")
add_executable(rl_environments_multirotor_ui ui.cpp)
target_link_libraries(
        rl_environments_multirotor_ui
        PRIVATE
        backprop_tools
        HighFive
        tensorboard_logger
        libattopng
        Boost::boost
        nlohmann_json::nlohmann_json
)
#target_compile_definitions(rl_environments_multirotor_ui PRIVATE BOOST_BEAST_USE_STD_STRING_VIEW)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_multirotor_ui)
BACKPROP_TOOLS_TARGET_LINK_DEPENDENCIES(rl_environments_multirotor_ui)
endif()

add_executable(rl_environments_multirotor_ui_training training.cpp)
target_link_libraries(
        rl_environments_multirotor_ui_training
        PRIVATE
        backprop_tools
        HighFive
        tensorboard_logger
        libattopng
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_multirotor_ui_training)
BACKPROP_TOOLS_TARGET_LINK_DEPENDENCIES(rl_environments_multirotor_ui_training)
add_compile_definitions(rl_environments_multirotor_ui_training BACKPROP_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD)

add_executable(rl_environments_multirotor_ui_ablation_study ablation_study.cpp)
target_link_libraries(
        rl_environments_multirotor_ui_ablation_study
        PRIVATE
        backprop_tools
        HighFive
        tensorboard_logger
        libattopng
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_multirotor_ui_ablation_study)
BACKPROP_TOOLS_TARGET_LINK_DEPENDENCIES(rl_environments_multirotor_ui_ablation_study)
add_compile_definitions(rl_environments_multirotor_ui_ablation_study BACKPROP_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD)
