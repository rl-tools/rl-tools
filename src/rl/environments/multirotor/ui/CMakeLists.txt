
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
        Boost::boost
        nlohmann_json::nlohmann_json
)
#target_compile_definitions(rl_environments_multirotor_ui PRIVATE BOOST_BEAST_USE_STD_STRING_VIEW)
endif()
