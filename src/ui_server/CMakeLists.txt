if(RL_TOOLS_ENABLE_JSON AND RL_TOOLS_ENABLE_BOOST_BEAST)
add_executable(ui_server server.cpp)
target_link_libraries(
        ui_server
        PRIVATE
        rl_tools_minimal
        nlohmann_json::nlohmann_json
        Boost::boost
)
endif()
