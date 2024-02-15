find_package(Threads)


if(RL_TOOLS_ENABLE_JSON AND RL_TOOLS_ENABLE_BOOST_BEAST AND RL_TOOLS_ENABLE_CLI11 AND Threads_FOUND)
add_executable(ui_server server.cpp)
target_link_libraries(
        ui_server
        PRIVATE
        RLtools::CoreDefinitions
        nlohmann_json::nlohmann_json
        Boost::filesystem
        CLI11::CLI11
        Threads::Threads
)

set(THREEJS_VERSION r156)



if(NOT EXISTS "${PROJECT_SOURCE_DIR}/include/rl_tools/ui_server/static/multirotor/lib/three.module.js")
file(DOWNLOAD "https://github.com/mrdoob/three.js/raw/${THREEJS_VERSION}/build/three.module.js"
    "${PROJECT_SOURCE_DIR}/include/rl_tools/ui_server/static/multirotor/lib/three.module.js"
)
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/include/rl_tools/ui_server/static/multirotor/lib/OrbitControls.js")
file(DOWNLOAD "https://github.com/mrdoob/three.js/raw/${THREEJS_VERSION}/examples/jsm/controls/OrbitControls.js"
        "${PROJECT_SOURCE_DIR}/include/rl_tools/ui_server/static/multirotor/lib/OrbitControls.js"
)
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/rl_tools/ui_server/static/
        DESTINATION share/rl_tools/ui_server/static
        FILES_MATCHING PATTERN "*.*"
)
RL_TOOLS_INSTALL(ui_server)
endif()
