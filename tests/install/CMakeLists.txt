project(RLtoolsUserTest)

find_package(RLtools REQUIRED)

add_executable(
        user_test_pendulum
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/rl/environments/pendulum/td3/cpu/standalone.cpp # this could be any source file which will have full access to rl_tools
)
target_link_libraries(
        user_test_pendulum
        PRIVATE
        RLtools::Core
)
