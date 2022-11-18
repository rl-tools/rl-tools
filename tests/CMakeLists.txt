include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Neural Network Test
add_executable(
        neural_network_test
        nn/test.cpp
)
target_include_directories(neural_network_test PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/lib
${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(
        neural_network_test
        GTest::gtest_main
        HDF5::HDF5
)

# RL Environment Pendulum test
add_executable(
        rl_environments_pendulum_test
        rl/environments/pendulum.cpp
)
target_include_directories(rl_environments_pendulum_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
        ${PROJECT_SOURCE_DIR}/include
        )
target_link_libraries(
        rl_environments_pendulum_test
        GTest::gtest_main
        HDF5::HDF5
)

include(GoogleTest)
gtest_discover_tests(neural_network_test)
gtest_discover_tests(rl_environments_pendulum_test)



