include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

## Neural Network Test
#add_executable(
#        neural_network_test
#        nn/test.cpp
#)
#message(${PROJECT_SOURCE_DIR}/include)
#target_include_directories(neural_network_test PRIVATE
#${CMAKE_CURRENT_SOURCE_DIR}/lib
#${PROJECT_SOURCE_DIR}/include
#)
#target_link_libraries(
#        neural_network_test
#        GTest::gtest_main
#        HDF5::HDF5
#)

# Neural Network Persistence Test
add_executable(
        neural_network_persistence_test
        nn/persisting.cpp
)
message(${PROJECT_SOURCE_DIR}/include)
target_include_directories(neural_network_persistence_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
        ${PROJECT_SOURCE_DIR}/include
        )
target_link_libraries(
        neural_network_persistence_test
        GTest::gtest_main
        HDF5::HDF5
)

## RL Environment Pendulum test
#add_executable(
#        rl_environments_pendulum_test
#        rl/environments/pendulum.cpp
#)
#target_include_directories(rl_environments_pendulum_test PRIVATE
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib
#        ${PROJECT_SOURCE_DIR}/include
#        )
#target_link_libraries(
#        rl_environments_pendulum_test
#        GTest::gtest_main
#        HDF5::HDF5
#)

## RL Algorithm off_policy_runner test
#add_executable(
#        rl_algorithms_off_policy_runner_test
#        rl/algorithms/off_policy_runner.cpp
#)
#target_include_directories(rl_algorithms_off_policy_runner_test PRIVATE
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib
#        ${PROJECT_SOURCE_DIR}/include
#        )
#target_link_libraries(
#        rl_algorithms_off_policy_runner_test
#        GTest::gtest_main
#        HDF5::HDF5
#)

## RL Algorithm td3 test
#add_executable(
#        rl_algorithms_td3_test
#        rl/algorithms/td3.cpp
#)
#target_include_directories(rl_algorithms_td3_test PRIVATE
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib
#        ${PROJECT_SOURCE_DIR}/include
#        )
#target_link_libraries(
#        rl_algorithms_td3_test
#        GTest::gtest_main
#        HDF5::HDF5
#)

include(GoogleTest)
gtest_discover_tests(neural_network_persistence_test)
#gtest_discover_tests(neural_network_test)
#gtest_discover_tests(rl_environments_pendulum_test)
#gtest_discover_tests(rl_algorithms_off_policy_runner_test)
#gtest_discover_tests(rl_algorithms_td3_test)



