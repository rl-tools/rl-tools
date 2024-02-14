option(RL_TOOLS_DOWNLOAD_GTEST OFF)

option(RL_TOOLS_TEST_LOCAL "Local dev" OFF)

#option(RL_TOOLS_TEST_RL_ENVIRONMENTS_PENDULUM_EVALUATE_VISUALLY "Visual Evaluation" ${RL_TOOLS_TEST_LOCAL})
option(RL_TOOLS_TEST_RL_ENVIRONMENTS_PENDULUM_EVALUATE_VISUALLY "Visual Evaluation" OFF)

option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_OUTPUT_PLOTS "Visual Evaluation" OFF)
#option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY "Visual Evaluation" ${RL_TOOLS_TEST_LOCAL})
option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_SECOND_STAGE_EVALUATE_VISUALLY "Visual Evaluation" OFF)

#option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS "Visual Evaluation" ON)
option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS "Visual Evaluation" OFF)
#option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY "Visual Evaluation" ${RL_TOOLS_TEST_LOCAL})
option(RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY "Visual Evaluation" OFF)

option(RL_TOOLS_TESTS_RL_ENVIRONMENTS_MULTIROTOR_UI_ENABLE "Multirotor UI" OFF)

set(INSTALL_GTEST OFF CACHE BOOL "")
set(BUILD_GMOCK OFF CACHE BOOL "")
add_subdirectory(lib/googletest EXCLUDE_FROM_ALL)
include(GoogleTest)

set(RL_TOOLS_TESTS_DATA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/data)
add_library(rl_tools_test INTERFACE)
target_link_libraries(rl_tools_test INTERFACE GTest::gtest_main)
target_compile_definitions(rl_tools_test INTERFACE RL_TOOLS_TESTS_DATA_PATH=${RL_TOOLS_TESTS_DATA_PATH})
add_library(RLtools::Test ALIAS rl_tools_test)


if(NOT APPLE)
#    add_subdirectory(lib/matplotlib EXCLUDE_FROM_ALL)
endif()


# function to do target_compile_definitions(test_neural_network_mlp PRIVATE RL_TOOLS_TESTS_DATA_PATH=${CMAKE_CURRENT_SOURCE_DIR}/data)






if(RL_TOOLS_TESTS_ENABLE_EIGEN)
find_package(Eigen3 REQUIRED)
endif()
#find_package(OpenMP REQUIRED)
if(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI)
find_package(Boost REQUIRED)
endif()


if(RL_TOOLS_BACKEND_ENABLE_CUDA)
#    set(CUTLASS_ENABLE_CUBLAS ON)
#    add_subdirectory(lib/cutlass EXCLUDE_FROM_ALL)
#    add_subdirectory(src/nn/cuda)
#    add_subdirectory(src/rl/cuda)
endif()

add_subdirectory(src)





## RL Algorithm td3 test training stats
#find_package(TBB REQUIRED)
#add_executable(
#        test_rl_algorithms_td3_pendulum_training_stats
#        src/rl/algorithms/td3/pendulum_training_stats.cpp
#)
#target_link_libraries(
#        test_rl_algorithms_td3_pendulum_training_stats
#        rl_tools
#        GTest::gtest_main
#        HighFive
#        matplotlib_cpp
#        TBB::tbb
#)
#target_include_directories(test_rl_algorithms_td3_pendulum_training_stats PRIVATE lib)
#gtest_discover_tests(test_rl_algorithms_td3_pendulum_training_stats)








## Container tests
#add_executable(
#        test_container_persist_code_load
#        src/container/persist_code_load.cpp
#)
#target_link_libraries(
#        test_container_persist_code_load
#        rl_tools
#        GTest::gtest_main
#)
#gtest_discover_tests(test_container_persist_code_load)





