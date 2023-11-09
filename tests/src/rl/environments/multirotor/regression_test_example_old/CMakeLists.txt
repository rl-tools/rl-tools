cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0097 NEW) # ExternalProject should not pull submodules by default

include(FetchContent)
FetchContent_Declare(test_rl_environments_multirotor_regression_test_code
        GIT_REPOSITORY file://${PROJECT_SOURCE_DIR}
#        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/regression_test
        GIT_TAG 9f6f106e54db69931320938720b7fcbcdea9d241
        GIT_SUBMODULES ""
        GIT_SHALLOW FALSE
)
FetchContent_Declare(test_rl_environments_multirotor_regression_test_code_new
        GIT_REPOSITORY file://${PROJECT_SOURCE_DIR}
        #        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/regression_test
        GIT_TAG 6635568fda3d54ccbb41f8799f04fa213d2d9967
        GIT_SUBMODULES ""
        GIT_SHALLOW FALSE
)

if(NOT test_rl_environments_multirotor_regression_test_code_POPULATED)
    FetchContent_Populate(test_rl_environments_multirotor_regression_test_code)
endif()
if(NOT test_rl_environments_multirotor_regression_test_code_new_POPULATED AND NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/include/backprop_tools_new)
    FetchContent_Populate(test_rl_environments_multirotor_regression_test_code_new) # we have to do it this way around because in the old version the internal imports are absolute <backprop_tools/...> etc. hence can not be renamed
    file(COPY ${test_rl_environments_multirotor_regression_test_code_new_SOURCE_DIR}/include/backprop_tools DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include)
    file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/include/backprop_tools ${CMAKE_CURRENT_BINARY_DIR}/include/backprop_tools_new)
endif()


add_executable(
        test_rl_environments_multirotor_regression_test
        comparison.cpp
)
target_link_libraries(
        test_rl_environments_multirotor_regression_test
#        backprop_tools
        GTest::gtest_main
)
target_include_directories(
        test_rl_environments_multirotor_regression_test
        PRIVATE
        ${test_rl_environments_multirotor_regression_test_code_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
)
#add_dependencies(
#        test_rl_environments_multirotor_regression_test
#        test_rl_environments_multirotor_regression_test_code)
#target_include_directories(
#        test_rl_environments_multirotor_regression_test
#        PUBLIC
#        ${CMAKE_SOURCE_DIR}/back
BACKPROP_TOOLS_ADD_DEFINITIONS(test_rl_environments_multirotor_regression_test)
