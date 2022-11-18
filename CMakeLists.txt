cmake_minimum_required(VERSION 3.19)
project(bare_metal)

set(CMAKE_C_FLAGS "-O3")

set(CMAKE_CXX_STANDARD 14)
find_package(HDF5 REQUIRED COMPONENTS C HL)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
        neural_network_test
        tests/neural_network_test.cpp
)
target_include_directories(neural_network_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib)
target_link_libraries(
        neural_network_test
        GTest::gtest_main
        HDF5::HDF5
)

include(GoogleTest)
gtest_discover_tests(neural_network_test)



add_executable(bare_metal main.cpp)

