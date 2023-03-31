project(layer_in_c_rl_algorithms_td3_wasm)
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s WASM=1")
#set(CMAKE_EXECUTABLE_SUFFIX ".html")
#set(CMAKE_EXECUTABLE_SUFFIX_FLAG "")
#set(emscripten_system_libs ON)
add_executable(
        test_rl_algorithms_td3_wasm_full_training
#        full_training.cpp
        test.cpp
)
#target_include_directories(
#        test_rl_algorithms_td3_wasm_full_training
#        PRIVATE
#        ../../../../../../include
#)
