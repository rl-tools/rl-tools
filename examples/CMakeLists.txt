#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

add_executable(example_td3_pendulum
        td3_pendulum/main.cpp
)
target_include_directories(example_td3_pendulum PRIVATE ${PROJECT_SOURCE_DIR}/include)