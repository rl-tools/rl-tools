add_executable(metra cli.cpp)
target_include_directories(metra PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(metra PRIVATE cxx_std_17)
