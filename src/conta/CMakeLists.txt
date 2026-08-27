add_executable(conta cli.cpp)
target_include_directories(conta PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(conta PRIVATE cxx_std_17)
