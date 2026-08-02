add_library(rendering_raytracing_backend INTERFACE)
target_link_libraries(rendering_raytracing_backend INTERFACE assimp::assimp)

add_library(rendering_raytracing_backend_depth INTERFACE)
target_link_libraries(rendering_raytracing_backend_depth INTERFACE assimp::assimp)

# Rule-13 guard: the freestanding kernels header must compile without any C++ standard library
# headers. Compile-only; never executed.
add_library(rendering_raytracing_generic_freestanding_check OBJECT freestanding_check.cpp)
target_include_directories(rendering_raytracing_generic_freestanding_check PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(rendering_raytracing_generic_freestanding_check PRIVATE -nostdinc++ -fno-exceptions -fno-rtti)
