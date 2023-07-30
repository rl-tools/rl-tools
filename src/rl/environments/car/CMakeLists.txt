add_executable(rl_environments_car_training car.cpp)
target_link_libraries(rl_environments_car_training
        PRIVATE
        backprop_tools
)
BACKPROP_TOOLS_TARGET_LINK_BLAS(rl_environments_car_training)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
add_executable(rl_environments_car_ui ui.cpp)
target_include_directories(rl_environments_car_ui PRIVATE ${GTK3_INCLUDE_DIRS})
target_compile_definitions(rl_environments_car_ui PRIVATE ${GTK3_CFLAGS_OTHER})
target_link_directories(rl_environments_car_ui PRIVATE ${GTK3_LIBRARY_DIRS})
find_package(SDL2 REQUIRED)
target_include_directories(rl_environments_car_ui PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(rl_environments_car_ui
        PRIVATE
        backprop_tools
        ${GTK3_LIBRARIES}
        ${SDL2_LIBRARIES}
)
