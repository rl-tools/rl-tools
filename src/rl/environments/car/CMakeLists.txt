
if(BACKPROP_TOOLS_ENABLE_GTK)
add_executable(rl_environments_car_training car.cpp)
target_link_libraries(rl_environments_car_training
        PRIVATE
        backprop_tools
)
target_compile_definitions(rl_environments_car_training PRIVATE BACKPROP_TOOLS_DISABLE_TENSORBOARD)
if(BACKPROP_TOOLS_ENABLE_GTK)
    target_include_directories(rl_environments_car_training PRIVATE ${GTK3_INCLUDE_DIRS})
#    target_compile_definitions(rl_environments_car_training PRIVATE ${GTK3_CFLAGS_OTHER})
    target_link_directories(rl_environments_car_training PRIVATE ${GTK3_LIBRARY_DIRS})
    target_link_libraries(rl_environments_car_training
            PRIVATE
            ${GTK3_LIBRARIES}
    )
endif()
endif()



if(BACKPROP_TOOLS_ENABLE_GTK AND BACKPROP_TOOLS_ENABLE_SDL2)
add_executable(rl_environments_car_ui ui.cpp)
target_include_directories(rl_environments_car_ui PRIVATE ${GTK3_INCLUDE_DIRS})
target_compile_definitions(rl_environments_car_ui PRIVATE ${GTK3_CFLAGS_OTHER})
target_link_directories(rl_environments_car_ui PRIVATE ${GTK3_LIBRARY_DIRS})
target_include_directories(rl_environments_car_ui PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(rl_environments_car_ui
        PRIVATE
        backprop_tools
        ${GTK3_LIBRARIES}
        ${SDL2_LIBRARIES}
)
BACKPROP_TOOLS_ADD_DEFINITIONS(rl_environments_car_ui)
endif()

if(BACKPROP_TOOLS_ENABLE_GTK AND BACKPROP_TOOLS_ENABLE_SDL2)
add_executable(rl_environments_car_track car_track.cpp)
target_include_directories(rl_environments_car_track PRIVATE ${GTK3_INCLUDE_DIRS})
target_compile_definitions(rl_environments_car_track PRIVATE ${GTK3_CFLAGS_OTHER})
target_link_directories(rl_environments_car_track PRIVATE ${GTK3_LIBRARY_DIRS})
target_include_directories(rl_environments_car_track PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(rl_environments_car_track
        PRIVATE
        backprop_tools
        ${GTK3_LIBRARIES}
        ${SDL2_LIBRARIES}
)
endif()
