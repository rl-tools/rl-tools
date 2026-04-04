if(RL_TOOLS_ENABLE_HDF5)
    add_executable(dyn_check_checkpoint main.cpp)
    target_link_libraries(dyn_check_checkpoint PRIVATE RLtools::RLtools)
endif()
