if(RL_TOOLS_ENABLE_HDF5)
    add_executable(foundation_policy_pre_training pre_training/main.cpp)
    target_link_libraries(foundation_policy_pre_training PRIVATE RLtools::RLtools)
endif()