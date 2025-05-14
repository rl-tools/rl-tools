if(RL_TOOLS_ENABLE_HDF5)
    add_executable(foundation_policy_pre_training_sample_dynamics_parameters pre_training/sample_dynamics_parameters.cpp)
    target_link_libraries(foundation_policy_pre_training_sample_dynamics_parameters PRIVATE RLtools::RLtools)

    add_executable(foundation_policy_pre_training pre_training/main.cpp)
    target_link_libraries(foundation_policy_pre_training PRIVATE RLtools::RLtools)

    add_executable(foundation_policy_post_training post_training/main.cpp)
    target_link_libraries(foundation_policy_post_training PRIVATE RLtools::RLtools)

#    add_executable(foundation_policy_post_training_sequential post_training/main_sequential.cpp)
#    target_link_libraries(foundation_policy_post_training_sequential PRIVATE RLtools::RLtools)

#    add_executable(foundation_policy_load_actor post_training/load_actor.cpp)
#    target_link_libraries(foundation_policy_load_actor PRIVATE RLtools::RLtools)

#    add_executable(foundation_policy_test_checkpoint post_training/test_checkpoint.cpp)
#    target_link_libraries(foundation_policy_test_checkpoint PRIVATE RLtools::RLtools)

    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_executable(foundation_policy_check_checkpoints post_training/check_checkpoints.cpp)
        target_link_libraries(foundation_policy_check_checkpoints PRIVATE RLtools::RLtools)
        target_link_libraries(foundation_policy_check_checkpoints PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_subdirectory(analysis)
endif()
