if(RL_TOOLS_ENABLE_JSON)
    if(RL_TOOLS_TEST_DATA_PATH)
        add_executable(test_rl_environments_l2f_validation validation.cpp)
        target_link_libraries(test_rl_environments_l2f_validation RLtools::RLtools GTest::gtest_main)
        gtest_discover_tests(test_rl_environments_l2f_validation)
    endif()

    add_executable(test_rl_environments_l2f_import_export import_export.cpp)
    target_link_libraries(test_rl_environments_l2f_import_export RLtools::RLtools GTest::gtest_main)
    gtest_discover_tests(test_rl_environments_l2f_import_export)

    add_executable(test_rl_environments_l2f_linear_acceleration_history linear_acceleration_history.cpp)
    target_link_libraries(test_rl_environments_l2f_linear_acceleration_history RLtools::RLtools GTest::gtest_main)
    gtest_discover_tests(test_rl_environments_l2f_linear_acceleration_history)

    add_executable(test_rl_environments_l2f_mahony mahony.cpp)
    target_link_libraries(test_rl_environments_l2f_mahony RLtools::RLtools GTest::gtest_main)
    gtest_discover_tests(test_rl_environments_l2f_mahony)

    add_executable(test_rl_environments_l2f_imu_measurement imu_measurement.cpp)
    target_link_libraries(test_rl_environments_l2f_imu_measurement RLtools::RLtools GTest::gtest_main)
    gtest_discover_tests(test_rl_environments_l2f_imu_measurement)

    add_executable(test_rl_environments_l2f_render_rotor_phase render_rotor_phase.cpp)
    target_link_libraries(test_rl_environments_l2f_render_rotor_phase RLtools::RLtools GTest::gtest_main)
    # the strip-invariance gate checks the semantic StateRender* no-read contract; fast-math
    # reassociation/contraction differences between the two state-type instantiations of step()
    # are compiler noise, not data flow, so this TU compiles with strict FP semantics
    # (source-file properties append after the RLtools interface flags, overriding -ffast-math)
    set_source_files_properties(render_rotor_phase.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    gtest_discover_tests(test_rl_environments_l2f_render_rotor_phase)
endif()

