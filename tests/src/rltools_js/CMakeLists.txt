if(RL_TOOLS_ENABLE_HDF5)
    add_executable(test_rltools_js_fixtures fixture_generator.cpp)
    target_link_libraries(test_rltools_js_fixtures PRIVATE RLtools::RLtools)
endif()
