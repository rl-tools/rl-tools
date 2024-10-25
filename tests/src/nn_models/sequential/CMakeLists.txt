
if(RL_TOOLS_ENABLE_HDF5)
add_executable(test_nn_models_sequential_persist persist.cpp)
target_link_libraries(test_nn_models_sequential_persist
    PRIVATE
    rl_tools
    rl_tools_test
)
gtest_discover_tests(test_nn_models_sequential_persist)

endif()

add_executable(test_nn_models_sequential_persist_code persist_code.cpp)
target_link_libraries(test_nn_models_sequential_persist_code
        PRIVATE
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_nn_models_sequential_persist_code)

if(EXISTS ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code.h)
add_executable(test_nn_models_sequential_persist_code_compile persist_code_compile.cpp)
target_link_libraries(test_nn_models_sequential_persist_code_compile
        PRIVATE
        rl_tools
        rl_tools_test
        )
gtest_discover_tests(test_nn_models_sequential_persist_code_compile)
else()
message(WARNING "test_nn_models_sequential_persist_code_compile is disabled because the file ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code.h")
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_2.h)
        add_executable(test_nn_models_sequential_persist_code_compile_model_2 persist_code_compile_model_2.cpp)
        target_link_libraries(test_nn_models_sequential_persist_code_compile_model_2
                PRIVATE
                rl_tools
                rl_tools_test
        )
        gtest_discover_tests(test_nn_models_sequential_persist_code_compile_model_2)
else()
        message(WARNING "test_nn_models_sequential_persist_code_compile_model_2 is disabled because the file ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_2.h")
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_2_forward.h)
        add_executable(test_nn_models_sequential_persist_code_compile_model_2_forward persist_code_compile_model_2_forward.cpp)
        target_link_libraries(test_nn_models_sequential_persist_code_compile_model_2_forward
                PRIVATE
                rl_tools
                rl_tools_test
        )
        gtest_discover_tests(test_nn_models_sequential_persist_code_compile_model_2_forward)
else()
        message(WARNING "test_nn_models_sequential_persist_code_compile_model_2_forward is disabled because the file ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_2_forward.h")
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_2_backward.h)
        add_executable(test_nn_models_sequential_persist_code_compile_model_2_backward persist_code_compile_model_2_backward.cpp)
        target_link_libraries(test_nn_models_sequential_persist_code_compile_model_2_backward
                PRIVATE
                rl_tools
                rl_tools_test
        )
        gtest_discover_tests(test_nn_models_sequential_persist_code_compile_model_2_backward)
else()
        message(WARNING "test_nn_models_sequential_persist_code_compile_model_2_backward is disabled because the file ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_2_backward.h")
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_mlp.h)
        add_executable(test_nn_models_sequential_persist_code_compile_model_mlp persist_code_compile_model_mlp.cpp)
        target_link_libraries(test_nn_models_sequential_persist_code_compile_model_mlp
                PRIVATE
                rl_tools
                rl_tools_test
        )
        gtest_discover_tests(test_nn_models_sequential_persist_code_compile_model_mlp)
else()
        message(WARNING "test_nn_models_sequential_persist_code_compile_model_mlp is disabled because the file ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_mlp.h")
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_mlp_forward.h)
        add_executable(test_nn_models_sequential_persist_code_compile_model_mlp_forward persist_code_compile_model_mlp_forward.cpp)
        target_link_libraries(test_nn_models_sequential_persist_code_compile_model_mlp_forward
                PRIVATE
                rl_tools
                rl_tools_test
        )
        gtest_discover_tests(test_nn_models_sequential_persist_code_compile_model_mlp_forward)
else()
        message(WARNING "test_nn_models_sequential_persist_code_compile_model_mlp_forward is disabled because the file ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_mlp_forward.h")
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_sample_and_squash_forward.h)
        add_executable(test_nn_models_sequential_persist_code_compile_model_sas_forward persist_code_compile_model_sample_and_squash_forward.cpp)
        target_link_libraries(test_nn_models_sequential_persist_code_compile_model_sas_forward
                PRIVATE
                rl_tools
                rl_tools_test
        )
        gtest_discover_tests(test_nn_models_sequential_persist_code_compile_model_sas_forward)
else()
        message(WARNING "test_nn_models_sequential_persist_code_compile_model_sas_forward is disabled because the file ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_sas_forward.h")
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_sample_and_squash_backward.h)
        add_executable(test_nn_models_sequential_persist_code_compile_model_sas_backward persist_code_compile_model_sample_and_squash_backward.cpp)
        target_link_libraries(test_nn_models_sequential_persist_code_compile_model_sas_backward
                PRIVATE
                rl_tools
                rl_tools_test
        )
        gtest_discover_tests(test_nn_models_sequential_persist_code_compile_model_sas_backward)
else()
        message(WARNING "test_nn_models_sequential_persist_code_compile_model_sas_backward is disabled because the file ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_sas_backward.h")
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_sample_and_squash_gradient.h)
        add_executable(test_nn_models_sequential_persist_code_compile_model_sas_gradient persist_code_compile_model_sample_and_squash_gradient.cpp)
        target_link_libraries(test_nn_models_sequential_persist_code_compile_model_sas_gradient
                PRIVATE
                rl_tools
                rl_tools_test
        )
        gtest_discover_tests(test_nn_models_sequential_persist_code_compile_model_sas_gradient)
else()
        message(WARNING "test_nn_models_sequential_persist_code_compile_model_sas_gradient is disabled because the file ${PROJECT_SOURCE_DIR}/tests/data/nn_models_sequential_persist_code_model_sas_gradient.h")
endif()

add_executable(
        test_nn_models_sequential_d_input_benchmark
        d_input_benchmark.cpp
)
target_link_libraries(
        test_nn_models_sequential_d_input_benchmark
        PRIVATE
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_nn_models_sequential_d_input_benchmark)

add_executable(test_nn_models_sequential_compose compose.cpp)
target_link_libraries(test_nn_models_sequential_compose
        PRIVATE
        rl_tools
        rl_tools_test
)
gtest_discover_tests(test_nn_models_sequential_compose)



if(RL_TOOLS_BACKEND_ENABLE_CUDA)
add_subdirectory(cuda)
endif()