add_executable(
        test_utils_typing
        typing.cpp
)
target_link_libraries(
        test_utils_typing
        PRIVATE
        RLtools::Core
        GTest::gtest_main
)
gtest_discover_tests(test_utils_typing)

add_executable(
        test_utils_conta
        conta.cpp
)
target_link_libraries(
        test_utils_conta
        PRIVATE
        RLtools::Core
        GTest::gtest_main
)
if(RL_TOOLS_ENABLE_ZLIB)
    target_compile_definitions(test_utils_conta PRIVATE CONTA_ENABLE_ZLIB)
    target_link_libraries(test_utils_conta PRIVATE ${RL_TOOLS_ZLIB_TARGET})
endif()
gtest_discover_tests(test_utils_conta)

add_executable(
        test_utils_metra
        metra.cpp
)
target_link_libraries(
        test_utils_metra
        PRIVATE
        RLtools::Core
        GTest::gtest_main
)
gtest_discover_tests(test_utils_metra)
