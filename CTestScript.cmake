set(CTEST_BINARY_DIRECTORY ".")
set(CTEST_SOURCE_DIRECTORY "..")
if(NOT CDASH_TOKEN)
    message(FATAL_ERROR "CDASH_TOKEN not set")
endif()
ctest_start(ExperimentalTest)
ctest_test()
ctest_submit(SUBMIT_URL https://my.cdash.org/submit.php?project=RLtools HTTPHEADER "Authorization: Bearer ${CDASH_TOKEN}")


#invoke by e.g. ctest -j4 -V -S ../rl_tools/CTestScript.cmake -DCDASH_TOKEN={TOKEN}
