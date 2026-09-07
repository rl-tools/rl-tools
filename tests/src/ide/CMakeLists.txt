set(RL_TOOLS_IDE_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/static/ide/build" CACHE PATH "Bundle root used by the IDE tests")
include("${CMAKE_CURRENT_SOURCE_DIR}/tests.cmake")
ide_add_tests(BUNDLE "${RL_TOOLS_IDE_OUTPUT_DIR}")
