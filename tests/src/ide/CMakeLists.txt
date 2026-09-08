set(RL_TOOLS_IDE_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/static/ide/build" CACHE PATH "Bundle root used by the IDE tests")
function(ide_register_test name)
    cmake_parse_arguments(PARSE_ARGV 1 test "" "PROGRAM;TIMEOUT;REQUIRED" "ARGS;FILES;LABELS")
    add_test(NAME ${name} COMMAND "${CMAKE_COMMAND}" "-DPROGRAM=${test_PROGRAM}" "-DARGUMENTS=${test_ARGS}"
        "-DFILES=${test_FILES}" "-DREQUIRED=${test_REQUIRED}" -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/run_test.cmake")
    set_tests_properties(${name} PROPERTIES TIMEOUT ${test_TIMEOUT} LABELS "${test_LABELS}")
    if(NOT test_REQUIRED)
        set_tests_properties(${name} PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP: IDE prerequisite")
    endif()
endfunction()

set(tests "${CMAKE_CURRENT_LIST_DIR}")
get_filename_component(repository "${tests}/../../.." ABSOLUTE)
find_program(IDE_NODE_EXECUTABLE node)
if(IDE_NODE_EXECUTABLE)
    set(node "${IDE_NODE_EXECUTABLE}")
else()
    set(node node)
endif()
file(GLOB unit_tests CONFIGURE_DEPENDS "${tests}/*.test.mjs")
ide_register_test(test_ide_unit PROGRAM "${node}" ARGS --test ${unit_tests}
    REQUIRED "${IDE_REQUIRE_TESTS}" TIMEOUT 120 LABELS ide ide-verify)
set(artifacts "${RL_TOOLS_IDE_OUTPUT_DIR}/toolchain/llvm.wasm" "${RL_TOOLS_IDE_OUTPUT_DIR}/toolchain/sysroot.tar"
    "${RL_TOOLS_IDE_OUTPUT_DIR}/toolchain/toolchain.json" "${RL_TOOLS_IDE_OUTPUT_DIR}/toolchain/SHA256SUMS"
    "${RL_TOOLS_IDE_OUTPUT_DIR}/rl_tools_include.tar" "${RL_TOOLS_IDE_OUTPUT_DIR}/manifest.json" "${RL_TOOLS_IDE_OUTPUT_DIR}/examples.json"
    "${RL_TOOLS_IDE_OUTPUT_DIR}/examples/pendulum_sac.cpp" "${RL_TOOLS_IDE_OUTPUT_DIR}/examples/smoke.cpp")
ide_register_test(test_ide_toolchain_provenance PROGRAM "${CMAKE_COMMAND}"
    ARGS "-DBUNDLE_DIR=${RL_TOOLS_IDE_OUTPUT_DIR}" "-DCONFIG=${IDE_CONFIGURATION}" -P "${tests}/provenance.cmake"
    FILES ${artifacts} REQUIRED "${IDE_REQUIRE_TESTS}" TIMEOUT 60 LABELS ide ide-verify)
ide_register_test(test_ide_toolchain PROGRAM "${node}" ARGS "${tests}/toolchain.mjs" --bundle "${RL_TOOLS_IDE_OUTPUT_DIR}"
    FILES ${artifacts} REQUIRED "${IDE_REQUIRE_TESTS}" TIMEOUT 300 LABELS ide ide-verify)
ide_register_test(test_ide_pipeline PROGRAM "${node}" ARGS "${tests}/pipeline.mjs" --bundle "${RL_TOOLS_IDE_OUTPUT_DIR}" --example smoke
    FILES ${artifacts} REQUIRED "${IDE_REQUIRE_TESTS}" TIMEOUT 300 LABELS ide ide-verify)
set(training_arguments "${tests}/pipeline.mjs" --bundle "${RL_TOOLS_IDE_OUTPUT_DIR}" --example pendulum_sac)
set(training_files ${artifacts})
if(IDE_REFERENCE)
    list(APPEND training_arguments --reference "${IDE_REFERENCE}")
    list(APPEND training_files "${IDE_REFERENCE}")
endif()
ide_register_test(test_ide_training PROGRAM "${node}" ARGS ${training_arguments}
    FILES ${training_files} REQUIRED "${IDE_REQUIRE_TESTS}" TIMEOUT 900 LABELS ide ide-training)
set(python "${repository}/.venv/bin/python")
if(WIN32)
    set(python "${repository}/.venv/Scripts/python.exe")
endif()
if(UNIX)
    ide_register_test(test_ide_build_graph PROGRAM "${python}" ARGS "${tests}/build_graph_test.py"
        REQUIRED "${IDE_REQUIRE_TESTS}" TIMEOUT 180 LABELS ide ide-verify)
endif()
set(browser_arguments "${tests}/browser_test.py" --root "${repository}" --bundle "${RL_TOOLS_IDE_OUTPUT_DIR}")
if(NOT IDE_REQUIRE_TESTS)
    list(APPEND browser_arguments --allow-missing)
endif()
find_program(IDE_BROWSER_EXECUTABLE NAMES google-chrome google-chrome-stable chromium chromium-browser firefox)
set(browser_labels ide ide-browser)
if(IDE_BROWSER_EXECUTABLE)
    list(APPEND browser_arguments --browser "${IDE_BROWSER_EXECUTABLE}")
    list(APPEND browser_labels ide-verify)
endif()
ide_register_test(test_ide_browser PROGRAM "${python}" ARGS ${browser_arguments}
    FILES ${artifacts} REQUIRED "${IDE_REQUIRE_TESTS}" TIMEOUT 660 LABELS ${browser_labels})
