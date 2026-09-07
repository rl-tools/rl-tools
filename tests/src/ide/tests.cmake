function(ide_register_test name)
    cmake_parse_arguments(PARSE_ARGV 1 test "" "PROGRAM;TIMEOUT;REQUIRED" "ARGS;FILES;LABELS")
    add_test(NAME ${name} COMMAND "${CMAKE_COMMAND}" "-DPROGRAM=${test_PROGRAM}" "-DARGUMENTS=${test_ARGS}"
        "-DFILES=${test_FILES}" "-DREQUIRED=${test_REQUIRED}" -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/run_test.cmake")
    set_tests_properties(${name} PROPERTIES TIMEOUT ${test_TIMEOUT} LABELS "${test_LABELS}")
    if(NOT test_REQUIRED)
        set_tests_properties(${name} PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP: IDE prerequisite")
    endif()
endfunction()

function(ide_add_tests)
    cmake_parse_arguments(PARSE_ARGV 0 suite "REQUIRED" "BUNDLE;REFERENCE" "")
    set(tests "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    get_filename_component(repository "${tests}/../../.." ABSOLUTE)
    find_program(IDE_NODE_EXECUTABLE node)
    if(IDE_NODE_EXECUTABLE)
        set(node "${IDE_NODE_EXECUTABLE}")
    else()
        set(node node)
    endif()
    file(GLOB unit_tests CONFIGURE_DEPENDS "${tests}/*.test.mjs")
    ide_register_test(test_ide_unit PROGRAM "${node}" ARGS --test ${unit_tests}
        REQUIRED "${suite_REQUIRED}" TIMEOUT 120 LABELS ide ide-verify)
    set(artifacts "${suite_BUNDLE}/toolchain/llvm.wasm" "${suite_BUNDLE}/toolchain/sysroot.tar"
        "${suite_BUNDLE}/toolchain/toolchain.json" "${suite_BUNDLE}/toolchain/SHA256SUMS"
        "${suite_BUNDLE}/rl_tools_include.tar" "${suite_BUNDLE}/manifest.json" "${suite_BUNDLE}/examples.json"
        "${suite_BUNDLE}/examples/pendulum_sac.cpp" "${suite_BUNDLE}/examples/smoke.cpp")
    ide_register_test(test_ide_toolchain_provenance PROGRAM "${CMAKE_COMMAND}"
        ARGS "-DBUNDLE_DIR=${suite_BUNDLE}" -P "${tests}/provenance.cmake"
        FILES ${artifacts} REQUIRED "${suite_REQUIRED}" TIMEOUT 60 LABELS ide ide-verify)
    ide_register_test(test_ide_toolchain PROGRAM "${node}" ARGS "${tests}/toolchain.mjs" --bundle "${suite_BUNDLE}"
        FILES ${artifacts} REQUIRED "${suite_REQUIRED}" TIMEOUT 300 LABELS ide ide-verify)
    ide_register_test(test_ide_pipeline PROGRAM "${node}" ARGS "${tests}/pipeline.mjs" --bundle "${suite_BUNDLE}" --example smoke
        FILES ${artifacts} REQUIRED "${suite_REQUIRED}" TIMEOUT 300 LABELS ide ide-verify)
    set(training_arguments "${tests}/pipeline.mjs" --bundle "${suite_BUNDLE}" --example pendulum_sac)
    set(training_files ${artifacts})
    if(suite_REFERENCE)
        list(APPEND training_arguments --reference "${suite_REFERENCE}")
        list(APPEND training_files "${suite_REFERENCE}")
    endif()
    ide_register_test(test_ide_training PROGRAM "${node}" ARGS ${training_arguments}
        FILES ${training_files} REQUIRED "${suite_REQUIRED}" TIMEOUT 900 LABELS ide ide-training)
    set(python "${repository}/.venv/bin/python")
    if(WIN32)
        set(python "${repository}/.venv/Scripts/python.exe")
    endif()
    if(UNIX)
        ide_register_test(test_ide_build_graph PROGRAM "${python}" ARGS "${tests}/build_graph_test.py"
            REQUIRED "${suite_REQUIRED}" TIMEOUT 180 LABELS ide ide-verify)
    endif()
    set(browser_arguments "${tests}/browser_test.py" --root "${repository}" --bundle "${suite_BUNDLE}")
    if(NOT suite_REQUIRED)
        list(APPEND browser_arguments --allow-missing)
    endif()
    find_program(IDE_BROWSER_EXECUTABLE NAMES google-chrome google-chrome-stable chromium chromium-browser firefox)
    set(browser_labels ide ide-browser)
    if(IDE_BROWSER_EXECUTABLE)
        list(APPEND browser_arguments --browser "${IDE_BROWSER_EXECUTABLE}")
        list(APPEND browser_labels ide-verify)
    endif()
    ide_register_test(test_ide_browser PROGRAM "${python}" ARGS ${browser_arguments}
        FILES ${artifacts} REQUIRED "${suite_REQUIRED}" TIMEOUT 660 LABELS ${browser_labels})
endfunction()
