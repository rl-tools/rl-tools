# Renders the golden corpus with the wasm reference module: raw dumps into DUMP_DIR (CTest fixture
# of the wasm comparator) and/or the corpus files into OUTPUT_DIR (regeneration). The scene is
# resolved natively with the conta tool because the module has no network; the module then only
# needs the scene's directory and the output directories preopened.
#   cmake -DCONTA=<conta tool> -DWASMTIME=<wasmtime> -DMODULE=<generate_golden.wasm> -DSCENE=<path|conta:sha1> [-DDUMP_DIR=<dir>] [-DOUTPUT_DIR=<corpus dir>] -P render.cmake
foreach(VARIABLE CONTA WASMTIME MODULE SCENE)
    if(NOT DEFINED ${VARIABLE})
        message(FATAL_ERROR "render.cmake: ${VARIABLE} is required")
    endif()
endforeach()
if(NOT DEFINED DUMP_DIR AND NOT DEFINED OUTPUT_DIR)
    message(FATAL_ERROR "render.cmake: DUMP_DIR or OUTPUT_DIR is required")
endif()
execute_process(COMMAND "${CONTA}" "${SCENE}" OUTPUT_VARIABLE SCENE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE CONTA_RESULT)
if(NOT CONTA_RESULT EQUAL 0 OR NOT EXISTS "${SCENE_PATH}")
    message(FATAL_ERROR "render.cmake: could not resolve the scene ${SCENE} (conta exit code ${CONTA_RESULT})")
endif()
get_filename_component(SCENE_DIRECTORY "${SCENE_PATH}" DIRECTORY)
set(PREOPENS --dir "${SCENE_DIRECTORY}")
set(ARGUMENTS --scene "${SCENE_PATH}")
if(DEFINED DUMP_DIR)
    file(REMOVE_RECURSE "${DUMP_DIR}")
    file(MAKE_DIRECTORY "${DUMP_DIR}")
    list(APPEND PREOPENS --dir "${DUMP_DIR}")
    list(APPEND ARGUMENTS --dump-dir "${DUMP_DIR}")
endif()
if(DEFINED OUTPUT_DIR)
    file(MAKE_DIRECTORY "${OUTPUT_DIR}")
    list(APPEND PREOPENS --dir "${OUTPUT_DIR}")
    list(APPEND ARGUMENTS --output-dir "${OUTPUT_DIR}")
endif()
execute_process(
    COMMAND "${WASMTIME}" run -W exceptions ${PREOPENS} "${MODULE}" ${ARGUMENTS}
    RESULT_VARIABLE RENDER_RESULT
)
if(NOT RENDER_RESULT EQUAL 0)
    message(FATAL_ERROR "render.cmake: the wasm reference module failed (exit code ${RENDER_RESULT})")
endif()
