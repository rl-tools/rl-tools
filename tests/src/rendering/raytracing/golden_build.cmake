# generation environment header for the golden generators, see golden_build.h.in; shared by the
# native generators and the wasm reference module (wasm/CMakeLists.txt)
function(rl_tools_configure_golden_generator TARGET_NAME BACKEND_NAME)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        RL_TOOLS_RENDERING_RAYTRACING_GOLDEN_BACKEND_NAME="${BACKEND_NAME}"
    )
    if(NOT RL_TOOLS_GOLDEN_TOOLCHAIN)
        set(RL_TOOLS_GOLDEN_TOOLCHAIN native)
    endif()
    configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/golden_build.h.in ${TARGET_NAME}-build.h.in @ONLY)
    file(GENERATE
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/${TARGET_NAME}/$<CONFIG>/golden_build.h
        INPUT ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-build.h.in
        CONDITION $<COMPILE_LANGUAGE:CXX>
        TARGET ${TARGET_NAME}
    )
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated/${TARGET_NAME}/$<CONFIG>)
endfunction()
