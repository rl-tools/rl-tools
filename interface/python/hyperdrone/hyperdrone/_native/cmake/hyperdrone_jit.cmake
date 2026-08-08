# Shared JIT machinery: one MODULE target per configs/<key>.txt file in the build tree.
# Each file holds NAME=VALUE compile definitions written by hyperdrone.jit; the target is
# named <COMPONENT>_<key> and its artifact lands in <build>/jit/<COMPONENT>_<key>.so.
#
# A component may define, before calling this function:
#   function(hyperdrone_component_libraries DEFINES OUT_LIBRARIES)
# to map a config's definition list to extra link libraries (set ${OUT_LIBRARIES} in
# PARENT_SCOPE); used e.g. by render to pick the backend device-program library.
function(hyperdrone_jit_add_configs)
    cmake_parse_arguments(ARG "" "COMPONENT" "SOURCES;LINK" ${ARGN})
    set(config_dir ${CMAKE_BINARY_DIR}/configs)
    file(MAKE_DIRECTORY ${config_dir})
    file(GLOB config_files ${config_dir}/*.txt)
    foreach(config_file IN LISTS config_files)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${config_file})
        get_filename_component(config_key ${config_file} NAME_WE)
        file(STRINGS ${config_file} config_defines)
        set(target ${ARG_COMPONENT}_${config_key})
        add_library(${target} MODULE ${ARG_SOURCES})
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../common)
        target_compile_definitions(${target} PRIVATE ${config_defines})
        set(extra_libraries "")
        if(COMMAND hyperdrone_component_libraries)
            hyperdrone_component_libraries("${config_defines}" extra_libraries)
        endif()
        target_link_libraries(${target} PRIVATE ${ARG_LINK} ${extra_libraries})
        set_target_properties(${target} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/jit
            PREFIX ""
        )
    endforeach()
endfunction()

# Reads NAME=VALUE from a definitions list into <PREFIX>_<NAME> variables in the caller's
# scope (helper for hyperdrone_component_libraries implementations).
function(hyperdrone_jit_parse_defines DEFINES PREFIX)
    foreach(define IN LISTS DEFINES)
        string(REGEX MATCH "^([A-Za-z0-9_]+)=(.*)$" _match "${define}")
        if(_match)
            set(${PREFIX}_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" PARENT_SCOPE)
        endif()
    endforeach()
endfunction()
