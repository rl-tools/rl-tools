# Shared JIT machinery: one MODULE target per configs/<key>.txt file in the build tree.
# Each file holds NAME=VALUE compile definitions written by hyperdrone._jit; the target is
# named <COMPONENT>_<key> and its artifact lands in <build>/jit/<COMPONENT>_<key>.so.
#
# A component may define, before calling this function:
#   function(hyperdrone_component_libraries DEFINES OUT_LIBRARIES)
# to map a config's definition list to extra link libraries (set ${OUT_LIBRARIES} in
# PARENT_SCOPE); used e.g. by render to pick the backend device-program library.
#
# Components with Python modules call hyperdrone_jit_find_nanobind() first; a macro so the
# variables nanobind's config sets stay in the caller's scope for nanobind_add_module.
macro(hyperdrone_jit_find_nanobind)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    find_package(nanobind CONFIG REQUIRED)
endmacro()

function(hyperdrone_jit_add_configs)
    cmake_parse_arguments(ARG "PYTHON_MODULE" "COMPONENT" "SOURCES;LINK" ${ARGN})
    set(config_dir ${CMAKE_BINARY_DIR}/configs)
    file(MAKE_DIRECTORY ${config_dir})
    file(GLOB config_files ${config_dir}/*.txt)
    foreach(config_file IN LISTS config_files)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${config_file})
        get_filename_component(config_key ${config_file} NAME_WE)
        file(STRINGS ${config_file} config_defines)
        set(target ${ARG_COMPONENT}_${config_key})
        if(ARG_PYTHON_MODULE)
            # NOMINSIZE: the simulation and renderer must not be built at nanobind's default -Os
            nanobind_add_module(${target} NOMINSIZE ${ARG_SOURCES})
            string(TOLOWER "${HYPERDRONE_VARIANT}" module_variant)
            target_compile_definitions(${target} PRIVATE HYPERDRONE_MODULE_NAME=hyperdrone_${ARG_COMPONENT}_${module_variant}_${config_key})
            set_target_properties(${target} PROPERTIES SUFFIX ".so")
        else()
            add_library(${target} MODULE ${ARG_SOURCES})
        endif()
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
