include(ExternalProject)

function(ide_external name)
    cmake_parse_arguments(PARSE_ARGV 1 stage "" "SOURCE;BINARY" "ARGS;TARGETS;DEPENDS;BYPRODUCTS")
    ExternalProject_Add(${name}
        PREFIX "${CMAKE_BINARY_DIR}/stamps/${name}"
        SOURCE_DIR "${stage_SOURCE}"
        BINARY_DIR "${stage_BINARY}"
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        CMAKE_GENERATOR "${CMAKE_GENERATOR}"
        LIST_SEPARATOR "|"
        CMAKE_ARGS ${stage_ARGS}
        CONFIGURE_HANDLED_BY_BUILD ON
        BUILD_COMMAND "${CMAKE_COMMAND}" -E env "SOURCE_DATE_EPOCH=${IDE_SOURCE_DATE_EPOCH}"
            "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel ${RL_TOOLS_IDE_JOBS} --target ${stage_TARGETS}
        BUILD_ALWAYS ON
        BUILD_BYPRODUCTS ${stage_BYPRODUCTS}
        INSTALL_COMMAND ""
        DEPENDS ${stage_DEPENDS}
        USES_TERMINAL_CONFIGURE ON
        USES_TERMINAL_BUILD ON
    )
    ExternalProject_Add_StepDependencies(${name} configure ${IDE_CONFIGURE_DEPENDS})
endfunction()

function(ide_select_tools output directory prefix)
    set(arguments)
    file(GLOB children CONFIGURE_DEPENDS RELATIVE "${directory}" "${directory}/*")
    foreach(child IN LISTS children)
        if(EXISTS "${directory}/${child}/CMakeLists.txt")
            string(TOUPPER "${child}" option)
            string(REPLACE "-" "_" option "${option}")
            if(child IN_LIST ARGN)
                set(enabled ON)
            else()
                set(enabled OFF)
            endif()
            list(APPEND arguments "-D${prefix}_TOOL_${option}_BUILD=${enabled}")
        endif()
    endforeach()
    set(${output} "${arguments}" PARENT_SCOPE)
endfunction()
