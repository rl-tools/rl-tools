set(SIM_SOURCE_IDENTITY "")
foreach(SOURCE CMakeLists.txt simulator_matrix_config.h.in simulator_matrix_identity.cmake
        simulator_matrix.cpp simulator_matrix.h simulator_matrix_main.cpp
        simulator_matrix_physics.h simulator_matrix_physics_contract.h
        simulator_matrix_physics.cu simulator_matrix_physics_cpu.cpp)
    set(SOURCE_PATH "${CMAKE_CURRENT_LIST_DIR}/${SOURCE}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SOURCE_PATH}")
    file(SHA256 "${SOURCE_PATH}" SOURCE_HASH)
    string(APPEND SIM_SOURCE_IDENTITY "${SOURCE}:${SOURCE_HASH}\n")
endforeach()
string(SHA256 SIM_SOURCE_HASH "${SIM_SOURCE_IDENTITY}")
