include(FetchContent)

FetchContent_Declare(
    procthor2glb_stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        f1c79c02822848a9bed4315b12c8c8f3761e1296
    EXCLUDE_FROM_ALL
)
rl_tools_fetchcontent_makeavailable_quiet(procthor2glb_stb)

FetchContent_Declare(
    procthor2glb_tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        0d9ce1a9561688c6c6e7471430d15a50a0b15f1a # v3.0.1
    EXCLUDE_FROM_ALL
)
rl_tools_fetchcontent_makeavailable_quiet(procthor2glb_tinygltf)

FetchContent_Declare(
    procthor2glb_basisu
    GIT_REPOSITORY https://github.com/BinomialLLC/basis_universal.git
    GIT_TAG        162ea47e2816dfdb4538352eea29854f6a4d230a
    EXCLUDE_FROM_ALL
)
rl_tools_fetchcontent_makeavailable_quiet(procthor2glb_basisu)

# the conversion core as a library: shared by the CLI below and the AI2ThorHab dataset source
# (include/rl_tools/rendering/datasets/procthor/conversion.h declares the API)
add_library(rl_tools_procthor_conversion STATIC
    conversion.cpp
    ${procthor2glb_basisu_SOURCE_DIR}/transcoder/basisu_transcoder.cpp
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set_source_files_properties(${procthor2glb_basisu_SOURCE_DIR}/transcoder/basisu_transcoder.cpp
        PROPERTIES COMPILE_OPTIONS "-Wno-stringop-overflow;-Wno-type-limits")
endif()
target_link_libraries(rl_tools_procthor_conversion PUBLIC RLtools::RLtools)
target_include_directories(rl_tools_procthor_conversion SYSTEM PRIVATE
    ${procthor2glb_stb_SOURCE_DIR}
    ${procthor2glb_tinygltf_SOURCE_DIR}/attic # tinygltf >= v3 is a C rewrite; the v2 C++ single-header lives in attic/
    ${procthor2glb_basisu_SOURCE_DIR}/transcoder
)

find_library(PROCTHOR2GLB_ZSTD_LIBRARY NAMES zstd)
if(PROCTHOR2GLB_ZSTD_LIBRARY)
    target_link_libraries(rl_tools_procthor_conversion PRIVATE ${PROCTHOR2GLB_ZSTD_LIBRARY})
else()
    target_compile_definitions(rl_tools_procthor_conversion PRIVATE BASISD_SUPPORT_KTX2_ZSTD=0)
endif()

add_executable(procthor2glb procthor2glb.cpp)
target_link_libraries(procthor2glb PRIVATE rl_tools_procthor_conversion)
