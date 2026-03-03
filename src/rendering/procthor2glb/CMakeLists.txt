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
    GIT_TAG        9ab0d0d5f73bfd8ae29d0eb43653b045d44f3ce1
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

add_executable(procthor2glb
    procthor2glb.cpp
    ${procthor2glb_basisu_SOURCE_DIR}/transcoder/basisu_transcoder.cpp
)
target_link_libraries(procthor2glb PRIVATE RLtools::RLtools)
target_include_directories(procthor2glb SYSTEM PRIVATE
    ${procthor2glb_stb_SOURCE_DIR}
    ${procthor2glb_tinygltf_SOURCE_DIR}
    ${procthor2glb_basisu_SOURCE_DIR}/transcoder
)

find_library(PROCTHOR2GLB_ZSTD_LIBRARY NAMES zstd)
if(PROCTHOR2GLB_ZSTD_LIBRARY)
    target_link_libraries(procthor2glb PRIVATE ${PROCTHOR2GLB_ZSTD_LIBRARY})
else()
    target_compile_definitions(procthor2glb PRIVATE BASISD_SUPPORT_KTX2_ZSTD=0)
endif()
