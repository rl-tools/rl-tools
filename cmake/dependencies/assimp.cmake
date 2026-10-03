include(FetchContent)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(RL_TOOLS_ASSIMP_VERSION 6.0.4)
set(RL_TOOLS_ASSIMP_ARCHIVE "${FETCHCONTENT_BASE_DIR}/assimp-download/assimp-v${RL_TOOLS_ASSIMP_VERSION}.tar.gz")
if(RL_TOOLS_OFFLINE_BUILD)
    set(RL_TOOLS_ASSIMP_URL "${RL_TOOLS_ASSIMP_ARCHIVE}")
else()
    set(RL_TOOLS_ASSIMP_URL "https://github.com/assimp/assimp/archive/refs/tags/v${RL_TOOLS_ASSIMP_VERSION}.tar.gz")
endif()
FetchContent_Declare(assimp
    URL "${RL_TOOLS_ASSIMP_URL}"
    URL_HASH SHA256=afa5487efdd285661afa842c85187cd8c541edad92e8d4aa85be4fca7476eccc
    DOWNLOAD_DIR "${FETCHCONTENT_BASE_DIR}/assimp-download"
    DOWNLOAD_NAME "assimp-v${RL_TOOLS_ASSIMP_VERSION}.tar.gz"
    EXCLUDE_FROM_ALL
)
