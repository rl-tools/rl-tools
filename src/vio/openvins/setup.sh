#!/usr/bin/env bash
# Fetches and builds OpenVINS (GPL-3.0 — kept out of the repo tree; the artifacts below are
# gitignored) for the visual_inertial_localization OpenVINS baseline binary.
# Pin: v2.7 plus the single upstream Ceres >= 2.2 compatibility commit (PR #520).
# Requires: libeigen3-dev libopencv-dev libboost-dev (+system/filesystem/thread/date_time) libceres-dev
set -euo pipefail
cd "$(dirname "$0")"
PREFIX="$(pwd)/prefix"
OPENVINS_TAG=v2.7
CERES_22_FIX=676042f779c9146ee3612ea74f05a29a3d5d317d # "State_JPLQuatLocal class has been updated" (ceres::Manifold, version-guarded)
if [ ! -d src ]; then
    git clone https://github.com/rpng/open_vins.git src
fi
cd src
git fetch origin --tags
git checkout "$OPENVINS_TAG"
git checkout "$CERES_22_FIX" -- ov_init/src/ceres/State_JPLQuatLocal.h ov_init/src/ceres/State_JPLQuatLocal.cpp
# Boost >= 1.90 (Ubuntu 26.04) no longer ships the header-only 'system' stub as a component
sed -i 's/find_package(Boost REQUIRED COMPONENTS system filesystem/find_package(Boost REQUIRED COMPONENTS filesystem/' ov_msckf/CMakeLists.txt ov_core/CMakeLists.txt ov_init/CMakeLists.txt
cmake -S ov_msckf -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS=-march=native \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DENABLE_ROS=OFF \
    -DENABLE_ARUCO_TAGS=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build build -j5
cmake --install build
echo "installed OpenVINS $OPENVINS_TAG (+$CERES_22_FIX) into $PREFIX"
