#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
APP_NAME="YawPredictor"
APP_BUNDLE="$SCRIPT_DIR/$APP_NAME.app"
BUILD_TMP="$SCRIPT_DIR/.build"

mkdir -p "$BUILD_TMP"

# --- Find HDF5 ---
HDF5_PREFIX="${HDF5_PREFIX:-$(brew --prefix hdf5 2>/dev/null || true)}"
if [ -z "$HDF5_PREFIX" ] || [ ! -d "$HDF5_PREFIX/include" ]; then
    echo "Error: HDF5 not found. Install with: brew install hdf5"
    exit 1
fi
echo "HDF5: $HDF5_PREFIX"

# --- Find HighFive headers ---
HIGHFIVE_INCLUDE=""
for candidate in \
    "$REPO_ROOT/.dependencies"/*/highfive-src/include \
    "$REPO_ROOT/build/../.dependencies"/*/highfive-src/include \
    "$(brew --prefix highfive 2>/dev/null)/include" \
    ; do
    if [ -d "$candidate" ] 2>/dev/null; then
        HIGHFIVE_INCLUDE="$candidate"
        break
    fi
done

if [ -z "$HIGHFIVE_INCLUDE" ]; then
    echo "Error: HighFive headers not found."
    echo "Either run a cmake configure first (so HighFive is fetched), or install: brew install highfive"
    exit 1
fi
echo "HighFive: $HIGHFIVE_INCLUDE"

CXX_FLAGS="-std=c++20 -Ofast -I $REPO_ROOT/include -I $HDF5_PREFIX/include -I $HIGHFIVE_INCLUDE -DRL_TOOLS_ENABLE_HDF5"

# --- Compile C++ inference library ---
echo "Compiling inference.cpp..."
clang++ $CXX_FLAGS -c "$SCRIPT_DIR/inference.cpp" -o "$BUILD_TMP/inference.o"
ar rcs "$BUILD_TMP/libinference.a" "$BUILD_TMP/inference.o"
echo "Built: $BUILD_TMP/libinference.a"

# --- Build h5_to_tar converter ---
echo "Compiling h5_to_tar..."
clang++ $CXX_FLAGS "$SCRIPT_DIR/h5_to_tar.cpp" -L "$HDF5_PREFIX/lib" -lhdf5 -lc++ -o "$BUILD_TMP/h5_to_tar"
echo "Built: $BUILD_TMP/h5_to_tar"

# --- Create .app bundle ---
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
cp "$SCRIPT_DIR/Info.plist" "$APP_BUNDLE/Contents/"

# --- Compile Swift ---
echo "Compiling Swift sources..."
SWIFT_SOURCES=$(find "$SCRIPT_DIR/Sources" -name "*.swift")

swiftc \
    -O \
    -import-objc-header "$SCRIPT_DIR/inference.h" \
    $SWIFT_SOURCES \
    "$BUILD_TMP/libinference.a" \
    -L "$HDF5_PREFIX/lib" -lhdf5 \
    -lc++ \
    -framework SwiftUI \
    -framework AVFoundation \
    -framework CoreImage \
    -framework AppKit \
    -framework CoreVideo \
    -o "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

# --- Sign with entitlements ---
codesign --force --sign - --entitlements "$SCRIPT_DIR/YawPredictor.entitlements" "$APP_BUNDLE"

echo ""
echo "Built: $APP_BUNDLE"
echo ""
echo "To convert HDF5 to tar format (for iOS):"
echo "  $BUILD_TMP/h5_to_tar <input.h5> <output.tar>"
echo ""
echo "Run:"
echo "  open $APP_BUNDLE --args <model.h5 or model.tar>"
