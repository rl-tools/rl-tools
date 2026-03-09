#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
BUILD_TMP="$SCRIPT_DIR/.build"

mkdir -p "$BUILD_TMP"

IOS_SDK=$(xcrun --sdk iphoneos --show-sdk-path)
IOS_MIN="16.0"

echo "iOS SDK: $IOS_SDK"

# --- Compile C++ inference library for iOS (no HDF5, tar only) ---
echo "Compiling inference.cpp for iOS..."
xcrun clang++ -std=c++20 -O3 -c \
    -target arm64-apple-ios${IOS_MIN} \
    -isysroot "$IOS_SDK" \
    -I "$REPO_ROOT/include" \
    "$SCRIPT_DIR/inference.cpp" \
    -o "$BUILD_TMP/inference_ios.o"

ar rcs "$BUILD_TMP/libinference_ios.a" "$BUILD_TMP/inference_ios.o"
echo "Built: $BUILD_TMP/libinference_ios.a"

# --- Find development team ---
TEAM_ID="${DEVELOPMENT_TEAM:-}"
if [ -z "$TEAM_ID" ]; then
    TEAM_ID=$(security find-identity -v -p codesigning 2>/dev/null | grep "Apple Development" | head -1 | sed 's/.*(\(.*\))/\1/' | tr -d '"' || true)
fi
if [ -z "$TEAM_ID" ]; then
    echo "Error: No development team found. Set DEVELOPMENT_TEAM=<team_id>"
    exit 1
fi
echo "Team: $TEAM_ID"

# --- Compile Swift for iOS ---
echo "Compiling Swift sources for iOS..."
SWIFT_SOURCES=$(find "$SCRIPT_DIR/Sources" -name "*.swift")

xcrun swiftc \
    -O \
    -target arm64-apple-ios${IOS_MIN} \
    -sdk "$IOS_SDK" \
    -import-objc-header "$SCRIPT_DIR/inference.h" \
    $SWIFT_SOURCES \
    "$BUILD_TMP/libinference_ios.a" \
    -lc++ \
    -framework SwiftUI \
    -framework AVFoundation \
    -framework CoreImage \
    -framework UIKit \
    -framework CoreVideo \
    -o "$BUILD_TMP/YawPredictor_ios"

# --- Create .app bundle ---
APP_BUNDLE="$BUILD_TMP/YawPredictor_ios.app"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE"

cp "$BUILD_TMP/YawPredictor_ios" "$APP_BUNDLE/YawPredictor"
cat > "$APP_BUNDLE/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.rltools.yaw-predictor</string>
    <key>CFBundleName</key>
    <string>YawPredictor</string>
    <key>CFBundleExecutable</key>
    <string>YawPredictor</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>MinimumOSVersion</key>
    <string>16.0</string>
    <key>NSCameraUsageDescription</key>
    <string>Camera access is needed to capture images for yaw angle prediction.</string>
    <key>UIDeviceFamily</key>
    <array>
        <integer>1</integer>
    </array>
    <key>UISupportedInterfaceOrientations</key>
    <array>
        <string>UIInterfaceOrientationPortrait</string>
    </array>
    <key>UILaunchScreen</key>
    <dict/>
</dict>
</plist>
PLIST

# --- Sign ---
codesign --force --sign "Apple Development" --entitlements "$SCRIPT_DIR/YawPredictor.entitlements" "$APP_BUNDLE"

echo ""
echo "Built: $APP_BUNDLE"
echo ""

# --- Install on device ---
echo "Installing on device..."
xcrun devicectl device install app --device "iPhone von Jonas" "$APP_BUNDLE" 2>&1 || {
    echo ""
    echo "Auto-install failed. Try:"
    echo "  xcrun devicectl device install app --device 'iPhone von Jonas' $APP_BUNDLE"
}
