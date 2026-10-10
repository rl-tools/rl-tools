#!/usr/bin/env bash
# Populates static/hyperdrone/assets/ for ?assets=local: drone GLB directory as $1,
# optional flight-controller TAR path as $2; the scene comes from tests/data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ASSETS_DIR="$SCRIPT_DIR/assets"
DRONE_SOURCE_DIR="${1:-}"
POLICY_SOURCE_PATH="${2:-}"
mkdir -p "$ASSETS_DIR"

link_asset() {
    local source_path="$1"
    local sha1="$2"
    local label="$3"
    if [ ! -f "$source_path" ]; then
        echo "missing: $label ($source_path)" >&2
        return 0
    fi
    local actual
    actual="$(shasum -a 1 "$source_path" | cut -d' ' -f1)"
    if [ "$actual" != "$sha1" ]; then
        echo "sha1 mismatch for $label: expected $sha1, got $actual — skipped" >&2
        return 0
    fi
    ln -f "$source_path" "$ASSETS_DIR/$sha1" 2>/dev/null || cp "$source_path" "$ASSETS_DIR/$sha1"
    echo "linked $label -> assets/$sha1"
}

link_asset "$REPO_ROOT/tests/data/ProcTHOR-Train-1.glb" 7f1c9129532798e0b63bc41edb6b4c09251cf8a0 "ProcTHOR-Train-1.glb"
if [ -n "$DRONE_SOURCE_DIR" ]; then
    link_asset "$DRONE_SOURCE_DIR/crazyflie.glb"           87d2b8b1d518445872d34906320a76656731902f "crazyflie.glb"
    link_asset "$DRONE_SOURCE_DIR/crazyflie_brushless.glb" eedb897237c1d77ed8ef0681554d92d06a7c27ca "crazyflie_brushless.glb"
    link_asset "$DRONE_SOURCE_DIR/savagebee_pusher.glb"    fd4eddc7c4a3f74e51823a2fa392bb96e095e53e "savagebee_pusher.glb"
    link_asset "$DRONE_SOURCE_DIR/arpl.glb"                8398a239fd1cd80df8adab67497baf7be3d8cda4 "arpl.glb"
    link_asset "$DRONE_SOURCE_DIR/x500.glb"                f6681e7a8b7fa7ef023bedcd795981becb731d0e "x500.glb"
    link_asset "$DRONE_SOURCE_DIR/soft.glb"                fcdf53a791f313b9bb07f775b10b9e4c2bfa0ae9 "soft.glb"
fi
if [ -n "$POLICY_SOURCE_PATH" ]; then
    link_asset "$POLICY_SOURCE_PATH" b6bd082c60fcb27f42a2b5f2f635587d2b60d170 "flight controller"
fi
