#!/usr/bin/env bash
# Self-containment rot guard: build the sdist, install it into a fresh venv outside the
# repository, and run a render smoke against the vendored rl-tools tree on the GENERIC
# backend. Intended for CI (clean container) and local verification.
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

python3 -m venv "${WORK_DIR}/venv"
"${WORK_DIR}/venv/bin/pip" install -q build
(cd "${PACKAGE_DIR}" && "${WORK_DIR}/venv/bin/python" -m build --sdist -o "${WORK_DIR}/dist")
"${WORK_DIR}/venv/bin/pip" install -q "${WORK_DIR}"/dist/hyperdrone-*.tar.gz numpy nanobind

cd /
env -u HYPERDRONE_RLTOOLS_ROOT \
    HYPERDRONE_RENDER_BACKEND=GENERIC \
    HYPERDRONE_CACHE_DIR="${WORK_DIR}/cache" \
    "${WORK_DIR}/venv/bin/python" - <<'PY'
import numpy as np
from hyperdrone import _jit as jit, render

root = jit.source_root()
assert "_vendor" in str(root), f"expected the vendored tree, resolved {root}"
dataset = render.datasets.Train200()
assert len(dataset) == 200
assert dataset["ProcTHOR-Train-1"].reference == "conta:07b9d8e44f3891573b6292f8cc47bf7725a43b89"
scene = render.Scene()
wall = render.Object(name="wall")
wall.add_mesh(render.Mesh(
    np.array([[2, -1, -1], [2, 1, -1], [2, 1, 1], [2, -1, 1]], dtype=np.float32),
    np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
))
scene.add_object(wall)
from hyperdrone.camera import CameraSpec, Depth
renderer = render.Renderer(width=16, height=16, cameras=(CameraSpec("front", (Depth(),)),), fidelity="low")
renderer.init(scene)
renderer.set_cameras(renderer.camera(position=(0, 0, 0), look_at=(1, 0, 0)))
renderer.render()
assert abs(renderer.outputs[Depth()].read()[0, 8, 8] - 2.0) < 1e-2
print("sdist self-containment: OK")
PY
