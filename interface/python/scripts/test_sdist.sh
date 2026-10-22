#!/usr/bin/env bash
# Self-containment rot guard: build both sdists, install exactly those two artifacts into a
# fresh venv outside the repository, and smoke the vendored rl-tools tree on the GENERIC
# backend: a render, a CPU dynamics step and a rotorcraft articulation. Intended for CI (clean
# container) and local use.
set -euo pipefail

PYTHON_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

python3 -m venv "${WORK_DIR}/venv"
"${WORK_DIR}/venv/bin/pip" install -q build
for package in hyperlux hyperdrone; do
    (cd "${PYTHON_DIR}/${package}" && "${WORK_DIR}/venv/bin/python" -m build --sdist -o "${WORK_DIR}/dist")
done
"${WORK_DIR}/venv/bin/pip" install -q "${WORK_DIR}"/dist/hyperlux-*.tar.gz "${WORK_DIR}"/dist/hyperdrone-*.tar.gz

cd /
env -u HYPERLUX_RLTOOLS_ROOT \
    HYPERLUX_BACKEND=GENERIC \
    HYPERLUX_CACHE_DIR="${WORK_DIR}/cache" \
    "${WORK_DIR}/venv/bin/python" - <<'PY'
import numpy as np
import hyperlux
from hyperlux import _jit as jit
from hyperlux.camera import CameraSpec, Depth
import hyperdrone
from hyperdrone import Rotorcraft, articulate, dynamics

root = jit.source_root()
assert "_vendor" in str(root), f"expected the vendored tree, resolved {root}"
assert hyperdrone.__version__.split(".post")[0] == hyperlux.__version__
dataset = hyperlux.datasets.procthor.Train200()
assert len(dataset) == 200
assert dataset["ProcTHOR-Train-1"].reference == "conta:07b9d8e44f3891573b6292f8cc47bf7725a43b89"


def quad(center_x, center_y, half_size, name):
    vertices = np.array([[center_x - half_size, center_y - half_size, 0], [center_x + half_size, center_y - half_size, 0],
                         [center_x + half_size, center_y + half_size, 0], [center_x - half_size, center_y + half_size, 0]], dtype=np.float32)
    obj = hyperlux.Object(name=name)
    obj.add_mesh(hyperlux.Mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)))
    return obj


def pose(z=0.0):
    out = np.eye(4, dtype=np.float32)[:3].copy()
    out[2, 3] = z
    return out


scene = hyperlux.Scene()
wall = hyperlux.Object(name="wall")
wall.add_mesh(hyperlux.Mesh(
    np.array([[2, -1, -1], [2, 1, -1], [2, 1, 1], [2, -1, 1]], dtype=np.float32),
    np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
))
scene.add_object(wall)
renderer = hyperlux.HyperLux(width=16, height=16, cameras=(CameraSpec("front", (Depth(),)),), fidelity="low")
renderer.init(scene)
renderer.set_cameras(renderer.camera(position=(0, 0, 0), look_at=(1, 0, 0)))
renderer.render()
assert abs(renderer.outputs[Depth()].read()[0, 8, 8] - 2.0) < 1e-2

sim = dynamics.Sim(2, seed=1)
sim.step(np.zeros((2, sim.action_dim), dtype=np.float32))
assert sim.observe().shape == (2, sim.observation_dim)

asset = hyperlux.ObjectAssembly()
asset.add_part(quad(0, 0, 0.1, "body"), pose())
asset.add_part(quad(0.1, 0.1, 0.05, "prop_0"), pose(z=0.05))
asset.add_part(quad(-0.1, 0.1, 0.05, "prop_1"), pose(z=0.05))
articulate(asset)
rotorcraft = Rotorcraft(asset)
assert asset.num_coordinates == 2 and rotorcraft.scale == 1
assert rotorcraft.coordinates(np.array([0.5, 0.25], dtype=np.float32)).shape == (2,)
print("sdist self-containment: OK")
PY
