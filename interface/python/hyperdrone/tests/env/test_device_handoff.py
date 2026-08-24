"""OptiX + CUDA integration: the sim's camera bases hand over to the renderer
device-to-device (no host synchronization in the loop) and produce frames
pixel-identical to the host camera path.

Runs in a subprocess: it needs the cuda dynamics variant, and one process can hold only
one variant — the rest of the env suite uses cpu.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from hyperdrone import render


def cuda_available():
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


HANDOFF_CHECK = """
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from hyperdrone import dynamics, render
from test_env import make_room

num_drones = 4
scene = make_room()
sim = dynamics.Sim(num_drones=num_drones, model="crazyflie", device="cuda")
renderer = render.Renderer(width=32, height=32, num_cameras=num_drones, output="rgb", fidelity="low")
renderer.init(scene)
assert renderer.backend == "optix" and sim.device == "cuda"

def spawn(positions):
    sim.reset(seed=0, sample_states=False)
    sim.state["position"] = positions
    renderer.set_cameras(sim.camera_bases(aspect=renderer.aspect), stream=sim.stream)

positions = np.array([[0.0, 0.0, 1.5], [1.0, 1.0, 1.5], [-1.0, 1.0, 1.0], [1.0, -1.0, 2.0]], dtype=np.float32)
spawn(positions)
actions = np.full((num_drones, sim.action_dim), 0.1, dtype=np.float32)
sim.step(actions)
renderer.set_cameras(sim.camera_bases(aspect=renderer.aspect), stream=sim.stream)
renderer.render()
device_frame = renderer.frame_raw().copy()

# replay the same step with the cameras routed through the host
spawn(positions)
sim.step(actions)
sim.synchronize()
renderer.set_cameras(sim.camera_bases_numpy(aspect=renderer.aspect))
renderer.render()
host_frame = renderer.frame_raw()

assert np.array_equal(device_frame, host_frame), "device hand-off diverged from host path"
print("DEVICE HANDOFF OK")
"""


@pytest.mark.skipif(
    render.backend() != "OPTIX" or not cuda_available(),
    reason="device hand-off requires the OptiX render backend and a CUDA driver",
)
def test_device_camera_handoff_matches_host():
    run = subprocess.run(
        [sys.executable, "-c", HANDOFF_CHECK, str(Path(__file__).parent)],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "DEVICE HANDOFF OK" in run.stdout
