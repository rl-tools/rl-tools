"""End-to-end hyperdrone stack: N drones spawned in free space, stepping L2F dynamics and
rendering their onboard cameras every step. On OptiX + CUDA the camera hand-off is fully
device-resident (no host synchronization in the loop); the same script runs — slowly — on
GENERIC + cpu anywhere.

  python drone_flythrough.py --drones 1024 --steps 200 --width 64 --height 64
"""
import argparse
import time
from pathlib import Path

import numpy as np

from hyperdrone import dynamics, env, render

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default=None, help="GLB scene (default: procedural room)")
parser.add_argument("--drones", type=int, default=64)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--height", type=int, default=64)
parser.add_argument("--drone-model", default="crazyflie")
parser.add_argument("--device", default="auto", help="dynamics device (cpu|cuda|auto)")
parser.add_argument("--save", default=None, help="save the final frame mosaic to this path")
arguments = parser.parse_args()


def build_room(rng, num_pillars=32, half=8.0, height=4.0):
    scene = render.Scene()
    room = render.Object(name="room")

    def quad(v0, v1, v2, v3, color):
        vertices = np.array([v0, v1, v2, v3], dtype=np.float32)
        indices = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], dtype=np.int32)
        room.add_mesh(render.Mesh(vertices, indices, color=color))

    quad((-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0), (0.5, 0.5, 0.5))
    quad((-half, -half, height), (half, -half, height), (half, half, height), (-half, half, height), (0.9, 0.9, 0.9))
    quad((-half, -half, 0), (half, -half, 0), (half, -half, height), (-half, -half, height), (0.8, 0.3, 0.3))
    quad((-half, half, 0), (half, half, 0), (half, half, height), (-half, half, height), (0.3, 0.8, 0.3))
    quad((-half, -half, 0), (-half, half, 0), (-half, half, height), (-half, -half, height), (0.3, 0.3, 0.8))
    quad((half, -half, 0), (half, half, 0), (half, half, height), (half, -half, height), (0.8, 0.8, 0.3))
    for _ in range(num_pillars):
        x, y = rng.uniform(-half * 0.8, half * 0.8, size=2)
        r = 0.25
        color = tuple(rng.uniform(0.2, 1.0, size=3))
        quad((x - r, y - r, 0), (x + r, y - r, 0), (x + r, y - r, height), (x - r, y - r, height), color)
        quad((x - r, y + r, 0), (x + r, y + r, 0), (x + r, y + r, height), (x - r, y + r, height), color)
        quad((x - r, y - r, 0), (x - r, y + r, 0), (x - r, y + r, height), (x - r, y - r, height), color)
        quad((x + r, y - r, 0), (x + r, y + r, 0), (x + r, y + r, height), (x + r, y - r, height), color)
    scene.add_object(room)
    scene.add_light(render.SceneLight.directional(direction=(0.3, 0.2, -1.0), color=(1.0, 1.0, 1.0)))
    scene.add_light(render.SceneLight.directional(direction=(-0.5, 0.4, -0.3), color=(0.4, 0.4, 0.5)))
    return scene


if arguments.model and Path(arguments.model).exists():
    scene = render.load_scene(arguments.model, shading="medium")
else:
    scene = build_room(np.random.default_rng(0))

sim = dynamics.Sim(num_drones=arguments.drones, model=arguments.drone_model, device=arguments.device)
renderer = render.Renderer(width=arguments.width, height=arguments.height,
                           num_cameras=arguments.drones, output="rgb", shading="medium")
world = env.World(scene, sim, renderer)

print(f"backend={renderer.backend} dynamics={sim.device} drones={arguments.drones} "
      f"device_handoff={'yes' if world._device_handoff else 'no (host cameras)'}")

sampler = env.FreeSpaceSampler(scene, clearance=0.5)
positions = sampler.sample(arguments.drones, seed=0)
world.spawn(positions)

rng = np.random.default_rng(1)
actions = np.zeros((arguments.drones, sim.action_dim), dtype=np.float32)
start = time.perf_counter()
for step in range(arguments.steps):
    actions[:] = rng.uniform(-0.1, 0.3, size=actions.shape)
    world.step(actions)
renderer.synchronize()
elapsed = time.perf_counter() - start

frames_per_second = arguments.steps * arguments.drones / elapsed
print(f"{arguments.steps} steps x {arguments.drones} drones in {elapsed:.2f}s "
      f"({arguments.steps / elapsed:.1f} steps/s, {frames_per_second:.0f} rendered frames/s)")

if arguments.save:
    renderer.save_image(arguments.save)
    print(f"saved {arguments.save}")
