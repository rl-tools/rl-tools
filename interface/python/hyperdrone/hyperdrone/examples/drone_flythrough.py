"""End-to-end hyperdrone stack: N drones spawned in free space, stepping L2F dynamics and
rendering their onboard cameras every step. On OptiX + CUDA the camera hand-off is fully
device-resident (no host synchronization in the loop); the same script runs — slowly — on
GENERIC + cpu anywhere.

  python -m hyperdrone.examples.drone_flythrough --drones 1024 --steps 200 --width 64 --height 64
  python -m hyperdrone.examples.drone_flythrough --drones 16 --steps 600 --width 128 --height 128 --video out.mp4
"""
import argparse
import math
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
parser.add_argument("--video", default=None, help="record a video of the onboard views to this path (mp4)")
parser.add_argument("--video-drones", type=int, default=16, help="number of onboard views tiled into the video")
parser.add_argument("--video-fps", type=float, default=30.0, help="target video frame rate; steps are strided to play at real time")
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
    scene = render.load_scene(arguments.model, fidelity="medium")
else:
    scene = build_room(np.random.default_rng(0))

sim = dynamics.Sim(num_drones=arguments.drones, model=arguments.drone_model, device=arguments.device)
renderer = render.Renderer(width=arguments.width, height=arguments.height,
                           num_cameras=arguments.drones, output="rgb", fidelity="medium")
world = env.World(scene, sim, renderer)

print(f"backend={renderer.backend} dynamics={sim.device} drones={arguments.drones} "
      f"device_handoff={'yes' if world._device_handoff else 'no (host cameras)'}")

sampler = env.FreeSpaceSampler(scene, clearance=0.5)
positions = sampler.sample(arguments.drones, seed=0)
world.spawn(positions)

# flythrough flight pattern: hover throttle + a small yaw differential (rotor pattern
# +,-,+,- spins the body around +Z while staying level) + an initial horizontal glide
# velocity that persists (no aero drag). The yaw differential costs a little collective
# thrust, so a minimal P-D altitude hold on all four rotors keeps the drones level at
# their spawn height — panning, gliding onboard views.
rng = np.random.default_rng(1)
hover = sim.parameters["hovering_throttle_relative"]
yaw_rates = rng.uniform(0.015, 0.035, size=arguments.drones) * rng.choice((-1.0, 1.0), size=arguments.drones)
base_actions = ((2.0 * hover - 1.0)[:, None] + yaw_rates[:, None] * np.array([1.0, -1.0, 1.0, -1.0]))
glide_angle = rng.uniform(0.0, 2.0 * np.pi, size=arguments.drones)
glide_speed = rng.uniform(0.2, 0.5, size=arguments.drones)
velocities = np.zeros((arguments.drones, 3), dtype=np.float32)
velocities[:, 0] = np.cos(glide_angle) * glide_speed
velocities[:, 1] = np.sin(glide_angle) * glide_speed
sim.state["linear_velocity"] = velocities
altitude_target = positions[:, 2].copy()


def flythrough_actions():
    position_z = sim.state.numpy("position")[:, 2]
    velocity_z = sim.state.numpy("linear_velocity")[:, 2]
    correction = 0.4 * (altitude_target - position_z) - 0.25 * velocity_z
    return np.clip(base_actions + correction[:, None], -1.0, 1.0).astype(np.float32)

video_writer = None
video_stride = 1
if arguments.video:
    import imageio

    video_drones = min(arguments.video_drones, arguments.drones)
    grid_columns = int(math.ceil(math.sqrt(video_drones)))
    grid_rows = int(math.ceil(video_drones / grid_columns))
    # stride steps so the video plays at real time: sim rate (1/dt) down to the target fps
    video_stride = max(1, round(1.0 / (sim.dt * arguments.video_fps)))
    video_writer = imageio.get_writer(arguments.video, fps=1.0 / (sim.dt * video_stride))

    def record_frame():
        views = renderer.frame()[:video_drones, :, :, :3]
        mosaic = np.zeros((grid_rows * arguments.height, grid_columns * arguments.width, 3), dtype=np.uint8)
        for view_index in range(video_drones):
            row, column = divmod(view_index, grid_columns)
            mosaic[row * arguments.height:(row + 1) * arguments.height,
                   column * arguments.width:(column + 1) * arguments.width] = views[view_index]
        video_writer.append_data(mosaic)

start = time.perf_counter()
for step in range(arguments.steps):
    world.step(flythrough_actions())
    if video_writer is not None and step % video_stride == 0:
        record_frame()
renderer.synchronize()
elapsed = time.perf_counter() - start

frames_per_second = arguments.steps * arguments.drones / elapsed
print(f"{arguments.steps} steps x {arguments.drones} drones in {elapsed:.2f}s "
      f"({arguments.steps / elapsed:.1f} steps/s, {frames_per_second:.0f} rendered frames/s)")

if video_writer is not None:
    video_writer.close()
    print(f"saved {arguments.video}")

if arguments.save:
    renderer.save_image(arguments.save)
    print(f"saved {arguments.save}")
