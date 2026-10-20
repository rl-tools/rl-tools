"""A spinning hinge and a block sliding along it: joint coordinates are interpolated across the
shutter before the kinematics, so the hinge keeps its full revolutions in the motion blur.
Run with python -m hyperdrone.examples.articulation; writes articulation.gif.
"""
import imageio.v2 as imageio
import numpy as np

from hyperdrone import render
from hyperdrone.camera import CameraSpec, Depth, RGB


def rectangle(half_y, half_z, color):
    vertices = np.array([[0, -half_y, -half_z], [0, half_y, -half_z],
                         [0, half_y, half_z], [0, -half_y, half_z]], dtype=np.float32)
    obj = render.Object()
    obj.add_mesh(render.Mesh(vertices, np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32), color=color))
    return obj


def main():
    identity = render.make_transform()
    asset = render.ObjectAssembly()
    hinge = asset.add_part(rectangle(0.8, 0.08, (1, 0.2, 0)), identity)
    block = asset.add_part(rectangle(0.15, 0.15, (0, 0.8, 1)), render.make_transform(position=(0, 0.5, 0)), parent=hinge)
    asset.set_joint(hinge, "revolute", axis=(1, 0, 0), coordinate=0)
    asset.set_joint(block, "prismatic", axis=(0, 1, 0), coordinate=1)
    pool = render.AssetPool()
    handle = pool.add_assembly(asset)
    scene = render.Scene()
    scene.add_object(rectangle(10, 10, (0.1, 0.1, 0.1)), render.make_transform(position=(8, 0, 0)))
    renderer = render.Renderer(width=128, height=128, fidelity="low", cameras=(CameraSpec("front", (RGB("rgba8"),)),),
                               num_probes=0, num_overlays=1, max_overlay_instances=2,
                               max_overlays_per_camera=1, motion_blur_samples=8,
                               dynamic_motion_blur=True)
    renderer.init(scene, asset_pool=pool)
    renderer.attach(0, 0)
    base = render.make_transform(position=(3, 0, 0))
    placement = renderer.spawn(0, handle, render.Pose(base))
    renderer.set_cameras(renderer.camera(position=(0, 0, 0), look_at=(1, 0, 0))[None])
    frames = []
    for frame in range(90):
        time_open, time_close = frame / 30, (frame + 0.75) / 30
        coordinates_open = [12 * time_open, 0.3 * np.sin(4 * time_open)]
        coordinates_close = [12 * time_close, 0.3 * np.sin(4 * time_close)]
        renderer.set_pose(placement, render.Pose(base), render.Pose(base), coordinates=coordinates_open, coordinates_close=coordinates_close)
        renderer.update()
        renderer.render()
        frames.append(renderer.outputs[RGB("rgba8")].read()[0, ..., :3])
    imageio.mimsave("articulation.gif", frames, duration=1000 / 30, loop=0)


if __name__ == "__main__":
    main()
