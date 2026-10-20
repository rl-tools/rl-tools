"""Run `ipython -i -m hyperdrone.examples.articulation_spy`, then edit the mechanism:

renderer.set_pose(placement, render.Pose(base), coordinates=[0.8, 0.3])
second = renderer.spawn(0, handle, render.Pose(render.make_transform(position=(2, 0, 1))))
renderer.despawn(second)
spy.close()
"""
from hyperdrone import render

rod = render.primitives.box(size=(0.12, 1.6, 0.12), color=(1, 0.2, 0))
cube = render.primitives.box(size=(0.3, 0.3, 0.3), color=(0, 0.8, 1))
asset = render.ObjectAssembly()
hinge = asset.add_part(rod, render.make_transform())
block = asset.add_part(cube, render.make_transform(position=(0, 0.5, 0)), parent=hinge)
asset.set_joint(hinge, "revolute", axis=(1, 0, 0), coordinate=0)
asset.set_joint(block, "prismatic", axis=(0, 1, 0), coordinate=1)
pool = render.AssetPool()
handle = pool.add_assembly(asset)
scene = render.Scene()
scene.set_environment(background=(0.035, 0.045, 0.06))
renderer = render.Renderer(width=256, height=256, fidelity="low", num_probes=0,
                           num_overlays=1, max_overlay_instances=8, max_overlays_per_camera=1)
renderer.init(scene, asset_pool=pool)
base = render.make_transform(position=(2, 0, 0))
placement = renderer.spawn(0, handle, render.Pose(base))
renderer.attach(0, 0)
renderer.set_cameras(renderer.camera(position=(0, 0, 0), look_at=(1, 0, 0))[None])
spy = renderer.spy(interactive=True)
