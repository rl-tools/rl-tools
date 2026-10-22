#!/usr/bin/env python3
"""Native smoke for an installed hyperlux + hyperdrone pair: a render, a CPU dynamics step, a
rotorcraft articulation and, given a scene, a CPU environment reset, step and render. Shared by
the sdist test, the publish rehearsal and CI, so one body gates checkouts and release artifacts
alike."""
import argparse

import numpy as np

import hyperlux
from hyperlux import _jit as jit
from hyperlux.camera import CameraSpec, Depth
import hyperdrone
from hyperdrone import Config, HyperDrone, Rotorcraft, articulate, dynamics


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


def render():
    scene = hyperlux.Scene()
    wall = hyperlux.Object(name="wall")
    wall.add_mesh(hyperlux.Mesh(np.array([[2, -1, -1], [2, 1, -1], [2, 1, 1], [2, -1, 1]], dtype=np.float32), np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)))
    scene.add_object(wall)
    renderer = hyperlux.HyperLux(width=16, height=16, cameras=(CameraSpec("front", (Depth(),)),), fidelity="low")
    renderer.init(scene)
    renderer.set_cameras(renderer.camera(position=(0, 0, 0), look_at=(1, 0, 0)))
    renderer.render()
    assert abs(renderer.outputs[Depth()].read()[0, 8, 8] - 2.0) < 1e-2


def step_dynamics():
    sim = dynamics.Sim(2, seed=1)
    sim.step(np.zeros((2, sim.action_dim), dtype=np.float32))
    assert sim.observe().shape == (2, sim.observation_dim)


def articulate_rotorcraft():
    asset = hyperlux.ObjectAssembly()
    asset.add_part(quad(0, 0, 0.1, "body"), pose())
    asset.add_part(quad(0.1, 0.1, 0.05, "prop_0"), pose(z=0.05))
    asset.add_part(quad(-0.1, 0.1, 0.05, "prop_1"), pose(z=0.05))
    articulate(asset)
    rotorcraft = Rotorcraft(asset)
    assert asset.num_coordinates == 2 and rotorcraft.scale == 1
    assert rotorcraft.coordinates(np.array([0.5, 0.25], dtype=np.float32)).shape == (2,)


def run_environment(scene):
    environment = HyperDrone([scene], config=Config(num_worlds=1, instances_per_world=2, width=16, height=16, fidelity="low", device="cpu"), seed=1)
    try:
        environment.reset()
        environment.step(np.zeros((environment.total_instances, environment.action_dim), dtype=np.float32))
        assert environment.observe_privileged().shape == (environment.total_instances, environment.observation_dim_privileged)
        environment.render()
    finally:
        environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendored", action="store_true", help="require the installed package's vendored rl-tools tree (a distribution, not a checkout)")
    parser.add_argument("--scene", help="a .glb scene for the environment case")
    args = parser.parse_args()
    root = jit.source_root()
    if args.vendored:
        assert "_vendor" in str(root), f"expected the vendored tree, resolved {root}"
    assert hyperdrone.__version__.split(".post")[0] == hyperlux.__version__
    dataset = hyperlux.datasets.procthor.Train200()
    assert len(dataset) == 200
    assert dataset["ProcTHOR-Train-1"].reference == "conta:07b9d8e44f3891573b6292f8cc47bf7725a43b89"
    render()
    step_dynamics()
    articulate_rotorcraft()
    if args.scene is not None:
        run_environment(args.scene)
    print(f"smoke: OK (sources {root}, environment {'checked' if args.scene is not None else 'skipped: no --scene'})")


if __name__ == "__main__":
    main()
