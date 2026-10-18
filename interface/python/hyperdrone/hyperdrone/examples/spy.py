"""Watch RAPTOR hold position: `python -m hyperdrone.examples.spy [scene.glb]`."""
import argparse
from pathlib import Path
import time

import numpy as np

from hyperdrone.env import EnvConfig, HyperDrone
from hyperdrone.examples.data import RIGGED_MODELS, procthor_scene_path


def position_targets(env):
    return np.asarray([
        [*point["scene_translation"], np.arctan2(point["scene_yaw"][1], point["scene_yaw"][0])]
        for point in env.spawn_points()
    ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", nargs="?", type=Path, help="local GLB (default: cached/downloaded ProcTHOR scene)")
    arguments = parser.parse_args()
    if arguments.scene is not None and not arguments.scene.is_file():
        parser.error("scene must be an existing local GLB file")

    env = HyperDrone(
        arguments.scene or procthor_scene_path(),
        config=EnvConfig(instances_per_world=4, width=64, height=64, preset="x500_fpv", control="position"),
        drone_asset=RIGGED_MODELS["x500"],
        parameters={"dynamics": {"mdp": {"init": {"guidance": 1}}}},
        seed=0,
    )
    try:
        env.reset()
        targets = position_targets(env)
        with env.spy() as spy:
            while True:
                env.render()
                if not spy.update():
                    break
                env.step(targets)
                time.sleep(env.dt)
    finally:
        env.close()


if __name__ == "__main__":
    main()
