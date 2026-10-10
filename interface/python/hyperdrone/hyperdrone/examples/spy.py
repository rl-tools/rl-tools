"""Run with `python -m hyperdrone.examples.spy /path/to/local/scene.glb`."""
import argparse
from pathlib import Path

import numpy as np

from hyperdrone.env import EnvConfig, MultiEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    arguments = parser.parse_args()
    if not arguments.scene.is_file():
        parser.error("scene must be an existing local GLB file")

    env = MultiEnvironment(arguments.scene, config=EnvConfig(instances_per_environment=4, width=64, height=64), seed=0)
    try:
        actions = np.zeros((env.total_instances, env.action_dim), dtype=np.float32)
        observations = np.empty((env.total_instances, env.observation_dim), dtype=np.float32)
        env.reset()
        with env.spy() as spy:
            step = 0
            while True:
                env.observe(out=observations)
                if not spy.update():
                    break
                env.step(actions)
                step += 1
                reset = env.terminated()
                if step % env.episode_step_limit == 0:
                    reset[:] = True
                if reset.any():
                    env.reset(reset)
    finally:
        env.close()


if __name__ == "__main__":
    main()
