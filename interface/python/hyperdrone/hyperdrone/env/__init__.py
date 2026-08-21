"""hyperdrone.env — environment setup: everything between "I have a GLB and a drone
model" and "N drones are flying through it".

The only layer that knows about both rendering and dynamics: free-space sampling for
spawn positions (probe-based, works on every render backend), and the World convenience
wiring sim.step -> camera_bases -> set_cameras -> render (device-resident on
OptiX + CUDA).
"""
from ._multi_environment import EnvConfig, MultiEnvironment
from ._sampling import FreeSpaceSampler
from ._world import World

__all__ = [
    "EnvConfig",
    "FreeSpaceSampler",
    "MultiEnvironment",
    "World",
]
