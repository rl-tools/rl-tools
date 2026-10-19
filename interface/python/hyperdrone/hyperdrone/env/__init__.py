"""hyperdrone.env — the RL environment: the C++ rl_tools rl::environments::HyperDrone<World>
driven through the exact batch verbs the C++ training targets use.

All environment semantics (reset, reward, termination, scene rotation, observation
composition) live on the C++ side; this package marshals tensors and nothing else. For
manual composition of the render and dynamics packages (no MDP), see
examples/drone_flythrough.py.
"""
from ._hyperdrone import EnvConfig, HyperDrone, ObservationLayout
from ..camera import ImageObservation

__all__ = [
    "EnvConfig",
    "HyperDrone",
    "ObservationLayout",
    "ImageObservation",
]


def __getattr__(name):
    if name == "MultiEnvironment":
        raise ImportError("MultiEnvironment is deprecated; use 'from hyperdrone import HyperDrone' instead.")
    raise AttributeError(f"module 'hyperdrone.env' has no attribute {name}")
