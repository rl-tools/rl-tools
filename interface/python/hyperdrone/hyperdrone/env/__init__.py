"""hyperdrone.env — the RL environment: the C++ rl_tools hyperdrone::MultiEnvironment<World>
driven through the exact batch verbs the C++ training targets use.

All environment semantics (reset, reward, termination, scene rotation, observation
composition) live on the C++ side; this package marshals tensors and nothing else. For
manual composition of the render and dynamics packages (no MDP), see
examples/drone_flythrough.py.
"""
from ._multi_environment import EnvConfig, MultiEnvironment, ObservationLayout
from ..camera import ImageObservation

__all__ = [
    "EnvConfig",
    "MultiEnvironment",
    "ObservationLayout",
    "ImageObservation",
    "Rotorcraft",
]


def __getattr__(name):
    if name == "Rotorcraft":
        from ._rotorcraft import Rotorcraft
        return Rotorcraft
    raise AttributeError(f"module 'hyperdrone.env' has no attribute {name}")
