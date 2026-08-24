"""hyperdrone.env — the RL environment: the C++ rl_tools hyperdrone::MultiEnvironment<World>
driven through the exact batch verbs the C++ training targets use.

All environment semantics (reset, reward, termination, scene rotation, observation
composition) live on the C++ side; this package marshals tensors and nothing else. For
manual composition of the render and dynamics packages (no MDP), see
examples/drone_flythrough.py.
"""
from ._multi_environment import EnvConfig, MultiEnvironment, ObservationLayout

__all__ = [
    "EnvConfig",
    "MultiEnvironment",
    "ObservationLayout",
]
