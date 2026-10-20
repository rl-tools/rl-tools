"""Removed: the environment lives at the package root; importing the names the paper appendix uses explains the rename."""


def __getattr__(name):
    if name in ("MultiEnvironment", "EnvConfig"):
        raise ImportError("hyperdrone: hyperdrone.env was removed; use 'from hyperdrone import Config, HyperDrone' (MultiEnvironment is now HyperDrone, EnvConfig is now Config, and its num_environments and instances_per_environment are now num_worlds and instances_per_world)")
    raise AttributeError(f"module 'hyperdrone.env' has no attribute {name}")
