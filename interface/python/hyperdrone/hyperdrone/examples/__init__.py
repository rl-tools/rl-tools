"""Runnable HyperDrone examples; the renderer-only ones live in hyperlux.examples.

Invoke a module with ``python -m hyperdrone.examples.<name>``.
"""

__all__ = ["procthor_scene_path", "rigged_model_path"]


def procthor_scene_path():
    from .data import procthor_scene_path as resolve
    return resolve()


def rigged_model_path(name):
    from .data import rigged_model_path as resolve
    return resolve(name)
