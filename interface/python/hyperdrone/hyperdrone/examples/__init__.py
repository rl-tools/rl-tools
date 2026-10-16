"""Runnable Hyperdrone examples.

Invoke a module with ``python -m hyperdrone.examples.<name>``.
"""

__all__ = ["procthor_scene_path"]


def procthor_scene_path():
    from .data import procthor_scene_path as resolve
    return resolve()
