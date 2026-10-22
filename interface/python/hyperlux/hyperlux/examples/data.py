"""Shared data resolution and download helpers for runnable examples."""

import os
from pathlib import Path

import conta

PROCTHOR_SCENE = {"description": "ai2thor-hab/glb/ProcTHOR-Train-1.glb", "hash": "75d1d67a17ecd16bc3a214baa895240da8060df7"}


def override_path(override_variable):
    override = os.environ.get(override_variable)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"hyperlux: {override_variable} does not exist: {path}")
        return path
    return None


def procthor_scene_path():
    """Return the local ProcTHOR scene, downloading and caching it when necessary.

    Resolution order: HYPERLUX_PROCTHOR_PATH, then the shared conta cache.
    """
    override = override_path("HYPERLUX_PROCTHOR_PATH")
    return override if override is not None else conta.resolve(PROCTHOR_SCENE)
