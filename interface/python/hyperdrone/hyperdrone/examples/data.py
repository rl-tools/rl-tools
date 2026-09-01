"""Shared data resolution and download helpers for runnable examples."""

import os
from pathlib import Path

from hyperdrone import conta, jit


PROCTHOR_SCENE = {"description": "ai2thor-hab/glb/ProcTHOR-Train-1.glb", "hash": "7f1c9129532798e0b63bc41edb6b4c09251cf8a0"}
X500_MODEL = {"description": "rigged/x500.glb", "hash": "f6681e7a8b7fa7ef023bedcd795981becb731d0e"}


def _test_data_path(filename, conta_entry, override_variable):
    override = os.environ.get(override_variable)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"hyperdrone: {override_variable} does not exist: {path}"
            )
        return path

    checkout_path = jit.source_root() / "tests" / "data" / filename
    if checkout_path.is_file():
        return checkout_path

    return conta.resolve(conta_entry)


def procthor_scene_path():
    """Return the local ProcTHOR scene, downloading and caching it when necessary.

    Resolution order: HYPERDRONE_PROCTHOR_PATH, an rl-tools checkout's tests/data,
    then the shared conta cache (content-addressed, same cache as the C++ client).
    """
    return _test_data_path(
        "ProcTHOR-Train-1.glb",
        PROCTHOR_SCENE,
        "HYPERDRONE_PROCTHOR_PATH",
    )


def x500_model_path():
    """Return the local x500 GLB, downloading and caching it when necessary.

    Resolution order: HYPERDRONE_X500_PATH, an rl-tools checkout's tests/data,
    then the shared conta cache (content-addressed, same cache as the C++ client).
    """
    return _test_data_path(
        "x500.glb",
        X500_MODEL,
        "HYPERDRONE_X500_PATH",
    )
