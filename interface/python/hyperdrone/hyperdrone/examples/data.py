"""Shared data resolution and download helpers for runnable examples."""

import os
from pathlib import Path

from hyperdrone import conta, jit


PROCTHOR_SCENE = {"description": "ai2thor-hab/glb/ProcTHOR-Train-1.glb", "hash": "75d1d67a17ecd16bc3a214baa895240da8060df7"}
X500_MODEL = {"description": "rigged/x500.glb", "hash": "f6681e7a8b7fa7ef023bedcd795981becb731d0e"}


def _override_path(override_variable):
    override = os.environ.get(override_variable)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"hyperdrone: {override_variable} does not exist: {path}"
            )
        return path

    return None


def _test_data_path(filename, conta_entry, override_variable):
    override = _override_path(override_variable)
    if override is not None:
        return override

    checkout_path = jit.source_root() / "tests" / "data" / filename
    if checkout_path.is_file():
        return checkout_path

    return conta.resolve(conta_entry)


def procthor_scene_path():
    """Return the local ProcTHOR scene, downloading and caching it when necessary.

    Resolution order: HYPERDRONE_PROCTHOR_PATH, then the shared conta cache.
    """
    override = _override_path("HYPERDRONE_PROCTHOR_PATH")
    return override if override is not None else conta.resolve(PROCTHOR_SCENE)


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
