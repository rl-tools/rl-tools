"""Shared data resolution and download helpers for runnable examples."""

import os
from pathlib import Path

import conta

from hyperdrone import jit


PROCTHOR_SCENE = {"description": "ai2thor-hab/glb/ProcTHOR-Train-1.glb", "hash": "75d1d67a17ecd16bc3a214baa895240da8060df7"}
# body/prop_* assemblies, each a drone table asset (hyperdrone.env.MultiEnvironment(drones=...))
RIGGED_MODELS = {
    "arpl": {"description": "rigged/arpl.glb", "hash": "8398a239fd1cd80df8adab67497baf7be3d8cda4"},
    "crazyflie_brushless": {"description": "rigged/crazyflie_brushless.glb", "hash": "eedb897237c1d77ed8ef0681554d92d06a7c27ca"},
    "crazyflie": {"description": "rigged/crazyflie.glb", "hash": "87d2b8b1d518445872d34906320a76656731902f"},
    "soft": {"description": "rigged/soft.glb", "hash": "fcdf53a791f313b9bb07f775b10b9e4c2bfa0ae9"},
    "x500": {"description": "rigged/x500.glb", "hash": "f6681e7a8b7fa7ef023bedcd795981becb731d0e"},
}
X500_MODEL = RIGGED_MODELS["x500"]


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


def rigged_model_path(name):
    """Return the local GLB of a RIGGED_MODELS entry, downloading and caching it when necessary.

    Resolution order: HYPERDRONE_<NAME>_PATH, an rl-tools checkout's tests/data,
    then the shared conta cache (content-addressed, same cache as the C++ client).
    """
    if name not in RIGGED_MODELS:
        raise KeyError(f"hyperdrone: unknown rigged model {name!r} (one of {sorted(RIGGED_MODELS)})")
    return _test_data_path(f"{name}.glb", RIGGED_MODELS[name], f"HYPERDRONE_{name.upper()}_PATH")


def x500_model_path():
    """Return the local x500 GLB (rigged_model_path("x500"))."""
    return rigged_model_path("x500")
