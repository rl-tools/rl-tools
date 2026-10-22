"""The public surface is locked: every name a module exports is listed here, and changing the
surface means editing this lock in the same commit — deliberate and reviewable. Additions fail
this test just like removals.
"""
import importlib
import inspect

import pytest

SURFACE = {
    "hyperdrone": ["Config", "HyperDrone", "ImageObservation", "Rotorcraft", "articulate"],
    "hyperdrone.dynamics": [
        "IDENTITY_MOUNT",
        "MODELS",
        "Sim",
    ],
}


def test_public_surface_is_locked():
    for package, expected in SURFACE.items():
        module = importlib.import_module(package)
        assert sorted(module.__all__) == sorted(expected), (
            f"{package}.__all__ changed — update the surface lock deliberately"
        )


def test_sim_has_no_mdp_surface():
    from hyperdrone.dynamics import Sim
    for name in ("set_compute_mdp", "rewards", "terminated"):
        assert not hasattr(Sim, name), (
            f"Sim.{name} was removed — reward/termination live in hyperdrone.HyperDrone"
        )


@pytest.mark.parametrize("name", ["MultiEnvironment", "EnvConfig"])
def test_removed_env_names_point_to_the_root(name):
    with pytest.raises(ImportError, match="from hyperdrone import Config, HyperDrone"):
        getattr(importlib.import_module("hyperdrone.env"), name)


SIGNATURES = {
    "hyperdrone.dynamics.Sim": ("num_drones",),
    "hyperdrone.dynamics.Sim.reset": ("mask",),
    "hyperdrone.dynamics.Sim.step": ("actions",),
    "hyperdrone.dynamics.Sim.camera_bases": (),
    "hyperdrone.dynamics.Sim.camera_bases_dlpack": (),
    "hyperdrone.Config": (),
    "hyperdrone.HyperDrone": ("scenes",),
    "hyperdrone.HyperDrone.reset": ("mask",),
    "hyperdrone.HyperDrone.step": ("actions",),
    "hyperdrone.HyperDrone.set_instance_parameters": ("overrides", "mask"),
    **{f"hyperdrone.HyperDrone.{name}": () for name in ("observe", "observe_privileged", "observe_imu", "rewards", "terminated", "render", "probe", "spy")},
    "hyperdrone.Rotorcraft": ("assembly",),
    "hyperdrone.articulate": ("assembly",),
}


def _resolve(name):
    parts = name.split(".")
    value = importlib.import_module(parts[0])
    for part in parts[1:]:
        value = getattr(value, part)
    return value


@pytest.mark.parametrize("name,allowed", SIGNATURES.items())
def test_keyword_only_signatures(name, allowed):
    signature = inspect.signature(_resolve(name))
    actual = tuple(parameter.name for parameter in signature.parameters.values() if parameter.name not in ("self", "cls") and parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD))
    assert actual == allowed
    assert all(parameter.kind != parameter.VAR_POSITIONAL for parameter in signature.parameters.values())


@pytest.mark.parametrize("name", ["hyperdrone.dynamics.Sim.camera_bases", "hyperdrone.dynamics.Sim.camera_bases_dlpack"])
def test_camera_aspect_is_required(name):
    aspect = inspect.signature(_resolve(name)).parameters["aspect"]
    assert aspect.kind is inspect.Parameter.KEYWORD_ONLY
    assert aspect.default is inspect.Parameter.empty
