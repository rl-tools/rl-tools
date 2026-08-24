"""The public surface is locked: every name a subpackage exports is listed here, and
changing the surface means editing this lock in the same commit — deliberate and
reviewable. Additions fail this test just like removals.
"""
import importlib

SURFACE = {
    "hyperdrone.render": [
        "AssetPool",
        "FIDELITY",
        "FreeSpaceSampler",
        "Mesh",
        "Object",
        "ObjectAssembly",
        "Renderer",
        "RendererConfig",
        "Scene",
        "SceneLight",
        "backend",
        "compose_transforms",
        "load_assembly",
        "load_object",
        "load_scene",
        "make_camera",
        "make_transform",
    ],
    "hyperdrone.dynamics": [
        "IDENTITY_MOUNT",
        "MODELS",
        "Sim",
        "SimConfig",
        "resolve_device",
    ],
    "hyperdrone.env": [
        "EnvConfig",
        "MultiEnvironment",
        "ObservationLayout",
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
            f"Sim.{name} was removed — reward/termination live in hyperdrone.env.MultiEnvironment"
        )
