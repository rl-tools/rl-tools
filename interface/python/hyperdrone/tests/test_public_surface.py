"""The public surface is locked: every name a subpackage exports is listed here, and
changing the surface means editing this lock in the same commit — deliberate and
reviewable. Additions fail this test just like removals.
"""
import importlib
import inspect

import pytest

SURFACE = {
    "hyperdrone": ["HyperDrone", "Rotorcraft", "articulate", "set_verbosity", "verbosity"],
    "hyperdrone.camera": ["CameraSpec", "RGB", "Depth", "Segmentation", "Normals", "Flow", "FlowStatus", "Probes", "ImageObservation", "OutputDescriptor"],
    "hyperdrone.render": [
        "AGENT_SEGMENTATION_CLASS",
        "AssetPool",
        "FIDELITY",
        "FlowStatus",
        "ClearanceSampler",
        "Mesh",
        "Object",
        "ObjectAssembly",
        "OverlayPlacement",
        "PlacementBatch",
        "Pose",
        "Renderer",
        "RendererConfig",
        "Scene",
        "SceneLight",
        "compose_transforms",
        "datasets",
        "load_assembly",
        "load_object",
        "load_scene",
        "make_camera",
        "make_transform",
        "primitives",
    ],
    "hyperdrone.render.datasets": ["procthor"],
    "hyperdrone.render.datasets.procthor": ["Train200"],
    "hyperdrone.dynamics": [
        "IDENTITY_MOUNT",
        "MODELS",
        "Sim",
    ],
    "hyperdrone.env": [
        "EnvConfig",
        "ImageObservation",
        "HyperDrone",
        "ObservationLayout",
    ],
}


def test_public_surface_is_locked():
    for package, expected in SURFACE.items():
        module = importlib.import_module(package)
        assert sorted(module.__all__) == sorted(expected), (
            f"{package}.__all__ changed — update the surface lock deliberately"
        )


def test_top_level_hyperdrone_is_the_environment():
    import hyperdrone
    from hyperdrone.env import HyperDrone
    assert hyperdrone.HyperDrone is HyperDrone


def test_sim_has_no_mdp_surface():
    from hyperdrone.dynamics import Sim
    for name in ("set_compute_mdp", "rewards", "terminated"):
        assert not hasattr(Sim, name), (
            f"Sim.{name} was removed — reward/termination live in hyperdrone.env.HyperDrone"
        )


SIGNATURES = {
    "hyperdrone.render.Renderer": ("width", "height"),
    "hyperdrone.render.RendererConfig": (),
    "hyperdrone.render.OverlayPlacement": (),
    "hyperdrone.render.Pose": ("base",),
    "hyperdrone.render.ClearanceSampler": ("scene",),
    "hyperdrone.render.ClearanceSampler.sample": ("count",),
    "hyperdrone.render.make_camera": ("position", "look_at"),
    "hyperdrone.render.load_scene": ("path",),
    "hyperdrone.render.load_object": ("path",),
    "hyperdrone.render.load_assembly": ("path",),
    "hyperdrone.render.Renderer.init": ("scene",),
    "hyperdrone.render.Renderer.camera": ("position", "look_at"),
    "hyperdrone.render.Renderer.generate_cameras": (),
    "hyperdrone.render.Renderer.set_cameras": ("cameras",),
    "hyperdrone.render.Renderer.set_motion_blur_cameras": ("cameras_open", "cameras_close"),
    "hyperdrone.render.Renderer.spawn": ("overlay", "asset", "pose"),
    "hyperdrone.render.Renderer.set_pose": ("placement", "pose", "close"),
    "hyperdrone.render.Renderer.placement_batch": ("placements",),
    **{f"hyperdrone.render.Renderer.{name}": () for name in ("render", "probe", "spy")},
    "hyperdrone.dynamics.Sim": ("num_drones",),
    "hyperdrone.dynamics.Sim.reset": ("mask",),
    "hyperdrone.dynamics.Sim.step": ("actions",),
    "hyperdrone.dynamics.Sim.camera_bases": (),
    "hyperdrone.dynamics.Sim.camera_bases_dlpack": (),
    "hyperdrone.env.EnvConfig": (),
    "hyperdrone.env.ObservationLayout": (),
    "hyperdrone.env.HyperDrone": ("scenes",),
    "hyperdrone.env.HyperDrone.reset": ("mask",),
    "hyperdrone.env.HyperDrone.step": ("actions",),
    "hyperdrone.env.HyperDrone.set_instance_parameters": ("overrides", "mask"),
    **{f"hyperdrone.env.HyperDrone.{name}": () for name in ("observe", "observe_privileged", "observe_imu", "rewards", "terminated", "render", "probe", "spy")},
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


@pytest.mark.parametrize("name", ["hyperdrone.render.make_camera", "hyperdrone.dynamics.Sim.camera_bases", "hyperdrone.dynamics.Sim.camera_bases_dlpack"])
def test_camera_aspect_is_required(name):
    aspect = inspect.signature(_resolve(name)).parameters["aspect"]
    assert aspect.kind is inspect.Parameter.KEYWORD_ONLY
    assert aspect.default is inspect.Parameter.empty


def test_native_options_are_keyword_only():
    import numpy as np
    from hyperdrone import render

    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    calls = (
        lambda: render.Mesh(vertices, faces, (1, 0, 0)),
        lambda: render.Object("object", 7),
        lambda: render.Scene().set_environment((0, 0, 0), (0, 0, 0)),
        lambda: render.SceneLight.point((0, 0, 0), (1, 1, 1), (1, 0, 0)),
        lambda: render.SceneLight.spot((0, 0, 0), (1, 0, 0), (1, 1, 1), 0.9, 0.8),
        lambda: render.Object().remap_segmentation_classes({}, "custom", {}),
        lambda: render.ObjectAssembly().remap_segmentation_classes({}, "custom", {}),
        lambda: render.AssetPool().remap_segmentation_classes({}, "custom", {}),
    )
    for call in calls:
        with pytest.raises(TypeError):
            call()
