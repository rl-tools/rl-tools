"""The public surface is locked: every name a module exports is listed here, and changing the
surface means editing this lock in the same commit — deliberate and reviewable. Additions fail
this test just like removals.
"""
import importlib
import inspect

import pytest

SURFACE = {
    "hyperlux": [
        "AGENT_SEGMENTATION_CLASS",
        "AssetPool",
        "FIDELITY",
        "FlowStatus",
        "ClearanceSampler",
        "Config",
        "HyperLux",
        "Mesh",
        "Object",
        "ObjectAssembly",
        "OverlayPlacement",
        "PlacementBatch",
        "Pose",
        "Scene",
        "SceneLight",
        "backend",
        "compose_transforms",
        "datasets",
        "load_assembly",
        "load_object",
        "load_scene",
        "make_camera",
        "make_transform",
        "primitives",
        "set_verbosity",
        "verbosity",
    ],
    "hyperlux.camera": ["CameraSpec", "RGB", "Depth", "Segmentation", "Normals", "Flow", "FlowStatus", "Probes", "OutputDescriptor"],
    "hyperlux.datasets": ["procthor"],
    "hyperlux.datasets.procthor": ["Train200"],
}


def test_public_surface_is_locked():
    for package, expected in SURFACE.items():
        module = importlib.import_module(package)
        assert sorted(module.__all__) == sorted(expected), (
            f"{package}.__all__ changed — update the surface lock deliberately"
        )


SIGNATURES = {
    "hyperlux.HyperLux": ("width", "height"),
    "hyperlux.Config": (),
    "hyperlux.OverlayPlacement": (),
    "hyperlux.Pose": ("base",),
    "hyperlux.ClearanceSampler": ("scene",),
    "hyperlux.ClearanceSampler.sample": ("count",),
    "hyperlux.make_camera": ("position", "look_at"),
    "hyperlux.load_scene": ("path",),
    "hyperlux.load_object": ("path",),
    "hyperlux.load_assembly": ("path",),
    "hyperlux.HyperLux.init": ("scene",),
    "hyperlux.HyperLux.camera": ("position", "look_at"),
    "hyperlux.HyperLux.generate_cameras": (),
    "hyperlux.HyperLux.set_cameras": ("cameras",),
    "hyperlux.HyperLux.set_motion_blur_cameras": ("cameras_open", "cameras_close"),
    "hyperlux.HyperLux.spawn": ("overlay", "asset", "pose"),
    "hyperlux.HyperLux.set_pose": ("placement", "pose", "close"),
    "hyperlux.HyperLux.placement_batch": ("placements",),
    **{f"hyperlux.HyperLux.{name}": () for name in ("render", "probe", "spy")},
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


def test_camera_aspect_is_required():
    aspect = inspect.signature(_resolve("hyperlux.make_camera")).parameters["aspect"]
    assert aspect.kind is inspect.Parameter.KEYWORD_ONLY
    assert aspect.default is inspect.Parameter.empty


def test_native_options_are_keyword_only():
    import numpy as np
    import hyperlux

    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    calls = (
        lambda: hyperlux.Mesh(vertices, faces, (1, 0, 0)),
        lambda: hyperlux.Object("object", 7),
        lambda: hyperlux.Scene().set_environment((0, 0, 0), (0, 0, 0)),
        lambda: hyperlux.SceneLight.point((0, 0, 0), (1, 1, 1), (1, 0, 0)),
        lambda: hyperlux.SceneLight.spot((0, 0, 0), (1, 0, 0), (1, 1, 1), 0.9, 0.8),
        lambda: hyperlux.Object().remap_segmentation_classes({}, "custom", {}),
        lambda: hyperlux.ObjectAssembly().remap_segmentation_classes({}, "custom", {}),
        lambda: hyperlux.AssetPool().remap_segmentation_classes({}, "custom", {}),
    )
    for call in calls:
        with pytest.raises(TypeError):
            call()
