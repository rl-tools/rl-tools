from hyperlux.camera import CameraSpec, RGB, Depth, Segmentation, Normals, Flow, FlowStatus, Probes
import numpy as np
import pytest

import hyperlux
from test_material_extensions import write_material_glb


@pytest.fixture
def environment_assets(tmp_path):
    material = {"pbrMetallicRoughness": {"baseColorFactor": [0.2, 0.1, 0.05, 1],
                                        "metallicFactor": 0, "roughnessFactor": 1}}
    environment = {"mode": "solid", "ambient": [0.6, 0.3, 0.2], "background": [0.02, 0.04, 0.08]}
    plain, authored = tmp_path / "plain.glb", tmp_path / "authored.glb"
    write_material_glb(plain, [material])
    write_material_glb(authored, [material], scene_extras={"rl_tools": {"environment": environment}})
    return plain, authored, environment


def frames(scene, fidelity):
    renderer = hyperlux.HyperLux(width=8, height=8, num_probes=0, cameras=(CameraSpec("camera", outputs=(RGB(format="rgba8"),)),), fidelity=fidelity)
    renderer.init(scene)
    output = []
    for look_at in [(1, 0.4, 0.3), (-1, 0, 0)]:
        camera = renderer.camera(position=(0, 0, 0), look_at=look_at, fov=45)
        renderer.set_cameras(camera[None])
        renderer.render()
        output.append(renderer.outputs[RGB(format="rgba8")].read().copy())
    return np.stack(output)


@pytest.mark.parametrize("fidelity", ["low", "medium", "high", "veryhigh"])
def test_scene_import_matches_explicit_environment(environment_assets, fidelity):
    plain, authored, environment = environment_assets
    expected = hyperlux.load_scene(plain, fidelity=fidelity)
    expected.set_environment(ambient=environment["ambient"], background=environment["background"])
    actual = hyperlux.load_scene(authored, fidelity=fidelity)
    np.testing.assert_allclose(frames(actual, fidelity), frames(expected, fidelity), atol=1)


def test_explicit_environment_overrides_import(environment_assets):
    plain, authored, _ = environment_assets
    scenes = [hyperlux.load_scene(path) for path in (plain, authored)]
    for scene in scenes:
        scene.set_environment(ambient=(0.25, 0.5, 0.75), background=(0.1, 0.2, 0.3))
    np.testing.assert_allclose(frames(scenes[0], "high"), frames(scenes[1], "high"), atol=1)


@pytest.mark.parametrize("loader", ["object", "assembly"])
def test_asset_import_preserves_parent_environment(environment_assets, loader):
    plain, authored, _ = environment_assets
    results = []
    for path in (plain, authored):
        scene = hyperlux.Scene()
        scene.set_environment(ambient=(0.15, 0.25, 0.35), background=(0.2, 0.1, 0.05))
        asset = getattr(hyperlux, f"load_{loader}")(path)
        getattr(scene, f"add_{loader}")(asset)
        results.append(frames(scene, "high"))
    np.testing.assert_allclose(results[0], results[1], atol=1)
