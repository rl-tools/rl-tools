import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hyperdrone import render
from hyperdrone.env import EnvConfig, MultiEnvironment


CONFIG = EnvConfig(instances_per_environment=2, width=16, height=16, fidelity="low",
                   output=("rgb", "semantic_segmentation"), device="cpu")
CLASSES = {0: "Unknown", 7: "Room"}


def room_glb(path, taxonomy="test", classes=None, extent=5):
    vertices = np.array([[x * extent, y * 3, z * extent]
                         for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype="<f4")
    indices = np.array([0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5,
                        0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6,
                        0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3], dtype="<u4")
    scene = {"nodes": [0]}
    node = {"mesh": 0}
    if taxonomy is not None:
        classes = CLASSES if classes is None else classes
        scene["extras"] = {"rl_tools": {"segmentation": {
            "taxonomy": taxonomy, "classes": {name: id for id, name in classes.items()}}}}
        node["extras"] = {"rl_tools": {"segmentation_class": 7}}
    data = vertices.tobytes() + indices.tobytes()
    document = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [scene], "nodes": [node],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(data)}],
        "bufferViews": [{"buffer": 0, "byteLength": vertices.nbytes},
                        {"buffer": 0, "byteOffset": vertices.nbytes, "byteLength": indices.nbytes}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3",
                       "min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
                      {"bufferView": 1, "componentType": 5125, "count": len(indices), "type": "SCALAR"}],
    }
    encoded = json.dumps(document).encode()
    encoded += b" " * (-len(encoded) % 4)
    path.write_bytes(struct.pack("<5I", 0x46546C67, 2, 28 + len(encoded) + len(data),
                                len(encoded), 0x4E4F534A) + encoded +
                     struct.pack("<2I", len(data), 0x004E4942) + data)
    return path


def test_dataset_config():
    assert CONFIG.dataset == "glb"
    assert CONFIG.key() != replace(CONFIG, dataset="procthor").key()
    assert CONFIG.defines()["HYPERDRONE_ENV_DATASET"] == 0
    with pytest.raises(ValueError, match="unknown dataset"):
        replace(CONFIG, dataset="missing")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_taxonomy_and_rotation(tmp_path, device):
    if device == "cuda" and render._backend() != "optix":
        pytest.skip("CUDA Worlds require OptiX")
    first = room_glb(tmp_path / "first.glb")
    second = room_glb(tmp_path / "second.glb", extent=6)
    for scenes in ([first, second, second, first], [second, first, first, second]):
        env = MultiEnvironment(scenes, config=replace(CONFIG, num_environments=2, device=device))
        try:
            assert env.segmentation_taxonomy == "test"
            assert env.segmentation_classes == CLASSES
            assert all(type(id) is int for id in env.segmentation_classes)
            edited = env.segmentation_classes
            edited[7] = "Changed"
            assert env.segmentation_classes == CLASSES
            with pytest.raises(AttributeError):
                env.segmentation_classes = {}
            env.reset()
            np.testing.assert_array_equal(env.semantic_segmentation(), 7)
            env.rotate_scene()
            env.reset()
            np.testing.assert_array_equal(env.semantic_segmentation(), 7)
            assert env.segmentation_taxonomy == "test"
            assert env.segmentation_classes == CLASSES
        finally:
            env.close()
        assert edited[7] == "Changed"


@pytest.mark.parametrize("taxonomy,classes", [("other", CLASSES), ("test", {0: "Unknown", 7: "Floor"})])
def test_conflicting_scenes(tmp_path, taxonomy, classes):
    first = room_glb(tmp_path / "first.glb")
    second = room_glb(tmp_path / "second.glb", taxonomy=taxonomy, classes=classes)
    with pytest.raises(ValueError, match="incompatible segmentation taxonomies.*scene"):
        MultiEnvironment([first, second], config=replace(CONFIG, num_environments=2))


def test_procthor_contract(tmp_path):
    config = replace(CONFIG, dataset="procthor")
    for name in (None, "custom"):
        scene = room_glb(tmp_path / "invalid.glb", taxonomy=name)
        with pytest.raises(ValueError, match="requires segmentation taxonomy 'ai2thor-hab'"):
            MultiEnvironment(scene, config=config)
    scene = room_glb(tmp_path / "procthor.glb", taxonomy="ai2thor-hab")
    env = MultiEnvironment(scene, config=config)
    try:
        assert env.segmentation_taxonomy == "ai2thor-hab"
        assert env.segmentation_classes == CLASSES
    finally:
        env.close()


@pytest.mark.parametrize("dataset", ["glb", "procthor"])
def test_directory_order(tmp_path, dataset):
    later = room_glb(tmp_path / "room10.glb", taxonomy="ai2thor-hab")
    earlier = room_glb(tmp_path / "room2.glb", taxonomy="ai2thor-hab", extent=6)
    env = MultiEnvironment(tmp_path, config=replace(CONFIG, dataset=dataset))
    try:
        for path in (earlier, later):
            env.reset()
            assert env.spawn_points()[0]["scene"] == hashlib.sha1(path.read_bytes()).hexdigest()
            assert env.segmentation_classes == CLASSES
            env.rotate_scene()
    finally:
        env.close()


def test_unnamed_and_nonsemantic(tmp_path):
    scene = room_glb(tmp_path / "unnamed.glb", taxonomy=None)
    env = MultiEnvironment(scene, config=CONFIG)
    try:
        assert env.segmentation_taxonomy == ""
        assert env.segmentation_classes == {}
        env.reset()
        np.testing.assert_array_equal(env.semantic_segmentation(), 0)
    finally:
        env.close()
    scenes = [room_glb(tmp_path / "one.glb"), room_glb(tmp_path / "two.glb", taxonomy="other")]
    for output in ("rgb", ("rgb", "instance_segmentation")):
        env = MultiEnvironment(scenes, config=replace(CONFIG, output=output, dataset="procthor"))
        try:
            assert env.segmentation_taxonomy == ""
            assert env.segmentation_classes == {}
        finally:
            env.close()


def test_custom_world_metadata(tmp_path):
    source = (Path(__file__).parent / "user_spec_header.h").read_text()
    source = source.replace("        static constexpr TI MAX_DRONES = 3;", """
        using CAMERA_OUTPUTS = rl_tools::rl::environments::hyperdrone::CameraOutputs<rl_tools::rendering::raytracing::outputs::List<rl_tools::rendering::raytracing::outputs::RGB<rl_tools::rendering::raytracing::outputs::format::Float32>, rl_tools::rendering::raytracing::outputs::Segmentation<rl_tools::rendering::raytracing::outputs::format::UInt32, rl_tools::rendering::raytracing::outputs::labels::Semantic>, rl_tools::rendering::raytracing::outputs::Probes>>;
""")
    header = tmp_path / "semantic.h"
    header.write_text(source)
    from hyperdrone._jit import source_root
    asset = source_root() / "tests" / "data" / "x500.glb"
    if not asset.exists():
        pytest.skip("no x500 test asset")
    scene = room_glb(tmp_path / "room.glb", taxonomy="ai2thor-hab")
    env = MultiEnvironment(scene, drone_asset=asset,
                           config=EnvConfig(spec_header=str(header), dataset="procthor", device="cpu"))
    try:
        assert env.segmentation_taxonomy == "ai2thor-hab"
        assert env.segmentation_classes == CLASSES
        env.reset()
        assert set(np.unique(env.semantic_segmentation())).issubset({0, 7, 0xFFFFFFFF})
    finally:
        env.close()
