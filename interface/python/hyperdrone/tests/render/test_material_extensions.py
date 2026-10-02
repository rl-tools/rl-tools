import json
import struct

import numpy as np
import pytest

from hyperdrone import render
from hyperdrone.render import _io


def write_material_glb(path, materials, texture=None, material_indices=(0,), nested=False):
    positions = np.array([[2, -10, -10], [2, 10, -10], [2, 10, 10], [2, -10, 10]], dtype="<f4")
    indices = np.array([0, 2, 1, 0, 3, 2], dtype="<u2")
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="<f4")
    binary = bytearray()
    views = []

    def append(data):
        index = len(views)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
        binary.extend(data)
        binary.extend(b"\0" * (-len(binary) % 4))
        return index

    position_view, index_view, uv_view = (append(data.tobytes()) for data in (positions, indices, uv))
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(material_indices)))}],
        "nodes": [{"mesh": i, "translation": [4 * (len(material_indices) - i - 1), 0, 0]}
                  for i in range(len(material_indices))],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 2},
                                    "indices": 1, "material": index}]} for index in material_indices],
        "materials": materials,
        "bufferViews": views,
        "accessors": [
            {"bufferView": position_view, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": [2, -10, -10], "max": [2, 10, 10]},
            {"bufferView": index_view, "componentType": 5123, "count": 6, "type": "SCALAR"},
            {"bufferView": uv_view, "componentType": 5126, "count": 4, "type": "VEC2"},
        ],
        "extensionsUsed": sorted({extension for material in materials for extension in material.get("extensions", {})}),
    }
    if nested:
        gltf["nodes"].append({"children": gltf["scenes"][0]["nodes"]})
        gltf["scenes"][0]["nodes"] = [len(gltf["nodes"]) - 1]
    if texture is not None:
        image_path = path.with_suffix(".png")
        _io.write_png(image_path, np.array([[texture]], dtype=np.uint8))
        image_view = append(image_path.read_bytes())
        gltf["images"] = [{"bufferView": image_view, "mimeType": "image/png"}]
        gltf["textures"] = [{"source": 0}]
    gltf["buffers"] = [{"byteLength": len(binary)}]
    encoded = json.dumps(gltf).encode()
    encoded += b" " * (-len(encoded) % 4)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary))
                     + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
                     + struct.pack("<II", len(binary), 0x004E4942) + binary)


def center_pixel(path, fidelity, loader="scene"):
    if loader == "scene":
        scene = render.load_scene(path, fidelity=fidelity)
    else:
        scene = render.Scene()
        asset = getattr(render, f"load_{loader}")(path, fidelity=fidelity)
        getattr(scene, f"add_{loader}")(asset)
    renderer = render.Renderer(width=8, height=8, num_cameras=1, num_probes=0,
                               output="rgb", fidelity=fidelity)
    renderer.init(scene)
    camera = renderer.camera(position=(0, 0, 0), look_at=(1, 0.4, 0.3), fov=45)
    renderer.set_cameras(camera[None])
    renderer.render()
    return renderer.frame()[0, 4, 4, :3].astype(float)


def srgb(linear):
    linear = np.clip(linear, 0, 1)
    return 255 * np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


@pytest.mark.parametrize("fidelity", ["high", "veryhigh"])
@pytest.mark.parametrize("strength", [None, "default", 0, 1, 4])
@pytest.mark.parametrize("textured", [False, True])
def test_emissive_strength_scales_linear_emission(tmp_path, fidelity, strength, textured):
    material = {
        "pbrMetallicRoughness": {"baseColorFactor": [0, 0, 0, 1],
                               "metallicFactor": 1, "roughnessFactor": 1},
        "emissiveFactor": [0.4, 0.2, 0.1],
    }
    if strength is not None:
        extension = {} if strength == "default" else {"emissiveStrength": strength}
        material["extensions"] = {"KHR_materials_emissive_strength": extension}
    expected = np.array(material["emissiveFactor"]) * (1 if strength in (None, "default") else strength)
    texture = (128, 192, 64, 255) if textured else None
    if textured:
        material["emissiveTexture"] = {"index": 0}
        values = np.asarray(texture[:3]) / 255
        expected *= np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    path = tmp_path / "emissive.glb"
    write_material_glb(path, [material], texture)
    np.testing.assert_allclose(center_pixel(path, fidelity), srgb(expected), atol=1.1)


@pytest.mark.parametrize("name", [None, "shared_name"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("loader", ["scene", "object", "assembly"])
def test_emissive_strength_material_association(tmp_path, name, nested, loader):
    materials = [{
        "pbrMetallicRoughness": {"baseColorFactor": [0, 0, 0, 1],
                               "metallicFactor": 1, "roughnessFactor": 1},
        "emissiveFactor": [0.1, 0.05, 0.025],
        "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": strength}},
    } for strength in (0, 4, 0)]
    if name is not None:
        for material in materials:
            material["name"] = name
    path = tmp_path / "materials.glb"
    write_material_glb(path, materials, material_indices=(2, 1), nested=nested)
    np.testing.assert_allclose(center_pixel(path, "high", loader), srgb([0.4, 0.2, 0.1]), atol=1.1)
