"""Generates drone.glb: a quadrotor with 5 glTF scene-root nodes (body, prop_0..prop_3).

The assembly loader (rl_tools::load(device, ObjectAssembly&, path)) turns each scene-root
node into one articulated part with the node origin as its pivot, so each prop node sits
at its hub and spins about its local Z. Geometry is authored in FLU (x forward, y left,
z up) and swizzled to glTF Y-up on emit; the renderer swizzles back at load time.
"""
import json
import math
import struct
import os

SPAN = 0.075            # hub offset in +-x and +-y (FLU, meters)
BODY_HALF = (0.05, 0.03, 0.015)
ARM_HALF = (0.053, 0.006, 0.004)
ARM_Z = 0.008
POD_HALF = (0.008, 0.008, 0.010)
POD_Z = 0.012
PROP_Z = 0.025          # prop plane height above body center (node origin z)
BLADE_HALF = (0.055, 0.007, 0.0015)
CAP_HALF = (0.007, 0.007, 0.0025)

BODY_COLOR = (0.42, 0.44, 0.48)
PROP_FRONT_COLOR = (0.85, 0.25, 0.05)
PROP_REAR_COLOR = (0.2, 0.2, 0.22)

HUBS = {  # name -> (sx, sy) in FLU
    "prop_0": (+1, +1),
    "prop_1": (+1, -1),
    "prop_2": (-1, +1),
    "prop_3": (-1, -1),
}


def flu_to_gltf(v):
    return (v[0], v[2], -v[1])


def cuboid(center, half, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    faces = [
        ((+1, 0, 0), (0, +1, 0), (0, 0, +1)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, +1)),
        ((0, +1, 0), (-1, 0, 0), (0, 0, +1)),
        ((0, -1, 0), (+1, 0, 0), (0, 0, +1)),
        ((0, 0, +1), (0, +1, 0), (+1, 0, 0)),
        ((0, 0, -1), (0, +1, 0), (-1, 0, 0)),
    ]
    positions, normals, indices = [], [], []
    for normal, tangent, bitangent in faces:
        base = len(positions)
        for du, dv in ((-1, -1), (+1, -1), (+1, +1), (-1, +1)):
            local = [
                (normal[i] + du * tangent[i] + dv * bitangent[i]) * half[i]
                for i in range(3)
            ]
            rotated = (
                local[0] * c - local[1] * s + center[0],
                local[0] * s + local[1] * c + center[1],
                local[2] + center[2],
            )
            positions.append(rotated)
            normals.append((normal[0] * c - normal[1] * s, normal[0] * s + normal[1] * c, normal[2]))
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    return positions, normals, indices


def merge(parts):
    positions, normals, indices = [], [], []
    for p, n, i in parts:
        offset = len(positions)
        positions += p
        normals += n
        indices += [offset + index for index in i]
    return positions, normals, indices


def body_geometry():
    parts = [cuboid((0, 0, 0), BODY_HALF)]
    for sx, sy in HUBS.values():
        hub = (sx * SPAN, sy * SPAN)
        yaw = math.atan2(hub[1], hub[0])
        parts.append(cuboid((hub[0] / 2, hub[1] / 2, ARM_Z), ARM_HALF, yaw))
        parts.append(cuboid((hub[0], hub[1], POD_Z), POD_HALF))
    return merge(parts)


def prop_geometry():
    return merge([
        cuboid((0, 0, 0), BLADE_HALF),
        cuboid((0, 0, -0.001), CAP_HALF),
    ])


def main():
    meshes = {
        "body": body_geometry(),
        "prop_front": prop_geometry(),
        "prop_rear": prop_geometry(),
    }
    materials = [
        {"name": name, "pbrMetallicRoughness": {"baseColorFactor": [*color, 1.0], "metallicFactor": 0.0, "roughnessFactor": roughness}}
        for name, color, roughness in [
            ("body", BODY_COLOR, 0.6),
            ("prop_front", PROP_FRONT_COLOR, 0.5),
            ("prop_rear", PROP_REAR_COLOR, 0.5),
        ]
    ]

    binary = b""
    buffer_views = []
    accessors = []
    gltf_meshes = []
    for mesh_index, (name, (positions, normals, indices)) in enumerate(meshes.items()):
        gltf_positions = [flu_to_gltf(p) for p in positions]
        gltf_normals = [flu_to_gltf(n) for n in normals]
        attribute_accessors = {}
        for attribute, data in (("POSITION", gltf_positions), ("NORMAL", gltf_normals)):
            payload = b"".join(struct.pack("<fff", *v) for v in data)
            buffer_views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload), "target": 34962})
            accessor = {"bufferView": len(buffer_views) - 1, "componentType": 5126, "count": len(data), "type": "VEC3"}
            if attribute == "POSITION":
                accessor["min"] = [min(v[i] for v in data) for i in range(3)]
                accessor["max"] = [max(v[i] for v in data) for i in range(3)]
            accessors.append(accessor)
            attribute_accessors[attribute] = len(accessors) - 1
            binary += payload
        payload = b"".join(struct.pack("<H", i) for i in indices)
        payload += b"\x00" * ((4 - len(payload) % 4) % 4)
        buffer_views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload), "target": 34963})
        accessors.append({"bufferView": len(buffer_views) - 1, "componentType": 5123, "count": len(indices), "type": "SCALAR"})
        binary += payload
        gltf_meshes.append({
            "name": name,
            "primitives": [{"attributes": attribute_accessors, "indices": len(accessors) - 1, "material": mesh_index}],
        })

    nodes = [{"mesh": 0, "name": "body"}]
    for name, (sx, sy) in HUBS.items():
        nodes.append({
            "mesh": 1 if sx > 0 else 2,
            "name": name,
            "translation": list(flu_to_gltf((sx * SPAN, sy * SPAN, PROP_Z))),
        })

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "materials": materials,
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    json_bytes = json.dumps(gltf).encode()
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    length = 12 + 8 + len(json_bytes) + 8 + len(binary)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drone.glb")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2, length))
        handle.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        handle.write(json_bytes)
        handle.write(struct.pack("<II", len(binary), 0x004E4942))
        handle.write(binary)
    print(f"wrote {path} ({length} bytes)")


if __name__ == "__main__":
    main()
