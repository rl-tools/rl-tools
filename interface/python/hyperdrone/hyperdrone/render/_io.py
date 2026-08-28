"""File output for rendered buffers, implemented in pure Python over the readback API.

The formats match the C++ save verbs in rl_tools' rendering/raytracing/save_cpu.h: images
are GRID_COLS x GRID_ROWS camera grids (cols = smallest c with c*c >= num_cameras) written
as RGBA8 PNGs with unused cells transparent black; depth_raw is a native little-endian
header (int32 num_cameras, height, width) followed by the float32 payload; probes is a
header (int32 num_cameras, num_probes) followed by packed (float32 distance, int32 hit)
records — the layouts consumed by tools/render_parity.
"""

import struct
import zlib

import numpy as np


def _png_chunk(tag, payload):
    data = tag + payload
    return struct.pack(">I", len(payload)) + data + struct.pack(">I", zlib.crc32(data) & 0xFFFFFFFF)


def write_png(path, rgba):
    """Write an (height, width, 4) uint8 array as an RGBA8 PNG (filter 0, one IDAT)."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    height, width = rgba.shape[0], rgba.shape[1]
    filtered = np.zeros((height, 1 + width * 4), dtype=np.uint8)
    filtered[:, 1:] = rgba.reshape(height, width * 4)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(b"IHDR", header))
        f.write(_png_chunk(b"IDAT", zlib.compress(filtered.tobytes())))
        f.write(_png_chunk(b"IEND", b""))


def read_png(path):
    """Read a PNG written by write_png back into an (height, width, 4) uint8 array."""
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    offset = 8
    width = height = None
    idat = b""
    while offset < len(blob):
        (length,) = struct.unpack(">I", blob[offset:offset + 4])
        tag = blob[offset + 4:offset + 8]
        payload = blob[offset + 8:offset + 8 + length]
        if tag == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", payload[:10])
            if bit_depth != 8 or color_type != 6:
                raise ValueError("read_png only supports 8-bit RGBA (write_png output)")
        elif tag == b"IDAT":
            idat += payload
        offset += 12 + length
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(height, 1 + width * 4)
    if np.any(raw[:, 0] != 0):
        raise ValueError("read_png only supports filter 0 scanlines (write_png output)")
    return raw[:, 1:].reshape(height, width, 4).copy()


def camera_grid(images):
    """Tile (num_cameras, height, width, 4) uint8 images into the canonical camera grid."""
    num_cameras, cam_height, cam_width = images.shape[0], images.shape[1], images.shape[2]
    cols = 1
    while cols * cols < num_cameras:
        cols += 1
    rows = (num_cameras + cols - 1) // cols
    grid = np.zeros((rows * cam_height, cols * cam_width, 4), dtype=np.uint8)
    for index in range(num_cameras):
        row, col = divmod(index, cols)
        grid[row * cam_height:(row + 1) * cam_height, col * cam_width:(col + 1) * cam_width] = images[index]
    return grid


def depth_to_gray(depth, camera_radius):
    """The depth colormap of the C++ save verb: min/max normalization over valid hits
    (0 < depth < camera_radius * 2 * 0.999), truncating to uint8; invalid pixels black."""
    depth = np.asarray(depth, dtype=np.float32)
    max_depth = np.float32(camera_radius * 2.0) if camera_radius > 0 else np.float32(1e30)
    valid_max_depth = max_depth * np.float32(0.999)
    valid = (depth > 0) & (depth < valid_max_depth)
    value = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        min_valid = depth[valid].min()
        max_valid = depth[valid].max()
        normalized = np.clip((depth - min_valid) / (max_valid - min_valid + np.float32(1e-6)), 0, 1)
        value[valid] = (normalized[valid] * np.float32(255.0)).astype(np.uint8)
    return value


def segmentation_to_rgba(ids):
    """The golden-ratio hue hash of the C++ save verb (float32 arithmetic); the miss
    sentinel 0xFFFFFFFF renders black."""
    ids = np.asarray(ids, dtype=np.uint32)
    hue = np.fmod(ids.astype(np.float32) * np.float32(0.61803398875), np.float32(1.0)) * np.float32(6.0)
    descending = np.float32(1.0) - np.abs(np.fmod(hue, np.float32(2.0)) - np.float32(1.0))
    sector = hue.astype(np.int32)
    one = np.ones_like(descending)
    zero = np.zeros_like(descending)
    red = np.select([sector == 0, sector == 1, sector == 2, sector == 3, sector == 4], [one, descending, zero, zero, descending], one)
    green = np.select([sector == 0, sector == 1, sector == 2, sector == 3, sector == 4], [descending, one, one, descending, zero], zero)
    blue = np.select([sector == 0, sector == 1, sector == 2, sector == 3, sector == 4], [zero, zero, descending, one, one], descending)
    rgba = np.empty(ids.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = (red * np.float32(255.0)).astype(np.uint8)
    rgba[..., 1] = (green * np.float32(255.0)).astype(np.uint8)
    rgba[..., 2] = (blue * np.float32(255.0)).astype(np.uint8)
    rgba[..., 3] = 255
    miss = ids == np.uint32(0xFFFFFFFF)
    rgba[miss] = (0, 0, 0, 255)
    return rgba


def gray_to_rgba(value):
    rgba = np.empty(value.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = value
    rgba[..., 1] = value
    rgba[..., 2] = value
    rgba[..., 3] = 255
    return rgba


def save_image(renderer, path):
    write_png(path, camera_grid(renderer.frame()))


def save_depth_image(renderer, path):
    gray = depth_to_gray(renderer.depth(), renderer.scene_bounds["camera_radius"])
    write_png(path, camera_grid(gray_to_rgba(gray)))


def save_depth_raw(renderer, path):
    depth = np.ascontiguousarray(renderer.depth(), dtype=np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<iii", renderer.num_cameras, renderer.height, renderer.width))
        f.write(depth.tobytes())


def save_segmentation_image(renderer, path):
    write_png(path, camera_grid(segmentation_to_rgba(renderer.segmentation())))


def save_probes(renderer, path):
    distances, hits = renderer.collisions()
    records = np.empty(distances.size, dtype=[("distance", "<f4"), ("hit", "<i4")])
    records["distance"] = distances.reshape(-1)
    records["hit"] = hits.reshape(-1)
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", renderer.num_cameras, renderer.num_probes))
        f.write(records.tobytes())
