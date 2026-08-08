import math

from ._component import load_core
from ._config import SHADING


def _shading_id(shading):
    return SHADING[shading.lower()] if isinstance(shading, str) else int(shading)


def load_scene(path, shading="high", rgb=True):
    scene = load_core().Scene()
    scene.load(str(path), _shading_id(shading), bool(rgb))
    return scene


def load_object(path, shading="high", rgb=True):
    return load_core().load_object(str(path), _shading_id(shading), bool(rgb))


def load_assembly(path, shading="high", rgb=True):
    return load_core().load_assembly(str(path), _shading_id(shading), bool(rgb))


def make_camera(position, look_at, up=(0.0, 0.0, 1.0), fov=math.radians(60.0), aspect=1.0):
    return load_core().make_camera(tuple(position), tuple(look_at), tuple(up), float(fov), float(aspect))


def make_transform(position=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0)):
    return load_core().make_transform(tuple(position), tuple(orientation_wxyz))


def compose_transforms(a, b):
    return load_core().compose_transforms(a, b)
