from ._component import load_core
from ._config import FIDELITY


def _fidelity_id(fidelity):
    return FIDELITY[fidelity.lower()] if isinstance(fidelity, str) else int(fidelity)


def load_scene(path, fidelity="high", rgb=True):
    scene = load_core().Scene()
    scene.load(str(path), _fidelity_id(fidelity), bool(rgb))
    return scene


def load_object(path, fidelity="high", rgb=True):
    return load_core().load_object(str(path), _fidelity_id(fidelity), bool(rgb))


def load_assembly(path, fidelity="high", rgb=True):
    return load_core().load_assembly(str(path), _fidelity_id(fidelity), bool(rgb))


def make_camera(position, look_at, up=(0.0, 0.0, 1.0), fov=60.0, aspect=1.0):
    return load_core().make_camera(tuple(position), tuple(look_at), tuple(up), float(fov), float(aspect))


def make_transform(position=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0)):
    return load_core().make_transform(tuple(position), tuple(orientation_wxyz))


def compose_transforms(a, b):
    return load_core().compose_transforms(a, b)
