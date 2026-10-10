from ._component import load_core
from ._config import FIDELITY


def _fidelity_id(fidelity):
    if not isinstance(fidelity, str) or fidelity.lower() not in FIDELITY:
        raise ValueError(f"fidelity must name one of {FIDELITY}")
    return FIDELITY.index(fidelity.lower())


def load_scene(path, *, fidelity="high", rgb=True, preserve_instances=False):
    """Load a GLB; preserve root instances for instance or semantic segmentation."""
    scene = load_core().Scene()
    scene._load(str(path), _fidelity_id(fidelity), bool(rgb), bool(preserve_instances))
    return scene


def load_object(path, *, fidelity="high", rgb=True):
    return load_core().load_object(str(path), _fidelity_id(fidelity), bool(rgb))


def load_assembly(path, *, fidelity="high", rgb=True, part_nodes=None):
    """One part per GLB scene root, or per named node in the given order (an enclosing node
    first) with nested nodes recording their parent part; see ObjectAssembly.part_parent."""
    return load_core().load_assembly(str(path), _fidelity_id(fidelity), bool(rgb),
                                     [] if part_nodes is None else [str(name) for name in part_nodes])


def make_camera(position, look_at, *, up=(0.0, 0.0, 1.0), fov=60.0, aspect):
    return load_core().make_camera(tuple(position), tuple(look_at), tuple(up), float(fov), float(aspect))


def make_transform(position=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0)):
    return load_core().make_transform(tuple(position), tuple(orientation_wxyz))


def compose_transforms(a, b):
    return load_core().compose_transforms(a, b)
