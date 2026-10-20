"""Procedural objects shared with the RLtools raytracer."""

from ._component import load_core


def box(size=(1.0, 1.0, 1.0), *, color=(0.8, 0.8, 0.8)):
    """A centered box with full FLU (X, Y, Z) dimensions and flat face normals."""
    return load_core()._primitive_box(tuple(size), tuple(color))
