"""hyperdrone's native components, built by hyperlux's JIT from hyperdrone/_native/<name>."""
from pathlib import Path

from hyperlux import _jit as jit

NATIVE_ROOT = Path(__file__).resolve().parent / "_native"


def component(name, variant, **arguments):
    return jit.Component(name=name, variant=variant, package="hyperdrone", source_dir=str(NATIVE_ROOT / name), **arguments)
