"""hyperdrone.dynamics — vectorized, stateful L2F multirotor simulator.

Self-contained peer of hyperdrone.render (never imports it): the coupling surface is a
plain DLPack tensor of packed camera bases produced by Sim.camera_bases(), which the
renderer consumes without knowing where it came from. Two JIT variants of one component:
cpu (portable reference) and cuda (batch-parallel kernels, device-resident buffers);
select via device= or HYPERDRONE_DYNAMICS_DEVICE (AUTO prefers cuda when available).
"""
from .. import cuda as _cuda
from ._component import component, load_core, resolve_device
from ._config import MODELS, STATE_COMPONENTS, SimConfig
from ._sim import IDENTITY_MOUNT, Sim

__all__ = [
    "IDENTITY_MOUNT",
    "MODELS",
    "Sim",
    "SimConfig",
    "resolve_device",
]

_cuda.register_provider(load_core)
