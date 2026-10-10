"""Vectorized L2F dynamics, independent of rendering; Sim defaults to CPU and exports device camera bases through camera_bases_dlpack()."""
from ._config import MODELS
from ._sim import IDENTITY_MOUNT, Sim

__all__ = ["IDENTITY_MOUNT", "MODELS", "Sim"]
