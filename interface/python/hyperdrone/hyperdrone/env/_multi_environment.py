"""ctypes surface over the C++ rl_tools MultiEnvironment<hyperdrone::World>.

The JIT artifact is a thin C shim over the rl_tools batch verbs; this class mirrors that
verb surface one-to-one (reset / render / observe / observe_privileged / step / rewards /
terminated / rotate_scene) so a notebook drives the exact environment the C++ training
targets use — no runners involved.
"""
import ctypes
import os
import sys
from dataclasses import dataclass

import numpy as np

from .. import jit

BACKENDS = ("OPTIX", "METAL", "VULKAN", "WEBGPU", "GENERIC")


def backend():
    value = os.environ.get("HYPERDRONE_ENV_BACKEND", os.environ.get("HYPERDRONE_RENDER_BACKEND", "AUTO")).upper()
    if value == "AUTO":
        value = "METAL" if sys.platform == "darwin" else "OPTIX"
    if value not in BACKENDS:
        raise jit.BuildError(
            f"hyperdrone: invalid HYPERDRONE_ENV_BACKEND {value} (OPTIX|METAL|VULKAN|WEBGPU|GENERIC)"
        )
    return value


def component():
    resolved = backend()
    return jit.Component(
        name="env",
        variant=resolved.lower(),
        needs_cuda=resolved == "OPTIX",
    )


_SHADING = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class EnvConfig:
    num_environments: int = 1
    instances: int = 4
    cam_width: int = 32
    cam_height: int = 32
    shading: str = "low"
    history_length: int = 1

    def defines(self):
        return {
            "HYPERDRONE_ENV_NUM_ENVIRONMENTS": self.num_environments,
            "HYPERDRONE_ENV_INSTANCES": self.instances,
            "HYPERDRONE_ENV_CAM_WIDTH": self.cam_width,
            "HYPERDRONE_ENV_CAM_HEIGHT": self.cam_height,
            "HYPERDRONE_ENV_SHADING": _SHADING[self.shading],
            "HYPERDRONE_ENV_HISTORY_LENGTH": self.history_length,
        }

    def canonical(self):
        return " ".join(f"{name}={value}" for name, value in sorted(self.defines().items()))

    def key(self):
        return jit.canonical_key(self.canonical())


class _Config(ctypes.Structure):
    _fields_ = [
        ("num_environments", ctypes.c_uint32),
        ("instances_per_environment", ctypes.c_uint32),
        ("total_instances", ctypes.c_uint32),
        ("n_agents", ctypes.c_uint32),
        ("cam_width", ctypes.c_uint32),
        ("cam_height", ctypes.c_uint32),
        ("image_channels", ctypes.c_uint32),
        ("observation_dim", ctypes.c_uint32),
        ("observation_dim_privileged", ctypes.c_uint32),
        ("action_dim", ctypes.c_uint32),
        ("episode_step_limit", ctypes.c_uint32),
    ]


_IFACE_VERSION = 1


def _load(config):
    artifact = jit.ensure(component(), config)
    library = ctypes.CDLL(str(artifact))
    library.hyperdrone_env_iface_version.restype = ctypes.c_int
    if library.hyperdrone_env_iface_version() != _IFACE_VERSION:
        raise jit.BuildError(
            f"hyperdrone: env iface version mismatch: artifact has "
            f"{library.hyperdrone_env_iface_version()}, python expects {_IFACE_VERSION}"
        )
    library.hyperdrone_env_create.restype = ctypes.c_void_p
    library.hyperdrone_env_destroy.argtypes = [ctypes.c_void_p]
    library.hyperdrone_env_config.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Config)]
    library.hyperdrone_env_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ulonglong]
    library.hyperdrone_env_reset.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
    library.hyperdrone_env_render.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
    library.hyperdrone_env_observe.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    library.hyperdrone_env_observe_privileged.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    library.hyperdrone_env_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    library.hyperdrone_env_rewards.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    library.hyperdrone_env_terminated.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
    library.hyperdrone_env_rotate_scene.argtypes = [ctypes.c_void_p]
    return library


def _float_ptr(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _uint8_ptr(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


class MultiEnvironment:
    """The C++ environment behind the same verb surface: construct with a scene directory,
    then reset(mask) -> render(mask) -> observe() -> step(actions) -> rewards()/terminated().
    """

    def __init__(self, scene_directory, config=None, seed=0):
        self.config = config if config is not None else EnvConfig()
        self._library = _load(self.config)
        self._handle = ctypes.c_void_p(self._library.hyperdrone_env_create())
        native = _Config()
        self._library.hyperdrone_env_config(self._handle, ctypes.byref(native))
        for name, _ in _Config._fields_:
            setattr(self, name, int(getattr(native, name)))
        self._library.hyperdrone_env_init(self._handle, str(scene_directory).encode(), seed)

    def close(self):
        if self._handle:
            self._library.hyperdrone_env_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _mask(self, mask):
        if mask is None:
            mask = np.ones(self.total_instances, dtype=np.uint8)
        return np.ascontiguousarray(np.asarray(mask, dtype=np.uint8).reshape(self.total_instances))

    def reset(self, mask=None):
        self._library.hyperdrone_env_reset(self._handle, _uint8_ptr(self._mask(mask)))

    def render(self, reset_mask=None):
        if reset_mask is None:
            reset_mask = np.zeros(self.total_instances, dtype=np.uint8)
        self._library.hyperdrone_env_render(self._handle, _uint8_ptr(self._mask(reset_mask)))

    def observe(self):
        observations = np.empty((self.total_instances, self.observation_dim), dtype=np.float32)
        self._library.hyperdrone_env_observe(self._handle, _float_ptr(observations))
        return observations

    def observe_privileged(self):
        observations = np.empty((self.total_instances, self.observation_dim_privileged), dtype=np.float32)
        self._library.hyperdrone_env_observe_privileged(self._handle, _float_ptr(observations))
        return observations

    def step(self, actions):
        actions = np.ascontiguousarray(np.asarray(actions, dtype=np.float32).reshape(self.total_instances, self.action_dim))
        self._library.hyperdrone_env_step(self._handle, _float_ptr(actions))

    def rewards(self):
        rewards = np.empty(self.total_instances, dtype=np.float32)
        self._library.hyperdrone_env_rewards(self._handle, _float_ptr(rewards))
        return rewards

    def terminated(self):
        flags = np.empty(self.total_instances, dtype=np.uint8)
        self._library.hyperdrone_env_terminated(self._handle, _uint8_ptr(flags))
        return flags.astype(bool)

    def rotate_scene(self):
        self._library.hyperdrone_env_rotate_scene(self._handle)

    def frames(self):
        """The latest visual observation as (total, height, width, channels) images."""
        return self.observe().reshape(
            self.total_instances, self.n_agents * self.cam_height, self.cam_width, self.image_channels
        )
