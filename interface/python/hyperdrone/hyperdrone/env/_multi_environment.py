"""ctypes surface over the C++ rl_tools hyperdrone::MultiEnvironment<World>.

The JIT artifact is a thin C shim over the rl_tools batch verbs; this class mirrors that
verb surface one-to-one (reset / render / observe / observe_privileged / step / rewards /
terminated / rotate_scene) so a notebook drives the exact environment the C++ training
targets use — no runners involved.

Configuration follows the C++ extension ladder: a preset names the platform
(`presets::*` on the C++ side), a task names the wrapper (`tasks::*`), and a handful of
universal knobs (instances, camera resolution, shading) complete the picture. Arbitrary
C++ specifications are reached through `spec_header=`: a path to a C++ header defining
`hyperdrone_env_user::WORLD` (including any rl_tools headers it needs); the header's
content hash is part of the JIT key, and preset/task/n_agents must stay at their
defaults since the user type supersedes them.
"""
import ctypes
import hashlib
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from .. import conta, jit

BACKENDS = ("OPTIX", "METAL", "VULKAN", "WEBGPU", "GENERIC")
PRESETS = {"crazyflie": 0, "x500_fpv": 1, "x500_fpv_imu": 2}
SELF_VISIBLE_PRESETS = ("x500_fpv", "x500_fpv_imu")
TASKS = {None: 0, "target_frame": 1, "moving_gate": 2, "visual_inertial_localization": 3}


def backend():
    from ..render._component import resolve_auto_backend

    value = os.environ.get("HYPERDRONE_ENV_BACKEND", os.environ.get("HYPERDRONE_RENDER_BACKEND", "AUTO")).upper()
    if value == "AUTO":
        value = resolve_auto_backend()
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
    preset: str = "crazyflie"
    task: str = None
    n_agents: int = 1
    spec_header: str = None

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"hyperdrone: unknown preset {self.preset!r} (one of {sorted(PRESETS)})")
        if self.task not in TASKS:
            raise ValueError(f"hyperdrone: unknown task {self.task!r} (one of {sorted(k for k in TASKS if k)} or None)")
        if self.shading not in _SHADING:
            raise ValueError(f"hyperdrone: unknown shading {self.shading!r} (one of {sorted(_SHADING)})")
        if self.n_agents < 1:
            raise ValueError("hyperdrone: n_agents must be >= 1")
        if self.n_agents > 1 and self.preset not in SELF_VISIBLE_PRESETS:
            raise ValueError(
                "hyperdrone: multi-agent needs a SELF_VISIBLE preset (agents seeing each "
                "other needs geometry to see) — use preset='x500_fpv'"
            )
        if self.task == "visual_inertial_localization":
            if self.preset != "x500_fpv_imu":
                raise ValueError(
                    "hyperdrone: task 'visual_inertial_localization' needs an IMU in the "
                    "dynamics state chain — use preset='x500_fpv_imu'"
                )
            if self.n_agents > 1:
                raise ValueError("hyperdrone: task 'visual_inertial_localization' is single-agent")
        if self.spec_header is not None:
            defaults = EnvConfig()
            if (self.preset, self.task, self.n_agents) != (defaults.preset, defaults.task, defaults.n_agents):
                raise ValueError(
                    "hyperdrone: spec_header supersedes preset/task/n_agents — leave them "
                    "at their defaults; the user header pins the World"
                )
            if not os.path.isfile(self.spec_header):
                raise ValueError(f"hyperdrone: spec_header not found: {self.spec_header}")

    @property
    def self_visible(self):
        """Whether the drone's own body is rendered into its cameras (needs drone_asset=);
        the visual_inertial_localization benchmark flies without self-occlusion."""
        return self.preset in SELF_VISIBLE_PRESETS and self.task != "visual_inertial_localization"

    def defines(self):
        values = {
            "HYPERDRONE_ENV_NUM_ENVIRONMENTS": self.num_environments,
            "HYPERDRONE_ENV_INSTANCES": self.instances,
            "HYPERDRONE_ENV_CAM_WIDTH": self.cam_width,
            "HYPERDRONE_ENV_CAM_HEIGHT": self.cam_height,
            "HYPERDRONE_ENV_SHADING": _SHADING[self.shading],
            "HYPERDRONE_ENV_HISTORY_LENGTH": self.history_length,
            "HYPERDRONE_ENV_PRESET": PRESETS[self.preset],
            "HYPERDRONE_ENV_TASK": TASKS[self.task],
            "HYPERDRONE_ENV_N_AGENTS": self.n_agents,
        }
        if self.spec_header is not None:
            values["HYPERDRONE_ENV_SPEC_HEADER"] = os.path.abspath(self.spec_header)
        return values

    def canonical(self):
        canonical = " ".join(f"{name}={value}" for name, value in sorted(self.defines().items()))
        if self.spec_header is not None:
            digest = hashlib.sha256(open(self.spec_header, "rb").read()).hexdigest()[:16]
            canonical += f" spec_hash={digest}"
        return canonical

    def key(self):
        return jit.canonical_key(self.canonical())


@dataclass(frozen=True)
class ObservationLayout:
    """Named blocks of an observation vector, parsed from the shim's layout export.

    For image observations (axis == "channel"), shape is (height, width, channels) and
    block offsets/sizes index the channel axis; for flat vectors (axis == "flat"), shape
    is (dim,) and blocks slice the vector directly.
    """
    shape: tuple
    axis: str
    blocks: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, text):
        shape = None
        axis = None
        blocks = {}
        for line in text.strip().splitlines():
            parts = line.split()
            if parts[0] == "shape":
                shape = tuple(int(value) for value in parts[1:])
            elif parts[0] == "axis":
                axis = parts[1]
            elif parts[0] == "block":
                blocks[parts[1]] = (int(parts[2]), int(parts[3]))
        if shape is None or axis not in ("channel", "flat"):
            raise jit.BuildError(f"hyperdrone: malformed observation layout: {text!r}")
        covered = sum(size for _, size in blocks.values())
        extent = shape[-1] if axis == "channel" else shape[0]
        if covered != extent:
            raise jit.BuildError(
                f"hyperdrone: observation layout blocks cover {covered} of {extent}: {text!r}"
            )
        return cls(shape=shape, axis=axis, blocks=blocks)


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
        ("observation_dim_imu", ctypes.c_uint32),
        ("frame_stride", ctypes.c_uint32),
        ("dt", ctypes.c_float),
    ]


_IFACE_VERSION = 6


def _load(config):
    artifact = jit.ensure(component(), config)
    library = ctypes.CDLL(str(artifact))
    library.hyperdrone_env_iface_version.restype = ctypes.c_int
    if library.hyperdrone_env_iface_version() != _IFACE_VERSION:
        raise jit.BuildError(
            f"hyperdrone: env iface version mismatch: artifact has "
            f"{library.hyperdrone_env_iface_version()}, python expects {_IFACE_VERSION}"
        )
    library.hyperdrone_env_config_string.restype = ctypes.c_char_p
    library.hyperdrone_env_observation_layout.restype = ctypes.c_char_p
    library.hyperdrone_env_observation_layout.argtypes = [ctypes.c_int]
    library.hyperdrone_env_create.restype = ctypes.c_void_p
    library.hyperdrone_env_destroy.argtypes = [ctypes.c_void_p]
    library.hyperdrone_env_config.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Config)]
    library.hyperdrone_env_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulonglong]
    library.hyperdrone_env_reset.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
    library.hyperdrone_env_render.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
    library.hyperdrone_env_observe.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    library.hyperdrone_env_observe_privileged.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    library.hyperdrone_env_observe_imu.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    library.hyperdrone_env_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    library.hyperdrone_env_rewards.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    library.hyperdrone_env_terminated.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
    library.hyperdrone_env_rotate_scene.argtypes = [ctypes.c_void_p]
    return library


def _float_ptr(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _uint8_ptr(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def _is_conta_reference(value):
    return isinstance(value, dict) or (isinstance(value, str) and value.startswith("conta:"))


def _resolve_asset(asset):
    if asset is None:
        return None
    if _is_conta_reference(asset):
        return str(conta.resolve(asset))
    return str(asset)


def _normalize_scenes(scenes, num_environments):
    """Turn the scenes argument into the newline-joined reference list the shim expects.

    Accepts a directory (the corpus is its sorted .glb files), a single reference, or a
    sequence of references — each a .glb path, a "conta:HASH" string, or a store manifest
    entry {"description": ..., "hash": ...}. Conta references are resolved (downloading
    into the shared conta cache) here, so failures raise instead of aborting the process
    through the C ABI; list order is the corpus order.
    """
    if isinstance(scenes, (str, os.PathLike, dict)):
        scenes = [scenes]
    references = [
        str(conta.resolve(entry)) if _is_conta_reference(entry) else str(entry)
        for entry in scenes
    ]
    if not references:
        raise ValueError("hyperdrone: scenes is empty")
    if len(references) == 1 and os.path.isdir(references[0]):
        count = sum(1 for name in os.listdir(references[0]) if name.endswith(".glb"))
        if count < num_environments:
            raise ValueError(
                f"hyperdrone: {count} .glb scene(s) in {references[0]}, but "
                f"num_environments={num_environments} needs at least one scene per environment"
            )
        return references
    missing = [reference for reference in references if not os.path.isfile(reference)]
    if missing:
        raise ValueError(f"hyperdrone: scene reference(s) not found: {missing}")
    if len(references) < num_environments:
        raise ValueError(
            f"hyperdrone: {len(references)} scene reference(s), but "
            f"num_environments={num_environments} needs at least one scene per environment"
        )
    return references


def _validate(config, native):
    if config.spec_header is not None:
        return
    expected = {
        "num_environments": config.num_environments,
        "instances_per_environment": config.instances,
        "n_agents": config.n_agents,
        "cam_width": config.cam_width,
        "cam_height": config.cam_height,
        "total_instances": config.num_environments * config.instances,
    }
    for name, value in expected.items():
        actual = int(getattr(native, name))
        if actual != value:
            raise jit.BuildError(
                f"hyperdrone: env config mismatch — the artifact reports {name}={actual}, "
                f"the EnvConfig requested {value}; the compile-time mirror has drifted"
            )


class MultiEnvironment:
    """The C++ environment behind the same verb surface: construct with scenes, then
    reset(mask) -> render(mask) -> observe() -> step(actions) -> rewards()/terminated().

    scenes: a directory of .glb scenes (sorted corpus), or a reference / sequence of
    references — .glb paths, "conta:HASH" strings, or conta store manifest entries
    {"description": ..., "hash": ...}; a reference list is the corpus in list order.
    drone_asset: body/prop_* GLB for SELF_VISIBLE presets (required when config.self_visible).
    gate_asset: gate GLB for the moving_gate task (required for task='moving_gate').
    Both assets accept conta references too.

    Tasks with an IMU stream (task='visual_inertial_localization') add observe_imu() — the
    IMU sample of the last step, named by observation_layout_imu — and render a camera frame
    only every frame_stride steps; their episodes are fixed-length and synchronized, so the
    reset mask must be all-or-none per environment. Autonomous tasks fly themselves:
    action_dim is 0 and step() takes no actions.
    """

    def __init__(self, scenes, config=None, seed=0, drone_asset=None, gate_asset=None):
        self.config = config if config is not None else EnvConfig()
        if self.config.self_visible and drone_asset is None:
            raise ValueError(
                f"hyperdrone: preset {self.config.preset!r} is SELF_VISIBLE and needs drone_asset= "
                "(a body/prop_* GLB, e.g. an x500 assembly)"
            )
        if self.config.task == "moving_gate" and gate_asset is None:
            raise ValueError("hyperdrone: task 'moving_gate' needs gate_asset= (the gate GLB)")
        scene_references = _normalize_scenes(scenes, self.config.num_environments)
        drone_asset = _resolve_asset(drone_asset)
        gate_asset = _resolve_asset(gate_asset)
        self._library = _load(self.config)
        self._handle = ctypes.c_void_p(self._library.hyperdrone_env_create())
        native = _Config()
        self._library.hyperdrone_env_config(self._handle, ctypes.byref(native))
        _validate(self.config, native)
        for name, ctype in _Config._fields_:
            setattr(self, name, (float if ctype is ctypes.c_float else int)(getattr(native, name)))
        self.config_string = self._library.hyperdrone_env_config_string().decode()
        self.observation_layout = ObservationLayout.parse(
            self._library.hyperdrone_env_observation_layout(0).decode()
        )
        self.observation_layout_privileged = ObservationLayout.parse(
            self._library.hyperdrone_env_observation_layout(1).decode()
        )
        self.observation_layout_imu = None
        if self.observation_dim_imu > 0:
            self.observation_layout_imu = ObservationLayout.parse(
                self._library.hyperdrone_env_observation_layout(2).decode()
            )
        drone_asset_encoded = drone_asset.encode() if drone_asset is not None else b""
        gate_asset_encoded = gate_asset.encode() if gate_asset is not None else b""
        self._library.hyperdrone_env_init(self._handle, "\n".join(scene_references).encode(), drone_asset_encoded, gate_asset_encoded, seed)

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

    def _check_synchronized(self, reset_mask):
        if self.config.task != "visual_inertial_localization":
            return
        per_environment = reset_mask.reshape(self.num_environments, self.instances_per_environment).astype(bool)
        if np.any(per_environment.any(axis=1) & ~per_environment.all(axis=1)):
            raise ValueError(
                "hyperdrone: task 'visual_inertial_localization' runs fixed-length synchronized "
                "episodes — the reset mask must be all-or-none per environment"
            )

    def render(self, reset_mask=None):
        if reset_mask is None:
            reset_mask = np.zeros(self.total_instances, dtype=np.uint8)
        reset_mask = self._mask(reset_mask)
        self._check_synchronized(reset_mask)
        self._library.hyperdrone_env_render(self._handle, _uint8_ptr(reset_mask))

    def observe(self):
        observations = np.empty((self.total_instances, self.observation_dim), dtype=np.float32)
        self._library.hyperdrone_env_observe(self._handle, _float_ptr(observations))
        return observations

    def observe_privileged(self):
        observations = np.empty((self.total_instances, self.observation_dim_privileged), dtype=np.float32)
        self._library.hyperdrone_env_observe_privileged(self._handle, _float_ptr(observations))
        return observations

    def observe_imu(self):
        """The IMU sample of the last step: (total, observation_dim_imu), named by observation_layout_imu."""
        if self.observation_dim_imu == 0:
            raise ValueError(f"hyperdrone: this configuration has no IMU observation (task={self.config.task!r})")
        observations = np.empty((self.total_instances, self.observation_dim_imu), dtype=np.float32)
        self._library.hyperdrone_env_observe_imu(self._handle, _float_ptr(observations))
        return observations

    def step(self, actions=None):
        if actions is None:
            if self.action_dim != 0:
                raise ValueError(
                    "hyperdrone: step() needs actions of shape (total_instances, action_dim); "
                    "only autonomous tasks (action_dim == 0) step without them"
                )
            actions = np.empty((self.total_instances, 0), dtype=np.float32)
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
        """The latest visual observation as (total, *observation_layout.shape) images."""
        return self.observe().reshape(self.total_instances, *self.observation_layout.shape)
