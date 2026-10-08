"""Direct bindings to the C++ rl_tools hyperdrone::MultiEnvironment<World>.

The JIT extension binds the rl_tools batch verbs; this class mirrors that
verb surface one-to-one (reset / render / observe / observe_privileged / step / rewards /
terminated / rotate_scene) so a notebook drives the exact environment the C++ training
targets use — no runners involved.

Configuration follows the C++ extension ladder: a preset names the platform
(`presets::*` on the C++ side), a task names the wrapper (`tasks::*`), and a handful of
universal knobs (instances, camera resolution, shading) complete the picture. Arbitrary
C++ specifications are reached through `spec_header=`: a path to a C++ header defining
`hyperdrone_env_user::WORLD` (including any rl_tools headers it needs); the header's
content hash is part of the JIT key, and preset/task/n_agents/max_drones must stay at their
defaults since the user type supersedes them.
"""
import copy
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from .. import conta, jit

PRESETS = {"crazyflie": 0, "x500_fpv": 1, "x500_fpv_imu": 2}
SELF_VISIBLE_PRESETS = ("x500_fpv", "x500_fpv_imu")
TASKS = {None: 0, "target_frame": 1, "moving_gate": 2, "visual_inertial_localization": 3}


def backend():
    from ..render._component import backend as render_backend

    return render_backend()


def component():
    resolved = backend()
    return jit.Component(
        name="env",
        variant=resolved.lower(),
        core_target="hyperdrone_env_core",
        needs_cuda=resolved == "OPTIX",
    )


_SHADING = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class EnvConfig:
    num_environments: int = 1
    instances: int = 4
    cam_width: int = 32
    cam_height: int = 32
    shading: str = "high"
    history_length: int = 1
    preset: str = "crazyflie"
    task: str = None
    n_agents: int = 1
    max_drones: int = 1
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
        if self.max_drones < 1:
            raise ValueError("hyperdrone: max_drones must be >= 1")
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
            if (self.preset, self.task, self.n_agents, self.max_drones) != (defaults.preset, defaults.task, defaults.n_agents, defaults.max_drones):
                raise ValueError(
                    "hyperdrone: spec_header supersedes preset/task/n_agents/max_drones — leave them "
                    "at their defaults; the user header pins the World"
                )
            if not os.path.isfile(self.spec_header):
                raise ValueError(f"hyperdrone: spec_header not found: {self.spec_header}")

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
            "HYPERDRONE_ENV_MAX_DRONES": self.max_drones,
        }
        if self.spec_header is not None:
            values["HYPERDRONE_ENV_SPEC_HEADER"] = os.path.abspath(self.spec_header)
        return values

    def canonical(self):
        canonical = " ".join(f"{name}={value}" for name, value in sorted(self.defines().items()))
        if self.spec_header is not None:
            with open(self.spec_header, "rb") as header:
                canonical += f" spec_hash={hashlib.sha256(header.read()).hexdigest()[:16]}"
        return canonical

    def key(self):
        return jit.canonical_key(self.canonical())


@dataclass(frozen=True)
class ObservationLayout:
    """Named blocks of an observation vector, described by the compiled environment.

    For image observations (axis == "channel"), shape is (height, width, channels) and
    block offsets/sizes index the channel axis; for flat vectors (axis == "flat"), shape
    is (dim,) and blocks slice the vector directly.
    """
    shape: tuple
    axis: str
    blocks: dict = field(default_factory=dict)



def _is_conta_reference(value):
    return isinstance(value, dict) or (isinstance(value, str) and value.startswith("conta:"))


def _resolve_asset(asset):
    if asset is None:
        return None
    if _is_conta_reference(asset):
        return str(conta.resolve(asset))
    return str(asset)


def _normalize_scenes(scenes, num_environments):
    """Normalize the scene corpus into a list of local paths.

    Accepts a directory (the corpus is its sorted .glb files), a single reference, or a
    sequence of references — each a .glb path, a "conta:HASH" string, or a store manifest
    entry {"description": ..., "hash": ...}. Conta references are resolved (downloading
    into the shared conta cache) here; list order is the corpus order. The corpus is
    validated here because the C++ dataset code exits the process on an invalid one.
    """
    if isinstance(scenes, (str, os.PathLike, dict)):
        scenes = [scenes]
    references = [
        str(conta.resolve(entry)) if _is_conta_reference(entry) else str(entry)
        for entry in scenes
    ]
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


def _merge(base, overrides, path=""):
    merged = dict(base)
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f"hyperdrone: unknown parameter {path}{key!r} (one of {sorted(base)})")
        if isinstance(base[key], dict):
            merged[key] = _merge(base[key], value, f"{path}{key}.")
            continue
        if np.shape(value) != np.shape(base[key]):
            raise ValueError(f"hyperdrone: parameter {path}{key} must have shape {np.shape(base[key])}, got {np.shape(value)}")
        merged[key] = value.tolist() if isinstance(value, (np.ndarray, np.generic)) else value
    return merged


class MultiEnvironment:
    """The C++ environment behind the same verb surface: construct with scenes, then
    reset(mask) -> render(mask) -> observe() -> step(actions) -> rewards()/terminated().

    scenes: a directory of .glb scenes (sorted corpus), or a reference / sequence of
    references — .glb paths, "conta:HASH" strings, or conta store manifest entries
    {"description": ..., "hash": ...}; a reference list is the corpus in list order.
    drones: the drone table every instance samples from uniformly at each reset, independently of
    the scene its environment shows: a list of {"asset": <GLB reference or None>, "parameters":
    {partial dict}}. The asset is the body/prop_* GLB rendered into the instance's cameras
    (required per entry when the compiled world is SELF_VISIBLE, rejected otherwise); the parameters are merged
    over the shared constructor base (see parameters) and may only touch what varies per drone: dynamics.dynamics
    (a "model" key loads an l2f registry entry), dynamics.imu and camera. The same GLB may appear
    in several entries; distinct GLBs load once. The table holds at most max_drones
    entries. drone_asset=X is the one-entry table [{"asset": X}]; without either, the table is
    the nominal drone.
    gate_asset: gate GLB for the moving_gate task (required for task='moving_gate').
    All assets accept conta references too.

    parameters: a partial dict in the schema of the parameters property, {"dynamics": <l2f parameters>, "camera": {"mount": {"pose": {"position", "orientation"}}, "fov"}}, merged over the preset's nominal values as the base of every drone entry (the specification's randomization ranges apply on top at reset). parameters["dynamics"]["dynamics"]["model"] loads an l2f registry entry (hyperdrone.dynamics.MODELS) before the rest is merged, J_inv follows J unless given, the camera orientation is a (w, x, y, z) quaternion mapping the camera's FLU axes into the body frame, and fov is the horizontal FOV in degrees.

    Tasks with an IMU stream (task='visual_inertial_localization') add observe_imu() — the
    IMU sample of the last step, named by observation_layout_imu — and render a camera frame
    only every frame_stride steps; their episodes are fixed-length and synchronized, so the
    reset mask must be all-or-none per environment. Autonomous tasks fly themselves:
    action_dim is 0 and step() takes no actions.
    """

    def __init__(self, scenes, config=None, seed=0, drone_asset=None, gate_asset=None, parameters=None, drones=None):
        self.config = config if config is not None else EnvConfig()
        if drones is not None and drone_asset is not None:
            raise ValueError("hyperdrone: pass either drone_asset= (one drone) or drones= (the table)")
        if drones is None:
            drones = [{"asset": drone_asset}] if drone_asset is not None else [{}]
        drones = [dict(entry) for entry in drones]
        if not drones:
            raise ValueError("hyperdrone: drones= needs at least one entry")
        for entry in drones:
            unknown = sorted(set(entry) - {"asset", "parameters"})
            if unknown:
                raise ValueError(f"hyperdrone: unknown drone entry keys {unknown} (asset, parameters)")
        if self.config.task == "moving_gate" and gate_asset is None:
            raise ValueError("hyperdrone: task 'moving_gate' needs gate_asset= (the gate GLB)")
        self._module = jit.module(component(), self.config)
        for name, value in self._module.dimensions.items():
            setattr(self, name, value)
        if len(drones) > self.max_drones:
            capacity_option = "MAX_DRONES in spec_header" if self.config.spec_header is not None else "EnvConfig(max_drones=...)"
            raise ValueError(
                f"hyperdrone: {len(drones)} drones but this configuration holds {self.max_drones}; "
                f"raise {capacity_option}"
            )
        for entry in drones:
            if self.self_visible and entry.get("asset") is None:
                raise ValueError("hyperdrone: this SELF_VISIBLE world needs an asset per drone; pass drone_asset= or drones=")
            if not self.self_visible and entry.get("asset") is not None:
                raise ValueError("hyperdrone: drone assets need a SELF_VISIBLE world, nothing would render them")
        scene_references = _normalize_scenes(scenes, self.num_environments)
        drone_assets = [_resolve_asset(entry.get("asset")) for entry in drones]
        gate_asset = _resolve_asset(gate_asset)
        self.observation_layout = ObservationLayout(**self._module.observation_layout)
        self.observation_layout_privileged = ObservationLayout(**self._module.observation_layout_privileged)
        self.observation_layout_imu = ObservationLayout(**self._module.observation_layout_imu) if self.observation_dim_imu > 0 else None
        self._environment = self._module.Environment()
        nominal = self._resolve(json.loads(self._environment.parameters()), parameters or {})
        drone_parameters = []
        for entry in drones:
            overrides = entry.get("parameters") or {}
            self._check_per_drone(overrides)
            drone_parameters.append(json.dumps(self._resolve(nominal, overrides)))
        self._environment.init(scene_references, drone_assets, drone_parameters, gate_asset, seed)
        self._sampled = np.zeros(self.total_instances, dtype=bool)

    @staticmethod
    def _check_per_drone(overrides):
        if "drone" in overrides:
            raise ValueError("hyperdrone: drone is sampled at reset from the drones= table")
        shared = sorted(set(overrides.get("dynamics", {})) - {"dynamics", "imu"})
        if shared:
            raise ValueError(f"hyperdrone: dynamics {shared} are shared by all instances; set them through parameters=")

    def _resolve(self, base, overrides):
        overrides = copy.deepcopy(overrides)
        dynamics = overrides.get("dynamics", {}).get("dynamics", {})
        model = dynamics.pop("model", None)
        if model is not None:
            entry = self._environment.model(model)
            if entry is None:
                raise ValueError(f"hyperdrone: unknown dynamics model {model!r} (see hyperdrone.dynamics.MODELS)")
            base = copy.deepcopy(base)
            base["dynamics"]["dynamics"] = json.loads(entry)
        merged = _merge(base, overrides)
        if "J" in dynamics and "J_inv" not in dynamics:
            merged["dynamics"]["dynamics"]["J_inv"] = np.linalg.inv(merged["dynamics"]["dynamics"]["J"]).tolist()
        return merged

    @property
    def parameters(self):
        """The resolved nominal parameters of drone 0, including its entry-specific overrides."""
        return json.loads(self._environment.parameters())

    @property
    def drones(self):
        """The drone table every instance samples from at reset: one dict per entry with the resolved asset path and the nominal parameters; the "drone" value of instance_parameters() indexes it."""
        return json.loads(self._environment.drones())

    def instance_parameters(self):
        """The current parameters of every instance, as sampled at its last reset: a list of dicts in the schema of parameters plus "drone", the index of the table entry the instance flies."""
        if not self._sampled.all():
            raise RuntimeError("hyperdrone: per-instance parameters exist once every instance has been reset")
        return json.loads(self._environment.instance_parameters())

    def set_instance_parameters(self, overrides, mask=None):
        """Merges overrides into the current parameters of the instances selected by mask (default: all) until each one's next reset. overrides is one dict for all selected instances or a list with one dict per selected instance (in instance order), in the schema of the parameters argument; only the drone itself varies per instance (dynamics.dynamics, dynamics.imu, camera), the rest is shared and set through parameters=, and the rendered body follows the drone sampled at reset."""
        mask = self._mask(mask)
        selected = np.flatnonzero(mask)
        if isinstance(overrides, dict):
            overrides = [overrides] * len(selected)
        if len(overrides) != len(selected):
            raise ValueError(f"hyperdrone: {len(overrides)} overrides for {len(selected)} selected instances")
        current = self.instance_parameters()
        merged = []
        for instance_i, instance_overrides in zip(selected, overrides):
            self._check_per_drone(instance_overrides)
            merged.append(self._resolve(current[instance_i], instance_overrides))
        self._environment.set_instance_parameters(mask, json.dumps(merged))

    @property
    def dt(self):
        return self.parameters["dynamics"]["integration"]["dt"]

    def close(self):
        self._environment = None

    def _mask(self, mask):
        if mask is None:
            mask = np.ones(self.total_instances, dtype=bool)
        return np.ascontiguousarray(np.asarray(mask, dtype=bool).reshape(self.total_instances))

    def reset(self, mask=None):
        mask = self._mask(mask)
        self._environment.reset(mask)
        self._sampled |= mask

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
            reset_mask = np.zeros(self.total_instances, dtype=bool)
        reset_mask = self._mask(reset_mask)
        self._check_synchronized(reset_mask)
        self._environment.render(reset_mask)

    def observe(self):
        observations = np.empty((self.total_instances, self.observation_dim), dtype=np.float32)
        self._environment.observe(observations)
        return observations

    def observe_privileged(self):
        observations = np.empty((self.total_instances, self.observation_dim_privileged), dtype=np.float32)
        self._environment.observe_privileged(observations)
        return observations

    def observe_imu(self):
        """The IMU sample of the last step: (total, observation_dim_imu), named by observation_layout_imu."""
        if self.observation_dim_imu == 0:
            raise ValueError(f"hyperdrone: this configuration has no IMU observation (task={self.config.task!r})")
        observations = np.empty((self.total_instances, self.observation_dim_imu), dtype=np.float32)
        self._environment.observe_imu(observations)
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
        self._environment.step(actions)

    def rewards(self):
        rewards = np.empty(self.total_instances, dtype=np.float32)
        self._environment.rewards(rewards)
        return rewards

    def terminated(self):
        flags = np.empty(self.total_instances, dtype=bool)
        self._environment.terminated(flags)
        return flags

    def rotate_scene(self):
        self._environment.rotate_scene()

    def frames(self):
        """The latest visual observation as (total, *observation_layout.shape) images."""
        return self.observe().reshape(self.total_instances, *self.observation_layout.shape)
