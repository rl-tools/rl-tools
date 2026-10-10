"""Direct bindings to the C++ rl_tools hyperdrone::MultiEnvironment<World>.

The JIT extension binds the rl_tools batch verbs; this class mirrors that
verb surface one-to-one (reset / observe / observe_privileged / step / rewards /
terminated / rotate_scene) so a notebook drives the exact environment the C++ training
targets use — no runners involved.

Configuration follows the C++ extension ladder: a preset names the platform
(`presets::*` on the C++ side), a task names the wrapper (`tasks::*`), and a handful of
universal knobs (instances_per_environment, camera resolution, fidelity, motion blur) complete the picture. Arbitrary
C++ specifications are reached through `spec_header=`: a path to a C++ header defining
`hyperdrone_env_user::WORLD` (including any rl_tools headers it needs); the header's
content hash is part of the JIT key, and preset/task/num_agents/num_cameras_per_agent must stay at their
defaults since the user type supersedes them.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field, fields

import numpy as np

import conta

from .. import _jit as jit
from .._dlpack import is_cuda
from .._outputs import VALID_MOTION_BLUR_SAMPLES, normalize_outputs, output_defines
from .._parameters import merge, resolve

PRESETS = {"crazyflie": 0, "x500_fpv": 1, "x500_fpv_imu": 2}
SELF_VISIBLE_PRESETS = ("x500_fpv", "x500_fpv_imu")
TASKS = {None: 0, "target_frame": 1, "moving_gate": 2, "visual_inertial_localization": 3}
DATASETS = {"glb": 0, "procthor": 1}


def _backend():
    from ..render._component import _backend as render_backend

    return render_backend()


def component():
    resolved = _backend()
    return jit.Component(
        name="env",
        variant=resolved.lower(),
        core_target="hyperdrone_env_core",
        needs_cuda=resolved == "optix",
    )


_SHADING = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True, kw_only=True)
class EnvConfig:
    num_environments: int = 1
    instances_per_environment: int = 4
    width: int = 32
    height: int = 32
    fidelity: str = "high"
    preset: str = "crazyflie"
    task: str = None
    num_agents: int = 1
    num_cameras_per_agent: int = 1
    spec_header: str = None
    device: str = None
    motion_blur_samples: int = 1
    output: tuple = ("rgb",)
    semantic_segmentation: bool = False
    dataset: str = "glb"

    def __post_init__(self):
        for name in ("num_environments", "instances_per_environment", "width", "height", "num_agents", "num_cameras_per_agent"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"hyperdrone: {name} must be a positive integer")
        object.__setattr__(self, "output", normalize_outputs(self.output))
        if "rgb" not in self.output:
            raise ValueError("hyperdrone: environment observations require rgb output")
        if self.semantic_segmentation and "segmentation" not in self.output:
            raise ValueError("hyperdrone: semantic_segmentation requires segmentation output")
        if self.dataset not in DATASETS:
            raise ValueError(f"hyperdrone: unknown dataset {self.dataset!r} (one of {sorted(DATASETS)})")
        if self.preset not in PRESETS:
            raise ValueError(f"hyperdrone: unknown preset {self.preset!r} (one of {sorted(PRESETS)})")
        if self.task not in TASKS:
            raise ValueError(f"hyperdrone: unknown task {self.task!r} (one of {sorted(k for k in TASKS if k)} or None)")
        if self.fidelity not in _SHADING:
            raise ValueError(f"hyperdrone: unknown fidelity {self.fidelity!r} (one of {sorted(_SHADING)})")
        if type(self.motion_blur_samples) is not int or self.motion_blur_samples not in VALID_MOTION_BLUR_SAMPLES:
            raise ValueError(f"hyperdrone: motion_blur_samples must be one of {VALID_MOTION_BLUR_SAMPLES}")
        if self.task is not None and self.num_agents > 1:
            raise ValueError(f"hyperdrone: task {self.task!r} is single-agent")
        if self.task == "target_frame" and self.num_cameras_per_agent != 1:
            raise ValueError("hyperdrone: task 'target_frame' requires one camera per agent")
        if self.device not in (None, "cuda", "cpu"):
            raise ValueError(f"hyperdrone: unknown device {self.device!r} (one of 'cuda', 'cpu' or None)")
        if self.num_agents > 1 and self.preset not in SELF_VISIBLE_PRESETS:
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
            if self.device == "cuda":
                raise ValueError("hyperdrone: task 'visual_inertial_localization' has no CUDA verbs — use device='cpu'")
        if self.spec_header is not None:
            defaults = EnvConfig()
            if self.output != defaults.output or self.semantic_segmentation:
                raise ValueError("hyperdrone: spec_header defines the outputs; leave output and semantic_segmentation at their defaults")
            superseded = [item.name for item in fields(self) if item.name not in ("num_environments", "spec_header", "device", "dataset") and getattr(self, item.name) != getattr(defaults, item.name)]
            if superseded:
                raise ValueError(f"hyperdrone: spec_header supersedes {', '.join(superseded)}; leave these fields at their defaults")
            if not os.path.isfile(self.spec_header):
                raise ValueError(f"hyperdrone: spec_header not found: {self.spec_header}")

    @property
    def world_device(self):
        """Where the World runs: device, or by default 'cuda' on the OptiX backend for the
        presets and tasks with CUDA verbs and 'cpu' otherwise (spec_header Worlds included, whose
        verbs may be host code)."""
        if self.device is not None:
            if self.device == "cuda" and _backend() != "optix":
                raise ValueError("hyperdrone: device='cuda' needs the OptiX backend")
            return self.device
        if _backend() == "optix" and self.spec_header is None and self.task != "visual_inertial_localization":
            return "cuda"
        return "cpu"

    def defines(self):
        values = {
            **output_defines(self.output, "HYPERDRONE_ENV"),
            "HYPERDRONE_ENV_SEMANTIC_SEGMENTATION": int(self.semantic_segmentation),
            "HYPERDRONE_ENV_DATASET": DATASETS[self.dataset],
            "HYPERDRONE_ENV_NUM_ENVIRONMENTS": self.num_environments,
            "HYPERDRONE_ENV_INSTANCES": self.instances_per_environment,
            "HYPERDRONE_ENV_CAM_WIDTH": self.width,
            "HYPERDRONE_ENV_CAM_HEIGHT": self.height,
            "HYPERDRONE_ENV_SHADING": _SHADING[self.fidelity],
            "HYPERDRONE_ENV_MB_SAMPLES": self.motion_blur_samples,
            "HYPERDRONE_ENV_PRESET": PRESETS[self.preset],
            "HYPERDRONE_ENV_TASK": TASKS[self.task],
            "HYPERDRONE_ENV_N_AGENTS": self.num_agents,
            "HYPERDRONE_ENV_N_CAMERAS_PER_AGENT": self.num_cameras_per_agent,
            "HYPERDRONE_ENV_DEVICE_CUDA": 1 if self.world_device == "cuda" else 0,
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


@dataclass(frozen=True, kw_only=True)
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


class MultiEnvironment:
    """The C++ environment behind the same verb surface: construct with scenes, then
    reset(mask) -> observe() -> step(actions) -> rewards()/terminated().

    scenes: a directory of .glb scenes (sorted corpus), or a reference / sequence of
    references — .glb paths, "conta:HASH" strings, or conta store manifest entries
    {"description": ..., "hash": ...}; a reference list is the corpus in list order.
    config.dataset: "glb" uses the authored taxonomy; "procthor" requires ai2thor-hab
    metadata for semantic output. The shared taxonomy covers all scenes and overlay assets.
    drones: the drone table every agent samples from uniformly at each reset, independently of
    the scene its environment shows: a list of {"asset": <GLB reference or None>, "parameters":
    {partial dict}}. The asset is the body/prop_* GLB rendered into the instance's cameras
    (required per entry when the compiled world is SELF_VISIBLE, rejected otherwise); the parameters are merged
    over the shared constructor base (see parameters) and may only touch what varies per drone: dynamics.dynamics
    (a "model" key loads an l2f registry entry), dynamics.imu and cameras. The same GLB may appear
    in several entries; distinct GLBs load once and need the same part count (the body swaps in
    place at reset). The table holds at most max_drone_types entries. drone_asset=X is the one-entry
    table [{"asset": X}]; without either, the table is the nominal drone.
    assets: task-owned references by name; moving_gate requires {"gate": reference}.
    All assets accept conta references too.

    parameters: a partial dict in the schema of the parameters property, {"dynamics": <l2f parameters>, "cameras": [{"mount": {"pose": {"position", "orientation"}}, "fov"}, ...]}, merged over the preset's nominal values as the base of every drone entry (the specification's randomization ranges apply on top at reset). cameras requires one partial dict per camera; {} keeps its nominal values. parameters["dynamics"]["dynamics"]["model"] loads an l2f registry entry (hyperdrone.dynamics.MODELS) before the rest is merged, J_inv follows J unless given, the camera orientation is a (w, x, y, z) quaternion mapping the camera's FLU axes into the body frame, and fov is the horizontal FOV in degrees.

    Tasks with an IMU stream (task='visual_inertial_localization') add observe_imu() — the
    IMU sample of the last step, named by observation_layout_imu — and render a camera frame
    only every frame_stride steps; their episodes are fixed-length and synchronized, so the
    reset mask must be all-or-none per environment. Autonomous tasks fly themselves:
    action_dim is 0 and step() takes no actions.

    Every verb takes host (numpy) arrays and, when the World runs on the GPU (device ==
    "cuda"), CUDA arrays of its GPU too (any DLPack producer, e.g. torch tensors); outputs are
    written into out= when given. CUDA arrays are used in place on the environment's
    execution stream (see the stream property); host arrays are staged and synchronized.
    """

    def __init__(self, scenes, *, config=None, seed=0, drone_asset=None, assets=None, parameters=None, drones=None, stream=None):
        self._spy = None
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
        self._module = jit.module(component(), self.config)
        self._synchronized_resets = self._module.synchronized_resets
        self.outputs = self._module.outputs
        self.segmentation_mode = self._module.segmentation_mode
        for name, value in self._module.dimensions.items():
            setattr(self, name, value)
        if len(drones) > self.max_drone_types:
            capacity_hint = "; raise MAX_DRONES in spec_header" if self.config.spec_header is not None else ""
            raise ValueError(f"hyperdrone: {len(drones)} drones but this configuration holds {self.max_drone_types}{capacity_hint}")
        for entry in drones:
            if self.self_visible and entry.get("asset") is None:
                raise ValueError("hyperdrone: this SELF_VISIBLE world needs an asset per drone; pass drone_asset= or drones=")
            if not self.self_visible and entry.get("asset") is not None:
                raise ValueError("hyperdrone: drone assets need a SELF_VISIBLE world, nothing would render them")
        scene_references = _normalize_scenes(scenes, self.num_environments)
        drone_assets = [_resolve_asset(entry.get("asset")) for entry in drones]
        assets = {name: _resolve_asset(asset) for name, asset in (assets or {}).items()}
        self.observation_layout = ObservationLayout(**self._module.observation_layout)
        self.observation_layout_privileged = ObservationLayout(**self._module.observation_layout_privileged)
        self.observation_layout_imu = ObservationLayout(**self._module.observation_layout_imu) if self.observation_dim_imu > 0 else None
        self._environment = self._module.Environment()
        if stream is not None:
            self.stream = stream
        nominal = self._resolve(json.loads(self._environment.parameters()), parameters or {})
        drone_parameters = []
        for entry in drones:
            overrides = entry.get("parameters") or {}
            if "drone" in overrides:
                raise ValueError("hyperdrone: drone is sampled at reset from the drones= table")
            self._check_per_drone(overrides)
            drone_parameters.append(json.dumps(self._resolve(nominal, overrides)))
        self._environment.init(scene_references, drone_assets, drone_parameters, assets, seed)

    @staticmethod
    def _check_per_drone(overrides):
        shared = sorted(set(overrides.get("dynamics", {})) - {"dynamics", "imu"})
        if shared:
            raise ValueError(f"hyperdrone: dynamics {shared} are shared by all instances; set them through parameters=")

    def _resolve(self, base, overrides):
        overrides = dict(overrides)
        if "cameras" in overrides:
            cameras = overrides["cameras"]
            if not isinstance(cameras, (list, tuple)) or len(cameras) != self.num_cameras_per_agent:
                raise ValueError(f"hyperdrone: provide {self.num_cameras_per_agent} camera overrides per agent")
            if any(not isinstance(camera, dict) for camera in cameras):
                raise ValueError("hyperdrone: each camera override must be a dict")
            overrides["cameras"] = [merge(nominal, camera, f"cameras[{index}].") for index, (nominal, camera) in enumerate(zip(base["cameras"], cameras))]
        return resolve(base, overrides, self._environment.model)

    @property
    def parameters(self):
        """The resolved nominal parameters of drone 0, including its entry-specific overrides."""
        return json.loads(self._environment.parameters())

    @property
    def drones(self):
        """The drone table every agent samples from at reset: one dict per entry with the resolved asset path and nominal parameters; each agent's "drone" value indexes it."""
        return json.loads(self._environment.drones())

    def instance_parameters(self):
        """Current episode parameters, one dict per instance. Single-agent entries use the parameters schema plus "drone", the model index; multi-agent entries contain "agents", a list of these dicts in agent order."""
        return json.loads(self._environment.instance_parameters())

    def set_instance_parameters(self, overrides, mask=None):
        """Overrides current parameters until reset: one dict for all selected instances or a list with one dict per selected instance. Multi-agent dicts contain "agents", one override dict per agent; use {} to keep an agent unchanged. Per-agent keys are dynamics.dynamics, dynamics.imu, cameras and "drone". cameras takes one partial dict per camera. Switching "drone" starts from that table entry's nominal before merging other overrides. Integration and other shared settings belong in constructor parameters=. Rotor overrides must preserve each rotor's XY quadrant and spin direction."""
        mask = self._host_mask(mask)
        selected = np.flatnonzero(mask)
        if isinstance(overrides, dict):
            overrides = [overrides] * len(selected)
        if len(overrides) != len(selected):
            raise ValueError(f"hyperdrone: {len(overrides)} overrides for {len(selected)} selected instances")
        current = self.instance_parameters()
        drones = self.drones
        merged = []
        for instance_i, instance_overrides in zip(selected, overrides):
            if self.num_agents == 1:
                per_agent_overrides = [instance_overrides]
                agents = [current[instance_i]]
            else:
                if not isinstance(instance_overrides, dict) or set(instance_overrides) != {"agents"}:
                    raise ValueError("hyperdrone: multi-agent overrides need an 'agents' list")
                per_agent_overrides = instance_overrides["agents"]
                if not isinstance(per_agent_overrides, (list, tuple)) or len(per_agent_overrides) != self.num_agents:
                    raise ValueError(f"hyperdrone: provide {self.num_agents} agent overrides per instance")
                agents = current[instance_i]["agents"]
            updated = []
            for base, agent_overrides in zip(agents, per_agent_overrides):
                if not isinstance(agent_overrides, dict):
                    raise ValueError("hyperdrone: each agent override must be a dict")
                self._check_per_drone(agent_overrides)
                if self.config.task == "target_frame" and ("cameras" in agent_overrides or "drone" in agent_overrides):
                    raise ValueError("hyperdrone: task 'target_frame' takes each episode's drone and camera from its target setup; set the nominal through parameters= and drones=")
                if "drone" in agent_overrides:
                    agent_overrides = dict(agent_overrides)
                    drone = agent_overrides.pop("drone")
                    if not isinstance(drone, (int, np.integer)) or not 0 <= drone < len(drones):
                        raise ValueError(f"hyperdrone: drone must index the drones table (0..{len(drones) - 1}), got {drone!r}")
                    base = dict(drones[drone]["parameters"], drone=int(drone))
                agent_parameters = self._resolve(base, agent_overrides)
                self._check_rotor_layout(drones[agent_parameters["drone"]]["parameters"], agent_parameters)
                updated.append(agent_parameters)
            merged.append(updated[0] if self.num_agents == 1 else {"agents": updated})
        self._environment.set_instance_parameters(mask, json.dumps(merged))

    def _check_rotor_layout(self, nominal, parameters):
        if not self.self_visible:
            return
        def layout(values):
            dynamics = values["dynamics"]["dynamics"]
            return np.sign(np.asarray(dynamics["rotor_positions"])[:, :2]).tolist(), np.sign(np.asarray(dynamics["rotor_torque_directions"])[:, 2]).tolist()
        if layout(parameters) != layout(nominal):
            raise ValueError("hyperdrone: the rendered props follow the table entry's rotors; per-instance rotor_positions and rotor_torque_directions must keep each rotor's XY quadrant and spin direction (switch the drone through \"drone\" instead)")

    @property
    def dt(self):
        return self.parameters["dynamics"]["integration"]["dt"]

    @property
    def device(self):
        """"cuda" when the World runs on the GPU (EnvConfig.device), "cpu" otherwise."""
        return "cuda" if self._module.CUDA_WORLD else "cpu"

    @property
    def render_backend(self):
        """The lowercase name of this environment's compiled raytracing backend."""
        return self._module.render_backend

    @property
    def device_id(self):
        """The CUDA device the World runs on (the current one at construction)."""
        return self._environment.device_id

    @property
    def stream(self):
        """The cudaStream_t every verb is enqueued on (0 on the cpu device). Set it to the stream
        your arrays are produced and consumed on (e.g. torch.cuda.current_stream().cuda_stream);
        work enqueued so far stays ordered before the switch."""
        return self._environment.stream

    @stream.setter
    def stream(self, value):
        self._environment.stream = int(value)

    def synchronize(self):
        self._environment.synchronize()

    def close(self):
        if self._spy is not None:
            self._spy.close()
        self._environment = None

    def spy(self):
        """Open the native camera mosaic and spectator on the main thread. Use `with env.spy() as spy:` and call spy.update() after normal observe/render calls; it never renders observations itself. Closing the window leaves this environment running."""
        from ._spy import Spy

        if self._environment is None:
            raise RuntimeError("hyperdrone: cannot open a spy on a closed environment")
        if self._spy is not None and self._spy._environment is not None:
            raise RuntimeError("hyperdrone: this environment already has an open spy")
        self._spy = Spy(self._environment)
        return self._spy

    def _host_mask(self, mask):
        if mask is None:
            mask = np.ones(self.total_instances, dtype=bool)
        return np.ascontiguousarray(np.asarray(mask, dtype=bool).reshape(self.total_instances))

    def _input(self, array, dtype, shape):
        if is_cuda(array):
            return array
        return np.ascontiguousarray(np.asarray(array, dtype=dtype).reshape(shape))

    def _output(self, out, shape, dtype):
        return np.empty(shape, dtype=dtype) if out is None else out

    def spawn_points(self):
        """Record each instance for reset(spawn_points=...): scene hash, one model index per agent in "drone", scene placement, per-agent poses (xyz + wxyz), shutter fraction. Camera mounts and dynamics come from the drone table."""
        return json.loads(self._environment.spawn_points())

    def reset(self, mask=None, *, spawn_points=None):
        """Resets the selected instances (default: all), optionally replaying one recorded spawn per selected instance (a host mask then). Replayed spawns fly their entry's nominal drone parameters (target frames follow the replayed placement); velocities, motor speeds and task-specific state still follow their configured initialization."""
        if spawn_points is None:
            self._environment.reset(None if mask is None else self._input(mask, bool, (self.total_instances,)))
            return
        if mask is not None and is_cuda(mask):
            raise ValueError("hyperdrone: spawn points take a host mask")
        mask = self._host_mask(mask)
        if len(spawn_points) != int(mask.sum()):
            raise ValueError("hyperdrone: provide one spawn per selected instance")
        for spawn in spawn_points:
            if set(spawn) != {"scene", "drone", "scene_translation", "scene_yaw", "poses", "shutter_fraction"}:
                raise ValueError("hyperdrone: invalid spawn fields (use spawn_points() to capture them)")
            if not isinstance(spawn["drone"], list) or len(spawn["drone"]) != self.num_agents or any(type(drone) is not int for drone in spawn["drone"]) or not isinstance(spawn["scene"], str):
                raise ValueError("hyperdrone: spawn needs one integer drone index per agent and a scene content hash")
            for name, shape in (("scene_translation", (3,)), ("scene_yaw", (2,)), ("poses", (self.num_agents, 7)), ("shutter_fraction", ())):
                values = np.asarray(spawn[name], dtype=float)
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError(f"hyperdrone: spawn {name} must have shape {shape} and finite values")
            if not np.isclose(np.linalg.norm(spawn["scene_yaw"]), 1) or not np.allclose(np.linalg.norm(np.asarray(spawn["poses"])[:, 3:], axis=1), 1):
                raise ValueError("hyperdrone: spawn yaw and orientation quaternions must have unit length")
            if not 0 <= spawn["shutter_fraction"] <= 1:
                raise ValueError("hyperdrone: spawn shutter_fraction must be in [0, 1]")
        self._environment.reset(mask, json.dumps(spawn_points, allow_nan=False))

    def observe(self, *, out=None):
        """The visual observation, (total, observation_dim) float32."""
        out = self._output(out, (self.total_instances, self.observation_dim), np.float32)
        self._environment.observe(out)
        return out

    def observe_privileged(self, *, out=None):
        out = self._output(out, (self.total_instances, self.observation_dim_privileged), np.float32)
        self._environment.observe_privileged(out)
        return out

    def observe_imu(self, *, out=None):
        """The IMU sample of the last step: (total, observation_dim_imu), named by observation_layout_imu."""
        if self.observation_dim_imu == 0:
            raise ValueError(f"hyperdrone: this configuration has no IMU observation (task={self.config.task!r})")
        out = self._output(out, (self.total_instances, self.observation_dim_imu), np.float32)
        self._environment.observe_imu(out)
        return out

    def step(self, actions=None):
        if actions is None:
            if self.action_dim != 0:
                raise ValueError(
                    "hyperdrone: step() needs actions of shape (total_instances, action_dim); "
                    "only autonomous tasks (action_dim == 0) step without them"
                )
            actions = np.empty((self.total_instances, 0), dtype=np.float32)
        self._environment.step(self._input(actions, np.float32, (self.total_instances, self.action_dim)))

    def rewards(self, *, out=None):
        out = self._output(out, self.total_instances, np.float32)
        self._environment.rewards(out)
        return out

    def terminated(self, *, out=None):
        out = self._output(out, self.total_instances, bool)
        self._environment.terminated(out)
        return out

    def rotate_scene(self):
        """Selects the next scene of each environment; reset every instance before using it."""
        self._environment.rotate_scene()

    def frames(self):
        """The latest visual observation as (total, *observation_layout.shape) images."""
        return self.observe().reshape(self.total_instances, *self.observation_layout.shape)

    def _render_output(self, name, dtype, channels, out):
        shape = (self.total_instances, self.num_views, self.height, self.width)
        if channels > 1:
            shape += (channels,)
        out = self._output(out, shape, dtype)
        getattr(self._environment, name)(out)
        return out

    def depth(self, *, out=None):
        """Camera-forward depth at pixel centres and shutter close, float32 (total_instances, num_views, H, W); misses are +inf. Acquires pending frames."""
        return self._render_output("depth", np.float32, 1, out)

    def segmentation(self, *, out=None):
        """Instance IDs or semantic classes, uint32 (total_instances, num_views, H, W); miss=0xFFFFFFFF."""
        return self._render_output("segmentation", np.uint32, 1, out)

    @property
    def segmentation_classes(self):
        """A copy of the shared {semantic ID: name} dictionary; empty without a named semantic taxonomy."""
        return self._environment.segmentation_classes

    @property
    def segmentation_taxonomy(self):
        """The shared semantic taxonomy name; empty when unnamed or semantic output is disabled."""
        return self._environment.segmentation_taxonomy

    def normals(self, *, out=None):
        """Scene-frame FLU geometric unit normals, float32 (total_instances, num_views, H, W, 3); miss=zero."""
        return self._render_output("normals", np.float32, 3, out)

    def flow(self, *, out=None):
        """Current-to-previous capture displacement in pixels, float32 (total_instances, num_views, H, W, 2).

        First captures and reset instances use identical poses at both instants. Every due capture is retained even when unread; repeated reads reuse the same output. Use flow_status() to mask it.
        """
        return self._render_output("flow", np.float32, 2, out)

    def flow_status(self, *, out=None):
        """FlowStatus codes, uint32 (total_instances, num_views, H, W)."""
        return self._render_output("flow_status", np.uint32, 1, out)
