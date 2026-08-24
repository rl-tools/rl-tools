import math

import numpy as np

from ._component import ensure_sim_library, load_core, resolve_device
from ._config import MODELS, STATE_COMPONENTS, SimConfig

IDENTITY_MOUNT = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32
)


class _DLPackBuffer:
    """DLPack producer over a live sim buffer; capsules are single-use, so each __dlpack__
    call mints a fresh one. The sim reference keeps the memory alive."""

    def __init__(self, sim, make_capsule, device_type):
        self._sim = sim
        self._make_capsule = make_capsule
        self._device_type = device_type

    def __dlpack__(self, stream=None, **_kwargs):
        return self._make_capsule()

    def __dlpack_device__(self):
        return (self._device_type, 0)


class SimParameters:
    """Per-drone runtime physical parameters (host numpy in/out). Writes persist until
    the next reset(), which restores the model preset before optional randomization."""

    NAMES = ("mass", "hovering_throttle_relative", "dt")

    def __init__(self, sim):
        self._sim = sim

    def __getitem__(self, name):
        out = np.empty(self._sim.num_drones, dtype=np.float32)
        self._sim._sim.read_parameter(name, out)
        return out

    def __setitem__(self, name, values):
        array = np.ascontiguousarray(np.broadcast_to(np.asarray(values, dtype=np.float32), (self._sim.num_drones,)))
        self._sim._sim.write_parameter(name, array)

    def keys(self):
        return self.NAMES


class SimState:
    """Zero-copy views of the published state component buffers, (num_drones, dim) each;
    device-resident on the CUDA variant. Refreshed in place by step()/reset()."""

    def __init__(self, sim):
        self._sim = sim

    def __getitem__(self, name):
        component, _ = STATE_COMPONENTS[name]
        jit = self._sim._sim
        return _DLPackBuffer(self._sim, lambda: jit.state_dlpack(component), jit.buffer_device_type)

    def __setitem__(self, name, values):
        component, dim = STATE_COMPONENTS[name]
        array = np.ascontiguousarray(values, dtype=np.float32).reshape(self._sim.num_drones, dim)
        self._sim._sim.set_state(component, array)

    def numpy(self, name):
        component, dim = STATE_COMPONENTS[name]
        out = np.empty((self._sim.num_drones, dim), dtype=np.float32)
        self._sim._sim.read_state(component, out)
        return out

    def keys(self):
        return STATE_COMPONENTS.keys()


class Sim:
    """Vectorized, stateful L2F multirotor simulator.

    Compile-time constants (num_drones, domain randomization) select a JIT-compiled
    library; the model preset, integration dt, and all physical parameters are runtime.
    On device="cuda" the state, observation, and camera-basis buffers are CUDA-resident
    and hand over to the renderer without touching the host.
    """

    def __init__(self, num_drones, model="crazyflie", dt=None, device="auto", domain_randomization=False):
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        self.config = SimConfig(num_drones=int(num_drones), domain_randomization=bool(domain_randomization))
        self.device = resolve_device(device).lower()
        library = ensure_sim_library(self.config, self.device)
        core = load_core(self.device)
        self._sim = core.JitSim(str(library), self.config.canonical())
        if not self._sim.set_model(model):
            raise ValueError(f"hyperdrone: unknown dynamics model {model}")
        self.model = model
        if dt is not None:
            self._sim.set_dt(float(dt))
        self.state = SimState(self)
        self.parameters = SimParameters(self)

    @property
    def num_drones(self):
        return self._sim.num_drones

    @property
    def action_dim(self):
        return self._sim.action_dim

    @property
    def observation_dim(self):
        return self._sim.observation_dim

    @property
    def dt(self):
        return self._sim.dt

    @dt.setter
    def dt(self, value):
        self._sim.set_dt(float(value))

    @property
    def stream(self):
        """cudaStream_t handle of the sim's stream (0 on the cpu variant); pass to
        renderer.set_cameras(..., stream=...) for event-ordered device hand-off."""
        return self._sim.stream

    def set_model(self, model):
        if not self._sim.set_model(model):
            raise ValueError(f"hyperdrone: unknown dynamics model {model}")
        self.model = model

    def reset(self, seed=0, sample_states=True, sample_parameters=False):
        """Deterministic given seed. Sampling runs on the host with the rl_tools CPU
        engine, so cpu and cuda variants reset to identical states and parameters."""
        self._sim.reset(int(seed), bool(sample_parameters), bool(sample_states))
        return self

    def step(self, actions, stream=0):
        """Advance all drones by one dt. actions: (num_drones, action_dim) float32 in
        [-1, 1] (normalized rotor commands), any DLPack producer; CUDA-resident tensors
        stay on the device (stream: the producer's cudaStream_t handle)."""
        device = getattr(actions, "__dlpack_device__", None)
        if device is not None and device()[0] == 2:  # kDLCUDA
            if not hasattr(self._sim, "step_device"):
                raise RuntimeError("hyperdrone: device-resident actions require the cuda dynamics variant")
            self._sim.step_device(actions, stream)
            return self
        array = np.ascontiguousarray(np.asarray(actions), dtype=np.float32)
        if array.shape != (self.num_drones, self.action_dim):
            raise ValueError(f"actions must have shape ({self.num_drones}, {self.action_dim})")
        self._sim.step(array)
        return self

    def synchronize(self):
        self._sim.synchronize()

    def observations(self):
        """Zero-copy view of the observation buffer; call update first via observe()."""
        self._sim.update_observations()
        return _DLPackBuffer(self, self._sim.observations_dlpack, self._sim.buffer_device_type)

    def observe(self):
        """Observations as a host numpy array, (num_drones, observation_dim)."""
        self._sim.update_observations()
        out = np.empty((self.num_drones, self.observation_dim), dtype=np.float32)
        self._sim.read_observations(out)
        return out

    def camera_bases(self, mount=None, fov=math.radians(80.0), aspect=1.0):
        """Packed camera ray-gen bases (num_drones, 12) derived from the drone poses:
        zero-copy DLPack producer, CUDA-resident on the cuda variant — feed directly to
        renderer.set_cameras(bases, stream=sim.stream). mount is a body-frame (3, 4)
        transform (camera looks along body +X by default)."""
        if mount is None:
            mount_array = IDENTITY_MOUNT
        else:
            mount_array = np.ascontiguousarray(np.asarray(mount), dtype=np.float32).reshape(3, 4)
        self._sim.update_camera_bases(mount_array, float(fov), float(aspect))
        return _DLPackBuffer(self, self._sim.camera_bases_dlpack, self._sim.buffer_device_type)

    def camera_bases_numpy(self, mount=None, fov=math.radians(80.0), aspect=1.0):
        self.camera_bases(mount=mount, fov=fov, aspect=aspect)
        out = np.empty((self.num_drones, 12), dtype=np.float32)
        self._sim.read_camera_bases(out)
        return out

