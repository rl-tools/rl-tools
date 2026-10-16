import json

import numpy as np

from .. import _cuda
from .._dlpack import DLPackBuffer as _DLPackBuffer, is_cuda
from .._parameters import resolve
from ._component import ensure_sim_library, load_core, _resolve_device
from ._config import STATE_COMPONENTS, SimConfig

IDENTITY_MOUNT = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)


def _load_cuda_core():
    return load_core("cuda")


class SimState:
    """Owned state snapshots; assignment updates the authoritative simulation state."""

    def __init__(self, sim):
        self._sim = sim

    def __getitem__(self, name):
        component, dim = STATE_COMPONENTS[name]
        out = np.empty((self._sim.num_drones, dim), dtype=np.float32)
        self._sim._sim.read_state(component, out)
        self._sim._pending_inputs.clear()
        return out

    def __setitem__(self, name, values):
        component, dim = STATE_COMPONENTS[name]
        array = np.ascontiguousarray(values, dtype=np.float32)
        if array.shape != (self._sim.num_drones, dim):
            raise ValueError(f"hyperdrone: state {name!r} must have shape ({self._sim.num_drones}, {dim})")
        self._sim._sim.set_state(component, array)
        self._sim._pending_inputs.clear()

    def keys(self):
        return STATE_COMPONENTS.keys()


class Sim:
    """Vectorized L2F dynamics with a fixed shared parameter set and explicit state updates.

    A seed initializes independent dynamics and observation streams once. Reset restores selected drones' nominal parameters and state; sample_states=True samples their initial states from a continuing host stream. Host accessors return owned arrays; *_dlpack accessors borrow publication buffers until the next producer update. Consumers must finish reading borrowed buffers before that update.
    """

    def __init__(self, num_drones, *, parameters=None, seed=0, device="cpu", control="motor"):
        self._config = SimConfig(num_drones=num_drones, control=control)
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("hyperdrone: seed must be an integer in [0, 2**32 - 1]")
        self.device = _resolve_device(device)
        library = ensure_sim_library(self._config, self.device)
        self._core = load_core(self.device)
        self._sim = self._core.JitSim(str(library), self._config.canonical())
        nominal = resolve(json.loads(self._sim.parameters()), {} if parameters is None else parameters, self._core.model_json)
        self._sim.init(seed, json.dumps(nominal))
        self._pending_inputs = []
        self.state = SimState(self)
        if self.device == "cuda":
            _cuda.register_provider(_load_cuda_core)

    @property
    def num_drones(self):
        return self._sim.num_drones

    @property
    def action_dim(self):
        return self._sim.action_dim

    @property
    def num_rotors(self):
        return self._sim.num_rotors

    @property
    def control(self):
        return self._config.control

    @property
    def action_low(self):
        return np.asarray(self._sim.action_low, dtype=np.float32)

    @property
    def action_high(self):
        return np.asarray(self._sim.action_high, dtype=np.float32)

    @property
    def observation_dim(self):
        return self._sim.observation_dim

    @property
    def parameters(self):
        return json.loads(self._sim.parameters())

    @property
    def dt(self):
        return self.parameters["dynamics"]["integration"]["dt"]

    @property
    def stream(self):
        """The CUDA stream handle (0 on CPU), for ordering device handoffs."""
        return self._sim.stream

    def _mask(self, mask):
        if mask is None:
            return np.ones(self.num_drones, dtype=np.bool_)
        array = np.ascontiguousarray(mask, dtype=np.bool_)
        if array.shape != (self.num_drones,):
            raise ValueError(f"hyperdrone: mask must have shape ({self.num_drones},)")
        return array

    def reset(self, mask=None, *, sample_states=False):
        self._sim.reset(self._mask(mask), bool(sample_states))
        self._pending_inputs.clear()

    def instance_parameters(self):
        return json.loads(self._sim.instance_parameters())

    def set_instance_parameters(self, overrides, mask=None):
        mask = self._mask(mask)
        selected = np.flatnonzero(mask)
        if isinstance(overrides, dict):
            overrides = [overrides] * len(selected)
        if len(overrides) != len(selected):
            raise ValueError(f"hyperdrone: {len(overrides)} overrides for {len(selected)} selected drones")
        current = self.instance_parameters()
        merged = [resolve(current[index], values, self._core.model_json) for index, values in zip(selected, overrides)]
        self._sim.set_instance_parameters(mask, json.dumps(merged))
        self._pending_inputs.clear()

    def step(self, actions, *, stream=0):
        """Advance one dt with the configured control commands of shape (num_drones, action_dim)."""
        if is_cuda(actions):
            if self.device != "cuda":
                raise RuntimeError("hyperdrone: device-resident actions require the cuda dynamics variant")
            self._sim.step_device(actions, int(stream))
            self._pending_inputs.append(actions)
            return
        array = np.ascontiguousarray(np.asarray(actions), dtype=np.float32)
        if array.shape != (self.num_drones, self.action_dim):
            raise ValueError(f"actions must have shape ({self.num_drones}, {self.action_dim})")
        self._sim.step(array)

    def synchronize(self):
        self._sim.synchronize()
        self._pending_inputs.clear()

    def _buffer(self, producer):
        return _DLPackBuffer(self, producer, self._sim.buffer_device_type, self._sim.buffer_device_id, self._sim.order_consumer)

    def state_dlpack(self, name):
        component, _ = STATE_COMPONENTS[name]
        return self._buffer(lambda: self._sim.state_dlpack(component))

    def observe_dlpack(self):
        self._sim.update_observations()
        return self._buffer(self._sim.observations_dlpack)

    def rotor_rpm(self):
        """Rotor speeds (num_drones, num_rotors) in RPM, using each drone's max_rpm calibration."""
        out = np.empty((self.num_drones, self.num_rotors), dtype=np.float32)
        self._sim.read_rotor_rpm(out)
        self._pending_inputs.clear()
        return out

    def observe(self):
        self._sim.update_observations()
        out = np.empty((self.num_drones, self.observation_dim), dtype=np.float32)
        self._sim.read_observations(out)
        self._pending_inputs.clear()
        return out

    def camera_bases_dlpack(self, *, mount=None, fov=80.0, aspect):
        """Borrow packed camera bases derived from authoritative drone poses; fov is in degrees."""
        mount_array = IDENTITY_MOUNT if mount is None else np.ascontiguousarray(mount, dtype=np.float32)
        if mount_array.shape != (3, 4):
            raise ValueError("hyperdrone: mount must have shape (3, 4)")
        if not np.isfinite(fov) or not 0 < fov < 180 or not np.isfinite(aspect) or aspect <= 0:
            raise ValueError("hyperdrone: fov must be in (0, 180) and aspect must be positive")
        self._sim.update_camera_bases(mount_array, float(fov), float(aspect))
        return self._buffer(self._sim.camera_bases_dlpack)

    def camera_bases(self, *, mount=None, fov=80.0, aspect):
        self.camera_bases_dlpack(mount=mount, fov=fov, aspect=aspect)
        out = np.empty((self.num_drones, 12), dtype=np.float32)
        self._sim.read_camera_bases(out)
        self._pending_inputs.clear()
        return out
