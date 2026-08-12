from dataclasses import dataclass

from .. import jit

MODELS = (
    "crazyflie",
    "crazyflie_openmv",
    "mrs",
    "x500",
    "x500_real",
    "x500_sim",
    "arpl",
    "fs_base",
    "flightmare",
    "soft",
    "soft_rigid",
)

STATE_COMPONENTS = {
    "position": (0, 3),
    "orientation": (1, 4),
    "linear_velocity": (2, 3),
    "angular_velocity": (3, 3),
    "rpm": (4, 4),
}


@dataclass(frozen=True)
class SimConfig:
    num_drones: int
    domain_randomization: bool

    def __post_init__(self):
        if self.num_drones < 1:
            raise ValueError("num_drones must be >= 1")

    def canonical(self):
        # must match build_config_string() in _native/dynamics/impl.cpp exactly
        return f"n={self.num_drones};dr={1 if self.domain_randomization else 0}"

    def key(self):
        return jit.canonical_key(self.canonical())

    def defines(self):
        return {
            "HYPERDRONE_DYNAMICS_NUM_DRONES": self.num_drones,
            "HYPERDRONE_DYNAMICS_DOMAIN_RANDOMIZATION": 1 if self.domain_randomization else 0,
        }
