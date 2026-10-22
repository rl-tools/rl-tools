from dataclasses import dataclass

from hyperlux import _jit as jit
from .._control import ACTUATIONS, CONTROLS, validate_control

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
    "crazyflie_brushless",
    "savagebee_pusher",
)

STATE_COMPONENTS = {
    "position": (0, 3),
    "orientation": (1, 4),
    "linear_velocity": (2, 3),
    "angular_velocity": (3, 3),
    "motor_command": (4, 4),
}


@dataclass(frozen=True, kw_only=True)
class Config:
    num_drones: int
    control: str = "motor"
    actuation: str = "rotors"

    def __post_init__(self):
        if type(self.num_drones) is not int or self.num_drones < 1:
            raise ValueError("num_drones must be a positive integer")
        validate_control(self.control, self.actuation)

    def canonical(self):
        # must match build_config_string() in _native/dynamics/impl.cpp exactly
        return f"n={self.num_drones} control={CONTROLS.index(self.control)} actuation={ACTUATIONS.index(self.actuation)}"

    def key(self):
        return jit.canonical_key(self.canonical())

    def defines(self):
        return {
            "HYPERDRONE_DYNAMICS_NUM_DRONES": self.num_drones,
            "HYPERDRONE_DYNAMICS_CONTROL": CONTROLS.index(self.control),
            "HYPERDRONE_DYNAMICS_ACTUATION": ACTUATIONS.index(self.actuation),
        }
