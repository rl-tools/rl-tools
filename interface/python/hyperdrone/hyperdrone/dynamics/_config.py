from dataclasses import dataclass

from .. import _jit as jit

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
class SimConfig:
    num_drones: int
    control: str = "motor"

    def __post_init__(self):
        if type(self.num_drones) is not int or self.num_drones < 1:
            raise ValueError("num_drones must be a positive integer")
        if self.control not in ("motor", "ctbr", "position"):
            raise ValueError("control must be 'motor', 'ctbr' or 'position'")

    def canonical(self):
        # must match build_config_string() in _native/dynamics/impl.cpp exactly
        return f"n={self.num_drones} control={('motor', 'ctbr', 'position').index(self.control)}"

    def key(self):
        return jit.canonical_key(self.canonical())

    def defines(self):
        return {
            "HYPERDRONE_DYNAMICS_NUM_DRONES": self.num_drones,
            "HYPERDRONE_DYNAMICS_CONTROL": ("motor", "ctbr", "position").index(self.control),
        }
