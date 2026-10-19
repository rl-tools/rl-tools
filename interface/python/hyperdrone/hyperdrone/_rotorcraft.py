import numpy as np

from .render import load_core


def _rotors(value, what):
    array = np.ascontiguousarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{what} must have shape (num_rotors, 3), got {array.shape}")
    return array


class Rotorcraft:
    """A body/prop_* assembly articulated as a rotorcraft: constructing it writes the joints onto
    assembly (part 0 is the body, every prop_* part spins about its hub, the centre of its
    geometry, every other part is fixed). Without rotors coordinate i is the phase of the i-th
    prop_* part spinning per Quad-X (front-right CCW) and scale is 1. With a dynamics model's
    rotor_positions and rotor_torque_directions, both (num_rotors, 3) as in
    sim.parameters["dynamics"]["dynamics"], each hub pairs with its nearest rotor: coordinate i is
    then the phase of rotor i spinning against its reaction torque, scale fits the assembly onto
    the rotors' wheelbase (pass it as Pose.scale), and rotor_position_deviation is the largest
    hub error as a fraction of the wheelbase (warned above 10%). Coordinates are radians;
    integrate sim.rotor_rpm() * 2 * pi / 60 over the time step to advance them."""

    def __init__(self, assembly, *, rotor_positions=None, rotor_torque_directions=None):
        if (rotor_positions is None) != (rotor_torque_directions is None):
            raise ValueError("rotor_positions and rotor_torque_directions come together")
        if rotor_positions is not None:
            rotor_positions = _rotors(rotor_positions, "rotor_positions")
            rotor_torque_directions = _rotors(rotor_torque_directions, "rotor_torque_directions")
        _, self.scale, self.rotor_position_deviation = load_core().rotorcraft(assembly, rotor_positions, rotor_torque_directions)
