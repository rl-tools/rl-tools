import numpy as np

from hyperlux import _jit as jit

from ._components import component


def _core():
    return jit.core(component("rotorcraft", "cpu", core_target="hyperdrone_rotorcraft_core"))


def _rotors(value, what):
    array = np.ascontiguousarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{what} must have shape (num_rotors, 3), got {array.shape}")
    return array


def articulate(assembly):
    """Write the joints of a body/prop_* drone asset: part 0 is the body; prop p, the p-th prop_*
    part, spins about +z through its hub, the centre of its geometry, as coordinate p; every other
    part is fixed. Joints bind when the assembly enters the pool, so articulate before add_assembly."""
    _core().articulate(assembly)


class Rotorcraft:
    """The convention that drives an articulated body/prop_* assembly (see articulate), read from
    its joints: prop p turns prop_spin[p] * phase[prop_rotor[p]], which coordinates(rotor_phases)
    evaluates. Without rotors prop p follows rotor p spinning per Quad-X (front-right CCW) and
    scale is 1. With a dynamics model's rotor_positions and rotor_torque_directions, both
    (num_rotors, 3) as in sim.parameters["dynamics"]["dynamics"], each hub pairs with its nearest
    rotor and spins against its reaction torque, scale fits the assembly onto the rotors'
    wheelbase (pass it as Pose.scale), and rotor_position_deviation is the largest hub error as a
    fraction of the wheelbase (warned above 10%). Phases are radians; integrate
    sim.rotor_rpm() * 2 * pi / 60 over the time step to advance them."""

    def __init__(self, assembly, *, rotor_positions=None, rotor_torque_directions=None):
        if (rotor_positions is None) != (rotor_torque_directions is None):
            raise ValueError("rotor_positions and rotor_torque_directions come together")
        if rotor_positions is not None:
            rotor_positions = _rotors(rotor_positions, "rotor_positions")
            rotor_torque_directions = _rotors(rotor_torque_directions, "rotor_torque_directions")
        _, self.scale, self.rotor_position_deviation, prop_rotor, prop_spin = _core().rotorcraft(assembly, rotor_positions, rotor_torque_directions)
        self.prop_rotor = np.asarray(prop_rotor, dtype=np.intp)
        self.prop_spin = np.asarray(prop_spin, dtype=np.float32)

    def coordinates(self, rotor_phases):
        """The assembly's joint coordinates at rotor_phases, (num_rotors,) radians or a batch
        (..., num_rotors): prop p turns prop_spin[p] * rotor_phases[prop_rotor[p]]."""
        phases = np.asarray(rotor_phases, dtype=np.float32)
        return phases[..., self.prop_rotor] * self.prop_spin
