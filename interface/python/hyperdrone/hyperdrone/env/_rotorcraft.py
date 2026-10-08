import numpy as np

from .. import jit, render
from ._multi_environment import component


def _rotors(value, what):
    array = np.ascontiguousarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{what} must have shape (num_rotors, 3), got {array.shape}")
    return array


class Rotorcraft(render.Rig):
    """The body/prop_* drone rig the environment renders for drone_asset, as a render.Rig:
    part 0 is the body, every prop_* part spins about its hub (the centre of its geometry),
    any other part rides the body. Without rotors, coordinate i is the phase of the i-th
    prop_* part and the spin follows the Quad-X convention (front-right CCW). With a
    dynamics model's rotor_positions and rotor_torque_directions, both (num_rotors, 3) as in
    Sim.model_parameters(), each hub pairs with its nearest rotor: coordinate i is then the
    phase of rotor i, spinning against the rotor's reaction torque, so sim.state["rpm"]
    integrates straight into the coordinates."""

    def __init__(self, assembly, rotor_positions=None, rotor_torque_directions=None):
        if (rotor_positions is None) != (rotor_torque_directions is None):
            raise ValueError("rotor_positions and rotor_torque_directions come together")
        if rotor_positions is not None:
            rotor_positions = _rotors(rotor_positions, "rotor_positions")
            rotor_torque_directions = _rotors(rotor_torque_directions, "rotor_torque_directions")
        render.load_core()  # registers RigModel, which the env core instantiates
        self._bind(jit.core(component()).rotorcraft_rig(assembly, rotor_positions, rotor_torque_directions))
