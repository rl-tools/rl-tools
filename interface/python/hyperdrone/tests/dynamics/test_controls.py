import copy

import numpy as np
import pytest

from hyperdrone import dynamics

N = 4
COMMAND = np.tile([0.2, -0.1, 0.3, 0.2], (N, 1)).astype(np.float32)
MASK = np.array([True, False, True, False])
COMPONENTS = ("position", "orientation", "linear_velocity", "angular_velocity", "motor_command")


@pytest.fixture(scope="module")
def ctbr():
    return dynamics.Sim(N, control="ctbr", device="cpu")


@pytest.fixture(scope="module")
def position_sims():
    return [dynamics.Sim(N, control="position", device="cpu", seed=3) for _ in range(3)]


def advance(sim, steps=8):
    for _ in range(steps):
        sim.step(COMMAND)


def prepare(sims):
    for sim in sims:
        sim.reset()
        advance(sim)


def copy_state(source, destination):
    for name in COMPONENTS:
        destination.state[name] = source.state[name]


def test_ctbr_physical_units_and_current_step_rates(ctbr):
    ctbr.reset()
    drone = ctbr.parameters["dynamics"]["dynamics"]
    hover = -drone["mass"] * drone["gravity"][2]
    actions = np.zeros((N, 4), dtype=np.float32)
    actions[0, 0] = hover
    actions[1, 3] = 2
    actions[2, 1] = 1
    actions[3, 2] = -1
    ctbr.step(actions)
    np.testing.assert_allclose(ctbr.state["position"][0], 0, atol=1e-7)
    np.testing.assert_allclose(ctbr.state["linear_velocity"][1:, 2], -9.81 * ctbr.dt, atol=1e-6)
    np.testing.assert_array_equal(ctbr.state["angular_velocity"], actions[:, 1:])
    np.testing.assert_allclose(ctbr.state["orientation"][1], [np.cos(ctbr.dt), 0, 0, np.sin(ctbr.dt)], atol=1e-6)
    actions[:, 1:] *= -1
    ctbr.step(actions)
    np.testing.assert_array_equal(ctbr.state["angular_velocity"], actions[:, 1:])
    np.testing.assert_allclose(ctbr.state["orientation"], np.tile([1, 0, 0, 0], (N, 1)), atol=1e-6)
    assert ctbr.num_rotors == ctbr.action_dim == 4
    assert ctbr.rotor_rpm().shape == (N, ctbr.num_rotors)
    np.testing.assert_array_equal(ctbr.action_low, [0, -20, -20, -20])
    np.testing.assert_array_equal(ctbr.action_high, [np.inf, 20, 20, 20])
    with pytest.raises(ValueError, match="shape"):
        ctbr.step(np.zeros((N, 3)))


def test_ctbr_limits_follow_per_instance_thrust(ctbr):
    ctbr.reset()
    drone = ctbr.parameters["dynamics"]["dynamics"]
    coefficients = np.asarray(drone["rotor_thrust_coefficients"])
    ctbr.set_instance_parameters({"dynamics": {"dynamics": {"rotor_thrust_coefficients": (coefficients * 2).tolist()}}}, MASK)
    actions = np.zeros((N, 4), dtype=np.float32)
    actions[:, 0] = 1e6
    ctbr.step(actions)
    maximum = drone["action_limit"]["max"]
    thrust = (coefficients[:, 0] + coefficients[:, 1] * maximum + coefficients[:, 2] * maximum ** 2).sum()
    expected = (thrust * np.where(MASK, 2, 1) / drone["mass"] + drone["gravity"][2]) * ctbr.dt
    np.testing.assert_allclose(ctbr.state["linear_velocity"][:, 2], expected, atol=1e-6)
    actions[:, 1:] = [100, -100, 100]
    ctbr.step(actions)
    np.testing.assert_array_equal(ctbr.state["angular_velocity"], np.tile([20, -20, 20], (N, 1)))


def test_position_masked_reset_preserves_other_rows(position_sims):
    selected, continuing, fresh = position_sims
    prepare(position_sims)
    selected.reset(MASK)
    fresh.reset()
    for sim in position_sims:
        advance(sim, 3)
    for name in COMPONENTS:
        np.testing.assert_array_equal(selected.state[name][MASK], fresh.state[name][MASK])
        np.testing.assert_array_equal(selected.state[name][~MASK], continuing.state[name][~MASK])
    assert np.isneginf(selected.action_low).all() and np.isposinf(selected.action_high).all()


def test_position_state_assignment_clears_memory(position_sims):
    selected, _, fresh = position_sims
    prepare(position_sims)
    selected.state["position"] = selected.state["position"]
    copy_state(selected, fresh)
    advance(selected, 3)
    advance(fresh, 3)
    for name in COMPONENTS:
        np.testing.assert_array_equal(selected.state[name], fresh.state[name])


def test_position_parameter_mutation_clears_selected_memory(position_sims):
    selected, continuing, fresh = position_sims
    prepare(position_sims)
    mass = selected.parameters["dynamics"]["dynamics"]["mass"]
    selected.set_instance_parameters({"dynamics": {"dynamics": {"mass": mass}}}, MASK)
    copy_state(selected, fresh)
    for sim in position_sims:
        advance(sim, 3)
    for name in COMPONENTS:
        np.testing.assert_array_equal(selected.state[name][MASK], fresh.state[name][MASK])
        np.testing.assert_array_equal(selected.state[name][~MASK], continuing.state[name][~MASK])
    assert not np.array_equal(selected.state["position"][MASK], continuing.state["position"][MASK])


def test_noncanonical_parameters_fail_transactionally(position_sims):
    selected, continuing, _ = position_sims
    prepare(position_sims)
    before = selected.instance_parameters()
    malformed = copy.deepcopy(before[0]["dynamics"]["dynamics"])
    malformed["rotor_positions"][0][1] *= -1
    override = {"dynamics": {"dynamics": malformed}}
    with pytest.raises(ValueError, match="canonical"):
        selected.set_instance_parameters(override, MASK)
    assert selected.instance_parameters() == before
    advance(selected, 3)
    advance(continuing, 3)
    for name in COMPONENTS:
        np.testing.assert_array_equal(selected.state[name], continuing.state[name])
    with pytest.raises(ValueError, match="canonical"):
        dynamics.Sim(N, control="position", parameters=override)
    motor = dynamics.Sim(N, parameters=override)
    assert motor.parameters["dynamics"]["dynamics"]["rotor_positions"] == malformed["rotor_positions"]
