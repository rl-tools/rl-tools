"""Dynamics backends for normalized-action gym-style use."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from rotorpy.vehicles.crazyflie_params import quad_params as default_quad_params
from rotorpy.vehicles.multirotor import Multirotor

ROTORPY_TO_L2F = np.array([1, 2, 3, 0])


def _scale_unit_to_range(x: np.ndarray, lo: float | np.ndarray, hi: float | np.ndarray) -> np.ndarray:
    """Map x from [-1, 1] to [lo, hi] with clipping."""
    x = np.clip(x, -1.0, 1.0)
    return (x + 1.0) * 0.5 * (hi - lo) + lo


# Action shape per control abstraction (normalized [-1,1] in env-space).
ACTION_DIMS: dict[str, int] = {
    "cmd_motor_speeds": 4,
    "cmd_motor_thrusts": 4,
    "cmd_ctbm": 4,    # [thrust, mx, my, mz]
    "cmd_ctbr": 4,    # [thrust, p, q, r]
    "cmd_ctatt": 4,   # [thrust, qx, qy, qz]   (qw inferred for unit quat)
    "cmd_vel": 3,     # [vx, vy, vz]
}

DYNAMICS_MODELS = ("rotorpy_crazyflie", "crazyflie_openmv")

CRAZYFLIE_OPENMV_PARAMS: dict[str, Any] = {
    "rotor_positions": np.array(
        [
            [0.028, -0.028, 0.0],
            [-0.028, -0.028, 0.0],
            [-0.028, 0.028, 0.0],
            [0.028, 0.028, 0.0],
        ],
        dtype=np.float64,
    ),
    "rotor_thrust_directions": np.array(
        [[0.0, 0.0, 1.0]] * 4,
        dtype=np.float64,
    ),
    "rotor_torque_directions": np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "rotor_thrust_coefficients": np.array(
        [[0.01244825, 0.05290432, 0.15135338]] * 4,
        dtype=np.float64,
    ),
    "rotor_torque_constants": np.array([4.665e-3] * 4, dtype=np.float64),
    "rotor_time_constants_rising": np.array([0.061] * 4, dtype=np.float64),
    "rotor_time_constants_falling": np.array([0.061] * 4, dtype=np.float64),
    "mass": 0.0477,
    "gravity": np.array([0.0, 0.0, -9.81], dtype=np.float64),
    "J": np.array(
        [
            [1.543e-5, 0.0, 0.0],
            [0.0, 1.946e-5, 0.0],
            [0.0, 0.0, 3.1957e-5],
        ],
        dtype=np.float64,
    ),
    "J_inv": np.array(
        [
            [64808.0, 0.0, 0.0],
            [0.0, 51387.0, 0.0],
            [0.0, 0.0, 31292.0],
        ],
        dtype=np.float64,
    ),
    "hovering_throttle_relative": 0.68,
    "action_limit": (0.0, 1.0),
}


def hover_rotor_speed(quad_params: dict, g: float = 9.81) -> float:
    """Per-rotor speed that produces a thrust equal to 1/N of weight."""
    return float(
        np.sqrt(quad_params["mass"] * g / (quad_params["num_rotors"] * quad_params["k_eta"]))
    )


def default_initial_state(quad_params: dict) -> dict:
    n = int(quad_params["num_rotors"])
    omega_h = hover_rotor_speed(quad_params)
    return {
        "x": np.zeros(3, dtype=np.float64),
        "v": np.zeros(3, dtype=np.float64),
        "q": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),  # xyzw
        "w": np.zeros(3, dtype=np.float64),
        "wind": np.zeros(3, dtype=np.float64),
        "rotor_speeds": np.full(n, omega_h, dtype=np.float64),
    }


def crazyflie_openmv_summary() -> dict[str, float | str]:
    params = CRAZYFLIE_OPENMV_PARAMS
    return {
        "model": "crazyflie_openmv",
        "mass": float(params["mass"]),
        "hover_command": float(params["hovering_throttle_relative"]),
        "motor_time_constant": float(params["rotor_time_constants_rising"][0]),
        "yaw_torque_per_thrust": float(params["rotor_torque_constants"][0]),
    }


def _crazyflie_openmv_quad_params() -> dict[str, Any]:
    params = CRAZYFLIE_OPENMV_PARAMS
    return {
        "num_rotors": 4,
        "mass": float(params["mass"]),
        "Ixx": float(params["J"][0, 0]),
        "Iyy": float(params["J"][1, 1]),
        "Izz": float(params["J"][2, 2]),
        "Ixy": float(params["J"][0, 1]),
        "Ixz": float(params["J"][0, 2]),
        "Iyz": float(params["J"][1, 2]),
        "rotor_speed_min": float(params["action_limit"][0]),
        "rotor_speed_max": float(params["action_limit"][1]),
        "rotor_speed_hover": float(params["hovering_throttle_relative"]),
        "tau_m": float(params["rotor_time_constants_rising"][0]),
    }


def _crazyflie_openmv_initial_state() -> dict:
    hover = float(CRAZYFLIE_OPENMV_PARAMS["hovering_throttle_relative"])
    return {
        "x": np.zeros(3, dtype=np.float64),
        "v": np.zeros(3, dtype=np.float64),
        "q": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "w": np.zeros(3, dtype=np.float64),
        "wind": np.zeros(3, dtype=np.float64),
        "rotor_speeds": np.full(4, hover, dtype=np.float64),
    }


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm == 0 or not np.isfinite(norm):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _quat_derivative_wxyz(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return 0.5 * np.array(
        [
            -q[1] * omega[0] - q[2] * omega[1] - q[3] * omega[2],
            q[0] * omega[0] + q[2] * omega[2] - q[3] * omega[1],
            q[0] * omega[1] + q[3] * omega[0] - q[1] * omega[2],
            q[0] * omega[2] + q[1] * omega[1] - q[2] * omega[0],
        ],
        dtype=np.float64,
    )


def _rotate_vector_by_quat_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_vec = q[1:]
    twice_cross = 2.0 * np.cross(q_vec, v)
    return v + q[0] * twice_cross + np.cross(q_vec, twice_cross)


class Dynamics:
    """Normalized-action dynamics facade with RotorPY and RLtools-compatible backends."""

    def __init__(
        self,
        control_mode: str = "cmd_motor_speeds",
        quad_params: dict | None = None,
        sim_rate: float = 100.0,
        aero: bool = True,
        dynamics_model: str = "rotorpy_crazyflie",
        # Limits for higher-level control modes. Match RotorPY's QuadrotorEnv defaults.
        max_body_rate: tuple[float, float, float] = (7.0, 7.0, 3.0),
        max_vel: float = 4.0 / np.sqrt(4),
    ):
        if control_mode not in ACTION_DIMS:
            raise ValueError(f"unknown control_mode {control_mode!r}; supported: {list(ACTION_DIMS)}")
        if dynamics_model not in DYNAMICS_MODELS:
            raise ValueError(f"unknown dynamics_model {dynamics_model!r}; supported: {list(DYNAMICS_MODELS)}")
        self.control_mode = control_mode
        self.dynamics_model = dynamics_model
        self.dt = 1.0 / float(sim_rate)
        self.max_body_rate = np.asarray(max_body_rate, dtype=np.float64)
        self.max_vel = float(max_vel)

        if dynamics_model == "crazyflie_openmv":
            if control_mode != "cmd_motor_speeds":
                raise ValueError("crazyflie_openmv currently supports only control_mode='cmd_motor_speeds'")
            if quad_params is not None:
                raise ValueError("quad_params cannot be combined with dynamics_model='crazyflie_openmv'")
            self.quad_params = _crazyflie_openmv_quad_params()
            self._vehicle = None
            self._rltools_params = deepcopy(CRAZYFLIE_OPENMV_PARAMS)
            self.max_thrust = self._rltools_total_thrust(float(self._rltools_params["action_limit"][1]))
            self.min_thrust = self._rltools_total_thrust(float(self._rltools_params["action_limit"][0]))
            self.state: dict = _crazyflie_openmv_initial_state()
            return

        self.quad_params = deepcopy(quad_params if quad_params is not None else default_quad_params)
        self._vehicle = Multirotor(
            quad_params=self.quad_params,
            initial_state=default_initial_state(self.quad_params),
            control_abstraction=control_mode,
            aero=aero,
        )

        # Cache thrust limits derived from rotor limits (used by ctbr/ctbm/ctatt).
        n = int(self.quad_params["num_rotors"])
        k_eta = float(self.quad_params["k_eta"])
        omega_max = float(self.quad_params["rotor_speed_max"])
        omega_min = float(self.quad_params["rotor_speed_min"])
        self.max_thrust = n * k_eta * omega_max ** 2
        self.min_thrust = n * k_eta * omega_min ** 2

        self.state: dict = default_initial_state(self.quad_params)

    # ---- public API ------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return ACTION_DIMS[self.control_mode]

    @property
    def num_rotors(self) -> int:
        return int(self.quad_params["num_rotors"])

    def default_initial_state(self) -> dict:
        if self.dynamics_model == "crazyflie_openmv":
            return _crazyflie_openmv_initial_state()
        return default_initial_state(self.quad_params)

    def reset(self, initial_state: dict | None = None) -> dict:
        self.state = (
            deepcopy(initial_state) if initial_state is not None
            else self.default_initial_state()
        )
        return deepcopy(self.state)

    def step(self, action_norm: np.ndarray) -> dict:
        action_norm = np.asarray(action_norm, dtype=np.float64).reshape(-1)
        if action_norm.shape[0] != self.action_dim:
            raise ValueError(f"action shape {action_norm.shape} != ({self.action_dim},)")
        if self.dynamics_model == "crazyflie_openmv":
            self.state = self._rltools_step(action_norm)
            return deepcopy(self.state)
        control = self._action_to_control(action_norm)
        self.state = self._vehicle.step(self.state, control, self.dt)
        return deepcopy(self.state)

    def flat_state(self) -> np.ndarray:
        """[pos(3), vel(3), quat_xyzw(4), body_rates(3)]  -> shape (13,) float32."""
        s = self.state
        return np.concatenate([s["x"], s["v"], s["q"], s["w"]]).astype(np.float32)

    # ---- internals -------------------------------------------------------
    def _action_to_control(self, a: np.ndarray) -> dict[str, Any]:
        m = self.control_mode
        if m == "cmd_motor_speeds":
            speeds = _scale_unit_to_range(
                a,
                float(self.quad_params["rotor_speed_min"]),
                float(self.quad_params["rotor_speed_max"]),
            )
            return {"cmd_motor_speeds": speeds}
        if m == "cmd_motor_thrusts":
            k_eta = float(self.quad_params["k_eta"])
            thrust_max = k_eta * float(self.quad_params["rotor_speed_max"]) ** 2
            thrust_min = k_eta * float(self.quad_params["rotor_speed_min"]) ** 2
            thrusts = _scale_unit_to_range(a, thrust_min, thrust_max)
            return {"cmd_motor_thrusts": thrusts}
        if m == "cmd_ctbr":
            thrust = _scale_unit_to_range(np.atleast_1d(a[0]), self.min_thrust, self.max_thrust)[0]
            cmd_w = a[1:] * self.max_body_rate
            return {"cmd_thrust": float(thrust), "cmd_w": cmd_w}
        if m == "cmd_ctbm":
            thrust = _scale_unit_to_range(np.atleast_1d(a[0]), self.min_thrust, self.max_thrust)[0]
            # Rough moment caps: max_thrust * arm length is conservative; use inertia*max_rate / dt-ish.
            inertia = np.diag([self.quad_params["Ixx"], self.quad_params["Iyy"], self.quad_params["Izz"]])
            max_moment = inertia @ self.max_body_rate / 0.1   # 100ms time-to-rate envelope
            cmd_moment = a[1:] * max_moment
            return {"cmd_thrust": float(thrust), "cmd_moment": cmd_moment}
        if m == "cmd_ctatt":
            thrust = _scale_unit_to_range(np.atleast_1d(a[0]), self.min_thrust, self.max_thrust)[0]
            qx, qy, qz = a[1] * 0.5, a[2] * 0.5, a[3] * 0.5  # bound to ~half-circle
            qw = float(np.sqrt(max(0.0, 1.0 - qx * qx - qy * qy - qz * qz)))
            return {"cmd_thrust": float(thrust), "cmd_q": np.array([qx, qy, qz, qw], dtype=np.float64)}
        if m == "cmd_vel":
            cmd_v = a * self.max_vel
            return {"cmd_v": cmd_v}
        raise AssertionError(f"unhandled control_mode {m}")

    def _rltools_total_thrust(self, command: float) -> float:
        coeff = self._rltools_params["rotor_thrust_coefficients"]
        thrust = coeff[:, 0] + coeff[:, 1] * command + coeff[:, 2] * command * command
        return float(np.sum(thrust * self._rltools_params["rotor_thrust_directions"][:, 2]))

    def _pack_rltools_state(self, state: dict) -> np.ndarray:
        rotor_speeds = np.asarray(state.get("rotor_speeds"), dtype=np.float64).reshape(4)
        return np.concatenate(
            [
                np.asarray(state["x"], dtype=np.float64).reshape(3),
                _quat_xyzw_to_wxyz(np.asarray(state["q"], dtype=np.float64).reshape(4)),
                np.asarray(state["v"], dtype=np.float64).reshape(3),
                np.asarray(state["w"], dtype=np.float64).reshape(3),
                rotor_speeds[ROTORPY_TO_L2F],
            ]
        )

    def _unpack_rltools_state(self, y: np.ndarray) -> dict:
        rpm_l2f = np.clip(y[13:17], self._rltools_params["action_limit"][0], self._rltools_params["action_limit"][1])
        rpm_rotorpy = np.empty(4, dtype=np.float64)
        rpm_rotorpy[ROTORPY_TO_L2F] = rpm_l2f
        return {
            "x": y[0:3].copy(),
            "v": y[7:10].copy(),
            "q": _quat_wxyz_to_xyzw(_normalize_quat(y[3:7])),
            "w": y[10:13].copy(),
            "wind": np.zeros(3, dtype=np.float64),
            "rotor_speeds": rpm_rotorpy,
        }

    def _rltools_step(self, action_norm: np.ndarray) -> dict:
        lo, hi = self._rltools_params["action_limit"]
        action_l2f = np.clip(action_norm, -1.0, 1.0)[ROTORPY_TO_L2F]
        motor_command = _scale_unit_to_range(action_l2f, float(lo), float(hi))
        y = self._pack_rltools_state(self.state)
        k1 = self._rltools_derivative(y, motor_command)
        k2 = self._rltools_derivative(y + 0.5 * self.dt * k1, motor_command)
        k3 = self._rltools_derivative(y + 0.5 * self.dt * k2, motor_command)
        k4 = self._rltools_derivative(y + self.dt * k3, motor_command)
        next_y = y + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_y[3:7] = _normalize_quat(next_y[3:7])
        next_y[13:17] = np.clip(next_y[13:17], lo, hi)
        return self._unpack_rltools_state(next_y)

    def _rltools_derivative(self, y: np.ndarray, motor_command: np.ndarray) -> np.ndarray:
        params = self._rltools_params
        q = y[3:7]
        velocity = y[7:10]
        omega = y[10:13]
        rpm = y[13:17]

        thrust = np.zeros(3, dtype=np.float64)
        torque = np.zeros(3, dtype=np.float64)
        coeff = params["rotor_thrust_coefficients"]
        for rotor_i in range(4):
            command = rpm[rotor_i]
            thrust_magnitude = (
                coeff[rotor_i, 0]
                + coeff[rotor_i, 1] * command
                + coeff[rotor_i, 2] * command * command
            )
            rotor_thrust = params["rotor_thrust_directions"][rotor_i] * thrust_magnitude
            thrust += rotor_thrust
            torque += params["rotor_torque_directions"][rotor_i] * (
                thrust_magnitude * params["rotor_torque_constants"][rotor_i]
            )
            torque += np.cross(params["rotor_positions"][rotor_i], rotor_thrust)

        y_dot = np.zeros_like(y)
        y_dot[0:3] = velocity
        y_dot[3:7] = _quat_derivative_wxyz(q, omega)
        y_dot[7:10] = _rotate_vector_by_quat_wxyz(q, thrust) / params["mass"] + params["gravity"]
        angular_momentum = params["J"] @ omega
        y_dot[10:13] = params["J_inv"] @ (torque - np.cross(omega, angular_momentum))
        tau = np.where(
            motor_command >= rpm,
            params["rotor_time_constants_rising"],
            params["rotor_time_constants_falling"],
        )
        y_dot[13:17] = (motor_command - rpm) / tau
        return y_dot
