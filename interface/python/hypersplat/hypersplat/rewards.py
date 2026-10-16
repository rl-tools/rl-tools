"""Default reward functions. Use any callable `(state, action_norm) -> float`."""
from __future__ import annotations

import numpy as np


def hover_reward(state: dict, action_norm: np.ndarray, target: np.ndarray | None = None) -> float:
    """Negative-cost hover reward: penalize position error, velocity, body rate."""
    target = np.zeros(3) if target is None else np.asarray(target)
    pos_err = np.linalg.norm(state["x"] - target)
    speed = np.linalg.norm(state["v"])
    body_rate = np.linalg.norm(state["w"])
    return float(-pos_err - 0.1 * speed - 0.01 * body_rate)
