import copy
import json

import numpy as np


def merge(base, overrides, path=""):
    if not isinstance(overrides, dict):
        raise ValueError(f"hyperdrone: parameter {path or '<root>'} must be a mapping")
    merged = dict(base)
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f"hyperdrone: unknown parameter {path}{key!r} (one of {sorted(base)})")
        if isinstance(base[key], dict):
            merged[key] = merge(base[key], value, f"{path}{key}.")
            continue
        if np.shape(value) != np.shape(base[key]):
            raise ValueError(f"hyperdrone: parameter {path}{key} must have shape {np.shape(base[key])}, got {np.shape(value)}")
        merged[key] = value.tolist() if isinstance(value, (np.ndarray, np.generic)) else value
    return merged


def resolve(base, overrides, model_lookup):
    overrides = copy.deepcopy(overrides)
    dynamics = overrides.get("dynamics", {}).get("dynamics", {})
    model = dynamics.pop("model", None)
    if model is not None:
        entry = model_lookup(model)
        if entry is None:
            raise ValueError(f"hyperdrone: unknown dynamics model {model!r} (see hyperdrone.dynamics.MODELS)")
        base = copy.deepcopy(base)
        base["dynamics"]["dynamics"] = json.loads(entry)
    merged = merge(base, overrides)
    if "J" in dynamics and "J_inv" not in dynamics:
        merged["dynamics"]["dynamics"]["J_inv"] = np.linalg.inv(merged["dynamics"]["dynamics"]["J"]).tolist()
    return merged
