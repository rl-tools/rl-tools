"""Rigged drone assets for runnable examples."""

import os
from pathlib import Path

import conta

__all__ = ["RIGGED_MODELS", "rigged_model_path"]

# body/prop_* assemblies, each a drone table asset (hyperdrone.HyperDrone(drones=...))
RIGGED_MODELS = {
    "arpl": {"description": "rigged/arpl_optimized_scaled.glb", "hash": "25401d66376e927dd71ca7cd9527a22c867eede6", "sha256": "5effcbd84845e6bc3cc8abbd0dfd2ce877a1114ba012db702032268832f41809"},
    "crazyflie_brushless": {"description": "rigged/crazyflie_brushless_optimized_scaled.glb", "hash": "b22399b2103e0e4559c7c14239e148f963817d77", "sha256": "7a634d5d79b228358ed83e39961c32976339de8ed3d14bd450092fde75d164f9"},
    "crazyflie": {"description": "rigged/crazyflie_optimized_scaled.glb", "hash": "41b0e8a02d13d1e82014cb3136a616d50c2d214e", "sha256": "c47bf3394e5493555b2f56ec67f4270112a478cbf6b96ffa04afdc34a5487f97"},
    "soft": {"description": "rigged/soft_optimized_scaled.glb", "hash": "d72a5b9f8d048846c660008daed65733c6003555", "sha256": "2643cf7a56252d099569f2aac289c2c4707ea17472e3b10d025985b444133358"},
    "savagebee_pusher": {"description": "rigged/savagebee_pusher_optimized_scaled.glb", "hash": "02e91faf581ba36d9befbf3dcb03da8340fbbd46", "sha256": "23c8644e572e3c4f3769f59a2dd81c3e164837c178054d5b951b97b086f8b674"},
    "x500": {"description": "rigged/x500_optimized_scaled.glb", "hash": "c0fda2143b7a2f0422148aa8f6c4e24db3f8a6e6", "sha256": "116a362405a7f790b33585d2fd663d84ee15ed53161209c3ac862af7266bab31"},
}


def rigged_model_path(name):
    """Return the local GLB of a RIGGED_MODELS entry, downloading and caching it when necessary.

    Resolution order: HYPERDRONE_<NAME>_PATH, then the shared conta cache.
    """
    if name not in RIGGED_MODELS:
        raise KeyError(f"hyperdrone: unknown rigged model {name!r} (one of {sorted(RIGGED_MODELS)})")
    override_variable = f"HYPERDRONE_{name.upper()}_PATH"
    override = os.environ.get(override_variable)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"hyperdrone: {override_variable} does not exist: {path}")
        return path
    return conta.resolve(RIGGED_MODELS[name])
