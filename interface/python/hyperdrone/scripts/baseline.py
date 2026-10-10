"""Run the five fixed Python workloads from a committed revision; print one JSON record per fresh child process. Timing covers batch step and host reads, excluding JIT, reset, initial acquisition, action construction, hashing and metra. Set CONTA_ROOT to an offline store; METRA_URL optionally enables logging."""

import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
STEPS = 100
SEED = 0
CONFIGURATIONS = ("single_camera", "target_frame", "moving_gate", "visual_inertial_localization", "manual")
AUTOPILOT = "b6bd082c60fcb27f42a2b5f2f635587d2b60d170"


def golden_actions(step, total, action_dim):
    # The same arithmetic as tests/env/test_golden_rollout.py and python_golden.cpp.
    actions = np.empty((total, action_dim), dtype=np.float32)
    for instance in range(total):
        for dim in range(action_dim):
            actions[instance, dim] = np.float32(-1.0 + 2.0 * (((step * 131 + instance * 31 + dim * 7) % 101) / 100.0))
    return actions


def fingerprint(streams):
    digest = hashlib.sha256()
    for name, values in sorted(streams.items()):
        digest.update(name.encode() + b"\0")
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def measure(step, observe, total, action_dim):
    initial = observe()
    result = {
        "steps": STEPS,
        "total_instances": total,
        "action_dim": action_dim,
        "streams": {name: {"shape": list(values.shape), "dtype": str(values.dtype)} for name, values in initial.items()},
        "reset_fingerprint": fingerprint(initial),
        "step_fingerprints": [],
    }
    del initial
    elapsed = 0.0
    for index in range(STEPS):
        actions = golden_actions(index, total, action_dim)
        start = time.perf_counter()
        step(actions)
        streams = observe()
        elapsed += time.perf_counter() - start
        result["step_fingerprints"].append(fingerprint(streams))
        del streams
    result["elapsed_seconds"] = elapsed
    result["steps_per_second"] = STEPS / elapsed
    result["fingerprint"] = hashlib.sha256(bytes.fromhex(result["reset_fingerprint"]) + b"".join(bytes.fromhex(value) for value in result["step_fingerprints"])).hexdigest()
    result["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    return result


def local_assets(name):
    import conta

    if not os.environ.get("CONTA_ROOT"):
        raise RuntimeError("set CONTA_ROOT to an offline conta store before running baselines")
    paths = {"scene": ROOT / "tests/data/ProcTHOR-Train-1.glb"}
    if name in ("target_frame", "moving_gate"):
        paths["x500"] = ROOT / "tests/data/x500.glb"
    if name == "visual_inertial_localization":
        paths["autopilot"] = Path(conta.resolve(AUTOPILOT))
    hashes = {}
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"baseline requires local {label}: {path}")
        with path.open("rb") as asset:
            if path.suffix == ".glb" and asset.read(4) != b"glTF":
                raise ValueError(f"baseline requires a GLB asset, not an LFS pointer: {path}")
            asset.seek(0)
            digest = hashlib.sha1()
            for chunk in iter(lambda: asset.read(1024 * 1024), b""):
                digest.update(chunk)
            hashes[label] = digest.hexdigest()
    return paths, hashes


def run_configuration(name):
    from hyperdrone import dynamics, render
    from hyperdrone.env import EnvConfig, MultiEnvironment

    paths, hashes = local_assets(name)
    if name == "manual":
        scene = render.load_scene(str(paths["scene"]), fidelity="medium")
        renderer = render.Renderer(64, 64, num_cameras=64, fidelity="medium")
        renderer.init(scene)
        sim = dynamics.Sim(64, seed=SEED, device="cpu")
        positions = render.ClearanceSampler(scene, clearance=0.5).sample(64, seed=SEED)
        sim.reset()
        sim.state["position"] = positions

        def observe():
            renderer.set_cameras(sim.camera_bases(aspect=renderer.aspect))
            renderer.render()
            return {"observation": renderer.frame(), "privileged": sim.observe()}

        result = measure(sim.step, observe, sim.num_drones, sim.action_dim)
        result.update(width=64, height=64, fidelity="medium", device=sim.device, backend=renderer.backend)
    else:
        settings = {
            "single_camera": {},
            "target_frame": {"preset": "x500_fpv", "task": "target_frame"},
            "moving_gate": {"task": "moving_gate"},
            "visual_inertial_localization": {"preset": "x500_fpv_imu", "task": "visual_inertial_localization", "instances_per_environment": 2, "width": 160, "height": 120},
        }
        config = EnvConfig(**({"instances_per_environment": 64, "width": 64, "height": 64, "device": "cpu"} | settings[name]))
        assets = {"gate": str(paths["x500"])} if name == "moving_gate" else None
        drone_asset = str(paths["x500"]) if name == "target_frame" else None
        env = MultiEnvironment([str(paths["scene"])], config=config, seed=SEED, drone_asset=drone_asset, assets=assets)
        try:
            env.reset()

            def observe():
                streams = {"observation": env.observe(), "privileged": env.observe_privileged()}
                if env.observation_dim_imu:
                    streams["imu"] = env.observe_imu()
                streams["rewards"] = env.rewards()
                streams["terminated"] = env.terminated()
                return streams

            result = measure(env.step, observe, env.total_instances, env.action_dim)
            result.update(width=env.width, height=env.height, fidelity=config.fidelity, device=config.world_device, backend=env.render_backend)
        finally:
            env.close()
    result.update(config=name, seed=SEED, assets=hashes)
    return result


def main():
    os.environ["HYPERDRONE_BUILD_JOBS"] = "1"
    if len(sys.argv) == 3 and sys.argv[1] == "--worker" and sys.argv[2] in CONFIGURATIONS:
        print(json.dumps(run_configuration(sys.argv[2]), sort_keys=True), flush=True)
        return
    if len(sys.argv) != 1:
        raise SystemExit("usage: .venv/bin/python interface/python/hyperdrone/scripts/baseline.py")
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    commit_time = int(subprocess.check_output(["git", "-C", str(ROOT), "show", "-s", "--format=%ct", "HEAD"], text=True).strip())
    for name in CONFIGURATIONS:
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", name], cwd=ROOT, capture_output=True, text=True)
        if child.stderr:
            print(child.stderr, end="", file=sys.stderr)
        if child.returncode:
            print(child.stdout, end="", file=sys.stderr)
            raise SystemExit(child.returncode)
        lines = child.stdout.splitlines()
        if len(lines) > 1:
            print("\n".join(lines[:-1]), file=sys.stderr)
        result = json.loads(lines[-1])
        result["commit"] = commit
        print(json.dumps(result, sort_keys=True), flush=True)
        if os.environ.get("METRA_URL"):
            import metra

            for metric in ("steps_per_second", "peak_rss_mb", "fingerprint"):
                metra.log(f"hyperdrone/python/{name}/{metric}", result[metric], commit=commit, commit_time=commit_time)


if __name__ == "__main__":
    main()
