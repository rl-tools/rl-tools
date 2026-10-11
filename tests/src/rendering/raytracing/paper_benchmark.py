"""Validate the May paper benchmark workloads against their pinned scene and camera contract."""

import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build-dir", type=Path, required=True)
parser.add_argument("--backend", choices=("metal", "vulkan", "optix"), required=True)
parser.add_argument("--with-procthor", action="store_true")
parser.add_argument("--timeout", type=float, default=20)
arguments = parser.parse_args()
build = arguments.build_dir.resolve()
binaries = build / "src/rendering/raytracing/benchmark"
paper_scene = build / "paper-benchmark/ProcTHOR-Train-1.glb"
paper_scene_sha1 = "7f1c9129532798e0b63bc41edb6b4c09251cf8a0"
num_cameras = 4096
for line in (build / "CMakeCache.txt").read_text().splitlines():
    if line.startswith("RL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS:"):
        num_cameras = int(line.split("=", 1)[1])


class PaperBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="rltools-paper-benchmark-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.invocations = 0

    def invoke(self, *options, profile=None, repeats=1, success=True, target=None):
        self.invocations += 1
        environment = dict(os.environ)
        environment.pop("METRA_URL", None)
        target = target or "rendering_raytracing_paper_benchmark"
        fidelity_options = [] if profile is None else ["--fidelity", profile]
        repeat_options = ([] if repeats is None or target != "rendering_raytracing_paper_benchmark"
                          else ["--repeats", str(repeats)])
        expected_repeats = 3 if repeats is None else repeats
        result = subprocess.run(
            [str(binaries / target), "--output-dir", str(self.root / str(self.invocations)),
             "--iterations", "3", "--warmup-seconds", "0", *fidelity_options, *repeat_options, *options],
            text=True, capture_output=True, timeout=arguments.timeout * expected_repeats, env=environment,
        )
        if not success:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertFalse(any(line.startswith("{") for line in result.stdout.splitlines()))
            return []
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        json_lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
        rows = [json.loads(line) for line in json_lines]
        self.assertTrue(rows, result.stdout)
        results_files = list((self.root / str(self.invocations)).glob(
            "rendering_raytracing_paper_benchmark_results_*.jsonl"))
        self.assertEqual(len(results_files), 1)
        self.assertEqual(results_files[0].read_text(), "\n".join(json_lines) + "\n")
        self.assertEqual(len({row["verification_png"] for row in rows}), len(rows))
        cells = [(row["shading_profile"], row["scene"], row["output"], row["workload"]) for row in rows]
        cells_per_repeat = len(set(cells))
        first_pass = cells[:cells_per_repeat]
        self.assertEqual(len(set(first_pass)), cells_per_repeat)
        self.assertEqual(cells, first_pass * expected_repeats)
        self.assertEqual([row["repeat"] for row in rows],
                         [repeat for repeat in range(1, expected_repeats + 1) for _ in first_pass])
        for row in rows:
            self.assertEqual(row["schema"], "render-v1")
            self.assertEqual(row["benchmark_contract"], "hyperdrone-paper-may2026")
            self.assertEqual(row["repeats"], expected_repeats)
            self.assertRegex(row["workload_source_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(row["historical_commit_attribution"], "inferred_from_paper_runs_and_history")
            self.assertTrue(row["rltools_commit"])
            coupled = row["workload"] == "render_physics_coupled"
            expected_commit = ("6c7fc4c6e0c912c451e40856b80740e65f129635" if coupled
                               else "10bee5cdc6616e6c42c16cf8966abff3f007ef06")
            self.assertEqual(row["historical_source_commit"], expected_commit)
            if row["workload"] == "render_only":
                self.assertIsNone(row["physics"])
            else:
                self.assertEqual(row["physics"], {
                    "model": "crazyflie", "action_interface": "direct_motor", "integrator": "rk4",
                    "dt": 0.01, "substeps": 1, "reset_interval": 10, "action_amplitude": 0.01,
                    "camera_coupled": coupled, "collision_response": False,
                })
            self.assertEqual(row["backend"], arguments.backend)
            self.assertTrue(row["device"])
            self.assertEqual(row["num_cameras"], num_cameras)
            self.assertEqual((row["width"], row["height"]), (64, 64))
            self.assertEqual(row["fov_degrees"], 80)
            self.assertEqual(row["orientation_mode"], "uniform_so3")
            expected_profiles = ("low", "medium", "high") if profile in (None, "all") else (profile,)
            self.assertIn(row["shading_profile"], expected_profiles)
            self.assertEqual(row["num_probes"], 0)
            self.assertEqual(row["aa_grid_size"], 1)
            self.assertEqual(row["samples_per_pixel"], 1)
            self.assertIs(row["motion_blur"], False)
            self.assertEqual(row["iterations"], 3)
            self.assertEqual(row["warmup_seconds"], 0)
            self.assertAlmostEqual(row["frames_per_s"] * row["elapsed_s"], 3 * num_cameras, delta=1e-5)
            self.assertAlmostEqual(row["pixels_per_s"], row["frames_per_s"] * 64 * 64, delta=1e-3)
            for field in ("elapsed_s", "frames_per_s", "pixels_per_s", "primary_mrays_per_s"):
                self.assertTrue(math.isfinite(row[field]), row)
                self.assertGreater(row[field], 0)
            if row["scene"] == "20_objects":
                self.assertEqual(row["scene_id"], "canonical_staggered_v1")
                self.assertIsNone(row["scene_sha256"])
                expected_position = (0, 0, 0)
            else:
                self.assertEqual(row["scene"], "procthor")
                self.assertEqual(row["scene_id"], paper_scene_sha1)
                self.assertEqual(row["scene_sha256"],
                                 "76c463017f2a25683abaea3b29786294602bd2629b2f0725b7215dabf58f35aa")
                expected_position = (-3.92, -5.67, 1)
            for actual, expected in zip(row["camera_position_flu"], expected_position):
                self.assertAlmostEqual(actual, expected, places=5)
            png = Path(row["verification_png"]).read_bytes()
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            width, height = struct.unpack(">II", png[16:24])
            self.assertEqual(width % 64, 0)
            self.assertEqual(height % 64, 0)
            self.assertGreaterEqual(width * height, num_cameras * 64 * 64)
        return rows

    def test_profiles_and_default_outputs(self):
        combined = self.invoke("--scene", "20_objects", "--workload", "render_only", profile="all", repeats=None)
        self.assertEqual(len(combined), 18)
        indexed = {(row["shading_profile"], row["output"]): row for row in combined[:6]}
        self.assertEqual(set(indexed), {
            (profile, output) for profile in ("low", "medium", "high") for output in ("rgb", "depth")
        })
        for row in combined:
            self.assertEqual(Path(row["verification_png"]).read_bytes(),
                             Path(indexed[(row["shading_profile"], row["output"])]["verification_png"]).read_bytes())
        for profile in ("medium", "high", "low"):
            with self.subTest(profile=profile):
                rows = self.invoke("--scene", "20_objects", "--workload", "render_only", profile=profile)
                self.assertEqual({row["output"] for row in rows}, {"rgb", "depth"})
                self.assertEqual(len(rows), 2)
                for row in rows:
                    self.assertEqual(row["workload"], "render_only")
                    self.assertEqual(row["seed"], 0)
                    self.assertEqual(row["sync_interval"], 10)
                    self.assertEqual(Path(row["verification_png"]).read_bytes(),
                                     Path(indexed[(profile, row["output"])]["verification_png"]).read_bytes())

    def test_camera_seed_and_partial_sync_batch(self):
        images = []
        for seed, interval in ((0, 1), (0, 10), (1, 10), (1, 1)):
            row = self.invoke("--scene", "20_objects", "--output", "rgb", "--workload", "render_only",
                              "--seed", str(seed), "--sync-interval", str(interval), profile="medium")[0]
            self.assertEqual(row["seed"], seed)
            self.assertEqual(row["sync_interval"], interval)
            images.append(Path(row["verification_png"]).read_bytes())
        self.assertEqual(images[0], images[1])
        self.assertEqual(images[2], images[3])
        self.assertNotEqual(images[0], images[2])

    def test_invalid_workload_and_nonpaper_target_reject_option(self):
        self.invoke("--scene", "20_objects", "--workload", "invalid", success=False)
        self.invoke("--scene", "20_objects", "--fidelity", "invalid", success=False)
        self.invoke("--scene", "20_objects", "--fidelity", success=False)
        self.invoke("--scene", "20_objects", "--orientation-mode", "random_yaw_pitch", success=False)
        self.invoke("--scene", "20_objects", "--workload", "render_only",
                    target="rendering_raytracing_sim_benchmark_medium", success=False)
        self.invoke("--scene", "20_objects", "--fidelity", "low",
                    target="rendering_raytracing_sim_benchmark_medium", success=False)
        self.invoke("--repeats", "3", target="rendering_raytracing_sim_benchmark_medium", success=False)

    def test_repeat_count_override_and_validation(self):
        rows = self.invoke("--scene", "20_objects", "--workload", "render_only", profile="medium", repeats=2)
        self.assertEqual(len(rows), 4)
        for value in ("0", "-1", "invalid", "1.5", "2147483648"):
            with self.subTest(value=value):
                self.invoke("--workload", "render_only", "--repeats", value, success=False)
        self.invoke("--workload", "render_only", "--repeats", success=False)

    @unittest.skipIf(arguments.backend == "optix", "OptiX supports the CUDA physics workloads")
    def test_other_backends_reject_physics_workloads(self):
        self.invoke(success=False)
        for workload in ("render_physics_static", "render_physics_coupled", "all"):
            with self.subTest(workload=workload):
                self.invoke("--scene", "20_objects", "--workload", workload, success=False)

    @unittest.skipUnless(arguments.backend == "optix", "requires CUDA physics and OptiX")
    def test_physics_static_keeps_the_rendered_cameras_unchanged(self):
        rows = self.invoke("--scene", "20_objects", "--workload", "all", profile="all")
        self.assertEqual(len(rows), 12)
        indexed = {(row["shading_profile"], row["workload"], row["output"]): row for row in rows}
        coupled_rows = self.invoke("--scene", "20_objects", "--workload", "render_physics_coupled")
        self.assertEqual(len(coupled_rows), 6)
        coupled_indexed = {(row["shading_profile"], row["output"]): row for row in coupled_rows}
        for profile in ("low", "medium", "high"):
            for output in ("rgb", "depth"):
                reference = indexed[(profile, "render_only", output)]
                static = indexed[(profile, "render_physics_static", output)]
                self.assertEqual(Path(reference["verification_png"]).read_bytes(),
                                 Path(static["verification_png"]).read_bytes())
                coupled = coupled_indexed[(profile, output)]
                self.assertNotEqual(Path(reference["verification_png"]).read_bytes(),
                                    Path(coupled["verification_png"]).read_bytes())

    @unittest.skipUnless(arguments.with_procthor, "requires the pinned May ProcTHOR asset")
    def test_default_scenes_and_outputs_use_the_may_asset(self):
        rows = self.invoke("--workload", "render_only", repeats=None)
        self.assertEqual(len(rows), 36)
        self.assertEqual({(row["shading_profile"], row["scene"], row["output"]) for row in rows}, {
            (profile, scene, output) for profile in ("low", "medium", "high")
            for scene in ("20_objects", "procthor") for output in ("rgb", "depth")
        })

    @unittest.skipUnless(arguments.backend == "optix" and arguments.with_procthor,
                         "requires CUDA physics, OptiX and the pinned May ProcTHOR asset")
    def test_defaults_repeat_all_24_cells_three_times_in_order(self):
        rows = self.invoke(repeats=None)
        self.assertEqual(len(rows), 72)
        self.assertEqual({(row["shading_profile"], row["scene"], row["output"], row["workload"]) for row in rows}, {
            (profile, scene, output, workload) for profile in ("low", "medium", "high")
            for scene in ("20_objects", "procthor") for output in ("rgb", "depth")
            for workload in ("render_only", "render_physics_static")
        })

    @unittest.skipUnless(arguments.with_procthor, "requires the pinned May ProcTHOR asset")
    def test_valid_but_unpinned_scene_is_rejected(self):
        content = bytearray(paper_scene.read_bytes())
        json_size, chunk_type = struct.unpack_from("<II", content, 12)
        self.assertEqual(chunk_type, 0x4E4F534A)
        document = json.loads(content[20:20 + json_size])
        document.setdefault("asset", {})["generator"] = "paper-benchmark-invalid-pin"
        changed = json.dumps(document, separators=(",", ":")).encode()
        changed += b" " * (-len(changed) % 4)
        tail = content[20 + json_size:]
        invalid = self.root / "different-valid-scene.glb"
        invalid.write_bytes(struct.pack("<IIIII", 0x46546C67, 2, 20 + len(changed) + len(tail),
                                        len(changed), 0x4E4F534A) + changed + tail)
        self.invoke("--scene", "procthor", "--output", "rgb", "--workload", "render_only",
                    "--procthor-path", str(invalid), success=False)


unittest.main(argv=[__file__], verbosity=2)
