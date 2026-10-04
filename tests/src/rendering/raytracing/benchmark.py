"""Exercise the render benchmark's synchronization, output records, and CLI contract."""
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
binaries = arguments.build_dir.resolve() / "src/rendering/raytracing/benchmark"


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='rltools-render-"json,\\\n')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.invocations = 0

    def invoke(self, *options, target="rendering_raytracing_sim_benchmark_medium", success=True):
        self.invocations += 1
        output = self.root / str(self.invocations)
        environment = dict(os.environ)
        environment.pop("METRA_URL", None)
        result = subprocess.run(
            [str(binaries / target), "--output-dir", str(output), *options],
            text=True, capture_output=True, timeout=arguments.timeout, env=environment,
        )
        if not success:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertFalse(any(line.startswith("{") for line in result.stdout.splitlines()))
            return []
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = []
        for line in result.stdout.splitlines():
            if line.startswith("{"):
                row = json.loads(line)
                self.assertEqual(row["schema"], "render-v1")
                self.assertEqual(row["backend"], arguments.backend)
                self.assertTrue(row["device"])
                for field in ("num_cameras", "width", "height", "aa_grid_size", "samples_per_pixel",
                              "num_probes", "seed", "sync_interval", "iterations"):
                    self.assertIs(type(row[field]), int, field)
                self.assertEqual(row["num_probes"], 0)
                self.assertIs(row["motion_blur"], False)
                for field in ("fov_degrees", "warmup_seconds", "elapsed_s", "frames_per_s",
                              "pixels_per_s", "primary_mrays_per_s"):
                    self.assertIsInstance(row[field], (int, float), field)
                    self.assertTrue(math.isfinite(row[field]), row)
                    if field != "warmup_seconds":
                        self.assertGreater(row[field], 0)
                for field in ("camera_position_flu", "camera_forward_flu", "camera_up_flu"):
                    if field != "camera_position_flu" and row["workload"] == "render_only":
                        self.assertIsNone(row[field])
                    else:
                        self.assertIsInstance(row[field], list)
                        self.assertEqual(len(row[field]), 3)
                        for value in row[field]:
                            self.assertIsInstance(value, (int, float))
                            self.assertTrue(math.isfinite(value))
                self.assertAlmostEqual(
                    row["frames_per_s"] * row["elapsed_s"],
                    row["num_cameras"] * row["iterations"], delta=1e-5,
                )
                png = Path(row["verification_png"]).read_bytes()
                self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
                width, height = struct.unpack(">II", png[16:24])
                self.assertEqual(width % row["width"], 0)
                self.assertEqual(height % row["height"], 0)
                self.assertGreaterEqual(width * height, row["num_cameras"] * row["width"] * row["height"])
                rows.append(row)
        self.assertTrue(rows, result.stdout)
        return rows

    def test_partial_batch_and_sync_interval_preserve_rgb_and_depth(self):
        base = ("--scene", "20_objects", "--output", "all", "--warmup-seconds", "0")
        reference = self.invoke(*base, "--iterations", "1", "--sync-interval", "1")
        self.assertEqual({row["output"] for row in reference}, {"rgb", "depth"})
        self.assertEqual(len(reference), 2)
        for interval in (1, 10):
            rows = self.invoke(*base, "--iterations", "11", "--sync-interval", str(interval))
            for expected, actual in zip(reference, rows):
                self.assertEqual(actual["iterations"], 11)
                self.assertEqual(actual["sync_interval"], interval)
                self.assertEqual(actual["workload"], "render_only")
                self.assertEqual(
                    Path(expected["verification_png"]).read_bytes(),
                    Path(actual["verification_png"]).read_bytes(),
                )

    def test_seed_is_repeatable_and_changes_the_cameras(self):
        base = ("--scene", "20_objects", "--warmup-seconds", "0", "--iterations", "1")
        images = []
        for seed in (0, 1, 1):
            row = self.invoke(*base, "--seed", str(seed))[0]
            self.assertEqual(row["seed"], seed)
            images.append(Path(row["verification_png"]).read_bytes())
        self.assertNotEqual(images[0], images[1])
        self.assertEqual(images[1], images[2])

    def test_duration_run(self):
        rows = self.invoke("--scene", "20_objects", "--warmup-seconds", "0",
                           "--seconds", "0.01", "--sync-interval", "2")
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(rows[0]["elapsed_s"], 0.01)
        self.assertEqual(rows[0]["iterations"] % 2, 0)

    def test_sweep_uses_the_same_schema(self):
        base = ("--scene", "20_objects", "--warmup-seconds", "0", "--iterations", "1")
        reference = self.invoke(*base)[0]
        rows = self.invoke(*base, "--sweep-config-count", "1", "--cooldown-seconds", "0",
                           target="rendering_raytracing_sim_benchmark_sweep")
        self.assertEqual(len(rows), 1)
        self.assertEqual(reference.keys(), rows[0].keys())
        self.assertEqual(Path(reference["verification_png"]).read_bytes(),
                         Path(rows[0]["verification_png"]).read_bytes())

    def test_sweep_selects_anti_aliasing_configuration(self):
        rows = self.invoke("--scene", "20_objects", "--warmup-seconds", "0", "--iterations", "1",
                           "--sweep-config-start", "3", "--sweep-config-count", "1",
                           target="rendering_raytracing_sim_benchmark_sweep")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["shading_profile"], "medium")
        self.assertEqual(rows[0]["aa_grid_size"], 2)
        self.assertEqual(rows[0]["samples_per_pixel"], 4)
        self.assertAlmostEqual(rows[0]["primary_mrays_per_s"],
                               rows[0]["pixels_per_s"] * 4 / 1e6, delta=1e-5)

    def test_invalid_or_removed_options_fail_before_rendering(self):
        cases = [
            ("--step-mode", "render_physics"), ("--warmup-iterations", "1"),
            ("--num-envs", "4096"), ("--resolution", "64"), ("--gpu-label", "gpu"),
            ("--iterations", "-1"), ("--iterations", "2147483648"),
            ("--sync-interval", "0"), ("--seconds", "nan"), ("--seconds", "inf"),
            ("--seconds", "1e309"),
            ("--warmup-seconds", "-1"), ("--seed", "-1"), ("--scene", "objects20"),
            ("--output", "invalid"), ("--position-flu", "0", "0", "0"),
            ("--cooldown-seconds", "0"), ("--iterations",),
        ]
        for options in cases:
            with self.subTest(options=options):
                self.invoke(*options, success=False)

    def test_empty_sweep_selection_fails(self):
        self.invoke("--scene", "20_objects", "--sweep-config-start", "999",
                    target="rendering_raytracing_sim_benchmark_sweep", success=False)

    @unittest.skipUnless(arguments.with_procthor, "requires the pinned ProcTHOR asset")
    def test_default_scene_and_frame_outputs(self):
        reference = self.invoke("--iterations", "1", "--warmup-seconds", "0")[0]
        self.assertEqual(reference["scene"], "procthor")
        self.assertEqual(reference["output"], "rgb")
        self.assertEqual(reference["shading_profile"], "medium")
        self.assertEqual(reference["orientation_mode"], "uniform_so3")
        frames = self.invoke("--output", "all", target="rendering_raytracing_procthor_frame_medium")
        self.assertEqual(len(frames), 2)
        self.assertEqual(len({row["verification_png"] for row in frames}), 2)
        for row in frames:
            self.assertEqual(row.keys(), reference.keys())
            self.assertEqual(row["workload"], "frame")
            self.assertEqual(row["num_cameras"], 1)
            self.assertEqual(row["width"], 1000)
            self.assertTrue(row["camera_forward_flu"])
        self.invoke("--output", "all", "--output-png", str(self.root / "ambiguous.png"),
                    target="rendering_raytracing_procthor_frame_medium", success=False)


unittest.main(argv=[__file__], verbosity=2)
