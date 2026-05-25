# Raytracing Benchmarks

This directory contains the benchmark targets for the OptiX raytracing renderer.
The simulator matrix target is the preferred benchmark for simulator-paper tables because it emits CSV rows and stitched PNG verification images.

The current shared renderer FOV is 80 degrees. AA and motion blur are disabled in the ordinary matrix targets; the single-target sweep below covers both no-AA and 2x AA. The simulator matrix samples one deterministic Haar-uniform SO(3) camera orientation per camera from `--seed` by default. `20_objects` is viewed from the scene origin and ProcTHOR from `[-3.92, -5.67, 1.0]`. The `20_objects` scene uses the shared `canonical_staggered_v1` staggered spatial layout with alternating box and sphere entries.

## Activate Environment

Run from the repository root:

```bash
cd ~/rl-tools2
source .venv/bin/activate
export CUDACXX=/usr/local/cuda-13.1/bin/nvcc
```

No package installation or environment setup is required by this document.

## Configure

If `build/` is already configured with raytracing enabled, skip this step. Otherwise:

```bash
CUDACXX=/usr/local/cuda-13.1/bin/nvcc cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DRL_TOOLS_ENABLE_TESTS=ON \
  -DRL_TOOLS_EXPERIMENTAL=ON \
  -DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO=ON \
  -DRL_TOOLS_NUMERIC_TYPES_ENABLE_BF16=ON \
  -DRL_TOOLS_ENABLE_TAR=ON \
  -DRL_TOOLS_RENDERING_ENABLE_RAYTRACING=ON
```

The simulator matrix target uses `4096` parallel camera environments by default. To rebuild for another value:

```bash
cmake -B build -DRL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS=8192
```

## Build

Build all benchmark targets explicitly. Some targets are `EXCLUDE_FROM_ALL`, so a plain full build may not include them.

```bash
cmake --build build --target \
  rendering_raytracing_benchmark \
  rendering_raytracing_benchmark_depth \
  rendering_raytracing_benchmark_rgbd \
  rendering_raytracing_sim_benchmark \
  rendering_raytracing_sim_benchmark_sweep \
  rendering_raytracing_sim_benchmark_medium \
  rendering_raytracing_sim_benchmark_high \
  rendering_raytracing_sim_benchmark_low \
  rendering_raytracing_sim_benchmark_rgb \
  rendering_raytracing_sim_benchmark_depth \
  -j5
```

## Run Simulator Matrix Benchmarks

The default timed duration is 10 seconds per row and the default warmup is 2 seconds. PNG stitching and CSV logging happen after the timed section.

Medium matrix:

```bash
OUT_MEDIUM=$(mktemp -d /tmp/rltools_rt_matrix_medium.XXXXXX)
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_medium \
  --scene all \
  --step-mode all \
  --output all \
  --seconds 10 \
  --warmup-seconds 2 \
  --gpu-label "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1) medium" \
  --output-dir "$OUT_MEDIUM" | tee "$OUT_MEDIUM/output.log"
echo "$OUT_MEDIUM"
```

High matrix:

```bash
OUT_HIGH=$(mktemp -d /tmp/rltools_rt_matrix_high.XXXXXX)
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_high \
  --scene all \
  --step-mode all \
  --output all \
  --seconds 10 \
  --warmup-seconds 2 \
  --gpu-label "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1) high" \
  --output-dir "$OUT_HIGH" | tee "$OUT_HIGH/output.log"
echo "$OUT_HIGH"
```

Low matrix:

```bash
OUT_LOW=$(mktemp -d /tmp/rltools_rt_matrix_low.XXXXXX)
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_low \
  --scene all \
  --step-mode all \
  --output all \
  --seconds 10 \
  --warmup-seconds 2 \
  --gpu-label "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1) low" \
  --output-dir "$OUT_LOW" | tee "$OUT_LOW/output.log"
echo "$OUT_LOW"
```

Each matrix covers:

| Axis | Values |
|---|---|
| Scene | `20_objects`, `procthor` |
| Output | `rgb`, `depth` |
| Step mode | `render_only`, `render_physics` |
| Profile | medium target, high target, low target |

`High` uses PBR shading, `Medium` uses textured/basic shading, and `Low` uses flat per-geometry color.

The benchmark writes one stitched PNG per row and CSV rows prefixed with `csv_result`.

Extract table rows:

```bash
rg '^csv_result' "$OUT_MEDIUM/output.log"
rg '^csv_result' "$OUT_HIGH/output.log"
rg '^csv_result' "$OUT_LOW/output.log"
```

List verification images:

```bash
ls -lh "$OUT_MEDIUM"/*.png
ls -lh "$OUT_HIGH"/*.png
ls -lh "$OUT_LOW"/*.png
```

## Run Single-Target Resolution and AA Sweep

This target avoids creating one CMake target per resolution/profile/AA combination. It runs the renderer configurations sequentially for resolutions `64, 128, 256, 512, 1024, 2048`, RGB profiles `medium, high, low`, depth once per resolution, and AA modes `none, aa2`. The sweep honors the usual simulator matrix scene, output, and step-mode filters. It keeps the 64x64 matrix workload as the reference and scales camera count down at higher resolutions to keep total pixels per iteration roughly constant.

```bash
OUT_SWEEP=$(mktemp -d /tmp/rltools_rt_matrix_sweep.XXXXXX)
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_sweep \
  --scene all \
  --step-mode all \
  --output all \
  --seconds 10 \
  --warmup-seconds 2 \
  --cooldown-seconds 10 \
  --gpu-label "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1) resolution-aa sweep" \
  --output-dir "$OUT_SWEEP" | tee "$OUT_SWEEP/output.log"
echo "$OUT_SWEEP"
```

With all scene, step, and output axes enabled, this emits 192 `csv_result` rows. The sweep CSV adds `shading_profile`, `anti_aliasing`, `aa_grid_size`, and `samples_per_pixel` columns.

The May 24, 2026 local benchmark run used:

```text
target: rendering_raytracing_sim_benchmark_sweep
output_dir: /tmp/rltools_rt_matrix_sweep_actual_20260524_03
gpu_label: NVIDIA GeForce RTX 4090 Laptop GPU resolution-aa sweep 2026-05-24
scene: all
step_mode: all
output: all
orientation_mode: uniform_so3
seed: 0
num_envs: 4096
seconds: 10
warmup_seconds: 2
warmup_iterations: 10
sync_interval: 10
cooldown_seconds: 10
rows: 192
```

Because the local command runner terminated very long single invocations, that run was split into chunks with `--sweep-config-start` and `--sweep-config-count 3`, with a 10 second sleep between chunk invocations. This chunking does not change the timed row parameters; it only limits wall time per process. A full unchunked invocation uses the same benchmark target and omits those two chunk flags.

## Run Output-Specific Simulator Targets

These targets are useful when validating compile-time RGB-only or depth-only code paths. The all-output matrix above already covers both outputs.

RGB-only:

```bash
OUT_RGB=$(mktemp -d /tmp/rltools_rt_matrix_rgb.XXXXXX)
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_rgb \
  --scene all \
  --step-mode all \
  --output rgb \
  --seconds 10 \
  --warmup-seconds 2 \
  --gpu-label "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1) rgb-only" \
  --output-dir "$OUT_RGB" | tee "$OUT_RGB/output.log"
echo "$OUT_RGB"
```

Depth-only:

```bash
OUT_DEPTH=$(mktemp -d /tmp/rltools_rt_matrix_depth.XXXXXX)
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_depth \
  --scene all \
  --step-mode all \
  --output depth \
  --seconds 10 \
  --warmup-seconds 2 \
  --gpu-label "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1) depth-only" \
  --output-dir "$OUT_DEPTH" | tee "$OUT_DEPTH/output.log"
echo "$OUT_DEPTH"
```

## Run Standalone Renderer Benchmarks

The standalone benchmark target in `benchmark.cpp` is older and model-file based. It includes collision probe rays and does not emit the simulator matrix CSV schema. It writes output files into the current working directory:

- RGB target: `owl_test.png`, `owl_probes.bin`
- Depth target: `owl_depth.png`, `owl_depth.bin`, `owl_probes.bin`
- RGBD target: RGB and depth files plus probes

Run it from a scratch directory so the outputs do not overwrite previous runs:

```bash
MODEL="$PWD/tests/data/ProcTHOR-Train-1.glb"
OUT_STANDALONE=$(mktemp -d /tmp/rltools_rt_standalone.XXXXXX)
pushd "$OUT_STANDALONE"

"$OLDPWD/build/src/rendering/raytracing/benchmark/rendering_raytracing_benchmark" \
  --model "$MODEL" | tee benchmark_rgb.log

"$OLDPWD/build/src/rendering/raytracing/benchmark/rendering_raytracing_benchmark_depth" \
  --model "$MODEL" | tee benchmark_depth.log

"$OLDPWD/build/src/rendering/raytracing/benchmark/rendering_raytracing_benchmark_rgbd" \
  --model "$MODEL" | tee benchmark_rgbd.log

popd
echo "$OUT_STANDALONE"
```

## Common Options

Simulator matrix options:

```text
--scene all|20_objects|procthor
--step-mode all|render_only|render_physics
--output all|rgb|depth
--seconds <timed seconds per row>
--iterations <fixed iterations per row>
--warmup-seconds <warmup seconds per row>
--warmup-iterations <count>
--sync-interval <count>
--cooldown-seconds <seconds>          # sweep target only
--sweep-config-start <index>          # sweep target only
--sweep-config-count <count>          # sweep target only
--seed <camera orientation seed>
--orientation-mode uniform_so3|random_yaw_pitch|look_at_scene_jitter
--gpu-label <label for CSV rows and PNG names>
--output-dir <directory for PNGs and output.log>
--procthor-path <path/to/ProcTHOR-Train-1.glb>
```

For paper-style numbers, keep the same `--seconds`, `--warmup-seconds`, `--seed`, FOV, resolution, and output settings across all simulator rows.
