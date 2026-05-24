# Raytracing Benchmarks

This directory contains the benchmark targets for the OptiX raytracing renderer.
The simulator matrix target is the preferred benchmark for simulator-paper tables because it emits CSV rows and stitched PNG verification images.

The current shared renderer FOV is 80 degrees. AA and motion blur are disabled in the benchmark targets documented here. The simulator matrix uses the fixed ProcTHOR camera position `[-3.92, -5.67, 1.0]` and deterministic random camera-orientation jitter from `--seed`. The `20_objects` scene uses the shared `canonical_staggered_v1` staggered spatial layout; sphere entries from the canonical layout are approximated as equal-sided boxes in this raytracing-only benchmark path.

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
  rendering_raytracing_sim_benchmark_high_fidelity \
  rendering_raytracing_sim_benchmark_fast_flat \
  rendering_raytracing_sim_benchmark_rgb \
  rendering_raytracing_sim_benchmark_depth \
  -j5
```

## Run Simulator Matrix Benchmarks

The default timed duration is 10 seconds per row and the default warmup is 2 seconds. PNG stitching and CSV logging happen after the timed section.

Low-fidelity matrix:

```bash
OUT_LOW=$(mktemp -d /tmp/rltools_rt_matrix_low.XXXXXX)
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark \
  --scene all \
  --step-mode all \
  --output all \
  --seconds 10 \
  --warmup-seconds 2 \
  --gpu-label "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1) low fidelity" \
  --output-dir "$OUT_LOW" | tee "$OUT_LOW/output.log"
echo "$OUT_LOW"
```

High-fidelity matrix:

```bash
OUT_HIGH=$(mktemp -d /tmp/rltools_rt_matrix_high.XXXXXX)
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_high_fidelity \
  --scene all \
  --step-mode all \
  --output all \
  --seconds 10 \
  --warmup-seconds 2 \
  --gpu-label "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1) high fidelity" \
  --output-dir "$OUT_HIGH" | tee "$OUT_HIGH/output.log"
echo "$OUT_HIGH"
```

Fast-flat matrix:

```bash
OUT_FAST=$(mktemp -d /tmp/rltools_rt_matrix_fast.XXXXXX)
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_fast_flat \
  --scene all \
  --step-mode all \
  --output all \
  --seconds 10 \
  --warmup-seconds 2 \
  --gpu-label "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1) fast flat" \
  --output-dir "$OUT_FAST" | tee "$OUT_FAST/output.log"
echo "$OUT_FAST"
```

Each matrix covers:

| Axis | Values |
|---|---|
| Scene | `20_objects`, `procthor` |
| Output | `rgb`, `depth` |
| Step mode | `render_only`, `render_physics` |
| Fidelity | basic target, high-fidelity target, fast-flat target |

The benchmark writes one stitched PNG per row and CSV rows prefixed with `csv_result`.

Extract table rows:

```bash
rg '^csv_result' "$OUT_LOW/output.log"
rg '^csv_result' "$OUT_HIGH/output.log"
rg '^csv_result' "$OUT_FAST/output.log"
```

List verification images:

```bash
ls -lh "$OUT_LOW"/*.png
ls -lh "$OUT_HIGH"/*.png
ls -lh "$OUT_FAST"/*.png
```

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
--seed <camera orientation seed>
--gpu-label <label for CSV rows and PNG names>
--output-dir <directory for PNGs and output.log>
--procthor-path <path/to/ProcTHOR-Train-1.glb>
```

For paper-style numbers, keep the same `--seconds`, `--warmup-seconds`, `--seed`, FOV, resolution, and output settings across all simulator rows.
