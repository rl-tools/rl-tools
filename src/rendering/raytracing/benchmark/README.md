# Raytracing benchmarks

The simulator benchmark supports OptiX, Metal, and Vulkan through the configured renderer backend. Its default workload matches `python -m hyperdrone.examples.benchmark`: ProcTHOR-Train-1, 4096 static cameras at 64×64, medium RGB, 80° FOV, no probes/AA/motion blur, seed 0, 2 seconds of warmup, and 10 seconds of measurement with synchronization every 10 launches. Python and C++ sample the same orientation distribution using different RNGs.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target rendering_raytracing_sim_benchmark_medium -j5
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_medium
```

`AUTO` selects the available backend. Use `-DRL_TOOLS_RENDERING_RAYTRACING_BACKEND=METAL`, `VULKAN`, or `OPTIX` to require one explicitly. OptiX requires CUDA; Vulkan requires headers and `glslangValidator`, plus a device with ray-query support at runtime. Vulkan can run on lavapipe for correctness testing. Neither Metal nor Vulkan requires CUDA.

The default scene resolves `conta:75d1d67a17ecd16bc3a214baa895240da8060df7`, the same content hash as the Python example; `--procthor-path` accepts a file or a conta reference. The procedural `--scene 20_objects` case needs no downloaded scene.

## Targets and cases

Canonical targets are `rendering_raytracing_sim_benchmark_{medium,high,very_high,low}`, `rendering_raytracing_sim_benchmark_sweep`, and `rendering_raytracing_procthor_frame_{medium,high,very_high,low}`. They are built explicitly. RGB and depth share each executable; select `--output rgb|depth|all`.

The render matrix is two scenes × two outputs:

```bash
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_medium \
  --scene all --output all --output-dir /tmp/rltools-render-matrix
```

Set the compile-time camera count with `-DRL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS=16` for small correctness runs. The normal resolution is 64×64; frame tools use one camera at 1000×1000. Target configuration is generated from `simulator_matrix_config.h.in` by CMake.

The sweep covers resolutions 64–2048, medium/high/low RGB and depth, with AA off and 2×2 AA. Camera count scales to preserve the normal target's pixel count, with a minimum of one camera. `--scene all --output all` produces 96 records. Configuration indices follow the selected outputs, with scenes sharing each configuration index.

```bash
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_sweep \
  --scene all --output all --output-dir /tmp/rltools-render-sweep
```

For a slice, use `--sweep-config-start` and `--sweep-config-count`; `--cooldown-seconds` defaults to 10 between configurations.

Frame tools accept a position and either a forward vector, look-at point, or quaternion, as shown by `--help`. With no pose arguments, they look from `[-3.92, -5.67, 1.0]` toward the scene center. `--output-png` requires a single output; otherwise artifact names are generated.

## Measurement and results

Warmup completes before timing. Even `--warmup-seconds 0` primes the renderer once. `--iterations N` selects a fixed count instead of duration; the final batch is drained even when it is shorter than `--sync-interval`. Scene loading, image saving, and reporting are outside the measured interval. Vulkan currently waits for the previous submission in each `render_launch`; equal sync intervals do not imply equal queue depth across backends.

Normal, sweep, and frame tools emit one `render-v1` JSON object per result on its own line. Numbers and booleans retain their JSON types; camera vectors are arrays, with unavailable forward/up vectors represented by `null`. Records identify the workload, backend, selected device, scene source, scene content identity, camera configuration, sampling, timing, and verification PNG. Ray throughput counts primary samples; shading may trace additional rays. Frame measurements have workload `frame`; benchmark measurements have workload `render_only`. Extract JSON Lines with `rg '^\{' output.log`. Throughput and elapsed time are also logged to metra when `METRA_URL` is set. Backend, device, workload, scene reference, profile, resolution, camera count, orientation, seed, and sync interval distinguish metric series.

This replaces the old CUDA-specific CSV/JSON layouts. The unused-camera `render_physics` workload and its CUDA bridge, redundant target aliases, separate RGB/depth targets, `--step-mode`, `--gpu-label`, `--warmup-iterations`, and runtime `--num-envs`/`--resolution` options have been removed. Physics benchmarking requires its own execution contract.

## Validation

After building the medium benchmark, sweep, and medium frame targets:

```bash
.venv/bin/python tests/src/rendering/raytracing/benchmark.py --build-dir build --backend metal --with-procthor
```

Select the matching backend and omit `--with-procthor` for procedural-only checks. Each subprocess has a 20-second timeout. Vulkan CI uses 16 cameras on lavapipe; production defaults remain 4096.
